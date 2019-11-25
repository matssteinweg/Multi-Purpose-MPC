import numpy as np
import cvxpy as cp


##################
# MPC Controller #
##################

class MPC:
    def __init__(self, model, T, Q, R, Qf, StateConstraints, InputConstraints,
                 Reference):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param T: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param Qf: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param Reference: reference values for state variables
        """

        # Parameters
        self.T = T  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.Qf = Qf  # weight matrix terminal

        # Model
        self.model = model

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Reference
        self.reference = Reference

        # Current control and prediction
        self.current_control = None
        self.current_prediction = None

        # Initialize Optimization Problem
        self.problem = self._init_problem()

    def _init_problem(self):
        """
        Initialize parametrized optimization problem to be solved at each
        time step.
        """

        # Instantiate optimization variables
        self.x = cp.Variable((self.model.n_states+1, self.T + 1))
        self.u = cp.Variable((2, self.T))

        # Instantiate optimization parameters
        self.kappa = cp.Parameter(self.T+1)
        self.x_0 = cp.Parameter(self.model.n_states+1, 1)
        self.A = cp.Parameter(self.model.A.shape)
        self.B = cp.Parameter(self.model.B.shape)

        # Initialize cost
        cost = 0

        # Initialize constraints
        constraints = [self.x[:, 0] == self.x_0]

        for t in range(self.T):

            # set dynamic constraints
            constraints += [self.x[:-1, t + 1] == self.A[:-1, :]
                            @ self.x[:, t] + self.B[:-1, :] @ self.u[:, t],
                            self.x[-1, t + 1] == self.kappa[t+1]]

            # set input constraints
            inputs = ['D', 'delta']
            for input_name, constraint in self.input_constraints.items():
                input_id = inputs.index(input_name)
                if constraint[0] is not None:
                    constraints.append(-self.u[input_id, t] <= -constraint[0])
                if constraint[1] is not None:
                    constraints.append(self.u[input_id, t] <= constraint[1])

            # Set state constraints
            for state_name, constraint in self.state_constraints.items():
                state_id = self.model.spatial_state.list_states(). \
                    index(state_name)
                if constraint[0] is not None:
                    constraints.append(-self.x[state_id, t] <= -constraint[0])
                if constraint[1] is not None:
                    constraints.append(self.x[state_id, t] <= constraint[1])

            # update cost function for states
            for state_name, state_reference in self.reference.items():
                state_id = self.model.spatial_state.list_states(). \
                    index(state_name)
                cost += cp.norm(self.x[state_id, t] - state_reference, 2) * self.Q[
                    state_id, state_id]

            # update cost function for inputs
            cost += cp.norm(self.u[0, t], 2) * self.R[0, 0]
            cost += cp.norm(self.u[1, t], 2) * self.R[1, 1]

        # set state constraints
        for state_name, constraint in self.state_constraints.items():
            state_id = self.model.spatial_state.list_states(). \
                index(state_name)
            if constraint[0] is not None:
                constraints.append(-self.x[state_id, self.T] <= -constraint[0])
            if constraint[1] is not None:
                constraints.append(self.x[state_id, self.T] <= constraint[1])

        # update cost function for states
        for state_name, state_reference in self.reference.items():
            state_id = self.model.spatial_state.list_states(). \
                index(state_name)
            cost += cp.norm(self.x[state_id, self.T] - state_reference, 2) * \
                    self.Qf[state_id, state_id]

        # sums problem objectives and concatenates constraints.
        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem

    def get_control(self):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # get current waypoint curvature
        kappa_ref = [wp.kappa for wp in self.model.reference_path.waypoints
        [self.model.wp_id:self.model.wp_id+self.T+1]]

        # Instantiate optimization parameters
        self.kappa.value = kappa_ref
        self.x_0.value = np.array(self.model.spatial_state[:] + [kappa_ref[0]])
        self.A.value = self.model.A
        self.B.value = self.model.B

        # Solve optimization problem
        self.problem.solve(solver=cp.ECOS, warm_start=True)

        # Store computed control signals and associated prediction
        try:
            self.current_control = self.u.value
            self.current_prediction = self.update_prediction(self.x.value)
        except TypeError:
            print('No solution found!')
            exit(1)

        # RCH - get first control signal
        D_0 = self.u.value[0, 0]
        delta_0 = self.u.value[1, 0]

        return D_0, delta_0

    def update_prediction(self, spatial_state_prediction):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # get current waypoint ID
        wp_id_ = np.copy(self.model.wp_id)

        for t in range(self.T):
            associated_waypoint = self.model.reference_path.waypoints[wp_id_+t]
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[:, t])
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred


import numpy as np
import cvxpy as cp

##################
# MPC Controller #
##################

class MPC:
    def __init__(self, model, T, Q, R, Qf, StateConstraints, InputConstraints, Reference):

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

    def get_control(self):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # get current waypoint curvature
        kappa_ref = self.model.reference_path.waypoints[self.model.wp_id].kappa

        # Set initial state
        x_0 = np.array(self.model.spatial_state[:] + [kappa_ref])

        # Instantiate optimization variables
        x = cp.Variable((len(x_0), self.T + 1))
        u = cp.Variable((2, self.T))

        # Instantiate optimization parameters
        kappa = cp.Parameter(value=kappa_ref)

        # Initialize cost
        cost = 0

        # Initialize constraints
        constraints = [x[:, 0] == x_0]

        for t in range(self.T):

            # update kappa value for next time step
            kappa.value = self.model.reference_path.waypoints[
                              self.model.wp_id + 1 + t].kappa - kappa_ref

            # set dynamic constraints
            constraints += [x[:-1, t + 1] == self.model.A[:-1, :]
                            @ x[:, t] + self.model.B[:-1, :] @ u[:, t],
                            x[-1, t + 1] == kappa]

            # set input constraints
            inputs = ['D', 'delta']
            for input_name, constraint in self.input_constraints.items():
                input_id = inputs.index(input_name)
                if constraint[0] is not None:
                    constraints.append(-u[input_id, t] <= -constraint[0])
                if constraint[1] is not None:
                    constraints.append(u[input_id, t] <= constraint[1])

            # Set state constraints
            for state_name, constraint in self.state_constraints.items():
                state_id = self.model.spatial_state.list_states().\
                    index(state_name)
                if constraint[0] is not None:
                    constraints.append(-x[state_id, t] <= -constraint[0])
                if constraint[1] is not None:
                    constraints.append(x[state_id, t] <= constraint[1])

            # update cost function for states
            for state_name, state_reference in self.reference.items():
                state_id = self.model.spatial_state.list_states().\
                    index(state_name)
                cost += cp.norm(x[state_id, t] - state_reference, 2) * self.Q[state_id, state_id]

            # update cost function for inputs
            cost += cp.norm(u[0, t], 2) * self.R[0, 0]
            cost += cp.norm(u[1, t], 2) * self.R[1, 1]

        # set state constraints
        for state_name, constraint in self.state_constraints.items():
            state_id = self.model.spatial_state.list_states(). \
                index(state_name)
            if constraint[0] is not None:
                constraints.append(-x[state_id, self.T] <= -constraint[0])
            if constraint[1] is not None:
                constraints.append(x[state_id, self.T] <= constraint[1])

        # update cost function for states
        for state_name, state_reference in self.reference.items():
            state_id = self.model.spatial_state.list_states(). \
                    index(state_name)
            cost += cp.norm(x[state_id, self.T] - state_reference, 2) * \
                    self.Qf[state_id, state_id]

        # sums problem objectives and concatenates constraints.
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.ECOS)

        # Store computed control signals and associated prediction
        try:
            self.current_control = u.value
            self.current_prediction = self.update_prediction(x.value)
        except TypeError:
            print('No solution found!')
            exit(1)

        # RCH - get first control signal
        D_0 = u.value[0, 0]
        delta_0 = u.value[1, 0]

        return D_0, delta_0

    def update_prediction(self, spatial_state_prediction):

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


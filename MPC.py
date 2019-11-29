import numpy as np
import cvxpy as cp
import osqp
import scipy as sp
from scipy import sparse

##################
# MPC Controller #
##################

class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param T: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param Reference: reference values for state variables
        """

        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal

        # Model
        self.model = model

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Current control and prediction
        self.current_prediction = None

        # Initialize Optimization Problem
        self.problem = self._init_problem()

    def _init_problem(self):
        """
        Initialize parametrized optimization problem to be solved at each
        time step.
        """

        # number of input and state variables
        nx = self.model.n_states
        nu = 1

        # system matrices
        self.A = cp.Parameter(shape=(nx, nx*self.N))
        self.B = cp.Parameter(shape=(nx, nu*self.N))
        self.A.value = np.zeros(self.A.shape)
        self.B.value = np.zeros(self.B.shape)

        # reference values
        xr = np.array([0., 0., -1.0])
        self.ur = cp.Parameter((nu, self.N))
        self.ur.value = np.zeros(self.ur.shape)

        # constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # initial state
        self.x_init = cp.Parameter(self.model.n_states)

        # Define problem
        self.u = cp.Variable((nu, self.N))
        self.x = cp.Variable((nx, self.N + 1))
        objective = 0
        constraints = [self.x[:, 0] == self.x_init]
        for n in range(self.N):
            objective += cp.quad_form(self.x[:, n] - xr, self.Q) + cp.quad_form(self.u[:, n] - self.ur[:, n], self.R)
            constraints += [self.x[:, n + 1] == self.A[:, n*nx:n*nx+nx] * self.x[:, n]
                            + self.B[:, n*nu] * (self.u[:, n] - self.ur[:, n])]
            constraints += [umin <= self.u[:, n], self.u[:, n] <= umax]
        objective += cp.quad_form(self.x[:, self.N] - xr, self.QN)
        constraints += [xmin <= self.x[:, self.N], self.x[:, self.N] <= xmax]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        return problem

    def get_control(self, v):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        nx = self.model.n_states
        nu = 1

        for n in range(self.N):
            current_waypoint = self.model.reference_path.waypoints[self.model.wp_id+n]
            next_waypoint = self.model.reference_path.waypoints[
                self.model.wp_id + n + 1]
            delta_s = next_waypoint - current_waypoint
            kappa_r = current_waypoint.kappa
            self.A.value[:, n*nx:n*nx+nx], self.B.value[:, n*nu:n*nu+nu] = self.model.linearize(v, kappa_r, delta_s)
            self.ur.value[:, n] = kappa_r

        self.x_init.value = np.array(self.model.spatial_state[:])
        self.problem.solve(solver=cp.OSQP, verbose=True)

        self.current_prediction = self.update_prediction(self.x.value)
        delta = np.arctan(self.u.value[0, 0] * self.model.l)

        return delta

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
        print('#########################')

        for n in range(self.N):
            associated_waypoint = self.model.reference_path.waypoints[self.model.wp_id+n]
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[:, n])
            print('delta: ', np.arctan(self.u.value[0, n] * self.model.l))
            print('e_y: ', spatial_state_prediction[0, n])
            print('e_psi: ', spatial_state_prediction[1, n])
            print('t: ', spatial_state_prediction[2, n])
            print('+++++++++++++++++++++++')

            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred


class MPC_OSQP:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param T: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param Reference: reference values for state variables
        """

        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal

        # Model
        self.model = model

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Current control and prediction
        self.current_prediction = None

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

    def _init_problem(self, v):
        """
        Initialize optimization problem for current time step.
        """

        # Number of state and input variables
        nx = self.model.n_states
        nu = 1

        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # LTV System Matrices
        A = np.zeros((nx * (self.N + 1), nx * (self.N + 1)))
        B = np.zeros((nx * (self.N + 1), nu * (self.N)))
        # Reference vector for state and input variables
        ur = np.zeros(self.N)
        xr = np.array([0.0, 0.0, -1.0])
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(self.N * nx)

        # Iterate over horizon
        for n in range(self.N):

            # Get information about current waypoint
            current_waypoint = self.model.reference_path.waypoints[
                self.model.wp_id + n]
            next_waypoint = self.model.reference_path.waypoints[
                self.model.wp_id + n + 1]
            delta_s = next_waypoint - current_waypoint
            kappa_r = current_waypoint.kappa

            # Compute LTV matrices
            A_lin, B_lin = self.model.linearize(v, kappa_r, delta_s)
            A[nx + n * nx:n * nx + 2 * nx, n * nx:n * nx + nx] = A_lin
            B[nx + n * nx:n * nx + 2 * nx, n * nu:n * nu + nu] = B_lin

            # Set kappa_r to reference for input signal
            ur[n] = kappa_r
            # Compute equality constraint offset (B*ur)
            uq[n * nx:n * nx + nx] = B_lin[:, 0] * kappa_r

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(self.N + 1),
                         -sparse.eye(nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * nx + self.N * nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([np.kron(np.ones(self.N + 1), xmin),
             np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([np.kron(np.ones(self.N + 1), xmax),
             np.kron(np.ones(self.N), umax)])
        # Get upper and lower bound vectors for inequality constraints
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
             sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        q = np.hstack(
            [np.kron(np.ones(self.N), -self.Q.dot(xr)), -self.QN.dot(xr),
             -self.R.A[0, 0] * ur])

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self, v):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # Number of state variables
        nx = self.model.n_states

        # Initialize optimization problem
        self._init_problem(v)

        # Solve optimization problem
        dec = self.optimizer.solve()
        x = np.reshape(dec.x[:(self.N+1)*nx], (self.N+1, nx))
        u = np.arctan(dec.x[-self.N] * self.model.l)
        self.current_prediction = self.update_prediction(u, x)

        return u

    def update_prediction(self, u, spatial_state_prediction):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # get current waypoint ID
        print('#########################')

        for n in range(self.N):
            associated_waypoint = self.model.reference_path.waypoints[self.model.wp_id+n]
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])
            print('delta: ', u)
            print('e_y: ', spatial_state_prediction[n, 0])
            print('e_psi: ', spatial_state_prediction[n, 1])
            print('t: ', spatial_state_prediction[n, 2])
            print('+++++++++++++++++++++++')

            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred


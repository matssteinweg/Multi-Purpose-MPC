import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# Colors
PREDICTION = '#BA4A00'

##################
# MPC Controller #
##################


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        """

        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal

        # Model
        self.model = model

        # Dimensions
        self.nx = self.model.n_states
        self.nu = 2

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Maximum lateral acceleration
        self.ay_max = ay_max

        # Current control and prediction
        self.current_prediction = None

        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0

        # Current control signals
        self.current_control = np.zeros((self.nu*self.N))

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

    def _init_problem(self):
        """
        Initialize optimization problem for current time step.
        """

        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # LTV System Matrices
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
        # Reference vector for state and input variables
        ur = np.zeros(self.nu*self.N)
        xr = np.zeros(self.nx*(self.N+1))
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(self.N * self.nx)
        # Dynamic state constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Dynamic input constraints
        umax_dyn = np.kron(np.ones(self.N), umax)
        # Get curvature predictions from previous control signals
        kappa_pred = np.tan(np.array(self.current_control[3::] +
                                     self.current_control[-1:])) / self.model.length

        # Iterate over horizon
        for n in range(self.N):

            # Get information about current waypoint
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            v_ref = current_waypoint.v_ref

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[(n+1) * self.nx: (n+2)*self.nx, n * self.nx:(n+1)*self.nx] = A_lin
            B[(n+1) * self.nx: (n+2)*self.nx, n * self.nu:(n+1)*self.nu] = B_lin

            # Set reference for input signal
            ur[n*self.nu:(n+1)*self.nu] = np.array([v_ref, kappa_ref])
            # Compute equality constraint offset (B*ur)
            uq[n * self.nx:(n+1)*self.nx] = B_lin.dot(np.array
                                            ([v_ref, kappa_ref])) - f

            # Constrain maximum speed based on predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            if vmax_dyn < umax_dyn[self.nu*n]:
                umax_dyn[self.nu*n] = vmax_dyn

        # Compute dynamic constraints on e_y
        ub, lb, _ = self.model.reference_path.update_path_constraints(
                    self.model.wp_id+1, self.N, 2*self.model.safety_margin,
            self.model.safety_margin)
        xmin_dyn[0] = self.model.spatial_state.e_y
        xmax_dyn[0] = self.model.spatial_state.e_y
        xmin_dyn[self.nx::self.nx] = lb
        xmax_dyn[self.nx::self.nx] = ub

        # Set reference for state as center-line of drivable area
        xr[self.nx::self.nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(self.N + 1),
                         -sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([xmin_dyn,
                           np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
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
            [-np.tile(np.diag(self.Q.A), self.N) * xr[:-self.nx],
             -self.QN.dot(xr[-self.nx:]),
             -np.tile(np.diag(self.R.A), self.N) * ur])

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # Number of state variables
        nx = self.model.n_states
        nu = 2

        # Update current waypoint
        self.model.get_current_waypoint()

        # Update spatial state
        self.model.spatial_state = self.model.t2s(reference_state=
            self.model.temporal_state, reference_waypoint=
            self.model.current_waypoint)

        # Initialize optimization problem
        self._init_problem()

        # Solve optimization problem
        dec = self.optimizer.solve()

        try:
            # Get control signals
            control_signals = np.array(dec.x[-self.N*nu:])
            control_signals[1::2] = np.arctan(control_signals[1::2] *
                                              self.model.length)
            v = control_signals[0]
            delta = control_signals[1]

            # Update control signals
            self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[:(self.N+1)*nx], (self.N+1, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x)

            # Get current control signal
            u = np.array([v, delta])

            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0

        except:

            print('Infeasible problem. Previously predicted'
                  ' control signal used!')
            id = nu * (self.infeasibility_counter + 1)
            u = np.array(self.current_control[id:id+2])

            # increase infeasibility counter
            self.infeasibility_counter += 1

        if self.infeasibility_counter == (self.N - 1):
            print('No control signal computed!')
            exit(1)

        return u

    def update_prediction(self, spatial_state_prediction):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, self.N):
            # Get associated waypoint
            associated_waypoint = self.model.reference_path.\
                get_waypoint(self.model.wp_id+n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred

    def show_prediction(self):
        """
        Display predicted car trajectory in current axis.
        """

        if self.current_prediction is not None:
            plt.scatter(self.current_prediction[0], self.current_prediction[1],
                    c=PREDICTION, s=30)


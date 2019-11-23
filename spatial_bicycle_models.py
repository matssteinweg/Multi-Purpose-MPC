import numpy as np
from abc import ABC, abstractmethod


#########################
# Temporal State Vector #
#########################

class TemporalState:
    def __init__(self, x, y, psi, v_x, v_y):
        """
        Temporal State Vector containing car pose (x, y, psi) and velocity
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: yaw angle | [rad]
        :param v_x: velocity in x direction (car frame) | [m/s]
        :param v_y: velocity in y direction (car frame) | [m/s]
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = v_x
        self.v_y = v_y


########################
# Spatial State Vector #
########################

class SpatialState(ABC):
    """
    Spatial State Vector - Abstract Base Class.
    """

    @abstractmethod
    def __init__(self):
        pass

    def __getitem__(self, item):
        return list(vars(self).values())[item]

    def __len__(self):
        return len(vars(self))

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """

        for state_id, state in enumerate(vars(self).values()):
            vars(self)[list(vars(self).keys())[state_id]] += other[state_id]
        return self

    def list_states(self):
        """
        Return list of names of all states.
        """
        return list(vars(self).keys())


class SimpleSpatialState(SpatialState):
    def __init__(self, e_y, e_psi, v):
        """
        Simplified Spatial State Vector containing orthogonal deviation from
        reference path (e_y), difference in orientation (e_psi) and velocity
        :param e_y: orthogonal deviation from center-line | [m]
        :param e_psi: yaw angle relative to path | [rad]
        :param v: absolute velocity | [m/s]
        """
        super(SimpleSpatialState, self).__init__()

        self.e_y = e_y
        self.e_psi = e_psi
        self.v = v


class ExtendedSpatialState(SpatialState):
    def __init__(self, e_y, e_psi, v_x, v_y, omega, t):
        """
        Extended Spatial State Vector containing separate velocities in x and
        y direction, angular velocity and time
        :param e_y: orthogonal deviation from center-line | [m]
        :param e_psi: yaw angle relative to path | [rad]
        :param v_x: velocity in x direction (car frame) | [m/s]
        :param v_y: velocity in y direction (car frame) | [m/s]
        :param omega: anglular velocity of the car | [rad/s]
        :param t: simulation time| [s]
        """
        super(ExtendedSpatialState, self).__init__()

        self.e_y = e_y
        self.e_psi = e_psi
        self.v_x = v_x
        self.v_y = v_y
        self.omega = omega
        self.t = t


####################################
# Spatial Bicycle Model Base Class #
####################################

class SpatialBicycleModel(ABC):
    def __init__(self, reference_path):
        """
        Abstract Base Class for Spatial Reformulation of Bicycle Model.
        :param reference_path: reference path object to follow
        """

        # Precision
        self.eps = 1e-12

        # Reference Path
        self.reference_path = reference_path

        # Set initial distance traveled
        self.s = 0.0

        # Set initial waypoint ID
        self.wp_id = 0

        # Set initial waypoint
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        # Declare spatial state variable | Initialization in sub-class
        self.spatial_state = None

        # Declare temporal state variable | Initialization in sub-class
        self.temporal_state = None

        # Declare system matrices of linearized model | Used for MPC
        self.A, self.B = None, None

    def s2t(self, reference_waypoint=None, reference_state=None):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return x, y, psi
        """

        # Compute spatial state for current waypoint if no waypoint given
        if reference_waypoint is None and reference_state is None:

            # compute temporal state variables
            x = self.current_waypoint.x - self.spatial_state.e_y * np.sin(
                self.current_waypoint.psi)
            y = self.current_waypoint.y + self.spatial_state.e_y * np.cos(
                self.current_waypoint.psi)
            psi = self.current_waypoint.psi + self.spatial_state.e_psi

        else:

            # compute temporal state variables
            x = reference_waypoint.x - reference_state[0] * np.sin(
                reference_waypoint.psi)
            y = reference_waypoint.y + reference_state[0] * np.cos(
                reference_waypoint.psi)
            psi = reference_waypoint.psi + reference_state[1]

        return x, y, psi

    def drive(self, D, delta):
        """
        Update states of spatial bicycle model. Model drive to the next
        waypoint on the reference path.
        :param D: acceleration command | [-1, 1]
        :param delta: angular velocity | [rad]
        """

        # Get spatial derivatives
        spatial_derivatives = self.get_spatial_derivatives(D, delta)

        # Get delta_s | distance to next waypoint
        next_waypoint = self.reference_path.waypoints[self.wp_id+1]
        delta_s = next_waypoint - self.current_waypoint

        # Update spatial state (Forward Euler Approximation)
        self.spatial_state += spatial_derivatives * delta_s

        # Assert that unique projections of car pose onto path exists
        assert self.spatial_state.e_y < (1 / (self.current_waypoint.kappa +
                                              self.eps))

        # Increase waypoint ID
        self.wp_id += 1

        # Update current waypoint
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        # Update temporal_state to match spatial state
        self.temporal_state = self.s2t()

        # Update s | total driven distance along path
        self.s += delta_s

        # Linearize model around new operating point
        self.A, self.B = self.linearize()

    @abstractmethod
    def get_spatial_derivatives(self, D, delta):
        pass

    @abstractmethod
    def linearize(self):
        pass


########################
# Simple Bicycle Model #
########################

class SimpleBicycleModel(SpatialBicycleModel):
    def __init__(self, reference_path, e_y, e_psi, v):
        """
        Simplified Spatial Bicycle Model. Spatial Reformulation of Kinematic
        Bicycle Model. Uses Simplified Spatial State.
        :param reference_path: reference path model is supposed to follow
        :param e_y: deviation from reference path | [m]
        :param e_psi: heading offset from reference path | [rad]
        :param v: initial velocity | [m/s]
        """

        # Initialize base class
        super(SimpleBicycleModel, self).__init__(reference_path)

        # Constants
        self.C1 = 0.5
        self.C2 = 17.06
        self.Cm1 = 12.0
        self.Cm2 = 2.17
        self.Cr2 = 0.1
        self.Cr0 = 0.6

        # Initialize spatial state
        self.spatial_state = SimpleSpatialState(e_y, e_psi, v)

        # Initialize temporal state
        self.temporal_state = self.s2t()

        # Compute linear system matrices | Used for MPC
        self.A, self.B = self.linearize()

    def s2t(self, reference_waypoint=None, reference_state=None):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return temporal state equivalent to self.spatial_state or provided
        reference state
        """

        if reference_state is None and reference_waypoint is None:
            # Get pose information from base class implementation
            x, y, psi = super(SimpleBicycleModel, self).s2t()
            # Compute simplified velocities
            v_x = self.spatial_state.v
            v_y = 0
        else:
            # Get pose information from base class implementation
            x, y, psi = super(SimpleBicycleModel, self).s2t(reference_waypoint,
                                                            reference_state)
            v_x = reference_state[2]
            v_y = 0

        return TemporalState(x, y, psi, v_x, v_y)

    def get_temporal_derivatives(self, D, delta):
        """
        Compute relevant temporal derivatives needed for state update.
        :param D: duty-cycle of DC motor | [-1, 1]
        :param delta: steering command | [rad]
        :return: temporal derivatives of distance, angle and velocity
        """

        # Compute velocity components | Approximation for small delta
        v_x = self.spatial_state.v
        v_y = self.spatial_state.v * delta * self.C1

        # Compute velocity along waypoint direction
        v_sigma = v_x * np.cos(self.spatial_state.e_psi) - v_y * np.sin(
            self.spatial_state.e_psi)

        # Compute velocity along path
        s_dot = 1 / (1 - (self.spatial_state.e_y * self.current_waypoint.kappa)) * v_sigma

        # Compute yaw angle rate of change
        psi_dot = self.spatial_state.v * delta * self.C2

        # Compute acceleration
        v_dot = (self.Cm1 - self.Cm2 * self.spatial_state.v) * D - self.Cr2 * (
                self.spatial_state.v ** 2) - self.Cr0 - (
                self.spatial_state.v * delta) ** 2 * self.C2 * self.C1 ** 2

        return s_dot, psi_dot, v_dot

    def get_spatial_derivatives(self, D, delta):
        """
        Compute spatial derivatives of all state variables for update.
        :param D: duty-cycle of DC motor | [-1, 1]
        :param delta: steering angle | [rad]
        :return: numpy array with spatial derivatives for all state variables
        """

        # Compute temporal derivatives
        s_dot, psi_dot, v_dot = self.get_temporal_derivatives(D, delta)

        # Compute spatial derivatives
        d_e_y_d_s = (self.spatial_state.v * np.sin(self.spatial_state.e_psi)
                 + self.spatial_state.v * delta * self.C1 * np.cos(
                    self.spatial_state.e_psi)) / s_dot
        d_e_psi_d_s = psi_dot / s_dot - self.current_waypoint.kappa
        d_v_d_s = v_dot / s_dot

        return np.array([d_e_y_d_s, d_e_psi_d_s, d_v_d_s])

    def linearize(self, D=0, delta=0):
        """
        Linearize the system equations around the current state and waypoint.
        :param delta: reference steering angle | [rad]
        :param D: reference duty-cycle of DC-motor | [-1, 1]
        """

        # Get current state | operating point to linearize around
        e_y = self.spatial_state.e_y
        e_psi = self.spatial_state.e_psi
        v = self.spatial_state.v

        # Get curvature of current waypoint
        kappa = self.reference_path.waypoints[self.wp_id].kappa

        # Get delta_s
        next_waypoint = self.reference_path.waypoints[self.wp_id+1]
        delta_s = next_waypoint - self.current_waypoint

        ##############################
        # Helper Partial Derivatives #
        ##############################

        # Compute velocity components
        v_x = v
        v_y = v * delta * self.C1

        # Compute partial derivatives of s_dot w.r.t. each state variable,
        # input variable and kappa
        s_dot = 1 / (1 - e_y*kappa) * (v_x * np.cos(e_psi) - v_y * np.sin(e_psi))
        d_s_dot_d_e_y = kappa / (1-e_y*kappa)**2 * (v_x * np.cos(e_psi) - v_y * np.sin(e_psi))
        d_s_dot_d_e_psi = 1 / (1 - e_y*kappa) * (-v_x * np.sin(e_psi) - v_y * np.cos(e_psi))
        d_s_dot_d_v = 1 / (1 - e_y*kappa) * (np.cos(e_psi) - delta * self.C1 * np.sin(e_psi))
        # d_s_dot_d_D = 0
        d_s_dot_d_delta = 1 / (1 - e_y*kappa) * (- v * self.C1 * np.sin(e_psi))
        d_s_dot_d_kappa = e_y / (1-e_y*kappa)**2 * (v_x * np.cos(e_psi) - v_y * np.sin(e_psi))

        # Compute partial derivatives of v_psi w.r.t. each state variable,
        # input variable and kappa
        v_psi = (v_x * np.sin(e_psi) + v_y * np.cos(e_psi))
        # d_v_psi_d_e_y = 0
        d_v_psi_d_e_psi = v_x * np.cos(e_psi) - v_y * np.sin(e_psi)
        d_v_psi_d_v = np.sin(e_psi) + self.C1 * delta * np.cos(e_psi)
        # d_v_psi_d_D = 0
        d_v_psi_d_delta = self.C1 * v * np.cos(e_psi)
        # d_v_psi_d_kappa = 0

        # Compute partial derivatives of psi_dot w.r.t. each state variable,
        # input variable and kappa
        psi_dot = v * delta * self.C2
        # d_psi_dot_d_e_y = 0
        # d_psi_dot_d_e_psi = 0
        d_psi_dot_d_v = delta * self.C2
        # d_psi_dot_d_D = 0
        d_psi_dot_d_delta = v * self.C2
        # d_psi_dot_d_kappa = 0

        # Compute partial derivatives of v_dot w.r.t. each state variable,
        # input variable and kappa
        v_dot = (self.Cm1 - self.Cm2 * v) * D - self.Cr2 * (v ** 2) - self.Cr0 \
                - (v * delta) ** 2 * self.C2 * (self.C1 ** 2)
        # d_v_dot_d_e_y = 0
        # d_v_dot_d_e_psi = 0
        d_v_dot_d_v = -self.Cm2 * D - 2 * self.Cr2 * v - 2 * v * (delta ** 2) \
                      * self.C2 * (self.C1 ** 2)
        d_v_dot_d_D = self.Cm1 - self.Cm2 * v
        d_v_dot_d_delta = -2 * (v ** 2) * delta * self.C2 * self.C1 ** 2
        # d_v_dot_d_kappa = 0

        #############################
        # State Partial Derivatives #
        #############################

        # Use pre-computed helper derivatives to compute spatial derivatives of
        # all state variables using Quotient Rule

        # Compute partial derivatives of e_y w.r.t. each state variable,
        # input variable and kappa
        # e_y = v_psi / s_dot
        d_e_y_d_e_y = - d_s_dot_d_e_y * v_psi / (s_dot**2)
        d_e_y_d_e_psi = (d_v_psi_d_e_psi * s_dot - d_s_dot_d_e_psi * v_psi) / (s_dot**2)
        d_e_y_d_v = (d_v_psi_d_v * s_dot - d_s_dot_d_v * v_psi) / (s_dot**2)
        d_e_y_d_D = 0
        d_e_y_d_delta = (d_v_psi_d_delta * s_dot - d_s_dot_d_delta * v_psi) / (s_dot**2)
        d_e_y_d_kappa = - d_s_dot_d_kappa * v_psi / (s_dot**2)

        # Compute partial derivatives of e_psi w.r.t. each state variable,
        # input variable and kappa
        # e_psi = psi_dot / s_dot - kappa
        d_e_psi_d_e_y = - d_s_dot_d_e_y * psi_dot / (s_dot**2)
        d_e_psi_d_e_psi = - d_s_dot_d_e_psi * psi_dot / (s_dot**2)
        d_e_psi_d_v = (d_psi_dot_d_v * s_dot - d_s_dot_d_v * psi_dot) / (s_dot**2)
        d_e_psi_d_D = 0
        d_e_psi_d_delta = (d_psi_dot_d_delta * s_dot - d_s_dot_d_delta * psi_dot) / (s_dot**2)
        d_e_psi_d_kappa = - d_s_dot_d_kappa * psi_dot / (s_dot**2) - 1

        # Compute partial derivatives of v w.r.t. each state variable,
        # input variable and kappa
        # v = v_dot / s_dot
        d_v_d_e_y = - d_s_dot_d_e_y * v_dot / (s_dot**2)
        d_v_d_e_psi = - d_s_dot_d_e_psi * v_dot / (s_dot**2)
        d_v_d_v = (d_v_dot_d_v * s_dot - d_s_dot_d_v * v_dot) / (s_dot**2)
        d_v_d_D = d_v_dot_d_D * s_dot / (s_dot**2)
        d_v_d_delta = (d_v_dot_d_delta * s_dot - d_s_dot_d_delta * v_dot) / (s_dot**2)
        d_v_d_kappa = - d_s_dot_d_kappa * v_dot / (s_dot**2)

        #############
        # Jacobians #
        #############

        # Construct Jacobian Matrix
        a_1 = np.array([d_e_y_d_e_y,    d_e_y_d_e_psi,      d_e_y_d_v,      d_e_y_d_kappa])
        a_2 = np.array([d_e_psi_d_e_y,  d_e_psi_d_e_psi,    d_e_psi_d_v,    d_e_psi_d_kappa])
        a_3 = np.array([d_v_d_e_y,      d_v_d_e_psi,        d_v_d_v,        d_v_d_kappa])

        b_1 = np.array([d_e_y_d_D, d_e_y_d_delta])
        b_2 = np.array([d_e_psi_d_D, d_e_psi_d_delta])
        b_3 = np.array([d_v_d_D, d_v_d_delta])

        # Add extra row for kappa | Allows for updating kappa during MPC
        # optimization
        a_4 = np.array([0, 0, 0, 0])
        b_4 = np.array([0, 0])

        Ja = np.stack((a_1, a_2, a_3, a_4), axis=0)
        Jb = np.stack((b_1, b_2, b_3, b_4), axis=0)

        ###################
        # System Matrices #
        ###################

        # Construct system matrices from Jacobians. Multiply by sampling
        # distance delta_s + add identity matrix (Forward Euler Approximation)
        A = Ja * delta_s + np.identity(Ja.shape[1])
        B = Jb * delta_s

        return A, B


##########################
# Extended Bicycle Model #
##########################

class ExtendedBicycleModel(SpatialBicycleModel):
    def __init__(self, reference_path, e_y, e_psi, v_x, v_y, omega, t):
        """
        Construct spatial bicycle model.
        :param e_y: initial deviation from reference path | [m]
        :param e_psi: initial heading offset from reference path | [rad]
        :param v: initial velocity | [m/s]
        :param reference_path: reference path model is supposed to follow
        """
        super(ExtendedBicycleModel, self).__init__(reference_path)

        # Constants
        self.m = 0.041
        self.Iz = 27.8e-6
        self.lf = 0.029
        self.lr = 0.033

        self.Cm1 = 0.287
        self.Cm2 = 0.0545
        self.Cr2 = 0.0518
        self.Cr0 = 0.00035

        self.Br = 3.3852
        self.Cr = 1.2691
        self.Dr = 0.1737
        self.Bf = 2.579
        self.Cf = 1.2
        self.Df = 0.192

        # Spatial state
        self.spatial_state = ExtendedSpatialState(e_y, e_psi, v_x, v_y, omega, t)

        # Temporal state
        self.temporal_state = self.s2t()

        # Linear System Matrices
        self.A, self.B = self.linearize()

    def s2t(self, reference_waypoint=None, predicted_state=None):
        """
        Convert spatial state to temporal state
        :return temporal state equivalent to self.spatial_state
        """

        # compute velocity information
        if predicted_state is None and reference_waypoint is None:
            # get information from base class
            x, y, psi = super(ExtendedBicycleModel, self).s2t()
            v_x = self.spatial_state.v_x
            v_y = self.spatial_state.v_y
        else:
            # get information from base class
            x, y, psi = super(ExtendedBicycleModel, self).s2t(reference_waypoint,
                                                            predicted_state)
            v_x = predicted_state[2]
            v_y = predicted_state[3]

        return TemporalState(x, y, psi, v_x, v_y)

    def get_forces(self, delta, D):
        """
        Compute forces required for temporal derivatives of v_x and v_y
        :param delta:
        :param D:
        :return:
        """

        F_rx = (self.Cm1 - self.Cm2 * self.spatial_state.v_x) * D - self.Cr0 - self.Cr2 * self.spatial_state.v_x ** 2

        alpha_f = - np.arctan2(self.spatial_state.omega*self.lf + self.spatial_state.v_y, self.spatial_state.v_x) + delta
        F_fy = self.Df * np.sin(self.Cf*np.arctan(self.Bf*alpha_f))

        alpha_r = np.arctan2(self.spatial_state.omega*self.lr - self.spatial_state.v_y, self.spatial_state.v_x)
        F_ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br*alpha_r))

        return F_rx, F_fy, F_ry, alpha_f, alpha_r

    def get_temporal_derivatives(self, delta, F_rx, F_fy, F_ry):
        """
        Compute temporal derivatives needed for state update.
        :param delta: steering command
        :param D: duty-cycle of DC motor
        :return: temporal derivatives of distance, angle and velocity
        """

        # velocity along path
        s_dot = 1 / (1 - (self.spatial_state.e_y * self.current_waypoint.kappa)) \
                * (self.spatial_state.v_x * np.cos(self.spatial_state.e_psi)
                   + self.spatial_state.v_y * np.sin(self.spatial_state.e_psi))

        # velocity in x and y direction
        v_x_dot = (F_rx - F_fy * np.sin(delta) + self.m * self.spatial_state.
                   v_y * self.spatial_state.omega) / self.m
        v_y_dot = (F_ry + F_fy * np.cos(delta) - self.m * self.spatial_state.
                   v_x * self.spatial_state.omega) / self.m

        # omega dot
        omega_dot = (F_fy * self.lf * np.cos(delta) - F_ry * self.lr) / self.Iz

        return s_dot, v_x_dot, v_y_dot, omega_dot

    def get_spatial_derivatives(self, delta, D):
        """
        Compute spatial derivatives of all state variables for update.
        :param delta: steering angle
        :param psi_dot: heading rate of change
        :param s_dot: velocity along path
        :param v_dot: acceleration
        :return: spatial derivatives for all state variables
        """

        # get required forces
        F_rx, F_fy, F_ry, _, _ = self.get_forces(delta, D)

        # Compute state derivatives
        s_dot, v_x_dot, v_y_dot, omega_dot = \
            self.get_temporal_derivatives(delta, F_rx, F_fy, F_ry)


        d_e_y = (self.spatial_state.v_x * np.sin(self.spatial_state.e_psi)
                 + self.spatial_state.v_y * np.cos(self.spatial_state.e_psi)) \
                / (s_dot + self.eps)
        d_e_psi = (self.spatial_state.omega / (s_dot + self.eps) - self.current_waypoint.kappa)

        d_v_x = v_x_dot / (s_dot + self.eps)
        d_v_y = v_y_dot / (s_dot + self.eps)
        d_omega = omega_dot / (s_dot + self.eps)
        d_t = 1 / (s_dot + self.eps)

        return np.array([d_e_y, d_e_psi, d_v_x, d_v_y, d_omega, d_t])

    def linearize(self, delta=0, D=0):
        """
        Linearize the system equations around the current state and waypoint.
        :param delta: reference steering angle
        :param D: reference dutycycle
        """

        # get current state
        e_y = self.spatial_state.e_y
        e_psi = self.spatial_state.e_psi
        v_x = self.spatial_state.v_x
        v_y = self.spatial_state.v_y
        omega = self.spatial_state.omega
        t = self.spatial_state.t

        # get information about current waypoint
        kappa = self.reference_path.waypoints[self.wp_id].kappa

        # get delta_s
        next_waypoint = self.reference_path.waypoints[self.wp_id + 1]
        delta_s = next_waypoint - self.current_waypoint

        # get temporal derivatives
        F_rx, F_fy, F_ry, alpha_f, alpha_r = self.get_forces(delta, D)
        s_dot, v_x_dot, v_y_dot, omega_dot = self.\
            get_temporal_derivatives(delta, F_rx, F_fy, F_ry)

        ##############################
        # Forces Partial Derivatives #
        ##############################

        d_alpha_f_d_v_x = 1 / (1 + ((omega * self.lf + v_y) / v_x)**2) * (omega * self.lf + v_y) / (v_x**2)
        d_alpha_f_d_v_y = - 1 / (1 + ((omega * self.lf + v_y) / v_x)**2) / v_x
        d_alpha_f_d_omega = - 1 / (1 + ((omega * self.lf + v_y) / v_x)**2) * (self.lf / v_x)
        d_alpha_f_d_delta = 1

        d_alpha_r_d_v_x = - 1 / (1 + ((omega * self.lr - v_y) / v_x)**2) * (omega * self.lr - v_y) / (v_x**2)
        d_alpha_r_d_v_y = - 1 / (1 + ((omega * self.lr - v_y) / v_x)**2) / v_x
        d_alpha_r_d_omega = 1 / (1 + ((omega * self.lr - v_y) / v_x)**2) * (self.lr * v_x)

        d_F_fy_d_v_x = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f)**2) * self.Bf * d_alpha_f_d_v_x
        d_F_fy_d_v_y = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f)**2) * self.Bf * d_alpha_f_d_v_y
        d_F_fy_d_omega = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f)**2) * self.Bf * d_alpha_f_d_omega
        d_F_fy_d_delta = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f)**2) * self.Bf * d_alpha_f_d_delta

        d_F_ry_d_v_x = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) * self.Cr / (1 + (self.Br * alpha_r)**2) * self.Br * d_alpha_r_d_v_x
        d_F_ry_d_v_y = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) * self.Cr / (1 + (self.Br * alpha_r)**2) * self.Br * d_alpha_r_d_v_y
        d_F_ry_d_omega = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) * self.Cr / (1 + (self.Br * alpha_r)**2) * self.Br * d_alpha_r_d_omega

        d_F_rx_d_v_x = - self.Cm2 * D - 2 * self.Cr2 * v_x
        d_F_rx_d_D = self.Cm1 - self.Cm2 * v_x

        ##############################
        # Helper Partial Derivatives #
        ##############################

        d_s_dot_d_e_y = kappa / (1-e_y*kappa)**2 * (v_x * np.cos(e_psi) - v_y * np.sin(e_psi))
        d_s_dot_d_e_psi = 1 / (1 - e_y*kappa) * (-v_x * np.sin(e_psi) - v_y * np.cos(e_psi))
        d_s_dot_d_v_x = 1 / (1 - e_y*kappa) * np.cos(e_psi)
        d_s_dot_d_v_y = -1 / (1 - e_y*kappa) * np.sin(e_psi)
        d_s_dot_d_omega = 0
        d_s_dot_d_t = 0
        d_s_dot_d_delta = 0
        d_s_dot_d_D = 0
        d_s_dot_d_kappa = e_y / (1-e_y*kappa)**2 * (v_x * np.cos(e_psi) - v_y * np.sin(e_psi))
        # Check

        c_1 = (v_x * np.sin(e_psi) + v_y * np.cos(e_psi))
        d_c_1_d_e_y = 0
        d_c_1_d_e_psi = v_x * np.cos(e_psi) - v_y * np.sin(e_psi)
        d_c_1_d_v_x = np.sin(e_psi)
        d_c_1_d_v_y = np.cos(e_psi)
        d_c_1_d_omega = 0
        d_c_1_d_t = 0
        d_c_1_d_delta = 0
        d_c_1_d_D = 0
        d_c_1_d_kappa = 0
        # Check

        d_v_x_dot_d_e_y = 0
        d_v_x_dot_d_e_psi = 0
        d_v_x_dot_d_v_x = (d_F_rx_d_v_x - d_F_fy_d_v_x * np.sin(delta)) / self.m
        d_v_x_dot_d_v_y = - (d_F_fy_d_v_y * np.sin(delta) + self.m * omega) / self.m
        d_v_x_dot_d_omega = - (d_F_fy_d_omega * np.sin(delta) + self.m * v_y) / self.m
        d_v_x_dot_d_t = 0
        d_v_x_dot_d_delta = - (F_fy * np.cos(delta) + d_F_fy_d_delta * np.sin(delta)) / self.m
        d_v_x_dot_d_D = d_F_rx_d_D / self.m
        d_v_x_dot_d_kappa = 0

        d_v_y_dot_d_e_y = 0
        d_v_y_dot_d_e_psi = 0
        d_v_y_dot_d_v_x = (d_F_ry_d_v_x + d_F_fy_d_v_x * np.cos(delta) - self.m * omega) / self.m
        d_v_y_dot_d_v_y = (d_F_ry_d_v_y + d_F_fy_d_v_y * np.cos(delta)) / self.m
        d_v_y_dot_d_omega = (d_F_ry_d_omega + d_F_fy_d_omega * np.cos(delta) - self.m * v_x) / self.m
        d_v_y_dot_d_t = 0
        d_v_y_dot_d_delta = d_F_fy_d_delta * np.cos(delta) / self.m
        d_v_y_dot_d_D = 0
        d_v_y_dot_d_kappa = 0

        d_omega_dot_d_e_y = 0
        d_omega_dot_d_e_psi = 0
        d_omega_dot_d_v_x = (d_F_fy_d_v_x * self.lf * np.cos(delta) - d_F_ry_d_v_x * self.lr) / self.Iz
        d_omega_dot_d_v_y = (d_F_fy_d_v_y * self.lf * np.cos(delta) - d_F_fy_d_v_y * self.lr) / self.Iz
        d_omega_dot_d_omega = (d_F_fy_d_omega * self.lf * np.cos(delta) - d_F_fy_d_omega * self.lr) / self.Iz
        d_omega_dot_d_t = 0
        d_omega_dot_d_delta = (- F_fy * np.sin(delta) + d_F_fy_d_delta * np.cos(delta)) / self.Iz
        d_omega_dot_d_D = 0
        d_omega_dot_d_kappa = 0

        #######################
        # Partial Derivatives #
        #######################

        # derivatives for E_Y
        d_e_y_d_e_y = -c_1 * d_s_dot_d_e_y / (s_dot**2)
        d_e_y_d_e_psi = (d_c_1_d_e_psi * s_dot - d_s_dot_d_e_psi * c_1) / (s_dot**2)
        d_e_y_d_v_x = (d_c_1_d_v_x * s_dot - d_s_dot_d_v_x * c_1) / (s_dot**2)
        d_e_y_d_v_y = (d_c_1_d_v_y * s_dot - d_s_dot_d_v_y * c_1) / (s_dot**2)
        d_e_y_d_omega = (d_c_1_d_omega * s_dot - d_s_dot_d_omega * c_1) / (s_dot**2)
        d_e_y_d_t = 0
        d_e_y_d_D = 0
        d_e_y_d_delta = (d_c_1_d_delta * s_dot - d_s_dot_d_delta * c_1) / (s_dot**2)
        d_e_y_d_kappa = -d_s_dot_d_kappa * c_1 / (s_dot**2)

        # derivatives for E_PSI
        d_e_psi_d_e_y = - omega * d_s_dot_d_e_y / (s_dot**2)
        d_e_psi_d_e_psi = - omega * d_s_dot_d_e_psi / (s_dot**2)
        d_e_psi_d_v_x = (- omega * d_s_dot_d_v_x) / (s_dot**2)
        d_e_psi_d_v_y = (- omega * d_s_dot_d_v_y) / (s_dot**2)
        d_e_psi_d_omega = (s_dot - omega * d_s_dot_d_omega) / (s_dot**2)
        d_e_psi_d_t = 0
        d_e_psi_d_delta = (- omega * d_s_dot_d_delta) / (s_dot**2)
        d_e_psi_d_D = (- omega * d_s_dot_d_D) / (s_dot**2)
        d_e_psi_d_kappa = -d_s_dot_d_kappa * omega / (s_dot**2) - 1

        # derivatives for V_X
        d_v_x_d_e_y = - d_s_dot_d_e_y * v_x_dot / (s_dot**2)
        d_v_x_d_e_psi = - d_s_dot_d_e_psi * v_x_dot / (s_dot**2)
        d_v_x_d_v_x = (d_v_x_dot_d_v_x * s_dot - d_s_dot_d_v_x * v_x_dot) / (s_dot**2)
        d_v_x_d_v_y = (d_v_x_dot_d_v_y * s_dot - d_s_dot_d_v_y * v_x_dot) / (s_dot**2)
        d_v_x_d_omega = (d_v_x_dot_d_omega * s_dot - d_s_dot_d_omega * v_x_dot) / (s_dot**2)
        d_v_x_d_t = 0
        d_v_x_d_delta = (d_v_x_dot_d_delta * s_dot - d_s_dot_d_delta * v_x_dot) / (s_dot**2)
        d_v_x_d_D = d_v_x_dot_d_D * s_dot / (s_dot**2)
        d_v_x_d_kappa = -d_s_dot_d_kappa * v_x_dot / (s_dot**2)

        # derivatives for V_Y
        d_v_y_d_e_y = - d_s_dot_d_e_y * v_y_dot / (s_dot ** 2)
        d_v_y_d_e_psi = - d_s_dot_d_e_psi * v_y_dot / (s_dot ** 2)
        d_v_y_d_v_x = (d_v_y_dot_d_v_x * s_dot - d_s_dot_d_v_x * v_y_dot) / (
                    s_dot ** 2)
        d_v_y_d_v_y = (d_v_y_dot_d_v_y * s_dot - d_s_dot_d_v_y * v_y_dot) / (
                    s_dot ** 2)
        d_v_y_d_omega = (d_v_y_dot_d_omega * s_dot - d_s_dot_d_omega * v_y_dot) / (
                                    s_dot ** 2)
        d_v_y_d_t = 0
        d_v_y_d_delta = (d_v_y_dot_d_delta * s_dot - d_s_dot_d_delta * v_y_dot) / (
                                    s_dot ** 2)
        d_v_y_d_D = d_v_y_dot_d_D * s_dot / (s_dot ** 2)
        d_v_y_d_kappa = -d_s_dot_d_kappa * v_y_dot / (s_dot ** 2)

        # derivatives for Omega
        d_omega_d_e_y = (d_omega_dot_d_e_y * s_dot - omega_dot * d_s_dot_d_e_y) / (s_dot**2)
        d_omega_d_e_psi = (d_omega_dot_d_e_psi * s_dot - omega_dot * d_s_dot_d_e_psi) / (s_dot**2)
        d_omega_d_v_x = (d_omega_dot_d_v_x * s_dot - omega_dot * d_s_dot_d_v_x) / (s_dot**2)
        d_omega_d_v_y = (d_omega_dot_d_v_y * s_dot - omega_dot * d_s_dot_d_v_y) / (s_dot**2)
        d_omega_d_omega = (d_omega_dot_d_omega * s_dot - omega_dot * d_s_dot_d_omega) / (s_dot**2)
        d_omega_d_t = (d_omega_dot_d_t * s_dot - omega_dot * d_s_dot_d_t) / (s_dot**2)
        d_omega_d_delta = (d_omega_dot_d_delta * s_dot - omega_dot * d_s_dot_d_delta) / (s_dot**2)
        d_omega_d_D = (d_omega_dot_d_D * s_dot - omega_dot * d_s_dot_d_D) / (s_dot**2)
        d_omega_d_kappa = (d_omega_dot_d_kappa * s_dot - omega_dot * d_s_dot_d_kappa) / (s_dot**2)

        # derivatives for T
        d_t_d_e_y = - d_s_dot_d_e_y / (s_dot**2)
        d_t_d_e_psi = - d_s_dot_d_e_psi / (s_dot ** 2)
        d_t_d_v_x = - d_s_dot_d_v_x / (s_dot ** 2)
        d_t_d_v_y = - d_s_dot_d_v_y / (s_dot ** 2)
        d_t_d_omega = - d_s_dot_d_omega / (s_dot ** 2)
        d_t_d_t = 0
        d_t_d_delta = - d_s_dot_d_delta / (s_dot ** 2)
        d_t_d_D = 0
        d_t_d_kappa = - d_s_dot_d_kappa / (s_dot ** 2)

        a_1 = np.array([d_e_y_d_e_y, d_e_y_d_e_psi, d_e_y_d_v_x, d_e_y_d_v_y, d_e_y_d_omega, d_e_y_d_t, d_e_y_d_kappa])
        a_2 = np.array([d_e_psi_d_e_y, d_e_psi_d_e_psi, d_e_psi_d_v_x, d_e_psi_d_v_y, d_e_psi_d_omega, d_e_psi_d_t, d_e_psi_d_kappa])
        a_3 = np.array([d_v_x_d_e_y, d_v_x_d_e_psi, d_v_x_d_v_x, d_v_x_d_v_y, d_v_x_d_omega, d_v_x_d_t, d_v_x_d_kappa])
        a_4 = np.array([d_v_y_d_e_y, d_v_y_d_e_psi, d_v_y_d_v_x, d_v_y_d_v_y, d_v_y_d_omega, d_v_y_d_t, d_v_y_d_kappa])
        a_5 = np.array([d_omega_d_e_y, d_omega_d_e_psi, d_omega_d_v_x, d_omega_d_v_y, d_omega_d_omega, d_omega_d_t, d_omega_d_kappa])
        a_6 = np.array([d_t_d_e_y, d_t_d_e_psi, d_t_d_v_x, d_t_d_v_y, d_t_d_omega, d_t_d_t, d_t_d_kappa])
        a_7 = np.array([0, 0, 0, 0, 0, 0, 1])
        A = np.stack((a_1, a_2, a_3, a_4, a_5, a_6, a_7), axis=0) * delta_s
        A[0, 0] += 1
        A[1, 1] += 1
        A[2, 2] += 1
        A[3, 3] += 1
        A[4, 4] += 1
        A[5, 5] += 1
        b_1 = np.array([d_e_y_d_D, d_e_y_d_delta])
        b_2 = np.array([d_e_psi_d_D, d_e_psi_d_delta])
        b_3 = np.array([d_v_x_d_D, d_v_x_d_delta])
        b_4 = np.array([d_v_y_d_D, d_v_y_d_delta])
        b_5 = np.array([d_omega_d_D, d_omega_d_delta])
        b_6 = np.array([d_t_d_D, d_t_d_delta])
        b_7 = np.array([0, 0])
        B = np.stack((b_1, b_2, b_3, b_4, b_5, b_6, b_7), axis=0) * delta_s

        # set system matrices
        return A, B

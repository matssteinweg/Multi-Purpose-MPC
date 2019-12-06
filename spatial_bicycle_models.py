import numpy as np
from abc import abstractmethod
try:
    from abc import ABC
except:
    # for Python 2.7
    from abc import ABCMeta
    class ABC(object):
        __metaclass__ = ABCMeta
        pass
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math

# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


#########################
# Temporal State Vector #
#########################

class TemporalState:
    def __init__(self, x, y, psi, v_x, v_y, omega, t):
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
        self.omega = omega
        self.t = t

        self.members = ['x', 'y', 'psi', 'v_x', 'v_y', 'omega', 't']

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """

        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self


########################
# Spatial State Vector #
########################

class SpatialState(ABC):
    """
    Spatial State Vector - Abstract Base Class.
    """

    @abstractmethod
    def __init__(self):
        self.members = None
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            members = [self.members[item]]
        else:
            members = self.members[item]
        return [vars(self)[key] for key in members]

    def __setitem__(self, key, value):
        vars(self)[self.members[key]] = value

    def __len__(self):
        return len(self.members)

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """

        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self

    def list_states(self):
        """
        Return list of names of all states.
        """
        return self.members


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

        self.members = ['e_y', 'e_psi', 'v_x', 'v_y', 'omega', 't']


####################################
# Spatial Bicycle Model Base Class #
####################################

class SpatialBicycleModel(ABC):
    def __init__(self, reference_path, length, width):
        """
        Abstract Base Class for Spatial Reformulation of Bicycle Model.
        :param reference_path: reference path object to follow
        """

        # Precision
        self.eps = 1e-12

        # Car Parameters
        self.l = length
        self.w = width
        self.safety_margin = self._compute_safety_margin()

        # Reference Path
        self.reference_path = reference_path

        # Set initial distance traveled
        self.s = 0.0

        # Set sampling time to None (Initialization required)
        self.sampling_time = None

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

    def t2s(self):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return x, y, psi
        """

        # compute temporal state variables
        e_y = np.cos(self.current_waypoint.psi) * \
              (self.temporal_state.y - self.current_waypoint.y) - \
        np.sin(self.current_waypoint.psi) * (self.temporal_state.x -
                                             self.current_waypoint.x)
        e_psi = self.temporal_state.psi - self.current_waypoint.psi
        e_psi = np.mod(e_psi + math.pi, 2*math.pi) - math.pi
        t = 0
        v_x = self.temporal_state.v_x
        v_y = self.temporal_state.v_y
        omega = self.temporal_state.omega

        return ExtendedSpatialState(e_y, e_psi, v_x, v_y, omega, t)

    def set_sampling_time(self, Ts):
        """
        Set sampling time of bicycle model.
        :param Ts: sampling time in s
        """
        self.Ts = Ts

    def drive(self, u):
        """
        Drive.
        :param u: input vector
        :return: numpy array with spatial derivatives for all state variables
        """

        # Get input signals
        v, delta = u

        # Compute temporal state derivatives
        x_dot = v * np.cos(self.temporal_state.psi)
        y_dot = v * np.sin(self.temporal_state.psi)
        psi_dot = v / self.l * np.tan(delta)
        temporal_derivatives = np.array([x_dot, y_dot, psi_dot, 0.0, 0.0, 0.0, 0.0])

        # Update spatial state (Forward Euler Approximation)
        self.temporal_state += temporal_derivatives * self.Ts

        # Compute velocity along path
        s_dot = 1 / (1 - self.spatial_state.e_y * self.current_waypoint.kappa) \
                * v * np.cos(self.spatial_state.e_psi)

        # Update distance travelled along reference path
        self.s += s_dot * self.Ts

        self.wp_id += 1

    def _compute_safety_margin(self):
        """
        Compute safety margin for car if modeled by its center of gravity.
        """

        # Model ellipsoid around the car
        length = self.l / np.sqrt(2)
        width = self.w / np.sqrt(2)

        return length, width

    def get_current_waypoint(self):
        """
        Create waypoint on reference path at current location of car by
        interpolation information from given path waypoints.
        """

        # Compute cumulative path length
        length_cum = np.cumsum(self.reference_path.segment_lengths)
        # Get first index with distance larger than distance traveled by car
        # so far
        greater_than_threshold = length_cum > self.s
        next_wp_id = greater_than_threshold.searchsorted(True)
        # Get previous index for interpolation
        prev_wp_id = next_wp_id - 1

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]

        if np.abs(self.s - s_next) < np.abs(self.s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.reference_path.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.reference_path.waypoints[prev_wp_id]
        #
        # # Weight for next waypoint
        # w = (s_next - self.s) / (s_next - s_prev)
        #
        # # Interpolate between the two waypoints
        # prev_wp = self.reference_path.waypoints[prev_wp_id]
        # next_wp = self.reference_path.waypoints[next_wp_id]
        # x = w * next_wp.x + (1 - w) * prev_wp.x
        # y = w * next_wp.y + (1 - w) * prev_wp.y
        # psi = w * next_wp.psi + (1 - w) * prev_wp.psi
        # kappa = w * next_wp.kappa + (1 - w) * prev_wp.kappa



    def show(self):
        """
        Display car on current axis.
        """

        # Get car's center of gravity
        cog = (self.temporal_state.x, self.temporal_state.y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(self.temporal_state.psi)
        # Draw rectangle
        car = plt_patches.Rectangle(cog, width=self.l, height=self.w,
                              angle=yaw, facecolor=CAR, edgecolor=CAR_OUTLINE, zorder=20)

        # Shift center rectangle to match center of the car
        car.set_x(car.get_x() - (self.l/2 * np.cos(self.temporal_state.psi) -
                                 self.w/2 * np.sin(self.temporal_state.psi)))
        car.set_y(car.get_y() - (self.w/2 * np.cos(self.temporal_state.psi) +
                                 self.l/2 * np.sin(self.temporal_state.psi)))

        # Show safety margin
        safety_margin = plt_patches.Ellipse(cog, width=2*self.safety_margin[0],
                                            height=2*self.safety_margin[1],
                                            angle=yaw,
                                            fill=False, edgecolor=CAR, zorder=20)

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(safety_margin)
        ax.add_patch(car)

    @abstractmethod
    def get_spatial_derivatives(self, state, input, kappa):
        pass

    @abstractmethod
    def linearize(self):
        pass


#################
# Bicycle Model #
#################

class BicycleModel(SpatialBicycleModel):
    def __init__(self, length, width, reference_path, e_y, e_psi, v_x, v_y, omega, t):
        """
        Simplified Spatial Bicycle Model. Spatial Reformulation of Kinematic
        Bicycle Model. Uses Simplified Spatial State.
        :param length: length of the car in m
        :param width: with of the car in m
        :param reference_path: reference path model is supposed to follow
        :param e_y: deviation from reference path | [m]
        :param e_psi: heading offset from reference path | [rad]
        """

        # Initialize base class
        super(BicycleModel, self).__init__(reference_path, length=length,
                                           width=width)

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

        # Initialize spatial state
        self.spatial_state = ExtendedSpatialState(e_y, e_psi, v_x, v_y, omega, t)

        # Number of spatial state variables
        self.n_states = len(self.spatial_state)

        # Initialize temporal state
        self.temporal_state = self.s2t()

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
            x, y, psi = super(BicycleModel, self).s2t()
            # Compute simplified velocities
        else:
            # Get pose information from base class implementation
            x, y, psi = super(BicycleModel, self).s2t(reference_waypoint,
                                                            reference_state)

        v_x = self.spatial_state.v_x
        v_y = self.spatial_state.v_y
        omega = self.spatial_state.omega
        t = self.spatial_state.omega

        return TemporalState(x, y, psi, v_x, v_y, omega, t)

    def get_forces(self, input):
        """
        Compute forces required for temporal derivatives of v_x and v_y
        :param delta:
        :param D:
        :return:
        """

        D, delta = input

        F_rx = (self.Cm1 - self.Cm2 * self.spatial_state.v_x) * D - \
               self.Cr0 - self.Cr2 * self.spatial_state.v_x ** 2

        alpha_f = - np.arctan2(
            self.spatial_state.omega * self.lf + self.spatial_state.v_y,
            self.spatial_state.v_x) + delta
        F_fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))

        alpha_r = np.arctan2(
            self.spatial_state.omega * self.lr - self.spatial_state.v_y,
            self.spatial_state.v_x)
        F_ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))

        return F_rx, F_fy, F_ry, alpha_f, alpha_r

    def get_temporal_derivatives(self, state, input, kappa, forces):
            """
            Compute temporal derivatives needed for state update.
            :param delta: steering command
            :param D: duty-cycle of DC motor
            :return: temporal derivatives of distance, angle and velocity
            """

            e_y, e_psi, v_x, v_y, omega, t = state
            D, delta = input
            F_rx, F_fy, F_ry = forces

            # velocity along path
            s_dot = 1 / (1 - (e_y * kappa)) * (v_x * np.cos(e_psi)
                       + v_y * np.sin(e_psi))

            # velocity in x and y direction
            v_x_dot = (F_rx - F_fy * np.sin(
                delta) + self.m * v_y * omega) / self.m
            v_y_dot = (F_ry + F_fy * np.cos(
                delta) - self.m * v_x * omega) / self.m

            # omega dot
            omega_dot = (F_fy * self.lf * np.cos(
                delta) - F_ry * self.lr) / self.Iz

            return s_dot, v_x_dot, v_y_dot, omega_dot

    def get_spatial_derivatives(self, state, input, kappa):
        """
        Compute spatial derivatives of all state variables for update.
        :param delta: steering angle
        :param psi_dot: heading rate of change
        :param s_dot: velocity along path
        :param v_dot: acceleration
        :return: spatial derivatives for all state variables
        """

        e_y, e_psi, v_x, v_y, omega, t = state
        D, delta = input

        # get required forces
        F_rx, F_fy, F_ry, _, _ = self.get_forces(input)
        forces = np.array([F_rx, F_fy, F_ry])

        # Compute state derivatives
        s_dot, v_x_dot, v_y_dot, omega_dot = \
            self.get_temporal_derivatives(state, input, kappa, forces)

        d_e_y = (v_x * np.sin(e_psi)
                 + v_y * np.cos(e_psi)) \
                / (s_dot + self.eps)
        d_e_psi = (omega / (s_dot + self.eps) - kappa)

        d_v_x = v_x_dot / (s_dot + self.eps)
        d_v_y = v_y_dot / (s_dot + self.eps)
        d_omega = omega_dot / (s_dot + self.eps)
        d_t = 1 / (s_dot + self.eps)

        return np.array([d_e_y, d_e_psi, d_v_x, d_v_y, d_omega, d_t])

    def linearize(self, state, input, kappa, delta_s):
        """
        Linearize the system equations around the current state and waypoint.
        :param kappa_r: kappa of waypoint around which to linearize
         """

        D, delta = input
        e_y, e_psi, v_x, v_y, omega, t = state

        ###################
        # System Matrices #
        ###################

        # get temporal derivatives
        F_rx, F_fy, F_ry, alpha_f, alpha_r = self.get_forces(input)
        forces = np.array([F_rx, F_fy, F_ry])
        s_dot, v_x_dot, v_y_dot, omega_dot = self. \
            get_temporal_derivatives(state, input, kappa, forces)

        ##############################
        # Forces Partial Derivatives #
        ##############################

        d_alpha_f_d_v_x = 1 / (1 + ((omega * self.lf + v_y) / v_x) ** 2) * (
                    omega * self.lf + v_y) / (v_x ** 2)
        d_alpha_f_d_v_y = - 1 / (1 + ((omega * self.lf + v_y) / v_x) ** 2) \
                          / v_x
        d_alpha_f_d_omega = - 1 / (1 + ((omega * self.lf + v_y) / v_x) ** 2) \
                            * (self.lf / v_x)
        d_alpha_f_d_delta = 1

        d_alpha_r_d_v_x = - 1 / (1 + ((omega * self.lr - v_y) / v_x) ** 2) * (
                    omega * self.lr - v_y) / (v_x ** 2)
        d_alpha_r_d_v_y = - 1 / (
                    1 + ((omega * self.lr - v_y) / v_x) ** 2) / v_x
        d_alpha_r_d_omega = 1 / (1 + ((omega * self.lr - v_y) / v_x) ** 2) * (
                    self.lr * v_x)

        d_F_fy_d_v_x = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) \
                       * self.Cf / (1 + (self.Bf * alpha_f) ** 2) * \
                       self.Bf * d_alpha_f_d_v_x
        d_F_fy_d_v_y = self.Df * np.cos(self.Cf * np.arctan(self.Bf * alpha_f)) \
                       * self.Cf / (1 + (self.Bf * alpha_f) ** 2) * self.Bf \
                       * d_alpha_f_d_v_y
        d_F_fy_d_omega = self.Df * np.cos(self.Cf * np.arctan(self.Bf *
                        alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f) ** 2) \
                         * self.Bf * d_alpha_f_d_omega
        d_F_fy_d_delta = self.Df * np.cos(self.Cf * np.arctan(self.Bf *
                        alpha_f)) * self.Cf / (1 + (self.Bf * alpha_f) ** 2) \
                         * self.Bf * d_alpha_f_d_delta

        d_F_ry_d_v_x = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) \
                       * self.Cr / (1 + (self.Br * alpha_r) ** 2) * self.Br * \
                       d_alpha_r_d_v_x
        d_F_ry_d_v_y = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) \
                       * self.Cr / (1 + (self.Br * alpha_r) ** 2) * self.Br \
                       * d_alpha_r_d_v_y
        d_F_ry_d_omega = self.Dr * np.cos(self.Cr * np.arctan(self.Br * alpha_r)) \
                         * self.Cr / (1 + (self.Br * alpha_r) ** 2) * self.Br \
                         * d_alpha_r_d_omega

        d_F_rx_d_v_x = - self.Cm2 * D - 2 * self.Cr2 * v_x
        d_F_rx_d_D = self.Cm1 - self.Cm2 * v_x

        ##############################
        # Helper Partial Derivatives #
        ##############################

        d_s_dot_d_e_y = kappa / (1 - e_y * kappa) ** 2 * (
                    v_x * np.cos(e_psi) - v_y * np.sin(e_psi))
        d_s_dot_d_e_psi = 1 / (1 - e_y * kappa) * (
                    -v_x * np.sin(e_psi) - v_y * np.cos(e_psi))
        d_s_dot_d_v_x = 1 / (1 - e_y * kappa) * np.cos(e_psi)
        d_s_dot_d_v_y = -1 / (1 - e_y * kappa) * np.sin(e_psi)
        d_s_dot_d_omega = 0
        d_s_dot_d_t = 0
        d_s_dot_d_delta = 0
        d_s_dot_d_D = 0

        c_1 = (v_x * np.sin(e_psi) + v_y * np.cos(e_psi))
        d_c_1_d_e_y = 0
        d_c_1_d_e_psi = v_x * np.cos(e_psi) - v_y * np.sin(e_psi)
        d_c_1_d_v_x = np.sin(e_psi)
        d_c_1_d_v_y = np.cos(e_psi)
        d_c_1_d_omega = 0
        d_c_1_d_t = 0
        d_c_1_d_delta = 0
        d_c_1_d_D = 0

        d_v_x_dot_d_e_y = 0
        d_v_x_dot_d_e_psi = 0
        d_v_x_dot_d_v_x = (d_F_rx_d_v_x - d_F_fy_d_v_x * np.sin(
            delta)) / self.m
        d_v_x_dot_d_v_y = - (
                    d_F_fy_d_v_y * np.sin(delta) + self.m * omega) / self.m
        d_v_x_dot_d_omega = - (
                    d_F_fy_d_omega * np.sin(delta) + self.m * v_y) / self.m
        d_v_x_dot_d_t = 0
        d_v_x_dot_d_delta = - (F_fy * np.cos(delta) + d_F_fy_d_delta * np.sin(
            delta)) / self.m
        d_v_x_dot_d_D = d_F_rx_d_D / self.m

        d_v_y_dot_d_e_y = 0
        d_v_y_dot_d_e_psi = 0
        d_v_y_dot_d_v_x = (d_F_ry_d_v_x + d_F_fy_d_v_x * np.cos(
            delta) - self.m * omega) / self.m
        d_v_y_dot_d_v_y = (d_F_ry_d_v_y + d_F_fy_d_v_y * np.cos(
            delta)) / self.m
        d_v_y_dot_d_omega = (d_F_ry_d_omega + d_F_fy_d_omega * np.cos(
            delta) - self.m * v_x) / self.m
        d_v_y_dot_d_t = 0
        d_v_y_dot_d_delta = d_F_fy_d_delta * np.cos(delta) / self.m
        d_v_y_dot_d_D = 0

        d_omega_dot_d_e_y = 0
        d_omega_dot_d_e_psi = 0
        d_omega_dot_d_v_x = (d_F_fy_d_v_x * self.lf * np.cos(
            delta) - d_F_ry_d_v_x * self.lr) / self.Iz
        d_omega_dot_d_v_y = (d_F_fy_d_v_y * self.lf * np.cos(
            delta) - d_F_ry_d_v_y * self.lr) / self.Iz
        d_omega_dot_d_omega = (d_F_fy_d_omega * self.lf * np.cos(
            delta) - d_F_ry_d_omega * self.lr) / self.Iz
        d_omega_dot_d_t = 0
        d_omega_dot_d_delta = (- F_fy * self.lf * np.sin(
            delta) + d_F_fy_d_delta * self.lf * np.cos(delta)) / self.Iz
        d_omega_dot_d_D = 0

        #######################
        # Partial Derivatives #
        #######################

        # derivatives for E_Y
        d_e_y_d_e_y = -c_1 * d_s_dot_d_e_y / (s_dot ** 2)
        d_e_y_d_e_psi = (d_c_1_d_e_psi * s_dot - d_s_dot_d_e_psi * c_1) / (
                    s_dot ** 2)
        d_e_y_d_v_x = (d_c_1_d_v_x * s_dot - d_s_dot_d_v_x * c_1) / (
                    s_dot ** 2)
        d_e_y_d_v_y = (d_c_1_d_v_y * s_dot - d_s_dot_d_v_y * c_1) / (
                    s_dot ** 2)
        d_e_y_d_omega = (d_c_1_d_omega * s_dot - d_s_dot_d_omega * c_1) / (
                    s_dot ** 2)
        d_e_y_d_t = 0
        d_e_y_d_D = 0
        d_e_y_d_delta = (d_c_1_d_delta * s_dot - d_s_dot_d_delta * c_1) / (
                    s_dot ** 2)

        # derivatives for E_PSI
        d_e_psi_d_e_y = - omega * d_s_dot_d_e_y / (s_dot ** 2)
        d_e_psi_d_e_psi = - omega * d_s_dot_d_e_psi / (s_dot ** 2)
        d_e_psi_d_v_x = (- omega * d_s_dot_d_v_x) / (s_dot ** 2)
        d_e_psi_d_v_y = (- omega * d_s_dot_d_v_y) / (s_dot ** 2)
        d_e_psi_d_omega = (s_dot - omega * d_s_dot_d_omega) / (s_dot ** 2)
        d_e_psi_d_t = 0
        d_e_psi_d_delta = (- omega * d_s_dot_d_delta) / (s_dot ** 2)
        d_e_psi_d_D = (- omega * d_s_dot_d_D) / (s_dot ** 2)

        # derivatives for V_X
        d_v_x_d_e_y = - d_s_dot_d_e_y * v_x_dot / (s_dot ** 2)
        d_v_x_d_e_psi = - d_s_dot_d_e_psi * v_x_dot / (s_dot ** 2)
        d_v_x_d_v_x = (d_v_x_dot_d_v_x * s_dot - d_s_dot_d_v_x * v_x_dot) / (
                    s_dot ** 2)
        d_v_x_d_v_y = (d_v_x_dot_d_v_y * s_dot - d_s_dot_d_v_y * v_x_dot) / (
                    s_dot ** 2)
        d_v_x_d_omega = (d_v_x_dot_d_omega * s_dot - d_s_dot_d_omega * v_x_dot) \
                        / (s_dot ** 2)
        d_v_x_d_t = 0
        d_v_x_d_delta = (d_v_x_dot_d_delta * s_dot - d_s_dot_d_delta * v_x_dot) / (
                                    s_dot ** 2)
        d_v_x_d_D = d_v_x_dot_d_D * s_dot / (s_dot ** 2)

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

        # derivatives for Omega
        d_omega_d_e_y = (d_omega_dot_d_e_y * s_dot - omega_dot * d_s_dot_d_e_y) / (
                                    s_dot ** 2)
        d_omega_d_e_psi = (d_omega_dot_d_e_psi * s_dot - omega_dot * d_s_dot_d_e_psi) / (
                                      s_dot ** 2)
        d_omega_d_v_x = (d_omega_dot_d_v_x * s_dot - omega_dot * d_s_dot_d_v_x) / (
                                    s_dot ** 2)
        d_omega_d_v_y = (d_omega_dot_d_v_y * s_dot - omega_dot * d_s_dot_d_v_y) / (
                                    s_dot ** 2)
        d_omega_d_omega = (d_omega_dot_d_omega * s_dot - omega_dot * d_s_dot_d_omega) / (
                                      s_dot ** 2)
        d_omega_d_t = (d_omega_dot_d_t * s_dot - omega_dot * d_s_dot_d_t) / (
                    s_dot ** 2)
        d_omega_d_delta = (d_omega_dot_d_delta * s_dot - omega_dot * d_s_dot_d_delta) / (
                                      s_dot ** 2)
        d_omega_d_D = (d_omega_dot_d_D * s_dot - omega_dot * d_s_dot_d_D) / (
                    s_dot ** 2)

        # derivatives for T
        d_t_d_e_y = - d_s_dot_d_e_y / (s_dot ** 2)
        d_t_d_e_psi = - d_s_dot_d_e_psi / (s_dot ** 2)
        d_t_d_v_x = - d_s_dot_d_v_x / (s_dot ** 2)
        d_t_d_v_y = - d_s_dot_d_v_y / (s_dot ** 2)
        d_t_d_omega = - d_s_dot_d_omega / (s_dot ** 2)
        d_t_d_t = 0
        d_t_d_delta = - d_s_dot_d_delta / (s_dot ** 2)
        d_t_d_D = 0

        # compute f
        e_y_dot = c_1 / s_dot
        e_psi_dot = omega / s_dot - kappa
        t_dot = 1 / s_dot
        f = np.array([e_y_dot, e_psi_dot, v_x_dot/s_dot, v_y_dot/s_dot, omega_dot/s_dot, t_dot]) * delta_s

        a_1 = np.array([d_e_y_d_e_y, d_e_y_d_e_psi, d_e_y_d_v_x, d_e_y_d_v_y,
                        d_e_y_d_omega, d_e_y_d_t])
        a_2 = np.array(
            [d_e_psi_d_e_y, d_e_psi_d_e_psi, d_e_psi_d_v_x, d_e_psi_d_v_y,
             d_e_psi_d_omega, d_e_psi_d_t])
        a_3 = np.array([d_v_x_d_e_y, d_v_x_d_e_psi, d_v_x_d_v_x, d_v_x_d_v_y,
                        d_v_x_d_omega, d_v_x_d_t])
        a_4 = np.array([d_v_y_d_e_y, d_v_y_d_e_psi, d_v_y_d_v_x, d_v_y_d_v_y,
                        d_v_y_d_omega, d_v_y_d_t])
        a_5 = np.array(
            [d_omega_d_e_y, d_omega_d_e_psi, d_omega_d_v_x, d_omega_d_v_y,
             d_omega_d_omega, d_omega_d_t])
        a_6 = np.array(
            [d_t_d_e_y, d_t_d_e_psi, d_t_d_v_x, d_t_d_v_y, d_t_d_omega,
             d_t_d_t])
        A = np.stack((a_1, a_2, a_3, a_4, a_5, a_6), axis=0) * delta_s
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
        B = np.stack((b_1, b_2, b_3, b_4, b_5, b_6), axis=0) * delta_s

        # set system matrices
        return f, A, B
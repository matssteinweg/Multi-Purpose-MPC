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
    def __init__(self, x, y, psi):
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

        self.members = ['x', 'y', 'psi']

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


class SimpleSpatialState(SpatialState):
    def __init__(self, e_y, e_psi, t):
        """
        Simplified Spatial State Vector containing orthogonal deviation from
        reference path (e_y), difference in orientation (e_psi) and velocity
        :param e_y: orthogonal deviation from center-line | [m]
        :param e_psi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        super(SimpleSpatialState, self).__init__()

        self.e_y = e_y
        self.e_psi = e_psi
        self.t = t

        self.members = ['e_y', 'e_psi', 't']


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

        return SimpleSpatialState(e_y, e_psi, t)

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
        temporal_derivatives = np.array([x_dot, y_dot, psi_dot])

        # Update spatial state (Forward Euler Approximation)
        self.temporal_state += temporal_derivatives * self.Ts

        # Compute velocity along path
        s_dot = 1 / (1 - self.spatial_state.e_y * self.current_waypoint.kappa) \
                * v * np.cos(self.spatial_state.e_psi)

        # Update distance travelled along reference path
        self.s += s_dot * self.Ts

    def _compute_safety_margin(self):
        """
        Compute safety margin for car if modeled by its center of gravity.
        """

        # Model ellipsoid around the car
        length = self.l / np.sqrt(2)
        width = self.w / np.sqrt(2)
        widht = 0
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
        #ax.add_patch(safety_margin)
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
    def __init__(self, length, width, reference_path, e_y, e_psi, t):
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

        # Initialize spatial state
        self.spatial_state = SimpleSpatialState(e_y, e_psi, t)

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

        return TemporalState(x, y, psi)

    def get_temporal_derivatives(self, state, input, kappa):
        """
        Compute relevant temporal derivatives needed for state update.
        :param state: state vector for which to compute derivatives
        :param input: input vector
        :param kappa: curvature of corresponding waypoint
        :return: temporal derivatives of distance, angle and velocity
        """

        e_y, e_psi, t = state
        v, delta = input

        # Compute velocity along path
        s_dot = 1 / (1 - (e_y * kappa)) * v * np.cos(e_psi)

        # Compute yaw angle rate of change
        psi_dot = v / self.l * np.tan(delta)

        return s_dot, psi_dot

    def get_spatial_derivatives(self, state, input, kappa):
        """
        Compute spatial derivatives of all state variables for update.
        :param state: state vector for which to compute derivatives
        :param input: input vector
        :param kappa: curvature of corresponding waypoint
        :return: numpy array with spatial derivatives for all state variables
        """

        e_y, e_psi, t = state
        v, delta = input

        # Compute temporal derivatives
        s_dot, psi_dot = self.get_temporal_derivatives(state, input, kappa)

        # Compute spatial derivatives
        d_e_y_d_s = v * np.sin(e_psi) / s_dot
        d_e_psi_d_s = psi_dot / s_dot - kappa
        d_t_d_s = 1 / s_dot

        return np.array([d_e_y_d_s, d_e_psi_d_s, d_t_d_s])

    def linearize(self, v=None, kappa_r=None, delta_s=None):
        """
        Linearize the system equations around the current state and waypoint.
        :param kappa_r: kappa of waypoint around which to linearize
         """

        # Get linearization state
        if kappa_r is None and delta_s is None:
            # Get curvature of linearization waypoint
            kappa_r = self.reference_path.waypoints[self.wp_id].kappa
            # Get delta_s
            next_waypoint = self.reference_path.waypoints[self.wp_id + 1]
            delta_s = next_waypoint - self.current_waypoint

        ###################
        # System Matrices #
        ###################

        # Construct Jacobian Matrix
        a_1 = np.array([1,                    delta_s,      0])
        a_2 = np.array([-kappa_r**2*delta_s,  1,            0])
        a_3 = np.array([-kappa_r/v*delta_s,   0,            1])

        b_1 = np.array([0,          0])
        b_2 = np.array([0,   delta_s])
        b_3 = np.array([-1/(v**2)*delta_s,          0])

        f = np.array([0.0, 0.0, 1/v*delta_s])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return f, A, B
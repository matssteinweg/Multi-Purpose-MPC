import numpy as np
from abc import ABC, abstractmethod


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

    def __setitem__(self, key, value):
        vars(self)[list(vars(self).keys())[key]] = value

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

    def drive(self, input, state=None, kappa=None, delta_s=None):
        """
        Drive.
        :param state: state vector for which to compute derivatives
        :param input: input vector
        :param kappa: curvature of corresponding waypoint
        :return: numpy array with spatial derivatives for all state variables
        """

        # Get spatial derivatives
        if state is None and kappa is None and delta_s is None:
            state = np.array(self.spatial_state[:])
            # Get delta_s | distance to next waypoint
            next_waypoint = self.reference_path.waypoints[self.wp_id + 1]
            delta_s = next_waypoint - self.current_waypoint
            # Get current curvature
            kappa = self.current_waypoint.kappa

            spatial_derivatives = self.get_spatial_derivatives(state, input, kappa)

            # Update spatial state (Forward Euler Approximation)
            self.spatial_state += spatial_derivatives * delta_s

            # Assert that unique projections of car pose onto path exists
            #assert self.spatial_state.e_y < (1 / (self.current_waypoint.kappa +
             #                                     self.eps))

            # Increase waypoint ID
            self.wp_id += 1

            # Update current waypoint
            self.current_waypoint = self.reference_path.waypoints[self.wp_id]

            # Update temporal_state to match spatial state
            self.temporal_state = self.s2t()

            # Update s | total driven distance along path
            self.s += delta_s

            # Linearize model around new operating point
            # self.A, self.B = self.linearize()

        else:

            spatial_derivatives = self.get_spatial_derivatives(state, input,
                                                               kappa)

            # Update spatial state (Forward Euler Approximation)
            state += spatial_derivatives * delta_s

            return state

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
    def __init__(self, reference_path, e_y, e_psi, t):
        """
        Simplified Spatial Bicycle Model. Spatial Reformulation of Kinematic
        Bicycle Model. Uses Simplified Spatial State.
        :param reference_path: reference path model is supposed to follow
        :param e_y: deviation from reference path | [m]
        :param e_psi: heading offset from reference path | [rad]
        :param v: initial velocity | [m/s]
        """

        # Initialize base class
        super(BicycleModel, self).__init__(reference_path)

        # Constants
        self.l = 0.06

        # Initialize spatial state
        self.spatial_state = SimpleSpatialState(e_y, e_psi, t)

        # Number of spatial state variables
        self.n_states = len(self.spatial_state)

        # Initialize temporal state
        self.temporal_state = self.s2t()

        # Compute linear system matrices | Used for MPC
        # self.A, self.B = self.linearize()

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
        a_3 = np.array([-kappa_r/v*delta_s,   0,            0])

        b_1 = np.array([0, ])
        b_2 = np.array([delta_s, ])
        b_3 = np.array([0, ])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return A, B
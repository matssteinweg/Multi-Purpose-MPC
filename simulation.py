from map import Map
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import SimpleBicycleModel, ExtendedBicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from time import time

if __name__ == '__main__':

    # Create Map
    map = Map(file_path='map_race.png', origin=[-1, -2], resolution=0.005)
    #map = Map(file_path='map_floor2.png')

    # Specify waypoints
    wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25,
            1.25, -0.75, -0.75, -0.25]
    wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5,
            -1.5]
    #wp_x = [-9.169, 11.9, 7.3, -6.95]
    #wp_y = [-15.678, 10.9, 14.5, -3.31]

    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # Smooth Path
    reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                   smoothing_distance=5)
    rx = [wp.x for wp in reference_path.waypoints]
    ry = [wp.y for wp in reference_path.waypoints]

    ################
    # Motion Model #
    ################

    # initial state
    e_y_0 = 0.0
    e_psi_0 = 0.0
    v_x_0 = 0.3
    v_y_0 = 0
    omega_0 = 0
    t_0 = 0

    # initialize car
    car = SimpleBicycleModel(reference_path=reference_path,
                                e_y=e_y_0, e_psi=e_psi_0, v=v_x_0)
    #car = ExtendedBicycleModel(reference_path=reference_path,
    #                         e_y=e_y_0, e_psi=e_psi_0, v_x=v_x_0, v_y=v_y_0,
    #                           omega=omega_0, t=t_0)

    ##############
    # Controller #
    ##############

    # path tracker
    T = 10
    Q = np.diag([0.1, 0.001, 0.1])
    Qf = Q
    #Q = np.diag([1, 0, 0, 0, 0, 0])
    #Qf = Q
    R = np.diag([0, 0])
    StateConstraints = {'e_y': (-0.1, 0.1), 'v': (0, 4)}
    InputConstraints = {'D': (-1, 1), 'delta': (-0.44, 0.44)}
    Reference = {'e_y': 0, 'e_psi': 0, 'v': 4.0}
    #Reference = {'e_y': 0, 'e_psi': 0, 'v_x': 1.0, 'v_y': 0, 'omega': 0, 't':0}
    mpc = MPC(car, T, Q, R, Qf, StateConstraints, InputConstraints, Reference)

    ##############
    # Simulation #
    ##############

    # logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]
    psi_log = [car.temporal_state.psi]
    v_log = [car.temporal_state.v_x]
    D_log = []
    delta_log = []

    start_time = time()

    # iterate over waypoints
    for wp_id in range(len(car.reference_path.waypoints)-T-1):

        print('V: {:.2f}'.format(car.temporal_state.v_x))

        # get control signals
        D, delta = mpc.get_control()

        # drive car
        car.drive(delta, D)

        # log current state
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)
        v_log.append(car.temporal_state.v_x)
        D_log.append(D)
        delta_log.append(delta)

        ###################
        # Plot Simulation #
        ###################
        # plot path
        car.reference_path.show()

        # plot car trajectory and velocity
        plt.scatter(x_log, y_log, c='g', s=15)

        # plot mpc prediction
        if mpc.current_prediction is not None:
            x_pred = mpc.current_prediction[0]
            y_pred = mpc.current_prediction[1]
            plt.scatter(x_pred, y_pred, c='b', s=10)

        plt.title('MPC Simulation: Position: {:.2f} m, {:.2f} m, Velocity: '
                  '{:.2f} m/s'.format(car.temporal_state.x,
                                      car.temporal_state.y, car.temporal_state.v_x))
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.0000001)

    end_time = time()
    print('Time Elapsed: {:.2f} s'.format(end_time-start_time))
    plt.close()

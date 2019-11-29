from map import Map
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC, MPC_OSQP
from scipy import sparse
import time


if __name__ == '__main__':

    # Select Simulation Mode | 'Race' or 'Q'
    sim_mode = 'Race'

    # Create Map
    if sim_mode == 'Race':
        map = Map(file_path='map_race.png', origin=[-1, -2], resolution=0.005)
        # Specify waypoints
        wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25,
                1.25, -0.75, -0.75, -0.25]
        wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0,
                -1.5, -1.5]
        # Specify path resolution
        path_resolution = 0.05  # m / wp
    elif sim_mode == 'Q':
        map = Map(file_path='map_floor2.png')
        wp_x = [-9.169, 11.9, 7.3, -6.95]
        wp_y = [-15.678, 10.9, 14.5, -3.31]
        # Specify path resolution
        path_resolution = 0.20  # m / wp
    else:
        print('Invalid Simulation Mode!')
        map, wp_x, wp_y, path_resolution = None, None, None, None
        exit(1)

    # Create smoothed reference path
    reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                   smoothing_distance=5)

    ################
    # Motion Model #
    ################

    # Initial state
    e_y_0 = 0.0
    e_psi_0 = 0.0
    t_0 = 0.0
    v = 1.0

    car = BicycleModel(reference_path=reference_path,
                                e_y=e_y_0, e_psi=e_psi_0, t=t_0)

    ##############
    # Controller #
    ##############

    N = 20
    Q = sparse.diags([0.01, 0.0, 0.4])
    R = sparse.diags([0.01])
    QN = Q
    InputConstraints = {'umin': np.array([-np.tan(0.44)/car.l]), 'umax': np.array([np.tan(0.44)/car.l])}
    StateConstraints = {'xmin': np.array([-0.2, -np.inf, 0]), 'xmax': np.array([0.2, np.inf, np.inf])}
    mpc = MPC_OSQP(car, N, Q, R, QN, StateConstraints, InputConstraints)

    ##############
    # Simulation #
    ##############

    # logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]

    # iterate over waypoints
    for wp_id in range(len(car.reference_path.waypoints)-mpc.N-1):

        # get control signals
        start = time.time()
        delta = mpc.get_control(v)
        end = time.time()
        u = np.array([v, delta])

        # drive car
        car.drive(u)

        # log
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)

        ###################
        # Plot Simulation #
        ###################
        # plot path
        car.reference_path.show()

        # plot car trajectory and velocity
        plt.scatter(x_log[:-1], y_log[:-1], c='g', s=15)

        plt.scatter(mpc.current_prediction[0], mpc.current_prediction[1], c='b', s=5)

        plt.title('MPC Simulation: Position: {:.2f} m, {:.2f} m'.
                  format(car.temporal_state.x, car.temporal_state.y))
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.00000001)
    plt.close()

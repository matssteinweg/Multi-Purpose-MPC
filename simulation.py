from map import Map
import numpy as np
from reference_path import ReferencePath, Obstacle
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
from time import time
from lidar_model import LidarModel


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
        # Create smoothed reference path
        reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                       smoothing_distance=5, max_width=0.23,
                                       circular=True)
    elif sim_mode == 'Q':
        map = Map(file_path='map_floor2.png')
        wp_x = [-9.169, 11.9, 7.3, -6.95]
        wp_y = [-15.678, 10.9, 14.5, -3.31]
        # Specify path resolution
        path_resolution = 0.20  # m / wp
        # Create smoothed reference path
        reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                       smoothing_distance=5, max_width=1.50,
                                       circular=False)
    else:
        print('Invalid Simulation Mode!')
        map, wp_x, wp_y, path_resolution, reference_path \
            = None, None, None, None, None
        exit(1)

    obs1 = Obstacle(cx=0.0, cy=0.0, radius=0.05)
    obs2 = Obstacle(cx=-0.8, cy=-0.5, radius=0.05)
    obs3 = Obstacle(cx=-0.7, cy=-1.5, radius=0.07)
    obs4 = Obstacle(cx=-0.3, cy=-1.0, radius=0.07)
    obs5 = Obstacle(cx=0.3, cy=-1.0, radius=0.05)
    obs6 = Obstacle(cx=0.75, cy=-1.5, radius=0.07)
    obs7 = Obstacle(cx=0.7, cy=-0.9, radius=0.08)
    obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
    obs9 = Obstacle(cx=0.7, cy=-0.1, radius=0.05)
    obs10 = Obstacle(cx=1.1, cy=-0.6, radius=0.07)
    reference_path.add_obstacles([obs1, obs2, obs3, obs4, obs5, obs6, obs7,
                                  obs8, obs9, obs10])

    ################
    # Motion Model #
    ################

    # Initial state
    e_y_0 = 0.0
    e_psi_0 = 0.0
    t_0 = 0.0

    car = BicycleModel(length=0.12, width=0.06, reference_path=reference_path,
                       e_y=e_y_0, e_psi=e_psi_0, t=t_0)

    ##############
    # Controller #
    ##############

    N = 30
    Q = sparse.diags([1.0, 0.0, 0.0])
    R = sparse.diags([1.0, 0.0])
    QN = sparse.diags([0.0, 0.0, 0.0])
    InputConstraints = {'umin': np.array([0.0, -np.tan(0.66)/car.l]),
                        'umax': np.array([2.5, np.tan(0.66)/car.l])}
    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    velocity_reference = 1.5  # m/s
    mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints,
              velocity_reference)

    #########
    # LiDAR #
    #########

    sensor = LidarModel(FoV=90, range=0.25, resolution=4.0)

    ##############
    # Simulation #
    ##############

    # Sampling time
    Ts = 0.05
    t = 0
    car.set_sampling_time(Ts)

    # Logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]

    # Until arrival at end of path
    while car.s < reference_path.length:

        # get control signals
        u = mpc.get_control()

        # drive car
        car.drive(u)

        # log
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)

        ###################
        # Plot Simulation #
        ###################

        # Plot path and drivable area
        reference_path.show()

        # Plot MPC prediction
        mpc.show_prediction()

        # Plot car
        car.show()

        t += Ts

        plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.
                  format(u[0], u[1], t))

        plt.pause(0.01)
    print('Final Time: {:.2f}'.format(t))
    plt.close()

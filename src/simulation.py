from map import Map, Obstacle
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse


if __name__ == '__main__':

    # Select Simulation Mode | 'Sim_Track' or 'Real_Track'
    sim_mode = 'Sim_Track'

    # Simulation Environment. Mini-Car on track specifically designed to show-
    # case time-optimal driving.
    if sim_mode == 'Sim_Track':

        # Load map file
        map = Map(file_path='maps/sim_map.png', origin=[-1, -2],
                  resolution=0.005)

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

        # Add obstacles
        use_obstacles = True
        if use_obstacles:
            obs1 = Obstacle(cx=0.0, cy=0.0, radius=0.05)
            obs2 = Obstacle(cx=-0.8, cy=-0.5, radius=0.08)
            obs3 = Obstacle(cx=-0.7, cy=-1.5, radius=0.05)
            obs4 = Obstacle(cx=-0.3, cy=-1.0, radius=0.08)
            obs5 = Obstacle(cx=0.27, cy=-1.0, radius=0.05)
            obs6 = Obstacle(cx=0.78, cy=-1.47, radius=0.05)
            obs7 = Obstacle(cx=0.73, cy=-0.9, radius=0.07)
            obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
            obs9 = Obstacle(cx=0.67, cy=-0.05, radius=0.06)
            map.add_obstacles([obs1, obs2, obs3, obs4, obs5, obs6, obs7,
                                          obs8, obs9])

        # Instantiate motion model
        car = BicycleModel(length=0.12, width=0.06,
                           reference_path=reference_path, Ts=0.05)

    # Real-World Environment. Track used for testing the algorithm on a 1:12
    # RC car.
    elif sim_mode == 'Real_Track':

        # Load map file
        map = Map(file_path='maps/real_map.png', origin=(-30.0, -24.0),
                  resolution=0.06)

        # Specify waypoints
        wp_x = [-9.169, 11.9, 7.3, -6.95]
        wp_y = [-15.678, 10.9, 14.5, -3.31]

        # Specify path resolution
        path_resolution = 0.20  # m / wp

        # Create smoothed reference path
        reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                       smoothing_distance=5, max_width=1.50,
                                       circular=False)

        # Add obstacles
        add_obstacles = False
        if add_obstacles:
            obs1 = Obstacle(cx=-6.3, cy=-11.1, radius=0.20)
            obs2 = Obstacle(cx=-2.2, cy=-6.8, radius=0.25)
            obs4 = Obstacle(cx=2.0, cy=-0.2, radius=0.25)
            obs8 = Obstacle(cx=6.0, cy=5.0, radius=0.3)
            obs9 = Obstacle(cx=7.42, cy=4.97, radius=0.3)
            map.add_obstacles([obs1, obs2, obs4, obs8, obs9])

        # Instantiate motion model
        car = BicycleModel(length=0.30, width=0.20,
                           reference_path=reference_path, Ts=0.05)

    else:
        print('Invalid Simulation Mode!')
        map, wp_x, wp_y, path_resolution, reference_path, car \
            = None, None, None, None, None, None
        exit(1)

    ##############
    # Controller #
    ##############

    N = 30
    Q = sparse.diags([1.0, 0.0, 0.0])
    R = sparse.diags([0.5, 0.0])
    QN = sparse.diags([1.0, 0.0, 0.0])

    v_max = 1.0  # m/s
    delta_max = 0.66  # rad
    ay_max = 4.0  # m/s^2
    InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                        'umax': np.array([v_max, np.tan(delta_max)/car.length])}
    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

    # Compute speed profile
    a_min = -0.1  # m/s^2
    a_max = 0.5  # m/s^2
    SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                               'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
    car.reference_path.compute_speed_profile(SpeedProfileConstraints)

    ##############
    # Simulation #
    ##############

    # Set simulation time to zero
    t = 0.0

    # Logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]
    v_log = [0.0]

    # Until arrival at end of path
    while car.s < reference_path.length:

        # Get control signals
        u = mpc.get_control()

        # Simulate car
        car.drive(u)

        # Log car state
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)
        v_log.append(u[0])

        # Increment simulation time
        t += car.Ts

        # Plot path and drivable area
        reference_path.show()

        # Plot car
        car.show()

        # Plot MPC prediction
        mpc.show_prediction()

        # Set figure title
        plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.format(u[0], u[1], t))
        plt.axis('off')
        plt.pause(0.001)

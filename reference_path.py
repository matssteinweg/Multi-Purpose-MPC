import numpy as np
import math
from map import Map
from bresenham import bresenham
import matplotlib.pyplot as plt


############
# Waypoint #
############

class Waypoint:
    def __init__(self, x, y, psi, kappa):
        """
        Waypoint object containing x, y location in global coordinate system,
        orientation of waypoint psi and local curvature kappa
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: orientation of waypoint | [rad]
        :param kappa: local curvature | [1 / m]
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.kappa = kappa

    def __sub__(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


##################
# Reference Path #
##################


class ReferencePath:
    def __init__(self, map, wp_x, wp_y, resolution, smoothing_distance, width_info=False):
        """
        Reference Path object. Create a reference trajectory from specified
        corner points with given resolution. Smoothing around corners can be
        applied.
        """
        # precision
        self.eps = 1e-12

        # map
        self.map = map

        # resolution of the path
        self.resolution = resolution

        # look ahead distance for path averaging
        self.smoothing_distance = smoothing_distance

        # waypoints with x, y, psi, k
        self.waypoints = self.construct_path(wp_x, wp_y)

        # path width
        self.get_width_info = width_info
        if self.get_width_info:
            self.width_info = self.compute_width()
            self.min_width = (np.min(self.width_info[0, :]),
                                np.min(self.width_info[3, :]))

    def construct_path(self, wp_x, wp_y):

        # Number of waypoints
        n_wp = [int(np.sqrt((wp_x[i + 1] - wp_x[i]) ** 2 +
                            (wp_y[i + 1] - wp_y[i]) ** 2) /
            self.resolution) for i in range(len(wp_x) - 1)]

        # Construct waypoints with specified resolution
        gp_x, gp_y = wp_x[-1], wp_y[-1]
        wp_x = [np.linspace(wp_x[i], wp_x[i+1], n_wp[i], endpoint=False).
                    tolist() for i in range(len(wp_x)-1)]
        wp_x = [wp for segment in wp_x for wp in segment] + [gp_x]
        wp_y = [np.linspace(wp_y[i], wp_y[i + 1], n_wp[i], endpoint=False).
                    tolist() for i in range(len(wp_y) - 1)]
        wp_y = [wp for segment in wp_y for wp in segment] + [gp_y]

        # smooth path
        wp_xs = []
        wp_ys = []
        for wp_id in range(self.smoothing_distance, len(wp_x) -
                                                    self.smoothing_distance):
            wp_xs.append(np.mean(wp_x[wp_id - self.smoothing_distance:wp_id
                                            + self.smoothing_distance + 1]))
            wp_ys.append(np.mean(wp_y[wp_id - self.smoothing_distance:wp_id
                                            + self.smoothing_distance + 1]))

        waypoints = list(zip(wp_xs, wp_ys))
        waypoints = self.spatial_reformulation(waypoints)
        return waypoints

    def spatial_reformulation(self, waypoints):
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa)
        :return: list of waypoint objects for entire reference path
        """

        waypoints_spatial = []
        for wp_id in range(len(waypoints) - 1):

            # get start and goal waypoints
            current_wp = np.array(waypoints[wp_id])
            next_wp = np.array(waypoints[wp_id + 1])

            # difference vector
            dif_ahead = next_wp - current_wp

            # angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            # distance to next waypoint
            dist_ahead = np.linalg.norm(dif_ahead, 2)

            # get x and y coordinates of current waypoint
            x = current_wp[0]
            y = current_wp[1]

            # first waypoint
            if wp_id == 0:
                kappa = 0
            else:
                prev_wp = np.array(waypoints[wp_id - 1])
                dif_behind = current_wp - prev_wp
                angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
                angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) \
                            - math.pi
                kappa = np.abs(angle_dif / (dist_ahead + self.eps))

            waypoints_spatial.append(Waypoint(x, y, psi, kappa))

        return waypoints_spatial

    def compute_width(self, max_dist=2.0):
        max_dist = max_dist  # m
        width_info = np.zeros((6, len(self.waypoints)))
        for wp_id, wp in enumerate(self.waypoints):
            for i, dir in enumerate(['left', 'right']):
                # get pixel coordinates of waypoint
                wp_x, wp_y = self.map.w2m(wp.x, wp.y)
                # get angle orthogonal to path in current direction
                if dir == 'left':
                    angle = np.mod(wp.psi + math.pi / 2 + math.pi,
                                 2 * math.pi) - math.pi
                else:
                    angle = np.mod(wp.psi - math.pi / 2 + math.pi,
                                   2 * math.pi) - math.pi
                # get closest cell to orthogonal vector
                t_x, t_y = self.map.w2m(wp.x + max_dist * np.cos(angle), wp.y + max_dist * np.sin(angle))
                # compute path between cells
                width_info[3*i:3*(i+1), wp_id] = self.get_min_dist(wp_x, wp_y, t_x, t_y, max_dist)
        return width_info

    def get_min_dist(self, wp_x, wp_y, t_x, t_y, max_dist):
        # get neighboring cells (account for discretization)
        neighbors_x, neighbors_y = [], []
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                neighbors_x.append(t_x + i)
                neighbors_y.append(t_y + j)

        # get bresenham paths to all neighboring cells
        paths = []
        for t_x, t_y in zip(neighbors_x, neighbors_y):
            path = list(bresenham(wp_x, wp_y, t_x, t_y))
            paths.append(path)

        min_dist = max_dist
        min_cell = self.map.m2w(t_x, t_y)
        for path in paths:
            for cell in path:
                t_x = cell[0]
                t_y = cell[1]
                # if path goes through occupied cell
                if self.map.data[t_y, t_x] == 0:
                    # get world coordinates
                    x, y = self.map.m2w(wp_x, wp_y)
                    c_x, c_y = self.map.m2w(t_x, t_y)
                    cell_dist = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                    if cell_dist < min_dist:
                        min_dist = cell_dist
                        min_cell = (c_x, c_y)
        dist_info = np.array([min_dist, min_cell[0], min_cell[1]])
        return dist_info

    def show(self):

        # plot map
        plt.clf()
        plt.imshow(np.flipud(self.map.data),cmap='gray',
                   extent=[self.map.origin[0], self.map.origin[0] +
                           self.map.width * self.map.resolution,
                           self.map.origin[1], self.map.origin[1] +
                           self.map.height * self.map.resolution])
        # plot reference path
        wp_x = np.array([wp.x for wp in self.waypoints])
        wp_y = np.array([wp.y for wp in self.waypoints])
        plt.scatter(wp_x, wp_y, color='k', s=5)

        if self.get_width_info:
            print('Min Width Left: {:f} m'.format(self.min_width[0]))
            print('Min Width Right: {:f} m'.format(self.min_width[1]))
            plt.quiver(wp_x, wp_y, self.width_info[1, :] - wp_x,
                       self.width_info[2, :] - wp_y, scale=1, units='xy',
                       width=0.05, color='#D4AC0D')
            plt.quiver(wp_x, wp_y, self.width_info[4, :] - wp_x,
                       self.width_info[5, :] - wp_y, scale=1, units='xy',
                       width=0.05, color='#BA4A00')


if __name__ == '__main__':

    # Create Map
    map = Map(file_path='map_race.png', origin=[-1, -2], resolution=0.005)

    # Specify waypoints
    # Floor 2
    # wp_x = [-9.169, -2.7, 11.9, 7.3, -6.95]
    # wp_y = [-15.678, -7.12, 10.9, 14.5, -3.31]
    # Race Track
    wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25]
    wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5]
    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # Smooth Path
    reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                          smoothing_distance=5)
    reference_path.show()
    plt.show()




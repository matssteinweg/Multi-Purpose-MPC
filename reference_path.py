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

        # information about drivable area at waypoint
        # upper and lower bound of drivable area orthogonal to
        # waypoint orientation
        self.lb = None
        self.ub = None
        self.border_cells = None

    def __sub__(self, other):
        """
        Overload subtract operator. Difference of two waypoints is equal to
        their euclidean distance.
        :param other: subtrahend
        :return: euclidean distance between two waypoints
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


##################
# Reference Path #
##################


class ReferencePath:
    def __init__(self, map, wp_x, wp_y, resolution, smoothing_distance, max_width):
        """
        Reference Path object. Create a reference trajectory from specified
        corner points with given resolution. Smoothing around corners can be
        applied. Waypoints represent center-line of the path with specified
        maximum width to both sides.
        :param map: map object on which path will be placed
        :param wp_x: x coordinates of corner points in global coordinates
        :param wp_y: y coordinates of corner points in global coordinates
        :param resolution: resolution of the path in m/wp
        :param smoothing_distance: number of waypoints used for smoothing the
        path by averaging neighborhood of waypoints
        :param max_width: maximum width of path to both sides in m
        """

        # Precision
        self.eps = 1e-12

        # Map
        self.map = map

        # Resolution of the path
        self.resolution = resolution

        # Look ahead distance for path averaging
        self.smoothing_distance = smoothing_distance

        # List of waypoint objects
        self.waypoints = self._construct_path(wp_x, wp_y)

        # Compute path width (attribute of each waypoint)
        self._compute_width(max_width=max_width)

    def _construct_path(self, wp_x, wp_y):
        """
        Construct path from given waypoints.
        :param wp_x: x coordinates of waypoints in global coordinates
        :param wp_y: y coordinates of waypoints in global coordinates
        :return: list of waypoint objects
        """

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

        # Smooth path
        wp_xs = []
        wp_ys = []
        for wp_id in range(self.smoothing_distance, len(wp_x) -
                                                    self.smoothing_distance):
            wp_xs.append(np.mean(wp_x[wp_id - self.smoothing_distance:wp_id
                                            + self.smoothing_distance + 1]))
            wp_ys.append(np.mean(wp_y[wp_id - self.smoothing_distance:wp_id
                                            + self.smoothing_distance + 1]))

        # Construct list of waypoint objects
        waypoints = list(zip(wp_xs, wp_ys))
        waypoints = self._construct_waypoints(waypoints)

        return waypoints

    def _construct_waypoints(self, waypoint_coordinates):
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """

        # List containing waypoint objects
        waypoints = []

        # Iterate over all waypoints
        for wp_id in range(len(waypoint_coordinates) - 1):

            # Get start and goal waypoints
            current_wp = np.array(waypoint_coordinates[wp_id])
            next_wp = np.array(waypoint_coordinates[wp_id + 1])

            # Difference vector
            dif_ahead = next_wp - current_wp

            # Angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            # Distance to next waypoint
            dist_ahead = np.linalg.norm(dif_ahead, 2)

            # Get x and y coordinates of current waypoint
            x, y = current_wp[0], current_wp[1]

            # Compute local curvature at waypoint
            # first waypoint
            if wp_id == 0:
                kappa = 0
            else:
                prev_wp = np.array(waypoint_coordinates[wp_id - 1])
                dif_behind = current_wp - prev_wp
                angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
                angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) \
                            - math.pi
                kappa = angle_dif / (dist_ahead + self.eps)

            waypoints.append(Waypoint(x, y, psi, kappa))

        return waypoints

    def _compute_width(self, max_width):
        """
        Compute the width of the path by checking the maximum free space to
        the left and right of the center-line.
        :param max_width: maximum width of the path.
        """

        # Iterate over all waypoints
        for wp_id, wp in enumerate(self.waypoints):
            # List containing information for current waypoint
            width_info = []
            # Check width left and right of the center-line
            for i, dir in enumerate(['left', 'right']):
                # Get angle orthogonal to path in current direction
                if dir == 'left':
                    angle = np.mod(wp.psi + math.pi / 2 + math.pi,
                                 2 * math.pi) - math.pi
                else:
                    angle = np.mod(wp.psi - math.pi / 2 + math.pi,
                                   2 * math.pi) - math.pi
                # Get closest cell to orthogonal vector
                t_x, t_y = self.map.w2m(wp.x + max_width * np.cos(angle), wp.y
                                        + max_width * np.sin(angle))
                # Compute distance to orthogonal cell on path border
                b_value, b_cell = self._get_min_width(wp, t_x, t_y, max_width)
                # Add information to list for current waypoint
                width_info.append(b_value)
                width_info.append(b_cell)

            # Set waypoint attributes with width to the left and right
            wp.ub = width_info[0]
            wp.lb = -1 * width_info[2]  # minus can be assumed as waypoints
            # represent center-line of the path
            # Set border cells of waypoint
            wp.border_cells = (width_info[1], width_info[3])

    def _get_min_width(self, wp, t_x, t_y, max_width):
        """
        Compute the minimum distance between the current waypoint and the
        orthogonal cell on the border of the path
        :param wp: current waypoint
        :param t_x: x coordinate of border cell in map coordinates
        :param t_y: y coordinate of border cell in map coordinates
        :param max_width: maximum path width in m
        :return: min_width to border and corresponding cell
        """

        # Get neighboring cells of orthogonal cell (account for
        # discretization inaccuracy)
        tn_x, tn_y = [], []
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                tn_x.append(t_x+i)
                tn_y.append(t_y+j)

        # Get pixel coordinates of waypoint
        wp_x, wp_y = self.map.w2m(wp.x, wp.y)

        # Get Bresenham paths to all possible cells
        paths = []
        for t_x, t_y in zip(tn_x, tn_y):
            path = list(bresenham(wp_x, wp_y, t_x, t_y))
            paths.append(path)

        # Compute minimum distance to border cell
        min_width = max_width
        # map inspected cell to world coordinates
        min_cell = self.map.m2w(t_x, t_y)
        for path in paths:
            for cell in path:
                t_x, t_y = cell[0], cell[1]
                # If path goes through occupied cell
                if self.map.data[t_y, t_x] == 0:
                    # Get world coordinates
                    c_x, c_y = self.map.m2w(t_x, t_y)
                    cell_dist = np.sqrt((wp.x - c_x) ** 2 + (wp.y - c_y) ** 2)
                    if cell_dist < min_width:
                        min_width = cell_dist
                        min_cell = (c_x, c_y)

        # decrease min_width by radius of circle around cell
        min_width -= 1 / np.sqrt(2) * self.map.resolution

        return min_width, min_cell

    def update_bounds(self, wp_id):
        """
        Compute upper and lower bounds of the drivable area orthogonal to
        the given waypoint.
        :param wp_id: ID of reference waypoint
        """

        # Get reference waypoint
        wp = self.waypoints[wp_id]

        # Get waypoint's border cells in map coordinates
        ub_p = self.map.w2m(wp.border_cells[0][0], wp.border_cells[0][1])
        lb_p = self.map.w2m(wp.border_cells[1][0], wp.border_cells[1][1])

        # Compute path from left border cell to right border cell
        path = list(bresenham(ub_p[0], ub_p[1], lb_p[0], lb_p[1]))

        # Initialize upper and lower bound of drivable area to
        # upper bound of path
        ub_o, lb_o = ub_p, ub_p

        # Initialize upper and lower bound of best segment to upper bound of
        # path
        ub_ls, lb_ls = ub_p, ub_p

        # Iterate over path from left border to right border
        for x, y in path:
            # If cell is free, update lower bound
            if self.map.data[y, x] == 1:
                lb_o = (x, y)
            # If cell is occupied, end segment. Update best segment if current
            # segment is larger than previous best segment. Then, reset upper
            # and lower bound to current cell
            elif self.map.data[y, x] == 0:
                if np.sqrt((ub_o[0]-lb_o[0])**2+(ub_o[1]-lb_o[1])**2) > \
                    np.sqrt((ub_ls[0]-lb_ls[0])**2+(ub_ls[1]-lb_ls[1])**2):
                    ub_ls = ub_o
                    lb_ls = lb_o
                # Start new segment
                ub_o = (x, y)
                lb_o = (x, y)

        # If no segment was set (no obstacle between left and right border),
        # return original bounds of path
        if ub_ls == ub_p and lb_ls == ub_p:
            return wp.lb, wp.ub

        # Transform upper and lower bound cells to world coordinates
        ub_ls = self.map.m2w(ub_ls[0], ub_ls[1])
        lb_ls = self.map.m2w(lb_ls[0], lb_ls[1])
        # Check sign of upper and lower bound
        angle_ub = np.mod(np.arctan2(ub_ls[1] - wp.y, ub_ls[0] - wp.x)
                          - wp.psi + math.pi, 2*math.pi) - math.pi
        angle_lb = np.mod(np.arctan2(lb_ls[1] - wp.y, lb_ls[0] - wp.x)
                          - wp.psi + math.pi, 2*math.pi) - math.pi
        sign_ub = np.sign(angle_ub)
        sign_lb = np.sign(angle_lb)
        # Compute upper and lower bound of largest drivable area
        ub = sign_ub * np.sqrt((ub_ls[0]-wp.x)**2+(ub_ls[1]-wp.y)**2)
        lb = sign_lb * np.sqrt((lb_ls[0]-wp.x)**2+(lb_ls[1]-wp.y)**2)

        # Update member variables of waypoint
        wp.ub = ub
        wp.lb = lb
        wp.border_cells = (ub_ls, lb_ls)

        return lb, ub

    def show(self, display_drivable_area=True):
        """
        Display path object on current figure.
        :param display_drivable_area: If True, display arrows indicating width
        of drivable area
        """

        # Clear figure
        plt.clf()

        # Plot map in gray-scale and set extent to match world coordinates
        plt.imshow(np.flipud(self.map.data), cmap='gray',
                   extent=[self.map.origin[0], self.map.origin[0] +
                           self.map.width * self.map.resolution,
                           self.map.origin[1], self.map.origin[1] +
                           self.map.height * self.map.resolution])

        # Get x and y coordinates for all waypoints
        wp_x = np.array([wp.x for wp in self.waypoints])
        wp_y = np.array([wp.y for wp in self.waypoints])

        # Get x and y locations of border cells for upper and lower bound
        wp_ub_x = np.array([wp.border_cells[0][0] for wp in self.waypoints])
        wp_ub_y = np.array([wp.border_cells[0][1] for wp in self.waypoints])
        wp_lb_x = np.array([wp.border_cells[1][0] for wp in self.waypoints])
        wp_lb_y = np.array([wp.border_cells[1][1] for wp in self.waypoints])

        # Plot waypoints
        plt.scatter(wp_x, wp_y, color='#99A3A4', s=3)

        # Plot arrows indicating drivable area
        if display_drivable_area:
            plt.quiver(wp_x, wp_y, wp_ub_x - wp_x, wp_ub_y - wp_y, scale=1,
                   units='xy', width=0.2*self.resolution, color='#2ECC71',
                   headwidth=1, headlength=2)
            plt.quiver(wp_x, wp_y, wp_lb_x - wp_x, wp_lb_y - wp_y, scale=1,
                   units='xy', width=0.2*self.resolution, color='#2ECC71',
                   headwidth=1, headlength=2)


if __name__ == '__main__':

    # Select Path | 'Race' or 'Q'
    path = 'Race'

    # Create Map
    if path == 'Race':
        map = Map(file_path='map_race.png', origin=[-1, -2], resolution=0.005)
        # Specify waypoints
        wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25,
                1.25, -0.75, -0.75, -0.25]
        wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0,
                -1.5, -1.5]
        # Specify path resolution
        path_resolution = 0.05  # m / wp
        reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                       smoothing_distance=5, max_width=0.22)
    elif path == 'Q':
        map = Map(file_path='map_floor2.png')
        wp_x = [-9.169, 11.9, 7.3, -6.95]
        wp_y = [-15.678, 10.9, 14.5, -3.31]
        # Specify path resolution
        path_resolution = 0.20  # m / wp
        reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                       smoothing_distance=5, max_width=1.5)
    else:
        reference_path = None
        print('Invalid path!')
        exit(1)

    reference_path.show()
    plt.show()




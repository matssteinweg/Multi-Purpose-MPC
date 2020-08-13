import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes
from PIL import Image
from skimage.draw import line_aa
import matplotlib.patches as plt_patches

# Colors
OBSTACLE = '#2E4053'


############
# Obstacle #
############

class Obstacle:
    def __init__(self, cx, cy, radius):
        """
        Constructor for a circular obstacle to be placed on a map.
        :param cx: x coordinate of center of obstacle in world coordinates
        :param cy: y coordinate of center of obstacle in world coordinates
        :param radius: radius of circular obstacle in m
        """
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def show(self):
        """
        Display obstacle on current axis.
        """

        # Draw circle
        circle = plt_patches.Circle(xy=(self.cx, self.cy), radius=
                                        self.radius, color=OBSTACLE, zorder=20)
        ax = plt.gca()
        ax.add_patch(circle)


#######
# Map #
#######

class Map:
    def __init__(self, file_path, origin, resolution, threshold_occupied=100):
        """
        Constructor for map object. Map contains occupancy grid map data of
        environment as well as meta information.
        :param file_path: path to image of map
        :param threshold_occupied: threshold value for binarization of map
        image
        :param origin: x and y coordinates of map origin in world coordinates
        [m]
        :param resolution: resolution in m/px
        """

        # Set binarization threshold
        self.threshold_occupied = threshold_occupied

        # Numpy array containing map data
        self.data = np.array(Image.open(file_path))[:, :, 0]

        # Process raw map image
        self.process_map()

        # Store meta information
        self.height = self.data.shape[0]  # height of the map in px
        self.width = self.data.shape[1]  # width of the map in px
        self.resolution = resolution  # resolution of the map in m/px
        self.origin = origin  # x and y coordinates of map origin
        # (bottom-left corner) in m

        # Containers for user-specified additional obstacles and boundaries
        self.obstacles = list()
        self.boundaries = list()

    def w2m(self, x, y):
        """
        World2Map. Transform coordinates from global coordinate system to
        map coordinates.
        :param x: x coordinate in global coordinate system
        :param y: y coordinate in global coordinate system
        :return: discrete x and y coordinates in px
        """
        dx = int(np.floor((x - self.origin[0]) / self.resolution))
        dy = int(np.floor((y - self.origin[1]) / self.resolution))

        return dx, dy

    def m2w(self, dx, dy):
        """
        Map2World. Transform coordinates from map coordinate system to
        global coordinates.
        :param dx: x coordinate in map coordinate system
        :param dy: y coordinate in map coordinate system
        :return: x and y coordinates of cell center in global coordinate system
        """
        x = (dx + 0.5) * self.resolution + self.origin[0]
        y = (dy + 0.5) * self.resolution + self.origin[1]

        return x, y

    def process_map(self):
        """
        Process raw map image. Binarization and removal of small holes in map.
        """

        # Binarization using specified threshold
        # 1 corresponds to free, 0 to occupied
        self.data = np.where(self.data >= self.threshold_occupied, 1, 0)

        # Remove small holes in map corresponding to spurious measurements
        self.data = remove_small_holes(self.data, area_threshold=5,
                                       connectivity=8).astype(np.int8)

    def add_obstacles(self, obstacles):
        """
        Add obstacles to the map.
        :param obstacles: list of obstacle objects
        """

        # Extend list of obstacles
        self.obstacles.extend(obstacles)

        # Iterate over list of new obstacles
        for obstacle in obstacles:

            # Compute radius of circular object in pixels
            radius_px = int(np.ceil(obstacle.radius / self.resolution))
            # Get center coordinates of obstacle in map coordinates
            cx_px, cy_px = self.w2m(obstacle.cx, obstacle.cy)

            # Add circular object to map
            y, x = np.ogrid[-radius_px: radius_px, -radius_px: radius_px]
            index = x ** 2 + y ** 2 <= radius_px ** 2
            self.data[cy_px-radius_px:cy_px+radius_px, cx_px-radius_px:
                                                cx_px+radius_px][index] = 0

    def add_boundary(self, boundaries):
        """
        Add boundaries to the map.
        :param boundaries: list of tuples containing coordinates of boundaries'
        start and end points
        """

        # Extend list of boundaries
        self.boundaries.extend(boundaries)

        # Iterate over list of boundaries
        for boundary in boundaries:
            sx = self.w2m(boundary[0][0], boundary[0][1])
            gx = self.w2m(boundary[1][0], boundary[1][1])
            path_x, path_y, _ = line_aa(sx[0], sx[1], gx[0], gx[1])
            for x, y in zip(path_x, path_y):
                self.data[y, x] = 0


if __name__ == '__main__':
    map = Map('maps/real_map.png')
    # map = Map('maps/sim_map.png')
    plt.imshow(np.flipud(map.data), cmap='gray')
    plt.show()

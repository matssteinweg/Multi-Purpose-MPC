import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes
from PIL import Image
from skimage.draw import line_aa


class Map:
    """
    Handle for map message. Contains a subscriber to the map topic and
    processes the map. Numpy array version of the
    map available as member variable.
    """
    def __init__(self, file_path, value_unknown=50, threshold_occupied=90, origin=[-30.0, -24.0], resolution=0.059999):

        self.value_unknown = value_unknown
        self.threshold_occupied = threshold_occupied
        # instantiate member variables
        self.data = np.array(Image.open(file_path))[:, :, 0]  # numpy array containing map data
        self.process_map()
        self.height = self.data.shape[0]  # height of the map in px
        self.width = self.data.shape[1]  # width of the map in px
        self.resolution = resolution  # resolution of the map in m/px
        self.origin = origin  # x and y coordinates of map origin
        # (bottom-left corner) in m

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
        d_x = np.floor((x - self.origin[0]) / self.resolution)
        d_y = np.floor((y - self.origin[1]) / self.resolution)

        return int(d_x), int(d_y)

    def m2w(self, dx, dy):
        """
        World2Map. Transform coordinates from global coordinate system to
        map coordinates.
        :param x: x coordinate in global coordinate system
        :param y: y coordinate in global coordinate system
        :return: discrete x and y coordinates in px
        """
        x = (dx + 0.5) * self.resolution + self.origin[0]
        y = (dy + 0.5) * self.resolution + self.origin[1]

        return x, y

    def add_obstacles(self, obstacles):
        """
        Add obstacles to the path.
        :param obstacles: list of obstacle objects
        """

        # Extend list of obstacles
        self.obstacles.extend(obstacles)

        # Iterate over list of obstacles
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

        # Extend list of boundaries
        self.boundaries.extend(boundaries)

        # Iterate over list of boundaries
        for boundary in boundaries:
            sx = self.w2m(boundary[0][0], boundary[0][1])
            gx = self.w2m(boundary[1][0], boundary[1][1])
            path_x, path_y, _ = line_aa(sx[0], sx[1], gx[0], gx[1])
            for x, y in zip(path_x, path_y):
                self.data[y, x] = 0

    def process_map(self):
        self.data = np.where(self.data >= 100, 1, 0)
        self.data = remove_small_holes(self.data, area_threshold=5,
                                       connectivity=8).astype(np.int8)



if __name__ == '__main__':
    map = Map('map_floor2.png')
    plt.imshow(map.data, cmap='gray')
    plt.show()

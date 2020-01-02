from map import Map
import matplotlib.pyplot as plt
import numpy as np
import math
import time

SCAN = '#5DADE2'


class LidarModel:
    """
    Lidar Model
    """
    def __init__(self, FoV, range, resolution):
        """
        Constructor for Lidar object.
        :param FoV: Sensor's Field of View in °
        :param range: range in m
        :param resolution: resolution in °
        """

        # set sensor parameters
        self.FoV = FoV
        self.range = range
        self.resolution = resolution

        # number of measurements
        self.n_measurements = int(self.FoV/self.resolution + 1)

        # construct measurement container
        angles = np.linspace(-math.pi / 360 * self.FoV,
                             math.pi / 360 * self.FoV,
                             self.n_measurements)
        ranges = np.ones(self.n_measurements) * self.range
        self.measurements = np.stack((angles, ranges), axis=0)

    def scan(self, car, map):
        """
        Get a Lidar Scan estimate
        :param car: state containing x and y coordinates of the sensor
        :param map: map object
        :return: self with updated self.measurements
        """

        start = time.time()
        # reset measurements
        self.measurements[1, :] = np.ones(self.n_measurements) * self.range

        # flip map upside-down to allow for normal indexing of y axis
        #map.data = np.flipud(map.data)

        # get sensor's map pose
        x, y = map.w2m(car.x, car.y)
        # get center of mass
        xc = x + 0.5
        yc = y + 0.5

        # get sensor range in px values
        range_px = int(self.range / map.resolution)

        # iterate over area within sensor's range
        for i in range(x - range_px, x + range_px + 1):
            if 0 <= i < map.width:
                for j in range(y - range_px, y + range_px + 1):
                    if 0 <= j < map.height:
                        # if obstacle detected
                        if map.data[j, i] == 0:

                            # get center of mass of cell
                            xc_target = i + 0.5
                            yc_target = j + 0.5

                            # check all corner's of cell
                            cell_angles = []
                            for k in range(-1, 2):
                                for l in range(-1, 2):
                                    dy = yc_target + l/2 - yc
                                    dx = xc_target + k/2 - xc
                                    cell_angle = np.arctan2(dy, dx) - car.psi
                                    if cell_angle < - math.pi:
                                        cell_angle = -np.mod(math.pi+cell_angle, 2*math.pi) + math.pi
                                    else:
                                        cell_angle = np.mod(math.pi+cell_angle, 2*math.pi) - math.pi
                                    cell_angles.append(cell_angle)

                            # get min and max angle hitting respective cell
                            min_angle = np.min(cell_angles)
                            max_angle = np.max(cell_angles)

                            # get distance to mass center of cell
                            cell_distance = np.sqrt(
                                (xc - xc_target)**2 + (yc - yc_target)**2)

                            # get IDs of all laser beams hitting cell
                            valid_beam_ids = []
                            if min_angle < -math.pi/2 and max_angle > math.pi/2:
                                for beam_id in range(self.n_measurements):
                                    if max_angle <= self.measurements[0, beam_id] <= min_angle:
                                        valid_beam_ids.append(beam_id)
                            else:
                                for beam_id in range(self.n_measurements):
                                    if min_angle <= self.measurements[0, beam_id] <= max_angle:
                                        valid_beam_ids.append(beam_id)

                            # update distance for all valid laser beams
                            for beam_id in valid_beam_ids:
                                if cell_distance < self.measurements[1, beam_id] / map.resolution:
                                    self.measurements[1, beam_id] = cell_distance * map.resolution

        #map.data = np.flipud(map.data)
        end = time.time()
        print('Time elapsed: ', end - start)

    def plot_scan(self, car):
        """
        Display current sensor measurements.
        :param car: state containing x and y coordinate of sensor
        """

        start = time.time()
        # get beam endpoints
        beam_end_x = self.measurements[1, :] * np.cos(self.measurements[0, :] + car.psi)
        beam_end_y = self.measurements[1, :] * np.sin(self.measurements[0, :] + car.psi)

        # plot all laser beams
        for i in range(self.n_measurements):
            plt.plot((car.x, car.x+beam_end_x[i]), (car.y, car.y+beam_end_y[i]), c=SCAN)
        end = time.time()
        print('Time elapsed: ', end - start)


if __name__ == '__main__':

    # Create Map
    map = Map('real_map.png')
    plt.imshow(map.data, cmap='gray',
               extent=[map.origin[0], map.origin[0] +
                       map.width * map.resolution,
                       map.origin[1], map.origin[1] +
                       map.height * map.resolution])

    car = BicycleModel(x=-4.9, y=-5.0, yaw=0.9)
    sensor = LidarModel(FoV=180, range=5, resolution=1)
    sensor.scan(car, map)
    sensor.plot_scan(car)

    plt.axis('equal')
    plt.show()

import pandas as pd
import numpy as np
import cv2
import copy
from math import sin, cos, pi


# Fast Frontier Detection method implemented in the class
class FFDdetectorCLass:

    def __init__(self):

        pass

    def get_laser_readings(self):

        # TODO: Implement get laser readings in real script
        pass

    def map_laser_readings(self, laser_readings_polar: np.ndarray, pos_of_robot_pix: dict, yaw_rot_of_robot):

        # Convert from polar coordinates to cartesian coordinates

        angle_of_reading = 0        # 360 laser readings in every message, each laser reading - int angle [0, 359]

        laser_readings_car_x = np.zeros(laser_readings_polar.shape)     # array of laser readings in x
                                                                        # cartesian coordinates

        laser_readings_car_y = np.zeros(laser_readings_polar.shape)     # array of laser readings in y
                                                                        # cartesian coordinates

        for laser_reading in laser_readings_polar:

            if laser_reading != 'inf':

                laser_readings_car_x[angle_of_reading] = float(laser_reading) * cos(angle_of_reading / 180 * pi)

                laser_readings_car_y[angle_of_reading] = float(laser_reading) * sin(angle_of_reading / 180 * pi)

            else:

                laser_readings_car_x[angle_of_reading] = 'inf'

                laser_readings_car_y[angle_of_reading] = 'inf'

            angle_of_reading += 1   # increasing angle for next laser reading

        laser_readings_car = np.vstack((laser_readings_car_x, laser_readings_car_y))

        # Transform /scan from robot's coordinate system to map coordinate system

        curr_laser_reading = {"x": None, "y": None}

        for laser_reading in range(0, laser_readings_car.shape[1] - 1):

            # getting x,y coordinates in robot's coordinate system
            curr_laser_reading["x"] = laser_readings_car[:, laser_reading][0]
            curr_laser_reading["y"] = laser_readings_car[:, laser_reading][1]




def main():

    # (*) loading up and setting up messages from ROS:

    map_msg_cont = pd.read_csv('map_msg_cont.csv', sep=',', header=None)    # continuous message
    scan_msg_cont = pd.read_csv('scan_msg_cont.csv', sep=',', header=None)  # continuous message

    # (*) setting up data as it was from ROS:

    map_msg = map_msg_cont.T[1]       # one sample of raw map with metadata
    scan_msg = scan_msg_cont.T[1]     # one sample of raw scan with metadata

    # (*) setting up map

    map_res = map_msg[5]                                                # map res
    map_shape = {"w": int(map_msg[6]), "h": int(map_msg[7])}            # shape of the map w - width, h - height
    map_msg_data = np.asarray(map_msg[15:])                             # one sample of raw map
    map_msg_data_reshape = map_msg_data.reshape((map_shape["h"],
                                                 map_shape["w"]))

    # DEBUG

    # cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
    # cv2.imshow('map', copy.deepcopy(map_msg_data_reshape).astype(np.uint8))  # show image on window 'win_name' made of numpy.ndarray
    # cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution
    #
    # cv2.waitKey(0)  # wait for key pressing
    # cv2.destroyAllWindows()  # close all windows

    # DEBUG END

    # (*) setting up scan

    scan_msg_data = np.asarray(scan_msg[11:371])    # laser readings in polar coordinates (counterclockwise from "x")

    pose_of_robot = {"j": 15, "i": 7}

    yaw_rot_of_robot = 0.785                # yaw rotation angle 0.785 [rad] = 45 [deg]

    ffd = FFDdetectorCLass()

    ffd.map_laser_readings(scan_msg_data, pose_of_robot, yaw_rot_of_robot)


if __name__ == '__main__':

    main()


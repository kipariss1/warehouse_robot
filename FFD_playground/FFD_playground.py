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

    def map_laser_readings(self, laser_readings_polar: np.ndarray, map_msg_data_reshape: np.ndarray,
                           pos_of_robot_m: dict, yaw_rot_of_robot, map_res):

        # Convert from polar coordinates to cartesian coordinates

        angle_of_reading = 0  # 360 laser readings in every message, each laser reading - int angle [0, 359]

        laser_readings_car_x = np.zeros(laser_readings_polar.shape)  # array of laser readings in x
        # cartesian coordinates

        laser_readings_car_y = np.zeros(laser_readings_polar.shape)  # array of laser readings in y
        # cartesian coordinatesTODO

        # DEBUG
        img = copy.deepcopy(map_msg_data_reshape).astype(np.uint8)

        # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
        img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # END DEBUG

        for laser_reading in laser_readings_polar:

            if laser_reading != 'inf':

                laser_readings_car_x[angle_of_reading] = float(laser_reading) * cos(angle_of_reading / 180 * pi)

                laser_readings_car_y[angle_of_reading] = float(laser_reading) * sin(angle_of_reading / 180 * pi)

            else:

                laser_readings_car_x[angle_of_reading] = 0

                laser_readings_car_y[angle_of_reading] = 0

            angle_of_reading += 1  # increasing angle for next laser reading

        laser_readings_car = np.vstack((laser_readings_car_x, laser_readings_car_y))

        # Transform /scan from robot's coordinate system to map (origin) coordinate system

        curr_laser_reading = {"x": None, "y": None}

        laser_readings_car_orig_x = np.zeros(laser_readings_polar.shape)  # array of laser readings in x
        # origin cartesian coordinates

        laser_readings_car_orig_y = np.zeros(laser_readings_polar.shape)  # array of laser readings in y
        # origin cartesian coordinates

        for laser_reading in range(0, laser_readings_car.shape[1] - 1):

            # getting x,y coordinates in robot's coordinate system
            curr_laser_reading["x"] = laser_readings_car[:, laser_reading][0]
            curr_laser_reading["y"] = laser_readings_car[:, laser_reading][1]

            robot_frame_cords = np.array([[curr_laser_reading["x"]], [curr_laser_reading["y"]], [1]])

            # setting up the transformation matrix for transforming from robot's frame to the map (origin) frame
            transf_matx = np.array([[cos(yaw_rot_of_robot), -sin(yaw_rot_of_robot), pos_of_robot_m["x"]],
                                    [sin(yaw_rot_of_robot), cos(yaw_rot_of_robot), pos_of_robot_m["y"]],
                                    [0, 0, 1]])

            # getting new coords in map (origin) frame:

            origin_map_frame_coords = np.matmul(transf_matx, robot_frame_cords)

            laser_readings_car_orig_x[laser_reading] = origin_map_frame_coords[0][0]
            laser_readings_car_orig_y[laser_reading] = origin_map_frame_coords[1][0]

            # DEBUG

            # j = int(laser_readings_car_orig_y[laser_reading] / map_res)
            # i = int(laser_readings_car_orig_x[laser_reading] / map_res)
            #
            # # getting the coordinates of the pixel, corresponding to the scan measurement
            # coords_pix = {"j": j if j < map_msg_data_reshape.shape[0] - 1 else map_msg_data_reshape.shape[0] - 1,
            #               "i": i if i < map_msg_data_reshape.shape[1] - 1 else map_msg_data_reshape.shape[1] - 1}
            #
            # cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
            #
            # img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 0])
            #
            # # cv2.circle(img_3d, (int(laser_readings_car_orig_x[laser_reading] / map_res),
            # #                     int(laser_readings_car_orig_y[laser_reading] / map_res)), 1, (255, 0, 0), 1)
            #
            # cv2.imshow('map', img_3d)  # show image on window 'win_name' made of numpy.ndarray
            # cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution
            #
            # cv2.waitKey(0)  # wait for key pressing
            # # cv2.destroyAllWindows()  # close all windows

            # DEBUG END

        laser_readings_car_orig = np.vstack((laser_readings_car_orig_x, laser_readings_car_orig_y))

        # returns laser readings in map (origin) frame
        return laser_readings_car_orig

    # function which gets line (all cells in line) from starting point (previous_point) and end point (curr_point)
    # (based on DDA algorithm)
    def DDALine(self, curr_point, previous_point, map_msg_data_reshape):

        di = abs(curr_point["i"] - previous_point["i"])
        dj = abs(curr_point["j"] - previous_point["j"])

        steps = max(di, dj)

        i_inc = float(di/steps)
        j_inc = float(dj/steps)

        points_list = [previous_point]  # creating list, containing first point in the line, to store the points
        new_point = {"i": previous_point["i"], "j": previous_point["j"]}

        # DEBUG

        img = copy.deepcopy(map_msg_data_reshape).astype(np.uint8)

        # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
        img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # END OF DEBUG

        for inc in range(0, steps):

            new_point["i"] += i_inc
            new_point["j"] += j_inc
            points_list.append({"i": int(new_point["i"]), "j": int(new_point["j"])})

            # DEBUG

            # i = int(new_point["i"])
            # j = int(new_point["j"])
            #
            # # getting the coordinates of the pixel, corresponding to the scan measurement
            # coords_pix = {"j": j if j < map_msg_data_reshape.shape[0] - 1 else map_msg_data_reshape.shape[0] - 1,
            #               "i": i if i < map_msg_data_reshape.shape[1] - 1 else map_msg_data_reshape.shape[1] - 1}
            #
            # cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
            #
            # img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 0])
            #
            # # cv2.circle(img_3d, (int(laser_readings_car_orig_x[laser_reading] / map_res),
            # #                     int(laser_readings_car_orig_y[laser_reading] / map_res)), 1, (255, 0, 0), 1)
            #
            # cv2.imshow('map', img_3d)  # show image on window 'win_name' made of numpy.ndarray
            # cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution
            #
            # cv2.waitKey(0)  # wait for key pressing
            # # cv2.destroyAllWindows()  # close all windows

            # DEBUG END

        return points_list, img_3d

    def get_lines_from_laser_readings(self, laser_readings_car_orig, map_msg_data_reshape, map_res):

        # get the starting point from from first laser reading
        previous_point = {"i": int(laser_readings_car_orig[:, 0][0]/map_res),
                          "j": int(laser_readings_car_orig[:, 0][1]/map_res)}

        lines_list = []     # create empty list for lines

        # getting the line
        for laser_reading in range(1, laser_readings_car_orig.shape[1]):

            curr_point = {"i": int(laser_readings_car_orig[:, laser_reading][0]/map_res),
                          "j": int(laser_readings_car_orig[:, laser_reading][1]/map_res)}

            if curr_point != previous_point:

                lines_list.append(self.DDALine(curr_point, previous_point, map_msg_data_reshape))

            else:

                lines_list.append([curr_point])     # appending list with single point in it

            previous_point = curr_point

        return lines_list


def main():
    # (*) loading up and setting up messages from ROS:

    map_msg_cont = pd.read_csv('map_msg_cont.csv', sep=',', header=None)  # continuous message
    scan_msg_cont = pd.read_csv('scan_msg_cont.csv', sep=',', header=None)  # continuous message

    # (*) setting up data as it was from ROS:

    map_msg = map_msg_cont.T[1]  # one sample of raw map with metadata
    scan_msg = scan_msg_cont.T[1]  # one sample of raw scan with metadata

    # (*) setting up map

    map_res = map_msg[5]  # map res
    map_shape = {"w": int(map_msg[6]), "h": int(map_msg[7])}  # shape of the map w - width, h - height
    map_msg_data = np.asarray(map_msg[15:])  # one sample of raw map
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

    scan_msg_data = np.asarray(scan_msg[11:371])  # laser readings in polar coordinates (counterclockwise from "x")

    pose_of_robot_pix = {"j": 15, "i": 7}

    yaw_rot_of_robot = 0  # yaw rotation angle 0.785 [rad] = 45 [deg]

    map_res = 0.15

    pose_of_robot_m = {"x": pose_of_robot_pix["i"] * map_res, "y": pose_of_robot_pix["j"] * map_res}

    ffd = FFDdetectorCLass()

    laser_readings_car_orig = ffd.map_laser_readings(scan_msg_data, map_msg_data_reshape, pose_of_robot_m,
                                                     yaw_rot_of_robot, map_res)

    lines_list = ffd.get_lines_from_laser_readings(laser_readings_car_orig, map_msg_data_reshape, map_res)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import cv2
import copy
from math import sin, cos, pi


# class for storing the frontiers from FFD method
class FFDfrontier:

    def __init__(self):

        self.list_of_points = []        # intialise empty list of points

    def add_point(self, point):

        self.list_of_points.append(point)

    def delete_all_points(self):

        self.list_of_points = []

    def calculate_centroid(self):

        pass

# Fast Frontier Detection method implemented in the class
class FFDdetectorCLass:

    def __init__(self):

        self.laser_readings_polar = None                            # scan readings in polar coordinates
        self.laser_readings_car_orig = None                         # scan readings in cartesian map frame
        self.map_msg_data_reshape = None                            # map
        self.map_res = None                                         # map resolution
        self.pos_of_robot = {"x": None, "y": None}                  # position of robot
        self.yaw_rot_of_robot = None                                # robot's rotation
        self.lines_list = []                                        # list of lines between scan points
        self.frontier_list = []                                     # list of frontiers

    def map_laser_readings(self, laser_readings_polar: np.ndarray):

        # Convert from polar coordinates to cartesian coordinates

        angle_of_reading = 0  # 360 laser readings in every message, each laser reading - int angle [0, 359]

        laser_readings_car_x = np.zeros(laser_readings_polar.shape)  # array of laser readings in x
        # cartesian coordinates

        laser_readings_car_y = np.zeros(laser_readings_polar.shape)  # array of laser readings in y
        # cartesian coordinatesTODO

        # DEBUG
        # img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)
        #
        # # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
        # img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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
            transf_matx = np.array([[cos(self.yaw_rot_of_robot), -sin(self.yaw_rot_of_robot), self.pos_of_robot["x"]],
                                    [sin(self.yaw_rot_of_robot), cos(self.yaw_rot_of_robot), self.pos_of_robot["y"]],
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
    def DDALine(self, curr_point, previous_point):

        di = abs(curr_point["i"] - previous_point["i"])
        dj = abs(curr_point["j"] - previous_point["j"])

        steps = max(di, dj)

        i_inc = float(di/steps)
        j_inc = float(dj/steps)

        points_list = [previous_point]  # creating list, containing first point in the line, to store the points
        new_point = {"i": previous_point["i"], "j": previous_point["j"]}

        # DEBUG

        # img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)
        #
        # # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
        # img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # END OF DEBUG

        for inc in range(0, steps):

            new_point["i"] += i_inc
            new_point["j"] += j_inc
            points_list.append({"i": int(new_point["i"]), "j": int(new_point["j"])})

            # DEBUG

            # i = int(new_point["i"])
            # j = int(new_point["j"])
            # # getting the coordinates of the pixel, corresponding to the scan measurement
            # coords_pix = {"j": j if j < map_msg_data_reshape.shape[0] - 1 else map_msg_data_reshape.shape[0] - 1,
            #               "i": i if i < map_msg_data_reshape.shape[1] - 1 else map_msg_data_reshape.shape[1] - 1}
            #
            # cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
            #
            # img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 0])
            #
            # cv2.imshow('map', img_3d)  # show image on window 'win_name' made of numpy.ndarray
            # cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution
            #
            # cv2.waitKey(0)  # wait for key pressing
            # # cv2.destroyAllWindows()  # close all windows

            # DEBUG END

        return points_list

    def get_lines_from_laser_readings(self, laser_readings_car_orig):

        # get the starting point from from first laser reading
        previous_point = {"i": int(laser_readings_car_orig[:, 0][0]/self.map_res),
                          "j": int(laser_readings_car_orig[:, 0][1]/self.map_res)}

        lines_list = []     # create empty list for lines

        # getting the line
        for laser_reading in range(1, laser_readings_car_orig.shape[1]):

            curr_point = {"i": int(laser_readings_car_orig[:, laser_reading][0]/self.map_res),
                          "j": int(laser_readings_car_orig[:, laser_reading][1]/self.map_res)}

            if curr_point != previous_point:

                lines_list.append(self.DDALine(curr_point, previous_point))

            else:

                lines_list.append([curr_point])     # appending list with single point in it

            previous_point = curr_point

        return lines_list

    def check_neighbours(self, point):

        bounds = 1                      # boundaries for checking the neighbours (if 1 -> checks 3x3 submap)

        submap = self.map_msg_data_reshape[point["j"] - bounds: point["j"] + bounds + 1,
                 point["i"] - bounds: point["i"] + bounds + 1]

        unkn_flag = '-1' in submap    # flag, indicating, there is unknown cells in the neighbouring cells in the map
        obst_flag = '100' in submap   # flag, indicating, there is an obstacle in the neighbouring cells in the map

        return unkn_flag, obst_flag

    def find_frontiers_in_lines(self, lines_list):

        frontier_list = []          # empty list to store frontiers
        new_frontier = FFDfrontier()

        # DEBUG
        #
        # img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)
        #
        # # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
        # img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #
        # END OF DEBUG

        # for cyclus to check all points in all lines to find frontiers
        for line in lines_list:

            for point in line:

                unkn_flag, obst_flag = self.check_neighbours(point)

                if unkn_flag and not obst_flag:

                    # add cell to frontier object
                    new_frontier.add_point(point)

                    # # DEBUG
                    #
                    # i = int(point["i"])
                    # j = int(point["j"])
                    # # getting the coordinates of the pixel, corresponding to the scan measurement
                    # coords_pix = {"j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[0] - 1,
                    #               "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[1] - 1}
                    #
                    # cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
                    #
                    # img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 0])
                    #
                    # cv2.imshow('map', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                    # cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution
                    #
                    # cv2.waitKey(0)  # wait for key pressing
                    # # cv2.destroyAllWindows()  # close all windows
                    #
                    # # DEBUG END

                # if the frontier isn't empty
                if obst_flag and new_frontier.list_of_points:

                    # add finished frontier to the list of frontiers, and send signal
                    # that it's time to delete all points from frontier object
                    frontier_list.append(copy.deepcopy(new_frontier))
                    new_frontier.delete_all_points()

        return frontier_list

    # main function of this class, using all of the above functions to find the goal and maintain found frontiers
    def do_ffd(self, pos_of_robot, yaw_rot_of_robot, laser_readings_polar, map_msg_data_reshape, map_res):

        # Getting actual data about robot's position and rotation
        self.pos_of_robot = pos_of_robot
        self.yaw_rot_of_robot = yaw_rot_of_robot

        # Getting actual map
        self.map_msg_data_reshape = map_msg_data_reshape
        self.map_res = map_res

        # Getting actual laser readings
        self.laser_readings_polar = laser_readings_polar

        # Maping laser readings to map
        self.laser_readings_car_orig = self.map_laser_readings(self.laser_readings_polar)

        # Getting lines between separate scan readings
        self.lines_list = self.get_lines_from_laser_readings(self.laser_readings_car_orig)

        # Finding frontiers in lines
        self.frontier_list = self.find_frontiers_in_lines(self.lines_list)

        # TODO: Finding the centroids from all frontiers

        # TODO: Implement returning list of frontier centroids
        return None




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

    # MAIN PART
    ffd = FFDdetectorCLass()

    ffd.do_ffd(pose_of_robot_m, yaw_rot_of_robot, scan_msg_data, map_msg_data_reshape, map_res)


if __name__ == '__main__':
    main()

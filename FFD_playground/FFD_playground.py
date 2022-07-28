import pandas as pd
import numpy as np
import cv2
import copy
from math import sin, cos, pi


# class for storing the frontiers from FFD method
class FFDfrontier:

    def __init__(self):

        self.list_of_points = []        # intialise empty list of points ( point is {"i": xx, "j": xx})

        self.centroid = {"i": None, "j": None}      # initialise centroid attribute

        self.num_of_elements = 0                    # number of cells in frontier

    def add_point(self, point):

        self.list_of_points.append(point)

    def delete_all_points(self):

        self.list_of_points.clear()

    def calculate_num_of_elements(self):

        self.num_of_elements = self.list_of_points.__len__()

    def calculate_centroid(self):

        sum_coord_i = 0
        sum_coord_j = 0

        for point in self.list_of_points:

            sum_coord_i += point["i"]
            sum_coord_j += point["j"]

        sum_coord_i = int(sum_coord_i/self.list_of_points.__len__())     # get mean i coordinates of frontier
        sum_coord_j = int(sum_coord_j/self.list_of_points.__len__())     # get mean j coordinates of frontier

        self.centroid = {"i": sum_coord_i, "j": sum_coord_j}


# Fast Frontier Detection method implemented in the class
class FFDdetectorCLass:

    def __init__(self, debug_flag=False):

        self.laser_readings_polar = None                            # scan readings in polar coordinates
        self.laser_readings_car_orig_pix = None                     # scan readings in cartesian map frame (in pixels)
        self.map_msg_data_reshape = None                            # map
        self.map_res = None                                         # map resolution
        self.pos_of_robot = {"x": None, "y": None}                  # position of robot
        self.yaw_rot_of_robot = None                                # robot's rotation
        self.lines_list = []                                        # list of lines between scan points
        self.frontier_list = []                                     # list of frontiers
        self.frontier_idx = 1                                       # the index of frontier to go to (idx of goal front)
        self.debug_flag = debug_flag                                # shows/not shows debug outputs
        self.do_ffd_state = 0                                       # state enumerator for do_ffd()
        self.map_origin_pos = {"x": None, "y": None}                # position of [0, 0] cell in map frame
        self.global_pool_of_frontiers = []                          # stores found old frontiers

    def map_laser_readings(self, laser_readings_polar: np.ndarray):

        # Convert from polar coordinates to cartesian coordinates

        angle_of_reading = 0  # 360 laser readings in every message, each laser reading - int angle [0, 359]

        laser_readings_car_x = np.zeros(laser_readings_polar.shape)  # array of laser readings in x
        # cartesian coordinates

        laser_readings_car_y = np.zeros(laser_readings_polar.shape)  # array of laser readings in y
        # cartesian coordinates

        if self.debug_flag:
            # DEBUG

            img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)

            # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
            img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # END DEBUG

        for laser_reading in laser_readings_polar:

            if laser_reading != float('inf'):

                laser_readings_car_x[angle_of_reading] = float(laser_reading) * cos(angle_of_reading / 180 * pi)

                laser_readings_car_y[angle_of_reading] = float(laser_reading) * sin(angle_of_reading / 180 * pi)

            else:

                laser_readings_car_x[angle_of_reading] = 0

                laser_readings_car_y[angle_of_reading] = 0

            angle_of_reading += 1  # increasing angle for next laser reading

        # deleting the 'inf' or [0, 0] scan values from array

        inf_idxs = np.where(np.logical_and(laser_readings_car_x == 0,
                                           laser_readings_car_y == 0))

        laser_readings_car_x = np.delete(laser_readings_car_x, inf_idxs)
        laser_readings_car_y = np.delete(laser_readings_car_y, inf_idxs)

        # stacking together laser x and y to common array

        laser_readings_car = np.vstack((laser_readings_car_x, laser_readings_car_y))

        # Transform /scan from robot's coordinate system to map (origin) coordinate system

        curr_laser_reading = {"x": None, "y": None}

        laser_readings_car_orig_x = np.zeros(laser_readings_car_x.shape)  # array of laser readings in x
        # origin cartesian coordinates

        laser_readings_car_orig_y = np.zeros(laser_readings_car_y.shape)  # array of laser readings in y
        # origin cartesian coordinates

        laser_readings_car_orig_i = np.zeros(laser_readings_car_x.shape)  # array of laser readings in i
        # origin cartesian coordinates

        laser_readings_car_orig_j = np.zeros(laser_readings_car_y.shape)  # array of laser readings in j
        # origin cartesian coordinates

        for laser_reading in range(0, laser_readings_car.shape[1]):

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

            laser_readings_car_orig_i[laser_reading] = int(origin_map_frame_coords[0][0]/self.map_res)
            laser_readings_car_orig_j[laser_reading] = int(origin_map_frame_coords[1][0]/self.map_res)

            if self.debug_flag:
                # DEBUG

                j = int(laser_readings_car_orig_y[laser_reading] / self.map_res)
                i = int(laser_readings_car_orig_x[laser_reading] / self.map_res)

                # getting the coordinates of the pixel, corresponding to the scan measurement
                coords_pix = {"j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[0] - 1,
                              "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[1] - 1}

                cv2.namedWindow('map laser readings', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

                img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 0])

                cv2.imshow('map laser readings', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                cv2.resizeWindow('map laser readings', 1600, 900)  # resizing window on my resolution

                cv2.waitKey(10)  # wait for key pressing
                # cv2.destroyAllWindows()  # close all windows

                # DEBUG END

        laser_readings_car_orig = np.vstack((laser_readings_car_orig_x, laser_readings_car_orig_y))
        laser_readings_car_orig_pix = np.vstack((laser_readings_car_orig_i, laser_readings_car_orig_j))
        laser_readings_car_orig_pix = laser_readings_car_orig_pix.astype(int)

        # leaving only unique values, saving the original indexes (without sorting)
        laser_readings_car_orig_pix = laser_readings_car_orig_pix[:, np.sort(np.unique(laser_readings_car_orig_pix,
                                                                                       return_index=True, axis=1)[1])]

        # returns laser readings in map (origin) frame
        return laser_readings_car_orig, laser_readings_car_orig_pix

    # function which gets line (all cells in line) from starting point (previous_point) and end point (curr_point)
    # (based on DDA algorithm)
    def DDALine(self, curr_point, previous_point):

        di = abs(curr_point["i"] - previous_point["i"])
        dj = abs(curr_point["j"] - previous_point["j"])

        steps = max(di, dj)

        if curr_point["i"] < previous_point["i"]:
            i_inc = -float(di/steps)
        else:
            i_inc = float(di/steps)

        if curr_point["j"] < previous_point["j"]:
            j_inc = -float(dj/steps)
        else:
            j_inc = float(dj/steps)

        points_list = [previous_point]  # creating list, containing first point in the line, to store the points
        new_point = {"i": previous_point["i"], "j": previous_point["j"]}

        if self.debug_flag:
            # DEBUG

            img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)

            # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
            img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # END OF DEBUG

        for inc in range(0, steps):

            new_point["i"] += i_inc
            new_point["j"] += j_inc
            points_list.append({"i": int(new_point["i"]), "j": int(new_point["j"])})

            if self.debug_flag:
                # DEBUG

                i = int(new_point["i"])
                j = int(new_point["j"])
                # getting the coordinates of the pixel, corresponding to the scan measurement
                coords_pix = {"j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[0] - 1,
                              "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[1] - 1}

                cv2.namedWindow('Drawing line', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

                img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([0, 0, 255])

                cv2.imshow('Drawing line', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                cv2.resizeWindow('Drawing line', 1600, 900)  # resizing window on my resolution

                # if big line is drawn - hold a picture a little bit longer
                if inc == steps-1 and steps > 1:

                    cv2.waitKey(10)  # wait for key pressing

                else:

                    cv2.waitKey(100)  # wait for key pressing

                # cv2.destroyAllWindows()  # close all windows

                # DEBUG END

        return points_list

    def get_lines_from_laser_readings(self, laser_readings_car_orig_pix):

        # get the starting point from from first laser reading
        previous_point = {"i": laser_readings_car_orig_pix[:, 0][0],
                          "j": laser_readings_car_orig_pix[:, 0][1]}

        lines_list = []     # create empty list for lines

        # getting the line
        for laser_reading in range(1, laser_readings_car_orig_pix.shape[1]):

            curr_point = {"i": laser_readings_car_orig_pix[:, laser_reading][0],
                          "j": laser_readings_car_orig_pix[:, laser_reading][1]}

            lines_list.append(self.DDALine(curr_point, previous_point))

            previous_point = curr_point

        # getting last line from last point to first point
        curr_point = {"i": laser_readings_car_orig_pix[:, 0][0],
                      "j": laser_readings_car_orig_pix[:, 0][1]}

        lines_list.append(self.DDALine(curr_point, previous_point))

        return lines_list

    def check_neighbours(self, point):

        bounds = 1                      # boundaries for checking the neighbours (if 1 -> checks 3x3 submap)

        submap = self.map_msg_data_reshape[point["j"] - bounds: point["j"] + bounds + 1,
                 point["i"] - bounds: point["i"] + bounds + 1]

        unkn_flag = -1 in submap    # flag, indicating, there is unknown cells in the neighbouring cells in the map
        obst_flag = 100 in submap   # flag, indicating, there is an obstacle in the neighbouring cells in the map

        return unkn_flag, obst_flag

    def find_frontiers_in_lines(self, lines_list):

        frontier_list = []          # empty list to store frontiers
        new_frontier = FFDfrontier()

        if self.debug_flag:
            # DEBUG

            img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)

            # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
            img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # END OF DEBUG

        # for cyclus to check all points in all lines to find frontiers
        for line in lines_list:

            for point in line:

                unkn_flag, obst_flag = self.check_neighbours(point)

                if unkn_flag and not obst_flag:

                    # add cell to frontier object
                    new_frontier.add_point(point)

                    if self.debug_flag:
                        # DEBUG

                        i = int(point["i"])
                        j = int(point["j"])
                        # getting the coordinates of the pixel, corresponding to the scan measurement
                        coords_pix = {"j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[0] - 1,
                                      "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[1] - 1}

                        cv2.namedWindow('finding frontiers', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

                        img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([0, 0, 255])

                        cv2.imshow('finding frontiers', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                        cv2.resizeWindow('finding frontiers', 1600, 900)  # resizing window on my resolution

                        cv2.waitKey(10)  # wait for key pressing
                        # cv2.destroyAllWindows()  # close all windows

                        # DEBUG END

                # if the frontier isn't empty
                if obst_flag and new_frontier.list_of_points:

                    # add finished frontier to the list of frontiers, and send signal
                    # that it's time to delete all points from frontier object
                    frontier_list.append(copy.deepcopy(new_frontier))
                    new_frontier.delete_all_points()

                if obst_flag:
                    if self.debug_flag:
                        # DEBUG

                        i = int(point["i"])
                        j = int(point["j"])
                        # getting the coordinates of the pixel, corresponding to the scan measurement
                        coords_pix = {
                            "j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[
                                                                                          0] - 1,
                            "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[
                                                                                          1] - 1}

                        cv2.namedWindow('finding frontiers', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

                        img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([255, 0, 255])  # purple

                        cv2.imshow('finding frontiers', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                        cv2.resizeWindow('finding frontiers', 1600, 900)  # resizing window on my resolution

                        cv2.waitKey(10)  # wait for key pressing
                        # cv2.destroyAllWindows()  # close all windows

                        # DEBUG END

                if not obst_flag and not unkn_flag:

                    if self.debug_flag:
                        # DEBUG

                        i = int(point["i"])
                        j = int(point["j"])
                        # getting the coordinates of the pixel, corresponding to the scan measurement
                        coords_pix = {
                            "j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[
                                                                                          0] - 1,
                            "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[
                                                                                          1] - 1}

                        cv2.namedWindow('finding frontiers', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

                        img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([192, 192, 192])     # silver

                        cv2.imshow('finding frontiers', img_3d)  # show image on window 'win_name' made of numpy.ndarray
                        cv2.resizeWindow('finding frontiers', 1600, 900)  # resizing window on my resolution

                        cv2.waitKey(10)  # wait for key pressing
                        # cv2.destroyAllWindows()  # close all windows

                        # DEBUG END

        return frontier_list

    # function returning position of point in origin or map frame
    def transf_map2odom_and_back(self, from_which_frame: str, to_which_frame: str, point):

        if from_which_frame == "orig" and to_which_frame == "map":

            # x_or is position of point in origin frame
            if "x" and "y" in point.keys():

                x_or = {"x": point["x"],
                        "y": point["y"]}

            elif "i" and "j" in point.keys():

                x_or = {"x": point["i"] * self.map_res,
                        "y": point["j"] * self.map_res}

            x_map_m = {"x": x_or["x"] + self.map_origin_pos["x"],
                       "y": x_or["y"] + self.map_origin_pos["y"]}

            x_map_pix = {"i": int(x_map_m["x"]/self.map_res),
                         "j": int(x_map_m["y"]/self.map_res)}

            # returns position of point in map frame in meters
            return x_map_m, x_map_pix

        elif from_which_frame == "map" and to_which_frame == "odom":

            # x_map is position of point in map frame
            if "x" and "y" in point.keys():

                x_map = {"x": point["x"],
                         "y": point["y"]}

            elif "i" and "j" in point.keys():

                x_map = {"x": point["i"] * self.map_res,
                         "y": point["j"] * self.map_res}

            x_or_m = {"x": x_map["x"] - self.map_origin_pos["x"],
                      "y": x_map["y"] - self.map_origin_pos["y"]}

            x_or_pix = {"i": int(x_or_m["x"]/self.map_res),
                        "j": int(x_or_m["y"]/self.map_res)}

            # returns position of point on origin frame in meters
            return x_or_m, x_or_pix

    # main function of this class, using all of the above functions to find the goal and maintain found frontiers
    def do_ffd(self, pos_of_robot, yaw_rot_of_robot, laser_readings_polar, map_msg_data_reshape, previous_map, map_res,
               map_origin_position):

        # Getting actual data about robot's position and rotation
        self.pos_of_robot = pos_of_robot
        self.yaw_rot_of_robot = yaw_rot_of_robot

        # Getting actual map
        self.map_msg_data_reshape = map_msg_data_reshape
        self.map_res = map_res
        self.map_origin_pos = map_origin_position

        # Getting actual laser readings
        self.laser_readings_polar = laser_readings_polar

        # Maping laser readings to map (returning laser values mapped on map in pix)
        _, self.laser_readings_car_orig_pix = self.map_laser_readings(self.laser_readings_polar)

        # Getting lines between separate scan readings
        self.lines_list = self.get_lines_from_laser_readings(self.laser_readings_car_orig_pix)

        # Finding frontiers in lines
        self.frontier_list = self.find_frontiers_in_lines(self.lines_list)

        # Finding the centroids from all frontiers
        for frontier in self.frontier_list:
            frontier.calculate_centroid()  # calculating centroid for each frontier
            frontier.calculate_num_of_elements()

        # Sort frontiers from big to small
        self.frontier_list.sort(key=lambda x: x.num_of_elements, reverse=True)

        # if ffd algorithm found frontiers -> go to biggest one newly found
        if self.frontier_list:
            goal = {"x": self.frontier_list[0].centroid["i"] * self.map_res,
                    "y": self.frontier_list[0].centroid["j"] * self.map_res}

            del self.frontier_list[0]  # deleting frontier from frontier list

        elif not self.frontier_list and self.global_pool_of_frontiers or self.map_msg_data_reshape == previous_map:

            obst_flag = False
            unkn_flag = False

            while not obst_flag and unkn_flag:

                # transforming back frontier's centroid from map frame, [i, j] --> orig frame, [i, j]
                goal, self.global_pool_of_frontiers[0].centroid = \
                    self.transf_map2odom_and_back("map", "orig", self.global_pool_of_frontiers[0].centroid)
                # checking if frontier's centroid is useful and not explored goal
                unkn_flag, obst_flag = self.check_neighbours(self.global_pool_of_frontiers[0].centroid)
                del self.global_pool_of_frontiers[0]

        else:

            rospy.signal_shutdown("All Frontiers are visited")

        # TODO: here translating other frontiers centroids

        # Writing down found frontier to the global pool to use later if scan won't find any new frontiers

        for frontier in self.frontier_list:

            # transforming frontiers centroid from origin frame, [i, j] --> map frame, [i, j]
            _, frontier.centroid = self.transf_map2odom_and_back("orig", "map", frontier.centroid)
            # deleting all points, bcs centroid is already calculated and wrote down and it's all we need
            frontier.delete_all_points()
            # appending frontier with no points, only with found centroid in map frame
            self.global_pool_of_frontiers.append(frontier)

        #  from origin frame to /map frame and saving them global frontier pool

        self.frontier_idx += 1

        if self.debug_flag:
            # DEBUG

            print("         ||FFD||         ")
            print("+-----------+-----------+")
            print("| front idx | front num |")
            print("+-----------+-----------+")
            print("|   %d       |     %d     |" % (self.frontier_idx, self.frontier_list.__len__()))
            print("+-----------+-----------+")

            # DEBUG

            # DEBUG

            img = copy.deepcopy(self.map_msg_data_reshape).astype(np.uint8)

            # change map from grayscale to bgr (map will stay the same, but i can add colours for debug)
            img_3d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            i = int(goal["x"] / self.map_res)
            j = int(goal["y"] / self.map_res)
            # getting the coordinates of the pixel, corresponding to the scan measurement
            coords_pix = {
                "j": j if j < self.map_msg_data_reshape.shape[0] - 1 else self.map_msg_data_reshape.shape[0] - 1,
                "i": i if i < self.map_msg_data_reshape.shape[1] - 1 else self.map_msg_data_reshape.shape[1] - 1}

            cv2.namedWindow('Goal', cv2.WINDOW_NORMAL)  # new window, named 'win_name'

            img_3d[coords_pix["j"], coords_pix["i"], :] = np.array([0, 255, 0])

            cv2.imshow('Goal', img_3d)  # show image on window 'win_name' made of numpy.ndarray
            cv2.resizeWindow('Goal', 1600, 900)  # resizing window on my resolution

            cv2.waitKey(1000)  # wait for key pressing
            cv2.destroyAllWindows()  # close all windows

            # DEBUG END

        return goal


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

    previous_map_msg_data_reshape = copy.deepcopy(map_msg_data_reshape)     # "previous" map
    previous_map_msg_data_reshape[0][0] = 100                               # changing "previous" map

    map_msg_data_reshape = map_msg_data_reshape.astype(int)
    previous_map_msg_data_reshape = previous_map_msg_data_reshape.astype(int)

    # DEBUG

    cv2.namedWindow('map', cv2.WINDOW_NORMAL)  # new window, named 'win_name'
    cv2.imshow('map', copy.deepcopy(map_msg_data_reshape).astype(np.uint8))  # show image on window 'win_name' made of numpy.ndarray
    cv2.resizeWindow('map', 1600, 900)  # resizing window on my resolution

    cv2.waitKey(1000)  # wait for key pressing
    cv2.destroyAllWindows()  # close all windows

    # DEBUG END

    # (*) setting up scan

    scan_msg_data = np.asarray(scan_msg[11:371])  # laser readings in polar coordinates (counterclockwise from "x")
    np.reshape(scan_msg_data, (360,))
    scan_msg_data = scan_msg_data.astype(np.float64)

    pose_of_robot_pix = {"j": 15, "i": 7}

    yaw_rot_of_robot = 0  # yaw rotation angle 0.785 [rad] = 45 [deg]

    map_res = 0.15

    pose_of_robot_m = {"x": pose_of_robot_pix["i"] * map_res, "y": pose_of_robot_pix["j"] * map_res}

    x_map2or = {"x": -2.927416274335609, "y": -2.5452358128875714}

    # MAIN PART
    ffd = FFDdetectorCLass(True)

    goal_coords = ffd.do_ffd(pose_of_robot_m, yaw_rot_of_robot, scan_msg_data, map_msg_data_reshape, previous_map_msg_data_reshape,
               map_res, x_map2or)


if __name__ == '__main__':
    main()

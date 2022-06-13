#! /usr/bin/env python
import random
import numpy as np
import tf2_ros  # lib for translating and rotating the frames
from random import randint
import copy
from math import sqrt
import cv2
import pandas as pd
import skimage


# datatype for storing the cluster
class Cluster:

    def __init__(self, starting_point: dict, map_id: str, cluster_id: int, has_walls_inside_flag: bool = False):
        self.starting_point = starting_point  # starting point from which cluster is being searched further
        self.list_of_cells = np.asarray([[], []], dtype=int)  # list of cells of cluster
        self.map_id = map_id  # id of the map, on which clustering is conducted
        self.cluster_id = cluster_id  # enumerator of the cluster
        self.number_of_elements: int = 0  # number of cells in cluster
        self.cluster_centroid = {"j": None, "i": None}  # centroid of the cluster

        # Appending starting point to the cluster
        self.list_of_cells = np.concatenate((self.list_of_cells, np.asarray([[starting_point["j"]],
                                                                             [starting_point["i"]]])), axis=1)
        self.list_of_cells.astype(int, copy=False)

        # Flag if cluster has walls inside itself
        self.has_walls_inside_flag = has_walls_inside_flag

    def calculate_number_of_elements(self):
        self.number_of_elements = self.list_of_cells.shape[1]

    def calculate_centroid(self):
        # calculate the average coordinate:

        sum_coord_i = 0
        sum_coord_j = 0

        for cell in range(self.number_of_elements):
            sum_coord_j += self.list_of_cells[0][cell]
            sum_coord_i += self.list_of_cells[1][cell]

        self.cluster_centroid["i"] = int(sum_coord_i / self.number_of_elements)
        self.cluster_centroid["j"] = int(sum_coord_j / self.number_of_elements)

    def add_pixel(self, j, i):

        # Appending pixel to the cluster
        self.list_of_cells = np.concatenate((self.list_of_cells, np.asarray([[j], [i]])), axis=1)
        self.list_of_cells.astype(int, copy=False)


# Yamauchi's frontier detection method implemented in the class
class DFDdetectorClass:

    def __init__(self, min_size_frontier: float, percentage_cutoff: float):

        self.min_size_frontier = min_size_frontier  # minimum size of the frontier

        if percentage_cutoff > 1 or percentage_cutoff < 0:
            raise ValueError("DFD Error: The value of percentage cutoff should be between 0 and 1!")

        self.map_resolution: float = 0  # map resolution to calculate min_number_of_elements
        self.min_num_of_elements: int = 0  # will be calculated later

        self.percentage_cutoff = percentage_cutoff  # specifying how much top percentage
        # of gradient to left (0.3 = 70 %)

        self.idx = 1  # global index

        self.safe_boundary_between_goal_and_walls = 1   # [cells]

        self.raw_map_data_numpy_reshape = np.zeros((0, 0))   # global map for the class

    def map_gradient(self):
        map_I_copy = copy.deepcopy(self.raw_map_data_numpy_reshape)

        # initializing gradient magnitude of the map
        nmap_mag = np.zeros((map_I_copy.shape[0], map_I_copy.shape[1]), dtype="uint16")

        map_I_copy = map_I_copy.astype(np.int16)
        # assigning uint8 data type to the map, so value "-1" -> 255 and the biggest gradient is between -1:
        # not known and 100: occupied

        # changing values, so i'll get the biggest gradient between free cells: 0 and unknown: -1
        map_I_copy = np.where(map_I_copy == 100, 50, map_I_copy)  # Known Occupied: 100 -> 50
        map_I_copy = np.where(map_I_copy == -1, 100, map_I_copy)  # Unknown: -1 -> 100

        # Sobel kernel
        Gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int16)
        Gy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int16)

        # Convoluting over whole map with mask, except 1 pixel border
        for j in range(1, map_I_copy.shape[0] - 2):
            for i in range(1, map_I_copy.shape[1] - 2):
                submap = map_I_copy[j - 1: j + 2, i - 1: i + 2]  # submap 3x3 pixels

                ncell_I_x = (submap * Gx).sum()  # element-wise multiplying and getting sum of all elements
                ncell_I_y = (submap * Gy).sum()  # element-wise multiplying and getting sum of all elements

                nmap_mag[j][i] = sqrt(
                    ncell_I_x ** 2 + ncell_I_y ** 2)  # get magnitude of gradient (we don't need theta)

        # Filtering out: leaving only the biggest gradient:
        max_gradient = np.amax(nmap_mag)  # get max value of gradient

        nmap_mag = np.where(nmap_mag < self.percentage_cutoff * max_gradient, 0, nmap_mag)  # filter out all cells,
        # except ones with
        # the highest gradient
        # (top 90%)

        nmap_mag = np.where(nmap_mag >= self.percentage_cutoff * max_gradient, 255, nmap_mag)  # make top highest
        # gradient:
        # 255, due to dtype
        # - highest
        # value (and for viz)

        nmap_mag = nmap_mag.astype(np.uint8)  # assing uint8 type to gradient map, because values are either 0 or 255

        return nmap_mag

    def clustering(self, nmap_mag, robot_position):

        cluster_list: list = []  # creating list, which will store all cluster objects
        nmap_mag_copy = copy.deepcopy(nmap_mag)  # making copy of magnitude map, to be able to delete pixels from it

        labeled_nmap_mag_copy = skimage.measure.label(nmap_mag_copy)    # CCL implemented labeling (returns ndarray:int)

        n_clusters = np.amax(labeled_nmap_mag_copy)     # number of clusters (0-th cluster is background)

        # DEBUG

        cv2.namedWindow('CLUSTER ' + str(n_clusters), cv2.WINDOW_NORMAL)  # new window, named 'win_name'
        cv2.imshow('CLUSTER ' + str(n_clusters), nmap_mag_copy)  # show image on window 'win_name' made of numpy.ndarray
        cv2.resizeWindow('CLUSTER ' + str(n_clusters), 1600, 900)  # resizing window on my resolution

        cv2.waitKey(0)  # wait for key pressing
        cv2.destroyAllWindows()  # close all windows

        # DEBUG END

        for i_cluster in range(1, n_clusters):

            cluster_indices = np.where(labeled_nmap_mag_copy == i_cluster)

            starting_point = {"j": cluster_indices[0][0], "i": cluster_indices[1][0]}

            # checking if there is a wall in the cluster
            if np.any(self.raw_map_data_numpy_reshape[cluster_indices]) == 100:

                # doesn't add cluster to cluster list (deleting the cluster from nmap_mag_copy for visualisation)
                nmap_mag_copy[cluster_indices] = 0

            else:

                # initialising new cluster
                new_cluster = Cluster(starting_point, 'nmap_mag', i_cluster)

                # adding all cells to cluster
                for pixel in range(1, len(cluster_indices[0])):

                    new_cluster.add_pixel(cluster_indices[0][pixel], cluster_indices[1][pixel])

                new_cluster.calculate_number_of_elements()

                # if cluster is bigger than cluster_min_size -> calculate its centroid
                if new_cluster.number_of_elements > self.min_num_of_elements:

                    new_cluster.calculate_centroid()

                    cluster_list.append(copy.deepcopy(new_cluster))  # appending cluster to the cluster list

                # (deleting the cluster from nmap_mag_copy for visualisation)

                # DEBUG
                if new_cluster.cluster_centroid["j"]:
                    nmap_mag_copy[new_cluster.cluster_centroid["j"]][new_cluster.cluster_centroid["i"]] = 120

                cv2.namedWindow('CLUSTER '+str(i_cluster), cv2.WINDOW_NORMAL)  # new window, named 'win_name'
                cv2.imshow('CLUSTER '+str(i_cluster), nmap_mag_copy)  # show image on window 'win_name' made of numpy.ndarray
                cv2.resizeWindow('CLUSTER '+str(i_cluster), 1600, 900)  # resizing window on my resolution

                cv2.waitKey(0)  # wait for key pressing
                cv2.destroyAllWindows()  # close all windows

                # DEBUG END

            nmap_mag_copy[cluster_indices] = 0

        return cluster_list

    def frontier_detection_DFD(self, raw_map_data_numpy_reshape: np.ndarray,
                               raw_costmap_data_numpy_reshape: np.ndarray, robot_position_pix,
                               map_resolution: float, previous_map_reshape) -> dict:

        self.raw_map_data_numpy_reshape = raw_map_data_numpy_reshape

        self.map_resolution = map_resolution
        self.min_num_of_elements = int(self.min_size_frontier / self.map_resolution)

        gradient = self.map_gradient()
        cluster_list = self.clustering(gradient, robot_position_pix)

        distance_from_centroid = []

        # DEBUG

        # print("| CLUSTER ID | CENTROID  | DIST2CENT | ROBOT POS |  \n")
        # print("__________________________________________________  \n")

        # END OF DEBUG

        # computing distances to centroid for all clusters
        for cluster in cluster_list:

            c_j = cluster.cluster_centroid["j"]
            c_i = cluster.cluster_centroid["i"]
            centroid_neighbourhood = raw_map_data_numpy_reshape[c_j - self.safe_boundary_between_goal_and_walls:
                                                                c_j + self.safe_boundary_between_goal_and_walls,
                                                                c_i - self.safe_boundary_between_goal_and_walls:
                                                                c_i + self.safe_boundary_between_goal_and_walls]

            # Check if cluster centroid or any of its neighbours are in the wall(if in the wall -> remove,
            #                                                                   else -> calculate centroid):
            if np.all(centroid_neighbourhood != 100):
                # print("There is NO wall around!!!")
                distance_from_centroid.append(sqrt((robot_position_pix["i"] - cluster.cluster_centroid["i"]) ** 2 +
                                                   (robot_position_pix["j"] - cluster.cluster_centroid["j"]) ** 2))

                # DEBUG

                # print("|   ", cluster.cluster_id, "      |   [", [c_j, c_i], "]    |   ", distance_from_centroid[-1],
                #       "    |   ", [robot_position_pix["j"], robot_position_pix["i"]], "      |  \n")
                # print("______________________________________  \n")

                # END OF DEBUG

            else:
                # print("There is wall around!!!")
                cluster_list.remove(cluster)

        # sorting dict from min distance_from_centroid to max

        distance_from_centroid_sorted = copy.deepcopy(distance_from_centroid)
        distance_from_centroid_sorted.sort()

        # condition to go to other frontiers (not the closest one) if map doesn't change,
        # if map changes, go to the closest one

        if np.all(raw_map_data_numpy_reshape == previous_map_reshape):

            curr_distance = distance_from_centroid_sorted[self.idx]

            curr_distance_idx = distance_from_centroid.index(curr_distance)

            goal_coords_pix = {"i": cluster_list[curr_distance_idx].cluster_centroid["i"],
                               "j": cluster_list[curr_distance_idx].cluster_centroid["j"]}

            self.idx += 1

            # DEBUG

            # print("______________________________________  \n")
            # print("      THIS IS NOT CLOSEST FRONTIER      \n")
            # print("|   ", cluster_list[curr_distance_idx].cluster_id, "      |   [",
            #       [cluster_list[curr_distance_idx].cluster_centroid["j"],
            #        cluster_list[curr_distance_idx].cluster_centroid["i"]], "]    |   ",
            #       distance_from_centroid[curr_distance_idx], "    |   ",
            #       [robot_position_pix["j"], robot_position_pix["i"]], "      |  \n")
            # print("______________________________________  \n")

            # END OF DEBUG

        else:

            min_distance = distance_from_centroid_sorted[0]

            min_distance_idx = distance_from_centroid.index(min_distance)

            goal_coords_pix = {"i": cluster_list[min_distance_idx].cluster_centroid["i"],
                               "j": cluster_list[min_distance_idx].cluster_centroid["j"]}

            self.idx = 1  # zeroing the counter
            # DEBUG

            # print("______________________________________  \n")
            # print("      THIS IS CLOSEST FRONTIER          \n")
            # print("|   ", cluster_list[min_distance_idx].cluster_id, "      |   [",
            #       [cluster_list[min_distance_idx].cluster_centroid["j"],
            #        cluster_list[min_distance_idx].cluster_centroid["i"]], "]    |   ",
            #       distance_from_centroid[min_distance_idx], "    |   ",
            #       [robot_position_pix["j"], robot_position_pix["i"]], "      |  \n")
            # print("______________________________________  \n")

            # END OF DEBUG

        goal_coords_m = {"x": goal_coords_pix["i"]*map_resolution,
                         "y": goal_coords_pix["j"]*map_resolution}

        print("This is what comes out of DFD in [m]: \n", goal_coords_m)

        # DEBUG
        # raw_costmap_data_numpy_reshape_copy = copy.deepcopy(raw_costmap_data_numpy_reshape)
        # raw_costmap_data_numpy_reshape_copy = raw_costmap_data_numpy_reshape_copy.astype(np.uint8)
        # raw_costmap_data_numpy_reshape_copy[cluster_list[min_distance_idx].cluster_centroid["i"]][
        #     cluster_list[min_distance_idx].cluster_centroid["j"]] = 175
        #
        # name = "Goal: "+str(cluster_list[min_distance_idx].cluster_centroid["i"])+" "\
        #            + str(cluster_list[min_distance_idx].cluster_centroid["j"])
        #
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # new window, named 'win_name'
        # cv2.imshow(name, raw_costmap_data_numpy_reshape_copy)  # show image on window 'win_name' made of numpy.ndarray
        # cv2.resizeWindow(name, 1600, 900)  # resizing window on my resolution
        #
        # cv2.waitKey(0)  # wait for key pressing
        # cv2.destroyAllWindows()  # close all windows

        # END OF DEBUG

        return goal_coords_m


def main():

    # load up the "previous" map
    previous_test_map = np.asarray(pd.read_csv('previous_map_ref_copy.csv', sep=',', header=None))
    # load up map
    test_map = np.asarray(pd.read_csv('map_ref_copy.csv', sep=',', header=None))
    # trim "previous" map
    previous_test_map = test_map[1:-1, 1:-1]
    # trim map
    test_map = test_map[1:-1, 1:-1]

    # init detector class
    detector_test = DFDdetectorClass(0.0026, 0.9)
    # get goal coords
    goal_coords = detector_test.frontier_detection_DFD(test_map,
                                                       test_map,
                                                       {"j": 39, "i": 61},
                                                       0.05,
                                                       previous_test_map)

    # goal_coords = frontier_detector.frontier_detection_DFD(raw_map_data_numpy_reshape,
    #                                                        raw_costmap_data_numpy_reshape,
    #                                                        pose_of_robot_pix,
    #                                                        raw_map.info.resolution,
    #                                                        previous_map_reshape)


if __name__ == "__main__":
    main()




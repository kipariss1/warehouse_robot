import pandas as pd
import numpy as np
import cv2
import time
from math import sqrt
from ClusterClass import Cluster
import copy

if __name__ == "__main__":

    # IMPORTING THE CSV MAP

    df = pd.read_csv('map.csv', sep=',', header=None)

    map = np.asarray(df)

    # END OF IMPORTING THE CSV MAP


class DFDdetectorClass:

    def __init__(self, min_size_frontier: float, percentage_cutoff: float):

        self.min_size_frontier = min_size_frontier      # minimum size of the frontier

        if percentage_cutoff > 1 or percentage_cutoff < 0:
            raise ValueError("DFD Error: The value of percentage cutoff should be between 0 and 1!")

        self.map_resolution: float = 0                      # map resolution to calculate min_number_of_elements
        self.min_num_of_elements: int = 0                   # will be calculated later

        self.percentage_cutoff = percentage_cutoff      # specifying how much top percentage
                                                        # of gradient to left (0.3 = 70 %)

    def map_gradient(self, map_I: np.ndarray):

        map_I_copy = copy.deepcopy(map_I)

        # assigning uint8 data type to the map, so value "-1" -> 255 and the biggest gradient is between -1: not known and
        # 100: occupied
        map_I_copy.astype(np.uint8)

        # initializing gradient magnitude of the map
        nmap_mag = np.zeros((map_I_copy.shape[0], map_I_copy.shape[1]), dtype="uint8")

        # Sobel kernel
        Gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="uint8")
        Gy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="uint8")

        # Convoluting over whole map with mask, except 1 pixel border
        for i in range(1, map_I_copy.shape[0] - 2):
            for j in range(1, map_I_copy.shape[1] - 2):

                submap = map_I_copy[i - 1: i + 2, j - 1: j + 2]      # submap 3x3 pixels

                ncell_I_x = (submap * Gx).sum()             # element-wise multiplying and getting sum of all elements
                ncell_I_y = (submap * Gy).sum()             # element-wise multiplying and getting sum of all elements

                nmap_mag[i][j] = sqrt(ncell_I_x**2 + ncell_I_y**2)  # get magnitude of gradient (we don't need theta)

        # Filtering out: leaving only the biggest gradient
        max_gradient = np.amax(nmap_mag)                                    # get max value of gradient

        nmap_mag = np.where(nmap_mag < self.percentage_cutoff * max_gradient, 0, nmap_mag)  # filter out all cells,
                                                                                            # except ones with
                                                                                            # the highest gradient
                                                                                            # (top 90%)

        nmap_mag = np.where(nmap_mag >= self.percentage_cutoff * max_gradient, 255, nmap_mag)   # make top highest
                                                                                                # gradient:
                                                                                                # 255, due to dtype
                                                                                                # - highest
                                                                                                # value (and for viz)

        return nmap_mag

    def check_neighbours(self, nmap_mag, i, j, cluster):

        # Checking if any cells in the neighbourhood == 255 (high gradient)

        if nmap_mag[i][j - 1] == 255:

            cluster.add_pixel(i, j - 1)
            nmap_mag[i][j - 1] = 0
            self.check_neighbours(nmap_mag, i, j - 1, cluster)

        if nmap_mag[i - 1][j] == 255:

            cluster.add_pixel(i - 1, j)
            nmap_mag[i - 1][j] = 0
            self.check_neighbours(nmap_mag, i - 1, j, cluster)

        if nmap_mag[i - 1][j - 1] == 255:

            cluster.add_pixel(i - 1, j - 1)
            nmap_mag[i - 1][j - 1] = 0
            self.check_neighbours(nmap_mag, i - 1, j - 1, cluster)

        if nmap_mag[i][j + 1] == 255:

            cluster.add_pixel(i, j + 1)
            nmap_mag[i][j + 1] = 0
            self.check_neighbours(nmap_mag, i, j + 1, cluster)

        if nmap_mag[i + 1][j] == 255:

            cluster.add_pixel(i + 1, j)
            nmap_mag[i + 1][j] = 0
            self.check_neighbours(nmap_mag, i + 1, j, cluster)

        if nmap_mag[i + 1][j + 1] == 255:

            cluster.add_pixel(i + 1,  j + 1)
            nmap_mag[i + 1][j + 1] = 0
            self.check_neighbours(nmap_mag, i + 1, j + 1, cluster)

        if nmap_mag[i - 1][j + 1] == 255:

            cluster.add_pixel(i - 1, j + 1)
            nmap_mag[i - 1][j + 1] = 0
            self.check_neighbours(nmap_mag, i - 1, j + 1, cluster)

        if nmap_mag[i + 1][j - 1] == 255:

            cluster.add_pixel(i + 1, j - 1)
            nmap_mag[i + 1][j - 1] = 0
            self.check_neighbours(nmap_mag, i + 1, j - 1, cluster)

        return True

    def clustering(self, nmap_mag):

        cluster_list: list = []                         # creating list, which will store all cluster objects
        nmap_mag_copy = copy.deepcopy(nmap_mag)         # making copy of magnitude map, to be able to delete pixels from it
        j_cl = 0                                        # numerator of clusters

        frontier_indices = np.where(nmap_mag_copy == 255)   # searching for all cells on magnitude map,
                                                            # which have high gradient

        frontier_indices = np.asarray(frontier_indices)     # transforming to np.ndarray

        while frontier_indices.any():

            starting_point = {"i": frontier_indices[0][0], "j": frontier_indices[1][0]}     # define starting point
                                                                                            # check_neighbours function

            nmap_mag_copy[frontier_indices[0][0]][frontier_indices[1][0]] = 0               # deleting pixel, that is
                                                                                            # already in cluster
            # Defining new cluster:
            # print(str(j_cl) + " cluster !!!")  # DEBUG
            new_cluster = Cluster(starting_point, 'nmap_mag', j_cl)

            self.check_neighbours(nmap_mag_copy,  frontier_indices[0][0], frontier_indices[1][0], new_cluster)

            # DEBUG

            # cv2.namedWindow('CLUSTER '+str(j_cl), cv2.WINDOW_NORMAL)  # new window, named 'win_name'
            # cv2.imshow('CLUSTER '+str(j_cl), nmap_mag_copy)  # show image on window 'win_name' made of numpy.ndarray
            # cv2.resizeWindow('CLUSTER '+str(j_cl), 1600, 900)  # resizing window on my resolution
            #
            # cv2.waitKey(0)  # wait for key pressing
            # cv2.destroyAllWindows()  # close all windows

            # DEBUG END

            new_cluster.calculate_number_of_elements()          # calculate the number of cells in cluster
            if new_cluster.number_of_elements > self.min_num_of_elements:
                new_cluster.calculate_centroid()                    # calculate centroid of the cluster

                # DEBUG

                # nmap_mag[new_cluster.cluster_centroid["i"]][new_cluster.cluster_centroid["j"]] = 120
                #
                # cv2.namedWindow('CLUSTER '+str(j_cl), cv2.WINDOW_NORMAL)  # new window, named 'win_name'
                # cv2.imshow('CLUSTER '+str(j_cl), nmap_mag)  # show image on window 'win_name' made of numpy.ndarray
                # cv2.resizeWindow('CLUSTER '+str(j_cl), 1600, 900)  # resizing window on my resolution
                #
                # cv2.waitKey(0)  # wait for key pressing
                # cv2.destroyAllWindows()  # close all windows

                # DEBUG END

                cluster_list.append(copy.deepcopy(new_cluster))     # appending cluster to the cluster list

                j_cl = j_cl + 1                                   # increase cluster enumerator

            frontier_indices = np.where(nmap_mag_copy == 255)   # searching again for all cells on magnitude map,
                                                                # which have gradient ==255, not 0
                                                                # after check_neighbours function

            frontier_indices = np.asarray(frontier_indices)     # transforming to np.ndarray

        return cluster_list

    def frontier_detection_DFD(self, raw_map_data_numpy_reshape: np.ndarray,
                               raw_costmap_data_numpy_reshape: np.ndarray, robot_position,
                               map_resolution: float) -> dict:

        self.map_resolution = map_resolution

        self.min_num_of_elements = int(self.min_size_frontier/self.map_resolution)

        gradient = self.map_gradient(raw_map_data_numpy_reshape)
        cluster_list = self.clustering(gradient)

        distance_from_centroid = []

        for cluster in cluster_list:

            distance_from_centroid.append(sqrt((robot_position["x"] - cluster.cluster_centroid["i"])**2
                                               + (robot_position["y"] - cluster.cluster_centroid["j"])**2))

        min_distance = min(distance_from_centroid)
        min_distance_idx = distance_from_centroid.index(min_distance)

        goal_coords = {"x": cluster_list[min_distance_idx].cluster_centroid["i"],
                       "y": cluster_list[min_distance_idx].cluster_centroid["j"]}

        return goal_coords


# END OF MAIN FUNCTIONS #

if __name__ == "__main__":          # if u run script directly

    # BEGIN OF SETUP FOR TESTING
    a = time.time_ns()
    detector = DFDdetectorClass(0.5, 0.9)
    goal_coords = detector.frontier_detection_DFD(map, None, {"x": 40, "y": 21}, 0.033)
    print("This is goal coords", goal_coords)
    b = time.time_ns()
    print("This is how much time it takes to do DFD: ", (b-a)*10**-9, "[s]")

    # CV2 SETUP

    # cv2.namedWindow('map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
    # cv2.imshow('map', map)                    # show image on window 'win_name' made of numpy.ndarray
    # cv2.resizeWindow('map', 1600, 900)             # resizing window on my resolution
    #
    # cv2.waitKey(0)                                      # wait for key pressing
    # cv2.destroyAllWindows()                             # close all windows

    # cv2.namedWindow('original_map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
    # cv2.imshow('original_map', origin_map)                    # show image on window 'win_name' made of numpy.ndarray
    # cv2.resizeWindow('original_map', 1600, 900)             # resizing window on my resolution
    #
    # cv2.waitKey(0)                                      # wait for key pressing
    # cv2.destroyAllWindows()                             # close all windows


    # cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
    # cv2.imshow('gradient', gradient)                    # show image on window 'win_name' made of numpy.ndarray
    # cv2.resizeWindow('gradient', 1600, 900)             # resizing window on my resolution
    #
    # cv2.waitKey(0)                                      # wait for key pressing
    # cv2.destroyAllWindows()                             # close all windows

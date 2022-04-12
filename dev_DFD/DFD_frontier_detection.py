import pandas as pd
import numpy as np
import cv2
import time
from math import sqrt
from ClusterClass import Cluster


## Importing the csv

df = pd.read_csv('map.csv', sep=',', header=None)

map = np.asarray(df)


# MAIN FUNCTIONS #
def map_gradient(map_I: np.ndarray):

    # assigning uint8 data type to the map, so value "-1" -> 255 and the biggest gradient is between -1: not known and
    # 100: occupied
    map_I.astype(np.uint8)

    # initializing gradient magnitude of the map
    nmap_mag = np.zeros((map_I.shape[0], map_I.shape[1]), dtype="uint8")

    # Sobel kernel
    Gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="uint8")
    Gy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="uint8")

    # Convoluting over whole map with mask, except 1 pixel border
    for i in range(1, map_I.shape[0] - 2):
        for j in range(1, map_I.shape[1] - 2):

            submap = map_I[i - 1: i + 2, j - 1: j + 2]      # submap 3x3 pixels

            ncell_I_x = (submap * Gx).sum()             # element-wise multiplying and getting sum of all elements
            ncell_I_y = (submap * Gy).sum()             # element-wise multiplying and getting sum of all elements

            nmap_mag[i][j] = sqrt(ncell_I_x**2 + ncell_I_y**2)  # get magnitude of gradient (we don't need theta)

    # Filtering out: leaving only the biggest gradient
    max_gradient = np.amax(nmap_mag)                                    # get max value of gradient

    percentage_cutoff = 0.9                                             # specifying how much top percentage
                                                                        # of gradient to left (0.3 = 70 %)

    nmap_mag = np.where(nmap_mag < percentage_cutoff * max_gradient, 0, nmap_mag)       # filter out all cells,
                                                                                        # except ones with
                                                                                        # the highest gradient (top 90%)

    nmap_mag = np.where(nmap_mag >= percentage_cutoff * max_gradient, 255, nmap_mag)    # make top highest gradient:
                                                                                        # 255, due to dtype - highest
                                                                                        # value (and for viz)

    # REMOVE return map_I, needed only for debugging
    return nmap_mag, map_I


def check_neighbours(nmap_mag, i, j, cluster):

    # Checking if any cells in the neighbourhood == 255 (high gradient)

    if nmap_mag[i][j - 1] == 255:

        cluster.add_pixel(i, j - 1)
        check_neighbours(nmap_mag, i, j - 1, cluster)

    if nmap_mag[i - 1][j] == 255:

        cluster.add_pixel(i - 1, j)
        check_neighbours(nmap_mag, i - 1, j, cluster)

    if nmap_mag[i - 1][j - 1] == 255:

        cluster.add_pixel(i - 1, j - 1)
        check_neighbours(nmap_mag, i - 1, j - 1, cluster)

    if nmap_mag[i][j + 1] == 255:

        cluster.add_pixel(i, j + 1)
        check_neighbours(nmap_mag, i, j + 1, cluster)

    if nmap_mag[i + 1][j] == 255:

        cluster.add_pixel(i + 1, j)
        check_neighbours(nmap_mag, i + 1, j, cluster)

    if nmap_mag[i + 1][j + 1] == 255:

        cluster.add_pixel(i + 1,  j + 1)
        check_neighbours(nmap_mag, i + 1, j + 1, cluster)

    if nmap_mag[i - 1][j + 1] == 255:

        cluster.add_pixel(i - 1, j + 1)
        check_neighbours(nmap_mag, i - 1, j + 1, cluster)

    if nmap_mag[i + 1][j - 1] == 255:

        cluster.add_pixel(i + 1, j - 1)
        check_neighbours(nmap_mag, i + 1, j - 1, cluster)

    return True


def clustering(nmap_mag):

    frontier_indices = np.where(nmap_mag == 255)    # searching for all cells on magnitude map, which have high gradient
    frontier_indices = np.asarray(frontier_indices)     # transforming to np.ndarray
    cluster_list: list                              # creating list, which will store all cluster data-types

    for i_fi in range(0, np.shape(frontier_indices)[1] - 1):

        starting_point = {"i": frontier_indices[0][i_fi], "j": frontier_indices[1][i_fi]}

        # Defining new cluster:
        j_cl = 1                   # numerator of clusters
        print(str(j_cl) + " cluster !!!")
        exec("global cluster_" + str(j_cl))                                         # making new cluster global
        exec("cluster_" + str(j_cl) + " = Cluster(starting_point, 'nmag_map')")     # initialising new cluster object

        results: bool = False

        # TODO: copy of nmap_mag, send it to check_neighbours and change value on it 255 -> 0

        exec("results = check_neighbours(nmap_mag,  frontier_indices[0][i_fi], "
             "frontier_indices[1][i_fi], cluster_" + str(j_cl) + ")")

        if results:                 # check_neighbours returns true if all neighbours == 255 are added to cluster

            exec("cluster_list.append(cluster_" + str(j_cl) + ")")     # appending cluster to the cluster list

            # deleting the starting point as it is in cluster now:
            np.delete(frontier_indices, 0, i_fi)
            np.delete(frontier_indices, 1, i_fi)

            j_cl = j_cl + 1                                   # increase cluster enumerator

    return cluster_list


def frontier_detection_DFD(raw_map_data_numpy_reshape: np.ndarray,
                           raw_costmap_data_numpy_reshape: np.ndarray) -> dict:
    pass


# END OF MAIN FUNCTIONS #
a = time.time_ns()
gradient, origin_map = map_gradient(map)
cluster_list = clustering(gradient)
b = time.time_ns()
print("This is how much time it takes to do DFD: ", (b-a)*10**-9, "[s]")

# # CV2 SETUP
#
# cv2.namedWindow('map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
# cv2.imshow('map', map)                    # show image on window 'win_name' made of numpy.ndarray
# cv2.resizeWindow('map', 1600, 900)             # resizing window on my resolution

# cv2.waitKey(0)                                      # wait for key pressing
# cv2.destroyAllWindows()                             # close all windows

# cv2.namedWindow('original_map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
# cv2.imshow('original_map', origin_map)                    # show image on window 'win_name' made of numpy.ndarray
# cv2.resizeWindow('original_map', 1600, 900)             # resizing window on my resolution
#
# cv2.waitKey(0)                                      # wait for key pressing
# cv2.destroyAllWindows()                             # close all windows


cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('gradient', gradient)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('gradient', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows

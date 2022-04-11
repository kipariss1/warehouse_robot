import pandas as pd
import numpy as np
import cv2
import time
from math import sqrt


## Importing the csv

df = pd.read_csv('map.csv', sep=',', header=None)

map = np.asarray(df, dtype="uint8")

# MAIN FUNCTIONS #
def map_gradient(map_I: np.ndarray):

    # CHANGE VALUES OF MAP TO GET BIGGER GRADIENT ON FRONTIER BETWEEN NOT-KNOWN AND FREE CELLS
    # VALUE CHANGES: 0 -> 0 (FREE), -1 -> 100 (NOT-KNOWN), 100 -> 50 (OCCUPIED)
    map_I = np.where(map_I == 100, 50, map_I)
    map_I = np.where(map_I == -1, 100, map_I)

    nmap_mag = np.zeros((map_I.shape[0], map_I.shape[1]), dtype="uint8")

    # Sobel kernel

    Gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="uint8")
    Gy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="uint8")

    # Convoluting over whole map with mask, except 1 pixel border
    for i in range(1, map_I.shape[0] - 2):
        for j in range(1, map_I.shape[1] - 2):

            submap = map_I[i - 1: i + 2, j - 1: j + 2]      # submap 3x3 pixels

            ncell_I_x = (submap * Gx).sum()             # elemnt-wise multiplying and getting sum of all elements
            ncell_I_y = (submap * Gy).sum()             # elemnt-wise multiplying and getting sum of all elements

            nmap_mag[i][j] = sqrt(ncell_I_x**2 + ncell_I_y**2)  # get magnitude of gradient (we don't need theta)

    # Filtering out: leaving only the biggest gradient
    max_gradient = np.amax(nmap_mag)                                    # get max value of gradient
    nmap_mag = np.where(nmap_mag < 0.9 * max_gradient, 0, nmap_mag)       # filter out all cells, except ones with
                                                                        # the highest gradient (top 90 %)

    nmap_mag = np.where(nmap_mag >= 0.9 * max_gradient, 255, nmap_mag)  # ONLY FOR DEBUG NEEDED

    # REMOVE returtn map_I, needed only for debugging
    return nmap_mag, map_I


def frontier_detection_DFD(raw_map_data_numpy_reshape: np.ndarray,
                           raw_costmap_data_numpy_reshape: np.ndarray) -> dict:
    pass


# END OF MAIN FUNCTIONS #

test_map = np.asarray([[0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100],
                       [0, 0, 0, -1, -1, -1, -1, 100, 100]], dtype="uint8")

gradient, origin_map = map_gradient(map)
print(gradient)


# CV2 SETUP

cv2.namedWindow('map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('map', map)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('map', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows

cv2.namedWindow('original_map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('original_map', origin_map)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('original_map', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows


cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('gradient', gradient)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('gradient', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows

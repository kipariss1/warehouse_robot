import pandas as pd
import numpy as np
import cv2
import time
from math import atan, sqrt


## Importing the csv

df = pd.read_csv('map.csv', sep=',', header=None)

map = np.asarray(df, dtype="int8")


# MAIN FUNCTIONS #

#TODO: reassign -1, 0, 100 on map to something different, so the contrast would be higher

def map_gradient(map_I: np.ndarray):

    nmap_mag = np.zeros((map_I.shape[0], map_I.shape[1]), dtype="uint8")
    nmap_theta = np.zeros((map_I.shape[0], map_I.shape[1]))

    #Sobel kernel

    Gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="uint8")
    Gy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="uint8")

    # Convoluting over whole map, except 1 pixel border
    for i in range(1, map_I.shape[0] - 2):
        for j in range(1, map_I.shape[1] - 2):

            submap = map_I[i - 1: i + 2, j - 1: j + 2]      # submap 3x3 pixels

            ncell_I_x = (submap * Gx).sum()             # elemnt-wise multiplying and getting sum of all elements
            ncell_I_y = (submap * Gy).sum()             # elemnt-wise multiplying and getting sum of all elements

            nmap_mag[i][j] = sqrt(ncell_I_x**2 + ncell_I_y**2)
            nmap_theta[i][j] = atan(ncell_I_y/ncell_I_y)

    return nmap_mag, nmap_theta


def frontier_detection_DFD(raw_map_data_numpy_reshape: np.ndarray,
                           raw_costmap_data_numpy_reshape: np.ndarray) -> dict:
    pass


# END OF MAIN FUNCTIONS #

test_map = np.asarray([[0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126],
                       [0, 0, 0, 0, 126, 126, 126, 126]], dtype="uint8")

gradient, _ = map_gradient(test_map)
print(gradient)


# CV2 SETUP

cv2.namedWindow('original_map', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('original_map', test_map)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('original_map', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows


cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)      # new window, named 'win_name'
cv2.imshow('gradient', gradient)                    # show image on window 'win_name' made of numpy.ndarray
cv2.resizeWindow('gradient', 1600, 900)             # resizing window on my resolution

cv2.waitKey(0)                                      # wait for key pressing
cv2.destroyAllWindows()                             # close all windows

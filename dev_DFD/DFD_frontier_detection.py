import pandas as pd
import numpy as np
import cv2


## Importing the csv

df = pd.read_csv('map.csv', sep=',', header=None)

map = np.asarray(df, dtype="uint8")
map2 = np.asarray([[100, 250, 22], [25, 65, 185], [45, 64, 200]], dtype="uint8")


def frontier_detection_DFD(raw_map_data_numpy_reshape: np.ndarray,
                           raw_costmap_data_numpy_reshape: np.ndarray) -> dict:

    gradient_raw_map_data_numpy_reshape = np.asarray(np.gradient(raw_map_data_numpy_reshape, 1))
    gradient_raw_map_data_numpy_reshape.astype(np.uint8)

    return gradient_raw_map_data_numpy_reshape


gradient = frontier_detection_DFD(map, None)

cv2.namedWindow('win_name', cv2.WINDOW_NORMAL)
cv2.imshow('win_name', gradient)
cv2.resizeWindow('win_name', 1600, 900)

cv2.waitKey(0)
cv2.destroyAllWindows()


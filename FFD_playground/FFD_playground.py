import pandas as pd
import numpy as np
import cv2
import copy


# Fast Frontier Detection method implemented in the class
class FFDdetectorCLass:

    def __init__(self):

        pass


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


if __name__ == '__main__':

    main()


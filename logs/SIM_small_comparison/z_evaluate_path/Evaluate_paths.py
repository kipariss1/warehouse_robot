import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from math import sqrt
import json


def main():

    path = os.getcwd()
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    for file in csv_files:

        if "naive" in file:

            naive_rob_pos_map_df = pd.read_csv(file)

        if "DFD" in file:

            DFD_rob_pos_map_df = pd.read_csv(file)

        if "FFD" in file:

            FFD_rob_pos_map_df = pd.read_csv(file)

    last_pos_of_origin_in_map = {"x": -2.9458618531968863, "y": -2.5629726113141174}
    map_res = 0.15

    # NAIVE

    rob_pos_map_frame_naive = pd.concat([naive_rob_pos_map_df.loc[:, "field.x"],
                                         naive_rob_pos_map_df.loc[:, "field.y"]], axis=1)

    previous_rob_pos_map_frame_m_naive = {"x": None, "y": None}
    curr_rob_pos_map_frame_m_naive = {"x": None, "y": None}
    previous_rob_pos_org_frame_m_naive = {"x": None, "y": None}
    curr_rob_pos_org_frame_m_naive = {"x": None, "y": None}
    curr_rob_pos_org_frame_pix_naive = {"i": None, "j": None}
    rob_pos_org_frame_pix_i_naive = []
    rob_pos_org_frame_pix_j_naive = []
    total_path_m_naive = 0

    for rob_pos in rob_pos_map_frame_naive.iterrows():

        # if it's first value
        if rob_pos[0] == 0:

            previous_rob_pos_map_frame_m_naive["x"] = rob_pos[1]["field.x"]
            previous_rob_pos_map_frame_m_naive["y"] = rob_pos[1]["field.y"]

            # calculating previous coords in origin frame in [m]

            previous_rob_pos_org_frame_m_naive["x"] = previous_rob_pos_map_frame_m_naive["x"] - last_pos_of_origin_in_map["x"]
            previous_rob_pos_org_frame_m_naive["y"] = previous_rob_pos_map_frame_m_naive["y"] - last_pos_of_origin_in_map["y"]

        else:

            curr_rob_pos_map_frame_m_naive["x"] = rob_pos[1]["field.x"]
            curr_rob_pos_map_frame_m_naive["y"] = rob_pos[1]["field.y"]

            # ------------------------------------------------------------------------

            # calculating total path:

            total_path_m_naive += sqrt((curr_rob_pos_map_frame_m_naive["x"] - previous_rob_pos_map_frame_m_naive["x"])**2 +
                                       (curr_rob_pos_map_frame_m_naive["y"] - previous_rob_pos_map_frame_m_naive["y"])**2)

            # calculating coords in origin frame in [m]

            curr_rob_pos_org_frame_m_naive["x"] = curr_rob_pos_map_frame_m_naive["x"] - last_pos_of_origin_in_map["x"]
            curr_rob_pos_org_frame_m_naive["y"] = curr_rob_pos_map_frame_m_naive["y"] - last_pos_of_origin_in_map["y"]

            # calculating coords in origin frame in [pix]

            curr_rob_pos_org_frame_pix_naive["i"] = int(curr_rob_pos_org_frame_m_naive["x"] / map_res)
            curr_rob_pos_org_frame_pix_naive["j"] = int(curr_rob_pos_org_frame_m_naive["y"] / map_res)

            rob_pos_org_frame_pix_i_naive.append(curr_rob_pos_org_frame_pix_naive["i"])
            rob_pos_org_frame_pix_j_naive.append(curr_rob_pos_org_frame_pix_naive["j"])

            # ------------------------------------------------------------------------

            previous_rob_pos_map_frame_m_naive["x"] = curr_rob_pos_map_frame_m_naive["x"]
            previous_rob_pos_map_frame_m_naive["y"] = curr_rob_pos_map_frame_m_naive["y"]

            previous_rob_pos_org_frame_m_naive["x"] = curr_rob_pos_org_frame_m_naive["x"]
            previous_rob_pos_org_frame_m_naive["y"] = curr_rob_pos_org_frame_m_naive["y"]


    # DFD

    rob_pos_map_frame_DFD = pd.concat([DFD_rob_pos_map_df.loc[:, "field.x"],
                                       DFD_rob_pos_map_df.loc[:, "field.y"]], axis=1)

    previous_rob_pos_map_frame_m_DFD = {"x": None, "y": None}
    curr_rob_pos_map_frame_m_DFD = {"x": None, "y": None}
    previous_rob_pos_org_frame_m_DFD = {"x": None, "y": None}
    curr_rob_pos_org_frame_m_DFD = {"x": None, "y": None}
    curr_rob_pos_org_frame_pix_DFD = {"i": None, "j": None}
    rob_pos_org_frame_pix_i_DFD = []
    rob_pos_org_frame_pix_j_DFD = []
    total_path_m_DFD = 0

    for rob_pos in rob_pos_map_frame_DFD.iterrows():

        # if it's first value
        if rob_pos[0] == 0:

            previous_rob_pos_map_frame_m_DFD["x"] = rob_pos[1]["field.x"]
            previous_rob_pos_map_frame_m_DFD["y"] = rob_pos[1]["field.y"]

            # calculating previous coords in origin frame in [m]

            previous_rob_pos_org_frame_m_DFD["x"] = previous_rob_pos_map_frame_m_DFD["x"] - last_pos_of_origin_in_map["x"]
            previous_rob_pos_org_frame_m_DFD["y"] = previous_rob_pos_map_frame_m_DFD["y"] - last_pos_of_origin_in_map["y"]

        else:

            curr_rob_pos_map_frame_m_DFD["x"] = rob_pos[1]["field.x"]
            curr_rob_pos_map_frame_m_DFD["y"] = rob_pos[1]["field.y"]

            # ------------------------------------------------------------------------

            # calculating total path:

            total_path_m_DFD += sqrt((curr_rob_pos_map_frame_m_DFD["x"] - previous_rob_pos_map_frame_m_DFD["x"]) ** 2 +
                                     (curr_rob_pos_map_frame_m_DFD["y"] - previous_rob_pos_map_frame_m_DFD["y"]) ** 2)

            # calculating coords in origin frame in [m]

            curr_rob_pos_org_frame_m_DFD["x"] = curr_rob_pos_map_frame_m_DFD["x"] - last_pos_of_origin_in_map["x"]
            curr_rob_pos_org_frame_m_DFD["y"] = curr_rob_pos_map_frame_m_DFD["y"] - last_pos_of_origin_in_map["y"]

            # calculating coords in origin frame in [pix]

            curr_rob_pos_org_frame_pix_DFD["i"] = int(curr_rob_pos_org_frame_m_DFD["x"] / map_res)
            curr_rob_pos_org_frame_pix_DFD["j"] = int(curr_rob_pos_org_frame_m_DFD["y"] / map_res)

            rob_pos_org_frame_pix_i_DFD.append(curr_rob_pos_org_frame_pix_DFD["i"])
            rob_pos_org_frame_pix_j_DFD.append(curr_rob_pos_org_frame_pix_DFD["j"])

            # ------------------------------------------------------------------------

            previous_rob_pos_map_frame_m_DFD["x"] = curr_rob_pos_map_frame_m_DFD["x"]
            previous_rob_pos_map_frame_m_DFD["y"] = curr_rob_pos_map_frame_m_DFD["y"]

            previous_rob_pos_org_frame_m_DFD["x"] = curr_rob_pos_org_frame_m_DFD["x"]
            previous_rob_pos_org_frame_m_DFD["y"] = curr_rob_pos_org_frame_m_DFD["y"]

    # FFD

    rob_pos_map_frame_FFD = pd.concat([FFD_rob_pos_map_df.loc[:, "field.x"],
                                       FFD_rob_pos_map_df.loc[:, "field.y"]], axis=1)

    previous_rob_pos_map_frame_m_FFD = {"x": None, "y": None}
    curr_rob_pos_map_frame_m_FFD = {"x": None, "y": None}
    previous_rob_pos_org_frame_m_FFD = {"x": None, "y": None}
    curr_rob_pos_org_frame_m_FFD = {"x": None, "y": None}
    curr_rob_pos_org_frame_pix_FFD = {"i": None, "j": None}
    rob_pos_org_frame_pix_i_FFD = []
    rob_pos_org_frame_pix_j_FFD = []
    total_path_m_FFD = 0

    for rob_pos in rob_pos_map_frame_FFD.iterrows():

        # if it's first value
        if rob_pos[0] == 0:

            previous_rob_pos_map_frame_m_FFD["x"] = rob_pos[1]["field.x"]
            previous_rob_pos_map_frame_m_FFD["y"] = rob_pos[1]["field.y"]

            # calculating previous coords in origin frame in [m]

            previous_rob_pos_org_frame_m_FFD["x"] = previous_rob_pos_map_frame_m_FFD["x"] - last_pos_of_origin_in_map["x"]
            previous_rob_pos_org_frame_m_FFD["y"] = previous_rob_pos_map_frame_m_FFD["y"] - last_pos_of_origin_in_map["y"]

        else:

            curr_rob_pos_map_frame_m_FFD["x"] = rob_pos[1]["field.x"]
            curr_rob_pos_map_frame_m_FFD["y"] = rob_pos[1]["field.y"]

            # ------------------------------------------------------------------------

            # calculating total path:

            total_path_m_FFD += sqrt((curr_rob_pos_map_frame_m_FFD["x"] - previous_rob_pos_map_frame_m_FFD["x"]) ** 2 +
                                     (curr_rob_pos_map_frame_m_FFD["y"] - previous_rob_pos_map_frame_m_FFD["y"]) ** 2)

            # calculating coords in origin frame in [m]

            curr_rob_pos_org_frame_m_FFD["x"] = curr_rob_pos_map_frame_m_FFD["x"] - last_pos_of_origin_in_map["x"]
            curr_rob_pos_org_frame_m_FFD["y"] = curr_rob_pos_map_frame_m_FFD["y"] - last_pos_of_origin_in_map["y"]

            # calculating coords in origin frame in [pix]

            curr_rob_pos_org_frame_pix_FFD["i"] = int(curr_rob_pos_org_frame_m_FFD["x"] / map_res)
            curr_rob_pos_org_frame_pix_FFD["j"] = int(curr_rob_pos_org_frame_m_FFD["y"] / map_res)

            rob_pos_org_frame_pix_i_FFD.append(curr_rob_pos_org_frame_pix_FFD["i"])
            rob_pos_org_frame_pix_j_FFD.append(curr_rob_pos_org_frame_pix_FFD["j"])

            # ------------------------------------------------------------------------

            previous_rob_pos_map_frame_m_FFD["x"] = curr_rob_pos_map_frame_m_FFD["x"]
            previous_rob_pos_map_frame_m_FFD["y"] = curr_rob_pos_map_frame_m_FFD["y"]

            previous_rob_pos_org_frame_m_FFD["x"] = curr_rob_pos_org_frame_m_FFD["x"]
            previous_rob_pos_org_frame_m_FFD["y"] = curr_rob_pos_org_frame_m_FFD["y"]

    fig3 = plt.figure(3)
    fig3.set_size_inches(20 / 2.54, 20 / 2.54)  # 20x20 cm size
    map_img = plt.imread('last_map.png')
    ax = plt.subplot(111)
    path_naive, = ax.plot(rob_pos_org_frame_pix_i_naive, rob_pos_org_frame_pix_j_naive, 'r-')
    path_DFD, = ax.plot(rob_pos_org_frame_pix_i_DFD, rob_pos_org_frame_pix_j_DFD, 'g-')
    path_FFD, = ax.plot(rob_pos_org_frame_pix_i_FFD, rob_pos_org_frame_pix_j_FFD, 'b-')
    ax.legend([path_naive, path_DFD, path_FFD], ['Path naive', 'Path DFD', 'Path FFD'])
    ax.imshow(map_img, zorder=0, origin='lower')
    ax.set_ylabel("Height of the map [pix], res=" + str(round(map_res, 2)))
    ax.set_xlabel("Width of the map [pix], res=" + str(round(map_res, 2)))
    plt.suptitle("Path of robot with naive, DFD, FFD \n"+
                 f" - naive total path = {round(total_path_m_naive, 2)} [m]\n"+
                 f" - DFD total path = {round(total_path_m_DFD, 2)} [m]\n"+
                 f" - FFD total path = {round(total_path_m_FFD, 2)} [m]\n",
                 fontweight="bold")

    plt.show()  # showing plot
    # fig3.savefig(os.path.join(path, "/TotalPathOfRobotAllMethodsSmallMap.png"))

if __name__ == '__main__':

    main()


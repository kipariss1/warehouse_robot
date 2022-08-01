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

        if "rob_pos_map" in file:

            rob_pos_map_df = pd.read_csv(file)

        elif "FFD" or "DFD" or "naive" in file:

            map_iters_metadata_df = pd.read_csv(file)

    # PLOTTING THE NUMBER OF KNOWN CELLS:

    number_of_discovered_cells = map_iters_metadata_df.loc[:, 'Number of discovered cells']
    time_passed_str = map_iters_metadata_df.loc[:, 'Time passed']
    n_of_iteration = map_iters_metadata_df.loc[:, 'N of iteration']
    time_passed = []

    for time in time_passed_str: time_passed.append(datetime.strptime(time, "%M:%S"))   # converting str -> datetime

    # Plot: 1 subplot (Number of discovered cells on time), 2 subplot (Number of discovered cells on iterations)
    fig1 = plt.figure(1)
    axs1 = [plt.subplot(211), plt.subplot(212)]
    fig1.set_size_inches(20 / 2.54, 20 / 2.54)                                       # 20x20 cm size
    axs1[0].plot(time_passed, number_of_discovered_cells)                        # first subplot
    axs1[0].set_ylabel('N of discovered cells [-]')
    axs1[0].set_xlabel('Time [mm:ss]')
    axs1[0].set_title('Number of discovered cells with time')
    axs1[0].xaxis.set_ticks(np.arange(datetime.strptime("0:00", "%M:%S"),  # arranging how many dates will be on
                                      time_passed[-1],  # x-axis
                                      timedelta(0, 60)))
    axs1[0].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))             # set the formatting of dates on min:sec
    axs1[1].plot(n_of_iteration, number_of_discovered_cells)                     # second subplot
    axs1[1].set_ylabel('N of discovered cells [-]')
    axs1[1].set_xlabel('Iteration [-]')
    axs1[1].set_title('Number of discovered cells with iterations')
    plt.subplots_adjust(left=0.1,                                               # adjusting gaps between subplots
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.3)
    plt.suptitle("Number of discovered cells with FFD", fontweight="bold")

    # TODO: save the plot into "Results of evaluation" folder

    # PLOTTING THE VELOCITY OF OPENING NEW CELLS:

    number_of_newly_discovered_cells = map_iters_metadata_df.loc[:, 'Number of newly discovered cells']
    num_of_newl_open_cells = pd.concat([number_of_newly_discovered_cells, pd.Series(time_passed), n_of_iteration], axis=1)
    num_of_newl_open_cells = num_of_newl_open_cells[num_of_newl_open_cells["Number of newly discovered cells"] != 0]

    # Calculating velocity:
    previous_n_of_open_cells = 0
    previous_time = datetime.strptime("0:00", "%M:%S")      # making first time as 0:00
    vel_of_opening_cells = []

    for meas in num_of_newl_open_cells.iterrows():

        # if it's first value -> velocity is number of open cells
        if meas[1].iloc[2] == 1:

            vel_of_opening_cells.append(meas[1].iloc[0])    # append number of newly opened cells
            previous_time = meas[1].iloc[1]                 # making previous time

        # if it's not first value -> vel of opening cells = number of open cells/timedelta
        else:

            num_of_cells = meas[1].iloc[0]
            curr_time = meas[1].iloc[1]
            time_delta = curr_time - previous_time
            vel_of_opening_cells.append(num_of_cells/time_delta.total_seconds())

    fig2 = plt.figure(2)
    axs2 = [plt.subplot(211), plt.subplot(212)]
    fig2.set_size_inches(20 / 2.54, 20 / 2.54)  # 20x20 cm size
    axs2[0].plot(num_of_newl_open_cells.iloc[:, 1], vel_of_opening_cells)  # first subplot
    axs2[0].set_ylabel('Velocity of discovering cells [cells/s]')
    axs2[0].set_xlabel('Time [mm:ss]')
    axs2[0].set_title('Velocity of discovering cells with time')
    axs2[0].xaxis.set_ticks(np.arange(datetime.strptime("0:00", "%M:%S"),  # arranging how many dates will be on
                                      time_passed[-1],  # x-axis
                                      timedelta(0, 60)))
    axs2[0].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))  # set the formatting of dates on min:sec
    axs2[1].plot(num_of_newl_open_cells.iloc[:, 2], num_of_newl_open_cells.iloc[:, 0])  # second subplot
    axs2[1].set_ylabel('N of newly discovered cells [-]')
    axs2[1].set_xlabel('Iteration [-]')
    axs2[1].set_title('Number of newly discovered cells with iterations')
    plt.subplots_adjust(left=0.1,  # adjusting gaps between subplots
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.3)
    plt.suptitle("Velocity of discovering cells with FFD", fontweight="bold")

    # TODO: save the plot into "Results of evaluation" folder

    # PLOTTING PATH ON THE LAST MAP

    last_pos_of_origin_in_map_str = map_iters_metadata_df.loc[:, "Pose of origin in map frame"].iloc[-1]
    last_pos_of_origin_in_map = json.loads(last_pos_of_origin_in_map_str.replace("'", '"'))

    map_res = map_iters_metadata_df.loc[:, "Map resolution"].iloc[-1]   # [m/cell]

    rob_pos_map_frame = pd.concat([rob_pos_map_df.loc[:, "field.x"], rob_pos_map_df.loc[:, "field.y"]], axis=1)

    previous_rob_pos_map_frame_m = {"x": None, "y": None}
    curr_rob_pos_map_frame_m = {"x": None, "y": None}
    previous_rob_pos_org_frame_m = {"x": None, "y": None}
    curr_rob_pos_org_frame_m = {"x": None, "y": None}
    curr_rob_pos_org_frame_pix = {"i": None, "j": None}
    rob_pos_org_frame_pix_i = []
    rob_pos_org_frame_pix_j = []
    total_path_m = 0

    for rob_pos in rob_pos_map_frame.iterrows():

        # if it's first value
        if rob_pos[0] == 0:

            previous_rob_pos_map_frame_m["x"] = rob_pos[1]["field.x"]
            previous_rob_pos_map_frame_m["y"] = rob_pos[1]["field.y"]

            # calculating previous coords in origin frame in [m]

            previous_rob_pos_org_frame_m["x"] = previous_rob_pos_map_frame_m["x"] - last_pos_of_origin_in_map["x"]
            previous_rob_pos_org_frame_m["y"] = previous_rob_pos_map_frame_m["y"] - last_pos_of_origin_in_map["y"]

        else:

            curr_rob_pos_map_frame_m["x"] = rob_pos[1]["field.x"]
            curr_rob_pos_map_frame_m["y"] = rob_pos[1]["field.y"]

            # ------------------------------------------------------------------------

            # calculating total path:

            total_path_m += sqrt((curr_rob_pos_map_frame_m["x"] - previous_rob_pos_map_frame_m["x"])**2 +
                                 (curr_rob_pos_map_frame_m["y"] - previous_rob_pos_map_frame_m["y"])**2)

            # calculating coords in origin frame in [m]

            curr_rob_pos_org_frame_m["x"] = curr_rob_pos_map_frame_m["x"] - last_pos_of_origin_in_map["x"]
            curr_rob_pos_org_frame_m["y"] = curr_rob_pos_map_frame_m["y"] - last_pos_of_origin_in_map["y"]

            # calculating coords in origin frame in [pix]

            curr_rob_pos_org_frame_pix["i"] = int(curr_rob_pos_org_frame_m["x"]/map_res)
            curr_rob_pos_org_frame_pix["j"] = int(curr_rob_pos_org_frame_m["x"] / map_res)

            rob_pos_org_frame_pix_i.append(curr_rob_pos_org_frame_pix["i"])
            rob_pos_org_frame_pix_j.append(curr_rob_pos_org_frame_pix["j"])

            # ------------------------------------------------------------------------

            previous_rob_pos_map_frame_m["x"] = curr_rob_pos_map_frame_m["x"]
            previous_rob_pos_map_frame_m["y"] = curr_rob_pos_map_frame_m["y"]

            previous_rob_pos_org_frame_m["x"] = curr_rob_pos_org_frame_m["x"]
            previous_rob_pos_org_frame_m["y"] = curr_rob_pos_org_frame_m["y"]

    print(total_path_m)

    fig3 = plt.figure(3)
    map_img = plt.imread('last_map.png')
    ax = plt.subplot(111)
    ax.plot(rob_pos_org_frame_pix_i, rob_pos_org_frame_pix_j, 'xr-')
    ax.imshow(map_img, zorder=0)



    # For showing all plots
    plt.show()  # showing plot


if __name__ == '__main__':

    main()

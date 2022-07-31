import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates


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
    fig, axs = plt.subplots(2)                                                  # 2 subplots
    fig.set_size_inches(20/2.54, 20/2.54)                                       # 20x20 cm size
    axs[0].plot(time_passed, number_of_discovered_cells)                        # first subplot
    axs[0].set_ylabel('N of discovered cells [-]')
    axs[0].set_xlabel('Time [mm:ss]')
    axs[0].set_title('Number of discovered cells with time')
    axs[0].xaxis.set_ticks(np.arange(datetime.strptime("0:00", "%M:%S"),        # arranging how many dates will be on
                                     time_passed[-1],                           # x-axis
                                     timedelta(0, 60)))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))             # set the formatting of dates on min:sec
    axs[1].plot(n_of_iteration, number_of_discovered_cells)                     # second subplot
    axs[1].set_ylabel('N of discovered cells [-]')
    axs[1].set_xlabel('Iteration [-]')
    axs[1].set_title('Number of discovered cells with iterations')
    plt.subplots_adjust(left=0.1,                                               # adjusting gaps between subplots
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.3)
    plt.suptitle("Numbered of discovered cells with FFD", fontweight="bold")
    plt.show()                                                                  # showing plot

    # TODO: save the plot into "Results of evaluation" folder


if __name__ == '__main__':

    main()

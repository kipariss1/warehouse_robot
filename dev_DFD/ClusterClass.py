# IT IS DATATYPE FOR STORING CELLS OF THE CLUSTER

import numpy as np


class Cluster:

    def __init__(self, starting_point: dict, map_id: str, cluster_id: int):

        self.starting_point = starting_point            # starting point from which cluster is being searched further
        self.list_of_cells = np.asarray([[], []], dtype=int)       # list of cells of cluster
        self.map_id = map_id                            # id of the map, on which clustering is conducted
        self.cluster_id = cluster_id                    # enumerator of the cluster
        self.number_of_elements: int = 0                # number of cells in cluster
        self.cluster_centroid = {"i": None, "j": None}  # centroid of the cluster

        # Appending starting point to the cluster
        self.list_of_cells = np.concatenate((self.list_of_cells, np.asarray([[starting_point["i"]],
                                                                             [starting_point["j"]]])), axis=1)
        self.list_of_cells.astype(int, copy=False)

    def calculate_number_of_elements(self):

        self.number_of_elements = self.list_of_cells.shape[1]

    def calculate_centroid(self):

        # calculate the average coordinate:

        sum_coord_i = 0
        sum_coord_j = 0

        for cell in range(self.number_of_elements):

            sum_coord_i += self.list_of_cells[0][cell]
            sum_coord_j += self.list_of_cells[1][cell]

        self.cluster_centroid["i"] = int(sum_coord_i/self.number_of_elements)
        self.cluster_centroid["j"] = int(sum_coord_j/self.number_of_elements)

        print(self.cluster_centroid)        # DEBUG

    def add_pixel(self, i, j):

        print("This point was added to cluster: ", "[", str(i), ", ", str(j), "] \n")  # DEBUG

        # Appending pixel to the cluster
        self.list_of_cells = np.concatenate((self.list_of_cells, np.asarray([[i], [j]])), axis=1)
        self.list_of_cells.astype(int, copy=False)

        print(self.list_of_cells)  # DEBUG

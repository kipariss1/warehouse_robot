# IT IS DATATYPE FOR STORING CELLS OF THE CLUSTER

import numpy as np


class Cluster:

    def __init__(self, starting_point: dict, map_id: str, cluster_id: int):

        self.starting_point = starting_point            # starting point from which cluster is being searched further
        self.list_of_cells = np.asarray([[], []])       # list of cells of cluster
        self.map_id = map_id                            # id of the map, on which clustering is conducted
        self.cluster_id = cluster_id                    # enumerator of the cluster

        # Appending starting point to the cluster
        np.append(self.list_of_cells[0], starting_point["i"])
        np.append(self.list_of_cells[1], starting_point["j"])

    def calculate_width(self):

        # TODO: function to calculate width after search is over

        pass

    def calculate_height(self):

        # TODO: function to calculate width after search is over

        pass

    def add_pixel(self, i, j):

        print("This point was added to cluster: ", "[", str(i), ", ", str(j), "] \n")

        np.append(self.list_of_cells[0], i)
        np.append(self.list_of_cells[1], j)

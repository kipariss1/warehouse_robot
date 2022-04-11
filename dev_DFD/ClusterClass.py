
## It will be datatype

class Cluster:

    def __init__(self, starting_point, map_for_searching):

        self.starting_point = starting_point        # starting point from which cluster is being searched further
        self.map_for_searching = map_for_searching  # map on which, the cluster is initialised
        self.width: int                             # width of cluster
        self.height: int                            # height of cluster
        self.top_b: int                             # top border of cluster
        self.bottom_b: int                          # bottom border of cluster
        self.left_b: int                            # left border of cluster
        self.right_b: int                           # right border of cluster
        self.number_of_elements: int                # number of pixels in cluster
        self.coords_of_elements: list               # list of all coordinates

    def calculate_width(self):

        # TODO: function to calculate width after search is over

        pass

    def calculate_height(self):

        # TODO: function to calculate width after search is over

        pass

    def add_pixel(self):

        # TODO: add pixel to the list of coordinates

        pass

import numpy as np


class CandidatesAgent:
    """
    this class is responsable for the creation of the Candidates Sets
    """

    def __init__(self, settings, dist_matrix):
        self.settings = settings
        self.dist = dist_matrix
        self.num_neig = settings.K
        self.distances = dist_matrix

    def create_candidate(self, city1, city2):
        # creation empty data collector
        CL = set()
        for city_c in [city1, city2]:
            neig_list = []
            for cit_n in np.argsort(self.distances[city_c])[1:]:
                if len(neig_list) < self.num_neig:
                    neig_list.append(cit_n)
                if len(neig_list) == self.num_neig:
                    break
            CL |= set(neig_list)
        return CL

    def cl_for_stats(self, city1):
        CL = []
        for city in np.argsort(self.distances[city1])[1:9]:
            CL.append(city)
        return CL

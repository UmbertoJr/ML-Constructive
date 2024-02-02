import numpy as np


class OutputHandler:

    def __init__(self, settings, opt_tour):
        self.settings = settings
        self.opt_tour = opt_tour
        self.n = len(opt_tour)

    def create_output(self, city1, city2):
        idx_ = np.argwhere(self.opt_tour == city1)[0][0]
        # pos1 = np.argwhere(np.argsort(self.dist[city1]) == city2)[0][0]
        # pos2 = np.argwhere(np.argsort(self.dist[city2]) == city1)[0][0]
        # pos = min(pos1, pos2)
        return 1 if city2 in [self.opt_tour[idx_ - self.n + 1], self.opt_tour[idx_ - 1]] else 0

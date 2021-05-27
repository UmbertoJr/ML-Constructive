import numpy as np


class OutputHandler:

    def __init__(self, settings, opt_tour):
        self.settings = settings
        self.opt_tour = opt_tour
        self.n = len(opt_tour)

    def create_output(self, city1, city2):
        idx_ = np.argwhere(self.opt_tour == city1)[0][0]
        return 1 if city2 in [self.opt_tour[idx_ - self.n + 1], self.opt_tour[idx_ - 1]] else 0

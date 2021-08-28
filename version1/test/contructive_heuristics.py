from time import time

import numpy as np

from InOut.tools import evaluate_solution
from test.our_constructive import PreSelection
from test.classic_constructive import MultiFragment, ClarkeWright, FarthestInsertion


class Constructive:

    def __init__(self, name_solver, admin):
        self.admin = admin
        solvers = {
            "MF": self.mf,
            "CW": self.ck,
            "FI": self.fi,
            "ML-C": self.select_method('our'),
            "F": self.select_method('first'),
            "S": self.select_method('second'),
            "Y": self.select_method('yes'),
            "AE": self.select_empirical('average empirical'),
            "BE": self.select_empirical("best empirical"),
            "ML-SC": self.select_method('optimal')
        }
        self.solve = solvers[name_solver]

    def mf(self, *args, **kwargs):
        start = time()
        return MultiFragment.mf(self.admin.dist_matrix), time() - start, None

    def ck(self, *args, **kwargs):
        start = time()
        return ClarkeWright.cw(self.admin.dist_matrix), time() - start, None

    def fi(self, *args, **kwargs):
        start = time()
        return FarthestInsertion.solve(self.admin.dist_matrix, self.admin.pos), time() - start, None

    def select_method(self, method):
        def our_method(prob):
            solver = PreSelection(self.admin, prob=prob, method=method)
            start = time()
            return solver.solve(), time() - start, solver
        return our_method

    def select_empirical(self, mode):
        def run_empirical(*args):
            sols, lens, times = [], [], []
            for _ in range(20):
                solver = self.select_method("empirical")
                solution, t, _ = solver(1.)
                sols.append(solution)
                lens.append(evaluate_solution(solution, self.admin.dist_matrix))
                times.append(t)

            if mode == "average empirical":
                mean_len = np.mean(lens)
                min_v = 1e10
                average_sol = None
                for i, case in enumerate(lens):
                    if np.abs(mean_len - case) < min_v:
                        min_v = np.abs(mean_len - case)
                        average_sol = sols[i]
                return average_sol, np.mean(times), None
            elif mode == "best empirical":
                min_case = 1e10
                best_solution = None
                for i, case in enumerate(lens):
                    if case < min_case:
                        min_case = case
                        best_solution = sols[i]
                return best_solution, np.sum(times), None
        return run_empirical
from time import time
from test.our_constructive import PreSelection
from test.classic_constructive import MultiFragment, ClarkeWright, FarthestInsertion


class Constructive:

    def __init__(self, name_solver, admin):
        self.admin = admin
        solvers = {
            "MF": self.mf,
            "CW": self.ck,
            "FI": self.fi,
            "ML-G": self.select_method('our'),
            "first": self.select_method('first'),
            "second": self.select_method('second'),
            "yes": self.select_method('yes'),
            "empirical": self.select_method('empirical'),
            "ML-SC": self.select_method('optimal')
        }
        if 'empirical' in name_solver:
            name_solver = 'empirical'
        self.solve = solvers[name_solver]

    def mf(self, *args):
        start = time()
        return MultiFragment.mf(self.admin.dist_matrix), time() - start

    def ck(self, *args):
        start = time()
        return ClarkeWright.cw(self.admin.dist_matrix), time() - start

    def fi(self, *args):
        start = time()
        return FarthestInsertion.solve(self.admin.dist_matrix, self.admin.pos), time() - start

    def select_method(self, method):
        def our_method(prob):
            solver = PreSelection(self.admin, prob=prob, method=method)
            start = time()
            return solver.solve(), time() - start, solver
        return our_method

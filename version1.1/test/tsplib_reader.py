import os

import matplotlib.pyplot as plt
import pyproj as proj
from typing import List

import torch
import tempfile
import numpy as np
from concorde.tsp import TSPSolver
from torch.utils.data import Dataset
from numpy.core._multiarray_umath import ndarray
from InOut.image_creator import ImageTrainDataCreator
# from test.utils import possible_plots

# setup your projections
crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
crs_bng = proj.Proj(init='epsg:27700')  # use a locally appropriate projected CRS


class Read_TSP_Files:
    name: str
    nPoints: int
    pos: ndarray
    lines: List[str]
    dist_matrix: ndarray
    optimal_tour: ndarray

    def __init__(self):
        self.path = "./data/test/TSPLIB/"
        self.files = [ #"Luca22.tsp"
            "kroA100.tsp", "kroC100.tsp", "rd100.tsp",
            "eil101.tsp", "lin105.tsp", "pr107.tsp", "pr124.tsp", "bier127.tsp",
            "ch130.tsp", "pr136.tsp", "gr137.tsp", "pr144.tsp", "kroA150.tsp", "pr152.tsp", "u159.tsp", "rat195.tsp",
            "d198.tsp",
            "kroA200.tsp", "gr202.tsp", "ts225.tsp", "tsp225.tsp", "pr226.tsp",
            "gr229.tsp", "gil262.tsp",
            "pr264.tsp", "a280.tsp", "pr299.tsp",
            "lin318.tsp", "rd400.tsp", "fl417.tsp",
            "gr431.tsp",
            "pr439.tsp", "pcb442.tsp", "d493.tsp",
            "att532.tsp", "u574.tsp",
            "rat575.tsp", "d657.tsp", "gr666.tsp",
            "u724.tsp", "rat783.tsp", "pr1002.tsp",
            "u1060.tsp", "vm1084.tsp", "pcb1173.tsp",
            "d1291.tsp", "rl1304.tsp", "rl1323.tsp",
            "nrw1379.tsp", "fl1400.tsp","u1432.tsp",
            "fl1577.tsp", "d1655.tsp", "vm1748.tsp",
        ]

        self.distance_formula_dict = {
            'EUC_2D': self.distance_euc,
            'ATT': self.distance_att,
            'GEO': self.distance_geo
        }

    def instances_generator(self):
        for file in self.files[:]:
            yield self.read_instance(self.path + file)

    def read_instance(self, name_tsp):
        # read raw data
        file_object = open(name_tsp)
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

        # store data set information
        # print(name_tsp)
        self.name = self.lines[0].split(' ')[1]
        self.nPoints = np.int32(self.lines[3].split(' ')[1])
        self.distance = self.lines[4].split(' ')[1]
        self.distance_formula = self.distance_formula_dict[self.distance]

        # read all data points and store them
        self.pos = np.zeros((self.nPoints, 2))
        for i in range(self.nPoints):
            line_i = self.lines[6 + i].split(' ')
            # if "gr137" in name_tsp:
            #     print(line_i)
            self.pos[i, 0] = float(line_i[1])
            self.pos[i, 1] = float(line_i[2])

        self.create_dist_matrix()
        if os.path.exists(f"{self.path}optimal/{self.name}.npy"):
            self.optimal_tour = self.optimal_solutions(self.name)
        else:
            self.optimal_tour = self.optimal_solver()

        # if self.name == "GEO"
        # x, y = proj.transform(crs_wgs, crs_bng, input_lon, input_lat)
        return self.nPoints, self.pos, self.dist_matrix, self.name, self.optimal_tour

    @staticmethod
    def distance_euc(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        return round(np.sqrt(delta_x ** 2 + delta_y ** 2), 0)

    @staticmethod
    def distance_att(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        rij = np.sqrt((delta_x ** 2 + delta_y ** 2) / 10.0)
        tij = float(rij)
        if tij < rij:
            dij = tij + 1
        else:
            dij = tij
        return dij

    @staticmethod
    def distance_geo(zi, zj):
        RRR = 6378.388
        lati_i, long_i = Read_TSP_Files.get_lat_long(zi)
        lati_j, long_j = Read_TSP_Files.get_lat_long(zj)
        q1 = np.cos(long_i - long_j)
        q2 = np.cos(lati_i - lati_j)
        q3 = np.cos(lati_i + lati_j)
        return float(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3
                                             )) + 1.0)


    @staticmethod
    def get_lat_long(z):
        lati = Read_TSP_Files.to_radiant(z[0])
        long = Read_TSP_Files.to_radiant(z[1])
        return lati, long

    @staticmethod
    def to_radiant(angle):
        _deg = float(angle)
        _min = angle - _deg
        return np.pi * (_deg + 5.0 * _min / 3.0) / 180.0

    def create_dist_matrix(self):
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = self.distance_formula(self.pos[i], self.pos[j])
        self.dist_matrix += self.dist_matrix.T

    def optimal_solver(self) -> ndarray:
        """

        :rtype: ndarray
        """
        dir = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)
            solver = TSPSolver.from_data(
                self.pos[:, 0],
                self.pos[:, 1],
                norm=self.distance
            )
            solution = solver.solve()
        os.chdir(dir)
        np.save(f"{self.path}optimal/{self.name}.npy", solution.tour)
        return solution.tour

    def optimal_solutions(self, name_instance):
        return np.load(f"{self.path}optimal/{self.name}.npy")


class EvalGenerator(Dataset):
    X: object
    Y: object

    def __init__(self, settings):
        super(EvalGenerator, self).__init__()
        self.reader = Read_TSP_Files()
        self.bs_test = 128
        self.image_creator = ImageTrainDataCreator(settings)
        self.create_new_data = self.image_creator.create_data_for_all
        # self.len = self.find_len()
        # self.len = 31025
        self.len = 31228
        self.create_testdata()

    def __len__(self):
        return self.len

    def create_testdata(self):
        all_input, all_output = [], []
        all_output2 = []
        for data in self.reader.instances_generator():
            number_cities, pos, dist_matrix, name, optimal_tour = data
            data_to_give = [number_cities, pos, optimal_tour]

            input_, output_ = self.create_new_data(data_to_give)
            all_input.append(input_)
            all_output.append(output_)

        self.X = torch.cat(tuple(all_input), dim=0)
        self.Y = torch.cat(tuple(all_output), dim=0)

    def __getitem__(self, item):
        return {"X": self.X[item], "Y": self.Y[item]}

    def find_len(self):
        tot_images = 0
        for data in self.reader.instances_generator():
            number_cities, pos, dist_matrix, name, optimal_tour = data
            tot_images += self.image_creator.get_num_of_images(number_cities, pos)
            # print(number_cities, tot_images)

        print(tot_images)
        return int(tot_images)


def test_TSPLIB_generator(settings):
    print('\n\n\n\n')
    instances_generator = Read_TSP_Files()
    for problem_data in instances_generator.instances_generator():
        npoints, pos, dist_, name, opt_tour = problem_data
        print(name, npoints, pos.shape, dist_.shape, instances_generator.distance)
        np.savetxt("dist_matrix.csv", dist_, delimiter=",")

        # plotter = possible_plots(pos, 0.)
        # plotter.plot_current_sol(pos, opt_tour)
        # plt.savefig('LucaOpt.png')
        # print(dist_)
        print(opt_tour)

        print()

if __name__ == '__main__':
    test_TSPLIB_generator(None)
import os
import h5py  
import tempfile
import numpy as np
from concorde.tsp import TSPSolver
from scipy.spatial.distance import pdist

def save(data, seed, hf, num_inst_file):
    all_dimensions, all_pos, all_tours = data

    # Define variable-length datatype for positions and tours
    float_vlen_type = h5py.special_dtype(vlen=np.dtype('float64'))
    int_vlen_type = h5py.special_dtype(vlen=np.dtype('int'))

    for it in range(num_inst_file):
        seed_to_add = seed + it
        group = hf.create_group(f'seed_{seed_to_add}')

        # Number of cities does not vary within an instance but across instances, so it can be saved as is
        group.create_dataset("num_cities", shape=(1,), dtype=np.int, chunks=True, data=np.array(all_dimensions[it]))

        # For positions and tours, use the variable-length datatype
        group.create_dataset("pos", shape=(len(all_pos[it]),), dtype=float_vlen_type, data=all_pos[it])

        group.create_dataset("optimal_tour", shape=(len(all_tours[it]),), dtype=int_vlen_type, data=all_tours[it])


# def save(data, seed, hf, num_inst_file):
#     all_dimensions, all_pos, all_tours = data

#     for it in range(num_inst_file):
#         seed_to_add = seed + it
#         group = hf.create_group(f'seed_{seed_to_add}')
#         group.create_dataset(f"num_cities", shape=(1,),
#                              dtype=np.int, chunks=True, data=np.array(all_dimensions[it]))

#         group.create_dataset(f"pos", shape=all_pos[it].shape,
#                              dtype=np.float, chunks=True, data=all_pos[it])

#         group.create_dataset(f"optimal_tour", shape=all_tours[it].shape,
#                              dtype=np.int, chunks=True,
#                              data=all_tours[it])


class GenerateInstances:

    def __init__(self, settings, stats=False):
        self.settings = settings
        if stats:
            self.range_down = 500
            self.range_up = 1000
        else:
            self.range_down = 100
            self.range_up = 300

    def create_instances(self, num_instances, seed_starting_from):
        data = [self.create_data(j + seed_starting_from) for j in range(num_instances)]
        return self.organize_data(data)

    def create_data(self, j):
        num_cities, pos = self.create_instance(j)
        directory = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)
            solver = TSPSolver.from_data(
                pos[:, 0] * 100000,
                pos[:, 1] * 100000,
                norm="EUC_2D"
            )
            solution = solver.solve()

        os.chdir(directory)

        return num_cities, pos, solution.tour

    def create_instance(self, j):
        np.random.seed(j)
        num_cities, pos = self.__call__()
        return num_cities, pos

    def __call__(self):
        number_cities = np.random.randint(self.range_down, self.range_up)
        pos = np.random.uniform(-0.5, 0.5, size=number_cities * 2).reshape((number_cities, 2))
        return number_cities, pos

    @staticmethod
    def distance_mat(pos):
        distance = GenerateInstances.create_upper_matrix(pdist(pos, "euclidean"), pos.shape[0])
        distance = np.round((distance.T + distance) * 1000, 0) / 1000
        return distance

    @staticmethod
    def organize_data(data):
        all_dimensions, all_pos, all_tours = ([] for _ in range(3))
        for i in range(len(data)):
            all_dimensions.append(data[i][0])
            all_pos.append(data[i][1])
            all_tours.append(data[i][2])

        all_dimensions, \
        all_pos, \
        all_tours = map(np.array, (all_dimensions,
                                   all_pos,
                                   all_tours))

        return all_dimensions, all_pos, all_tours

    @staticmethod
    def create_upper_matrix(values, size):
        """
        builds an upper matrix
        @param values: to insert in the matrix
        @param size: of the matrix
        @return:
        """
        upper = np.zeros((size, size))
        r = np.arange(size)
        mask = r[:, None] < r
        upper[mask] = values
        return upper

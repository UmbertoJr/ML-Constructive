import os
import h5py  
import tempfile
import numpy as np
from concorde.tsp import TSPSolver
from scipy.spatial.distance import pdist

def save(data, seed, hf, num_inst_file):
    all_dimensions, all_pos, all_tours = data
    
    for it in range(num_inst_file):
        seed_to_add = seed + it
        group = hf.create_group(f'seed_{seed_to_add}')
        
        # Save the number of cities directly, since it's scalar and consistent within an instance
        group.create_dataset("num_cities", data=np.array([all_dimensions[it]]))

        # Save positions
        pos_dtype = h5py.special_dtype(vlen=float)  # Define variable-length float type
        pos_dataset = group.create_dataset("pos", (len(all_pos[it]),), dtype=pos_dtype)
        for idx, pos in enumerate(all_pos[it]):
            pos_dataset[idx] = pos  # Assigning each position array to the dataset
       
        # Save optimal tours directly as an array
        tour_dataset = group.create_dataset("optimal_tour", data=all_tours[it])
       


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
        original_directory = os.getcwd()  # Store the original directory
        try:
            # Use a try-finally block to ensure the directory is always reverted
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)  # Change to the temporary directory
                solver = TSPSolver.from_data(
                    pos[:, 0] * 100000,
                    pos[:, 1] * 100000,
                    norm="EUC_2D"
                )
                solution = solver.solve(verbose=0)  # Suppress Concorde's output
        finally:
            os.chdir(original_directory)  # Revert to the original directory regardless of success or failure

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
        all_dimensions = []
        all_pos = []
        all_tours = []
        for num_cities, pos, tour in data:
            all_dimensions.append(num_cities)
            all_pos.append(pos) 
            all_tours.append(tour) 

        all_dimensions = np.array(all_dimensions)  

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

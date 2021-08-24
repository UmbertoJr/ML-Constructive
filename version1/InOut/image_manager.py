import os
from abc import ABC

import torch
import numpy as np
from h5py import File
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist

from InOut.tools import DirManager
from InOut.image_creator import ImageTrainDataCreator, ImageTestCreator
from InOut.utils import sort_the_list_of_files, slice_iterator, to_torch, plot_single_cv
from InOut.output_agent import OutputHandler


class DatasetHandler(Dataset):

    def __init__(self, settings, path=False):
        self.file = False
        self.train = False
        self.settings = settings
        self.current_initial = 0
        self.starting_instance = 0
        self.current_available = 0
        self.ls_ind = 0
        self.slice = slice(0, 1)
        self.cases = settings.cases_in_L_P
        self.dir_ent = DirManager(settings)
        if path:
            self.eval = True
            self.path = path
            self.files_saved = os.listdir(self.path)
            self.num_instances_x_file = 1000
            self.starting_seed = 123
            self.tot_num_of_instances = 1000
            self.new_instances = 5
        else:
            self.eval = False
            self.path = self.dir_ent.folder_instances
            self.files_saved = sort_the_list_of_files(self.path)[:settings.last_file]
            self.num_instances_x_file = settings.num_instances_x_file
            self.starting_seed = 10000
            self.tot_num_of_instances = self.settings.total_number_instances
            self.new_instances = 10
        self.file = File(f"{self.path}/{self.files_saved[0]}", "r")
        self.image_creator = ImageTrainDataCreator(settings, cases=self.cases)

        # self.len = self.find_len()
        self.len = 1206121
        # self.len = 7880633
        self.create_new_data = self.image_creator.create_data_for_all

    def __len__(self):
        return self.len

    def __getitem__(self, slice_):
        if str(slice_).isdigit():
            n = int(slice_)
            if n >= self.current_available:
                self.ls_ind = n - self.current_initial
                self.load_new_pics()
                self.ls_ind = n - self.current_initial
                return {"X": self.X[self.ls_ind], "Y": self.Y[self.ls_ind]}
            else:
                self.ls_ind = n - self.current_initial
                return {"X": self.X[self.ls_ind], "Y": self.Y[self.ls_ind]}

    def load_new_pics(self):
        slice_ = slice(self.starting_instance, self.starting_instance + self.new_instances)
        self.starting_instance += self.new_instances
        self.slice = slice_

        if self.chek_if_to_load:
            self.index_file = self.from_slice_to_indexfile()
            self.load_new_file()

        count_new_images = 0
        all_input, all_output = [], []
        index_slice = [self.starting_seed + j for j in slice_iterator(slice_)]
        initial_key = index_slice[0]
        for _ in range(self.new_instances):
            data, new_images = self.load_data_from_file(initial_key)
            count_new_images += new_images

            input_, output_ = self.create_new_data(data)

            all_input.append(input_)
            all_output.append(output_)
            initial_key += 1

            if initial_key >= self.starting_seed + self.starting_instance:
                break
        new_X = torch.cat(tuple(all_input), dim=0)
        new_Y = torch.cat(tuple(all_output), dim=0)

        self.current_initial += self.ls_ind
        self.X, self.Y = new_X, new_Y
        self.current_available += count_new_images

    @property
    def chek_if_to_load(self):
        slice = self.slice
        start_sl = str(slice).replace("(", ",").replace(")", ",").split(',')[1]
        if int(start_sl) % self.settings.num_instances_x_file == 0:
            return True
        else:
            return False

    def from_slice_to_indexfile(self):
        slice_ = self.slice
        start_sl = str(slice_).replace("(", ",").replace(")", ",").split(',')[1]
        return int(start_sl) // self.settings.num_instances_x_file

    def find_len(self):
        tot_images = 0
        key = self.starting_seed
        for i in range(len(self.files_saved)):
            self.index_file = i
            self.load_new_file()
            for j in range(self.num_instances_x_file):
                number_cities = self.file[f'//seed_{key}'][f'num_cities'][...]
                pos = self.file[f'//seed_{key}'][f'pos'][...]
                tot_images += self.image_creator.get_num_of_images(number_cities[0], pos)
                key += 1

        print(tot_images)
        return tot_images

    def load_new_file(self):
        self.file.close()
        self.file = File(f"{self.path}/{self.files_saved[self.index_file]}", "r")

    def load_data_from_file(self, key):
        actual_key = key
        number_cities = self.file[f'//seed_{actual_key}'][f'num_cities'][...]
        pos = self.file[f'//seed_{actual_key}'][f'pos'][...]
        tour = self.file[f'//seed_{actual_key}'][f'optimal_tour'][...]
        number_cities = number_cities[0]
        new_images = self.image_creator.get_num_of_images(number_cities, pos)
        return (number_cities, pos, tour), new_images


class OnlineDataSetHandler(Dataset, ABC):

    def __init__(self, settings, model, mode='train'):
        self.file = False
        self.settings = settings
        self.cases = settings.cases_in_L_P
        self.dir_ent = DirManager(settings)
        self.path = self.dir_ent.folder_instances
        self.mode = mode
        if mode == 'eval':
            self.files_saved = os.listdir("./data/eval/")
            self.num_instances_x_file = 1000
            self.tot_num_of_instances = 1000
        else:
            self.files_saved = sort_the_list_of_files(self.path)[:settings.last_file]
            self.num_instances_x_file = settings.num_instances_x_file
            self.tot_num_of_instances = self.settings.total_number_instances

        self.len = 256
        self.model = model
        if next(model.parameters()).is_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # self.model.load_state_dict(torch.load(f'./data/net_weights/CL_{self.settings.cases_in_L_P}/best_diff.pth',
        #                                       map_location=self.device))
        self.model.load_state_dict(torch.load(f'./data/net_weights/CL_{self.settings.cases_in_L_P}/checkpoint.pth',
                                              map_location=self.device))

    def get_data(self):
        x, y = [], []
        for _ in range(3):
            file_name = np.random.choice(self.files_saved, size=1)[0]
            if self.mode == 'eval':
                initial_key = 123
            else:
                initial_key = int(file_name[:5])
            file = File(f"{self.path}/{file_name}", "r")
            actual_key = initial_key + np.random.randint(self.num_instances_x_file)
            number_cities = file[f'//seed_{actual_key}'][f'num_cities'][...]
            number_cities = number_cities[0]
            pos = file[f'//seed_{actual_key}'][f'pos'][...]
            opt_tour = file[f'//seed_{actual_key}'][f'optimal_tour'][...]

            output_handler = OutputHandler(self.settings, opt_tour)
            image_creator = ImageTestCreator(self.settings, pos=pos)
            partial_solution = {str(i): [] for i in range(number_cities)}
            dist_matrix = self.distance_mat(pos)
            L_P = self.create_LP(dist_matrix)
            for i, j in L_P:
                if self.condition_to_enter_sol(i, j, partial_solution):
                    image, too_close = image_creator.get_image(i, j, partial_solution)
                    x.append(image)
                    if too_close:
                        self.add_to_sol(i, j, partial_solution)
                        continue
                    else:
                        out_ = output_handler.create_output(i, j)
                    y.append(out_)
                    self.model.eval()
                    image_ = torch.tensor(image, dtype=torch.float).to(self.device)
                    image_ = image_[None, :, :, :]
                    image_ = image_.permute(0, 3, 1, 2)
                    ret = self.model(image_)
                    ret = ret.detach().cpu().numpy()[0]
                    # print(image_.shape)
                    # print(out_, ret)
                    # plot_single_cv(image)
                    if ret[0] < 0.01:
                        self.add_to_sol(i, j, partial_solution)

        X = to_torch(np.stack(x, axis=0)).to(self.device)
        Y = torch.tensor(np.stack(y), dtype=torch.long).to(self.device)
        return X[:self.settings.bs], Y[:self.settings.bs]

    def add_to_sol(self, node1, node2, dict_sol):
        dict_sol[str(node1)].append(node2)
        dict_sol[str(node2)].append(node1)
        return dict_sol

    def __len__(self):
        return self.len

    def create_LP(self, dist_matrix):
        len_neig = self.settings.cases_in_L_P
        LP_v = {i: {} for i in range(len_neig)}
        keys = []
        return_list = []
        n = dist_matrix.shape[0]
        neighborhood = self.create_neigs(dist_matrix, n)
        for in_cl in range(len_neig):
            for node in range(n):
                h = neighborhood[node][in_cl]
                if (node, h) not in keys and (h, node) not in keys:
                    LP_v[in_cl][(node, h)] = dist_matrix[node, h]
                    keys.append((node, h))

        for in_cl in range(len_neig):
            return_list.extend([k for k, v in sorted(LP_v[in_cl].items(), key=lambda item: item[1])])
        return return_list

    def create_neigs(self, dist_matrix, n):
        neigs = {}
        for i in range(n):
            neigs[i] = np.argsort(dist_matrix[i])[1: self.settings.K + 1]
        return neigs

    def condition_to_enter_sol(self, node1, node2, dict_sol):
        if self.check_if_one(node1, node2, dict_sol):
            if self.innerLoopTracker([node1, node2], dict_sol):
                return True
        elif self.check_if_available(node1, node2, dict_sol):
            return True
        return False

    def check_if_available(self, n1, n2, sol):
        return True if (bool(n1 != n2) and len(sol[str(n1)]) < 2 and len(sol[str(n2)]) < 2) else False

    def check_if_one(self, n1, n2, sol):
        return True if (bool(n1 != n2) and len(sol[str(n1)]) == 1 and len(sol[str(n2)]) == 1) else False

    @staticmethod
    def innerLoopTracker(edge_to_append, sol):
        n1, n2 = edge_to_append
        if len(sol[str(n1)]) == 0:
            return True
        if len(sol[str(n2)]) == 0:
            return True
        cur_city = sol[str(n1)][0]
        partial_tour = [n1, cur_city]
        while True:
            if len(sol[str(cur_city)]) == 2:
                for i in sol[str(cur_city)]:
                    if i not in partial_tour:
                        cur_city = i
                        partial_tour.append(cur_city)
                        if cur_city == n2:
                            return False
            else:
                return True

    def distance_mat(self, pos):
        distance = self.create_upper_matrix(pdist(pos, "euclidean"), pos.shape[0])
        distance = np.round((distance.T + distance) * 1000, 0) / 1000
        return distance

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

import os

import torch
from h5py import File
from torch.utils.data import Dataset

from InOut.tools import DirManager
from InOut.utils import sort_the_list_of_files, slice_iterator
from InOut.image_creator import ImageTrainDataCreator


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

        self.len = self.find_len()
        # self.len = 1206121
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
                # tot_images += number_cities[0]
                key += 1

        # len_tot_images = tot_images * self.cases
        # return int(len_tot_images)
        print(tot_images)
        return tot_images

    def load_new_file(self):
        self.file.close()
        self.file = File(f"{self.path}/{self.files_saved[self.index_file]}", "r")

    def load_data_from_file(self, key):
        actual_key = key
        # while actual_key < self.starting_seed + self.tot_num_of_instances:
        number_cities = self.file[f'//seed_{actual_key}'][f'num_cities'][...]
        pos = self.file[f'//seed_{actual_key}'][f'pos'][...]
        tour = self.file[f'//seed_{actual_key}'][f'optimal_tour'][...]
        number_cities = number_cities[0]
        new_images = self.image_creator.get_num_of_images(number_cities, pos)
        return (number_cities, pos, tour), new_images



import os
import h5py
from tqdm import tqdm
from time import time
from InOut.tools import DirManager, create_folder
from instances_generator.generator import GenerateInstances, save


def create_instances(settings):
    start = time()
    dir_ent = DirManager(settings)

    number_instances_per_file = settings.num_instances_x_file
    number_files = settings.total_number_instances // number_instances_per_file
    
    iterator_on_files = tqdm(range(number_files))

    gt = GenerateInstances(settings)
    for file in iterator_on_files:
        seed = file * number_instances_per_file + 10000

        # Define the file path
        file_path = f"{dir_ent.folder_instances}{seed}_file.h5"

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping or handling accordingly.")
            continue  # Skip this iteration and hence, don't overwrite the file

        # to save created data
        data = gt.create_instances(number_instances_per_file, seed)

        # Open the file in write mode, which will create the file
        with h5py.File(file_path, "w") as hf:
            save(data, seed, hf, number_instances_per_file)

    print(f"the total time to create the instances is : {time() - start} secs")



def create_statistical_study_data(settings):
    start = time()
    number_instances_per_file = 1000
    gt = GenerateInstances(settings, stats=True)

    # to save created data
    data = gt.create_instances(number_instances_per_file, 123)
    path = create_folder("data/eval/")
    hf = h5py.File(f"{path}evaluation.h5", "w")
    save(data, 123, hf, number_instances_per_file)
    hf.close()

    print(f"the total time to create the instances is : {time() - start} secs")

import h5py
from tqdm import tqdm
from time import time
from InOut.tools import DirManager, create_folder
from instances_generator.generator import Generate_Instances, save


def create_instances(settings):
    start = time()
    dir_ent = DirManager(settings)

    number_instances_per_file = settings.num_instances_x_file
    number_files = settings.total_number_instances // number_instances_per_file
    print(f"total number of files {number_files}")

    iterator_on_files = tqdm(range(number_files))

    gt = Generate_Instances(settings)
    for file in iterator_on_files:
        seed = file * number_instances_per_file + 10000

        # to save created data
        data = gt.create_instances(number_instances_per_file, seed)

        hf = h5py.File(f"{dir_ent.folder_instances}{seed}_file.h5", "w")
        save(data, seed, hf, number_instances_per_file)
        hf.close()

    print(f"the total time to create the instances is : {time() - start} secs")


def create_statistical_study_data(settings):
    start = time()
    number_instances_per_file = 1000
    gt = Generate_Instances(settings, stats=True)

    # to save created data
    data = gt.create_instances(number_instances_per_file, 123)
    path = create_folder("data/eval/")
    hf = h5py.File(f"{path}evaluation.h5", "w")
    save(data, 123, hf, number_instances_per_file)
    hf.close()

    print(f"the total time to create the instances is : {time() - start} secs")

from InOut.tools import print_sett
from InOut import test_images_created
from model.network import check_network
from test import test_metrics_on_TSPLIB
from instances_generator.test import stat_plots
from instances_generator import create_instances, create_statistical_study_data
from test.looking_for_best_configuration import show_results, train_the_best_configuration


class Settings:
    operation: str = 'create instances'
    K: int = 30
    num_pixels: int = 96
    ray_dot: int = 1
    thickness_edge: int = 1
    total_number_instances: int = 40960
    num_instances_x_file: int = 1280
    cases_in_L_P: int = 2
    last_file: int = 30
    bs: int = 256

    def __init__(self):
        pass


operations = {
    "create training instances": create_instances,
    "create evaluation instances": create_statistical_study_data,
    "statistical study": stat_plots,
    'test images': test_images_created,
    "test net": check_network,
    "train": train_the_best_configuration,
    "show results train": show_results,
    "test on TSPLIB": test_metrics_on_TSPLIB,
    "": None
}

if __name__ == '__main__':
    settings = Settings()
    # settings.operation = 'create statistical study instances'
    # settings.operation = 'statistical study'
    # settings.operation = 'test net'
    # settings.operation = 'test images'
    settings.operation = 'train'
    # settings.operation = 'show results train'
    # settings.operation = "get the best network parameters after first train"
    # # settings.operation = "test reconstruction"
    settings.operation = 'test on TSPLIB'
    print_sett(settings)
    operations[settings.operation](settings)

from InOut import test_images_created
from InOut.tools import print_sett
from instances_generator.test import stat_plots
from instances_generator import create_instances, create_statistical_study_data
from model import train_the_net
from model.network import check_network
from test.looking_for_best_configuration import show_results, train_the_best_configuration
from test import test_on_constr, test_metrics_on_TSPLIB, average_on_different_checks


class Settings:
    operation: str = 'create instances'
    K: int = 15
    num_pixels: int = 96
    ray_dot: int = 1
    thickness_edge: int = 1
    total_number_instances: int = 40960
    num_instances_x_file: int = 1280
    cases_in_L_P: int = 3
    last_file: int = 15
    bs: int = 132  #256

    def __init__(self):
        pass


operations = {
    "create instances": create_instances,
    "create statistical study instances": create_statistical_study_data,
    "statistical study": stat_plots,
    'test input images': test_images_created,
    "test net": check_network,
    "train": train_the_net,
    "show results train": show_results,
    "get the best network parameters after first train": train_the_best_configuration,
    # "test": test_on_eval,
    # "test TSPLIB files": test_TSPLIB_generator,
    "test reconstruction": test_on_constr,
    "test metrics on TSPLIB": test_metrics_on_TSPLIB,
    # "check distributions": check_distributions_across_different_heuristics,
    "test variations": average_on_different_checks,
    "": None
}

if __name__ == '__main__':
    settings = Settings()
    # settings.operation = 'create statistical study instances'
    # settings.operation = 'statistical study'
    # settings.operation = 'test net'
    # settings.operation = 'train'
    # settings.operation = 'show results train'
    settings.operation = "get the best network parameters after first train"
    # settings.operation = "test reconstruction"
    # settings.operation = 'test metrics on TSPLIB'
    # settings.operation = 'test variations'
    print_sett(settings)
    operations[settings.operation](settings)

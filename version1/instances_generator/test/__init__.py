import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from InOut.utils import distance_mat
from InOut.tools import create_folder, SampleCreator, plot_histogram, with_hue, change_width
from InOut.candidateSets import CandidatesAgent
from InOut.image_manager import DatasetHandler, slice_iterator
from InOut.output_agent import OutputHandler


class CheckCL(DatasetHandler):

    def __init__(self, settings):
        super(CheckCL, self).__init__(settings, path='data/eval/')

    def __len__(self):
        return self.len

    def take_distribution(self, from_, to_):
        slice_ = slice(from_, to_)
        self.slice = slice_
        if self.chek_if_to_load:
            self.index_file = self.from_slice_to_indexfile()
            self.load_new_file()

        index_slice = [self.starting_seed + j for j in slice_iterator(slice_)]
        tot_number_positive_cases = 0
        counts_cl = np.zeros(8)
        for key in index_slice:
            number_cities = self.file[f'//seed_{key}'][f'num_cities'][...]
            pos = self.file[f'//seed_{key}'][f'pos'][...]
            tour = self.file[f'//seed_{key}'][f'optimal_tour'][...]
            number_cities = number_cities[0]
            assert pos.shape[0] == number_cities, f"shape pos {pos.shape},  numb cities {number_cities}, seed {key}"

            tot_number_positive_cases += number_cities
            array_counts = distribution_fun(self.settings, number_cities, pos, tour)
            counts_cl += array_counts

        return counts_cl, tot_number_positive_cases


def stat_plots(settings):
    generator = CheckCL(settings)
    bs = 100
    print('\ncandidate list distribution\n')
    data_logger = tqdm(range(10))
    tot_counts_cl, tot_p = np.zeros(8), 0
    for i in data_logger:
        counts_cl, tp = generator.take_distribution(i * bs, (i + 1) * bs)
        tot_counts_cl += counts_cl
        tot_p += tp

    data = pd.DataFrame({"Position in the CL": np.arange(1, 9), "Positive Cases PDF": tot_counts_cl / (tot_p * 2)})
    ax = sns.barplot(x="Position in the CL", y="Positive Cases PDF", data=data, color='tab:blue')
    with_hue(ax, data, 8, 1)
    create_folder(f'data/images/')
    plt.savefig('data/images/opt_distr_CL.png')
    plt.show()

    # print('\n\nTrue Positive Rate and False Positive Rate\n')
    # check_distributions_across_different_heuristics(settings)


def distribution_fun(settings, number_cities, pos, optimal_tour):
    """
    this function is used to check the distribution of the outputs in the training dataset

    :param settings: the settings for the experiment
    :param number_cities: cities's number in the current instance
    :param pos: coordinates for the current instance
    :param optimal_tour: optimal tour solved using Concorde
    :return: a list called output that has angles or index neigbors depending on the settings
    """
    # creation empty data collector
    counts_CL = np.zeros(8)

    dist_matrix = distance_mat(pos)
    pos -= np.mean(pos, axis=0)

    # preprocessing for candidates
    candidates_agent = CandidatesAgent(settings, dist_matrix)

    # init the output handler
    output_handler = OutputHandler(settings, optimal_tour)
    for city in range(number_cities):
        # select the candidate set for the current city
        CL = candidates_agent.cl_for_stats(city)

        for position, city2 in enumerate(CL):
            #
            counts_CL[position] += output_handler.create_output(city, city2)

    return counts_CL



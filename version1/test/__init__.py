import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Tester_on_eval, Logger

from test.utils import OutcomeAdmin
from InOut.tools import DirManager
from test.local_search import LocalSearch
from test.contructive_heuristics import Constructive
from instances_generator.test import SampleCreator, plot_histogram
from test.tsplib_reader import Read_TSP_Files

to_plot = False
just_one = to_plot


def test_on_eval(settings):
    dir_ent = DirManager(settings)
    device = 'cpu'
    log_str_fun = Logger.log_pred
    tester = Tester_on_eval(settings, dir_ent, log_str_fun, settings.cases_in_L_P, device)


def test_on_constr(settings):
    generator_instance = Read_TSP_Files()
    constructive_algs = ['MF', 'CW', 'FI', 'ML-G']
    data = {}
    all_df = []
    metrics = ["probability"]
    for solv in constructive_algs:
        metrics.extend([f"gap {solv}", f"acc {solv}", f"time {solv}"])

    # for prob in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, .08, 0.85, 0.9, 0.95, 1]:
    # for prob in np.linspace(0.94, 1., 7):
    prob = 0.88
    for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
        admin = OutcomeAdmin(problem_data, constructive_algs, settings)
        if to_plot: admin.plot_solution(admin.optimal_tour, "ottimo")
        data[admin.name] = [prob]
        for constr in constructive_algs:
            constr_heur = Constructive(constr, admin)
            # create simple solution
            sol_no_pre, time_to_solve = constr_heur.solve(prob)
            if to_plot: admin.plot_solution(sol_no_pre, constr)
            admin.save(sol_no_pre, method=constr, time=time_to_solve)

        list_to_save = create_list_to_save(constructive_algs, admin)
        data[admin.name].extend(list_to_save)
        if just_one: break

    df_result = pd.DataFrame.from_dict(data, orient='index', columns=metrics)
    df_result.loc['mean'] = df_result.mean()
    df_result.loc['std'] = df_result.std()
    # all_df.append(df_result)

    # big_df = pd.concat(all_df, ignore_index=True)
    # big_df.to_csv('./data/test/all_results_random.csv')
    # big_df.to_csv('./data/test/selected_results_rete2.csv')
    df_result.to_csv('./data/test/reconstruction/results_rete_prob08.csv')


def create_solvers_names(constructive_algs, improvement):
    solvers = []
    for const in constructive_algs:
        for impr in improvement:
            solvers.extend([f'{const}', f'{const} + {impr}'])
    return solvers


def create_list_to_save(solvers, admin):
    list_to_save = []
    for solver in solvers:
        list_to_save.extend([admin.gaps[solver],
                             admin.accs[solver],
                             admin.time[solver]])
    return list_to_save


def test_metrics_on_TSPLIB(settings):
    constructive_algs = ['first', 'ML-G', 'ML-SC']
    data_p = {'Method': [], 'Position in the CL': [], 'True Positive Rate': []}
    data_n = {'Method': [], 'Position in the CL': [], 'False Positive Rate': []}
    prob = 0.88
    data = {}
    metrics = ["probability"]
    for solv in constructive_algs:
        metrics.extend([f"gap {solv}", f"acc {solv}", f"time {solv}"])
    generator_instance = Read_TSP_Files()

    for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
        admin = OutcomeAdmin(problem_data, constructive_algs, settings)
        sc = SampleCreator(admin.dist_matrix, admin.optimal_tour)
        data[admin.name] = [prob]
        for constructive in constructive_algs:
            greedy_heuristic = Constructive(constructive, admin)

            # create simple solution
            sol_no_pre, time_to_solve = greedy_heuristic.solve(prob)
            admin.save(sol_no_pre, method=constructive, time=time_to_solve)

            data_p, data_n = sc.save_new_data(data_p, data_n, admin.sols[constructive], constructive)

        list_to_save = create_list_to_save(constructive_algs, admin)
        data[admin.name].extend(list_to_save)

    plot_histogram(data_p, case='True Positive Rate')
    plot_histogram(data_n, case='False Positive Rate')

    df_result = pd.DataFrame.from_dict(data, orient='index', columns=metrics)
    df_result.loc['mean'] = df_result.mean()
    df_result.loc['std'] = df_result.std()
    # all_df.append(df_result)

    # big_df = pd.concat(all_df, ignore_index=True)
    # big_df.to_csv('./data/test/all_results_random.csv')
    # big_df.to_csv('./data/test/selected_results_rete2.csv')
    df_result.to_csv('./data/test/reconstruction/results_ML-G.csv')


def average_on_different_checks(settings):
    generator_instance = Read_TSP_Files()
    # constructive_algs = ['first', 'second', 'yes',
    #                      'no', 'empirical', 'our', 'optimal']
    constructive_algs = ["our",
                         "empirical1", "empirical2", "empirical3", "empirical4", "empirical5",
                         "empirical6", "empirical7", "empirical8", "empirical9", "empirical10",
                         "empirical11", "empirical12", "empirical13", "empirical14", "empirical15",
                         "empirical16", "empirical17", "empirical18", "empirical19", "empirical20",
                         ]
    data = {}
    all_df = []
    metrics = ["probability"]
    for solv in constructive_algs:
        metrics.extend([f"gap {solv}", f"acc {solv}", f"time {solv}"])

    prob = 0.88
    for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
        admin = OutcomeAdmin(problem_data, constructive_algs, settings)
        data[admin.name] = [prob]
        for constr in constructive_algs:
            constr_heur = Constructive(constr, admin)
            # create simple solution
            sol_no_pre, time_to_solve = constr_heur.solve(prob)
            admin.save(sol_no_pre, method=constr, time=time_to_solve)

        list_to_save = create_list_to_save(constructive_algs, admin)
        data[admin.name].extend(list_to_save)

    df_result = pd.DataFrame.from_dict(data, orient='index', columns=metrics)
    df_result.loc['mean'] = df_result.mean()
    df_result.loc['std'] = df_result.std()
    # all_df.append(df_result)

    # big_df = pd.concat(all_df, ignore_index=True)
    # big_df.to_csv('./data/test/all_results_random.csv')
    # big_df.to_csv('./data/test/selected_results_rete2.csv')
    df_result.to_csv('./data/test/reconstruction/test_diversi_rico_policies.csv')


def check_distributions_across_different_heuristics(settings) -> None:
    constructive_algs = ['MF', 'CW']
    data_p = {'Method': [], 'Position in the CL': [], 'True Positive Rate': []}
    data_n = {'Method': [], 'Position in the CL': [], 'False Positive Rate': []}
    prob = 0.88
    generator_instance = Read_TSP_Files()

    for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
        admin = OutcomeAdmin(problem_data, constructive_algs, settings)
        # print(admin.optimal_tour)
        sc = SampleCreator(admin.dist_matrix, admin.optimal_tour)
        for constructive in constructive_algs:
            greedy_heuristic = Constructive(constructive, admin)

            # create simple solution
            sol_no_pre, time_to_solve = greedy_heuristic.solve(prob)
            admin.save(sol_no_pre, method=constructive, time=time_to_solve)

            data_p, data_n = sc.save_new_data(data_p, data_n, admin.sols[constructive], constructive)

    plot_histogram(data_p, case='True Positive Rate')
    plot_histogram(data_n, case='False Positive Rate')

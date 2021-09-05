import pandas as pd
from tqdm import tqdm
from test.utils import Tester_on_eval, Logger

from test.utils import OutcomeAdmin
from InOut.tools import DirManager, create_folder
from test.local_search import LocalSearch
from test.contructive_heuristics import Constructive
from InOut.tools import SampleCreator, plot_histogram
from test.tsplib_reader import Read_TSP_Files

to_plot = False
just_one = to_plot


def test_on_eval(settings):
    dir_ent = DirManager(settings)
    device = 'cpu'
    log_str_fun = Logger.log_pred
    tester = Tester_on_eval(settings, dir_ent, log_str_fun, settings.cases_in_L_P, device)


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
    # constructive_algs = ['MF', 'FI', 'CW', 'F', 'S', 'Y', 'AE',  'BE', 'ML-C', 'ML-SC']
    constructive_algs = ['F', 'ML-C']
    # constructive_algs = ['MF', 'FI', 'CW']
    data_p = {'Method': [], 'Position in the CL': [], 'True Positive Rate': []}
    data_n = {'Method': [], 'Position in the CL': [], 'False Positive Rate': []}
    for prob in [1e-2]:
        metrics = []
        data = {}
        for solv in constructive_algs:
            metrics.extend([f"gap {solv}", f"acc {solv}", f"time {solv}"])
        generator_instance = Read_TSP_Files()
        # sc = Sampler()
        for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
            admin = OutcomeAdmin(problem_data, constructive_algs, settings)
            sc = SampleCreator(admin.dist_matrix, admin.optimal_tour)
            data[admin.name] = []
            for constructive in constructive_algs:
                greedy_heuristic = Constructive(constructive, admin)

                # create simple solution
                sol_no_pre, time_to_solve, solver = greedy_heuristic.solve(prob=[0.01, 0.01])
                admin.save(sol_no_pre, method=constructive, time=time_to_solve)

                print(constructive, admin.gaps[constructive], admin.name)
                print("\n\n")
                # sc.save_from_constructor(solver.P, solver.N, solver.TP, solver.TN, solver.cases)
                data_p, data_n = sc.save_new_data(data_p, data_n, admin.sols[constructive], constructive)

            # input()
            list_to_save = create_list_to_save(constructive_algs, admin)
            data[admin.name].extend(list_to_save)

        df_result = pd.DataFrame.from_dict(data, orient='index', columns=metrics)
        print(f" prob = {prob}")
        print(df_result.mean())
        print('\n\n')
        # break
        df_result.loc['mean'] = df_result.mean()
        df_result.loc['std'] = df_result.std()
        create_folder(folder_name_to_create=f"test/reconstruction/CL_{settings.cases_in_L_P}/",
                      starting_folder='./data/')
        df_result.to_csv(F'./data/test/reconstruction/CL_{settings.cases_in_L_P}/results_prob_{prob}.csv')

    # print(sc.samples)
    df_positive = pd.DataFrame(data_p)
    df_positive.to_csv(f'./data/test/reconstruction/CL_{settings.cases_in_L_P}/positive_cases_ML-G.csv')
    print(df_positive.groupby(['Method', 'Position in the CL']).mean())
    print(df_positive[df_positive['True Positive Rate'] == 0].groupby(['Method', 'Position in the CL']).count())
    print(df_positive[df_positive['True Positive Rate'] == 1].groupby(['Method', 'Position in the CL']).count())
    print(df_positive.groupby(['Method', 'Position in the CL']).count())
    print('\n')

    df_negative = pd.DataFrame(data_n)
    df_negative.to_csv(f'./data/test/reconstruction/CL_{settings.cases_in_L_P}/negative_cases_ML-G.csv')
    print(df_negative.groupby(['Method', 'Position in the CL']).mean())
    print(df_negative[df_negative['False Positive Rate'] == 0].groupby(['Method', 'Position in the CL']).count())
    print(df_negative[df_negative['False Positive Rate'] == 1].groupby(['Method', 'Position in the CL']).count())
    print(df_negative.groupby(['Method', 'Position in the CL']).count())

    # df_result = pd.DataFrame.from_dict(data, orient='index', columns=metrics)
    # df_result.loc['mean'] = df_result.mean()
    # df_result.loc['std'] = df_result.std()
    # all_df.append(df_result)

    # big_df = pd.concat(all_df, ignore_index=True)
    # big_df.to_csv('./data/test/all_results_random.csv')
    # big_df.to_csv('./data/test/selected_results_rete2.csv')
    # df_result.to_csv('./data/test/reconstruction/results_ML-G.csv')
    # df_result.to_csv('./data/test/reconstruction/various_ML-G.csv')

    # plot_histogram(data_n, case='False Positive Rate')
    # plot_histogram(data_p, case='True Positive Rate')


def stats_on_constructives(settings):
    generator_instance = Read_TSP_Files()
    constructive_algs = ['MF', "FI", "CW"]
    data_p = {'Method': [], 'Position in the CL': [], 'True Positive Rate': []}
    data_n = {'Method': [], 'Position in the CL': [], 'False Positive Rate': []}
    print('\n\nTrue Positive Rate and False Positive Rate for MF, FI and CW\n')
    for problem_data in tqdm(generator_instance.instances_generator(), total=len(generator_instance.files)):
        admin = OutcomeAdmin(problem_data, constructive_algs, settings)
        sc = SampleCreator(admin.dist_matrix, admin.optimal_tour)
        for constr in constructive_algs:
            constr_heur = Constructive(constr, admin)
            # create simple solution
            sol_no_pre, time_to_solve, _ = constr_heur.solve()
            admin.save(sol_no_pre, method=constr, time=time_to_solve)

            data_p, data_n = sc.save_new_data(data_p, data_n, admin.sols[constr], constr)

    df_positive = pd.DataFrame(data_p)
    df_positive.to_csv(f'./data/test/reconstruction/CL_{settings.cases_in_L_P}/positive_cases_constructives.csv')
    print(df_positive.groupby(['Method', 'Position in the CL']).mean())
    print(df_positive[df_positive['True Positive Rate'] == 0].groupby(['Method', 'Position in the CL']).count())
    print(df_positive[df_positive['True Positive Rate'] == 1].groupby(['Method', 'Position in the CL']).count())
    print(df_positive.groupby(['Method', 'Position in the CL']).count())
    print('\n')

    df_negative = pd.DataFrame(data_n)
    df_negative.to_csv(f'./data/test/reconstruction/CL_{settings.cases_in_L_P}/negative_cases_constructives.csv')
    print(df_negative.groupby(['Method', 'Position in the CL']).mean())
    print(df_negative[df_negative['False Positive Rate'] == 0].groupby(['Method', 'Position in the CL']).count())
    print(df_negative[df_negative['False Positive Rate'] == 1].groupby(['Method', 'Position in the CL']).count())
    print(df_negative.groupby(['Method', 'Position in the CL']).count())

    plot_histogram(data_n, case='False Positive Rate')
    plot_histogram(data_p, case='True Positive Rate')

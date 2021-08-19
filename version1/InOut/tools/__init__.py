import os
import pandas as pd
import seaborn as sbn
import torch
import numpy as np
import matplotlib.pyplot as plt


class DirManager:
    def __init__(self, settings):
        self.settings = settings
        self.folder_instances = create_folder(folder_name_to_create="data/train/",
                                              starting_folder='./')

        self.folder_train = f'./data/net_weights/CL_{settings.cases_in_L_P}/'

    def save_model(self, model, epoch, iteration):
        name_model = f"model_ep{epoch}_it{iteration}"
        torch.save(model.state_dict(),
                   self.folder_train + '/' + name_model)

    def create_folder_for_train(self, cl):
        self.folder_train = create_folder(folder_name_to_create=f"net_weights/CL_{cl}/",
                                          starting_folder='./data/')
        return self.folder_train


def evaluate_solution(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node

    total_length += dist_matrix[from_node, starting_node]
    return total_length


def print_sett(settings):
    for k, v in vars(settings).items():
        print(f'{k} = "{v}"')


def create_folder(folder_name_to_create, starting_folder="./"):
    folders_list = folder_name_to_create.split("/")
    folder_name = f"{starting_folder}"
    for i in range(len(folders_list[:-1])):
        folder_name += folders_list[i] + '/'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    return folder_name


def to_generator(list_, num_cit):
    for el in list_:
        yield el // num_cit, el % num_cit


def plot_points(pos, tour_found):
    plt.scatter(pos[:, 0], pos[:, 1], color='gray')
    colors = ['red']  # First city red
    trip = pos[tour_found]
    z, y = trip[:, 0], trip[:, 1]
    for _ in range(len(z) - 1):
        colors.append('blue')
    plt.scatter(z, y, color=colors)
    for i, txt in enumerate(tour_found[:-1]):  # tour_found[:-1]
        plt.annotate(txt, (z[i], y[i]))


def plot_tour(pos, tour_found, color='b--'):
    trip = pos[tour_found]
    tour = np.array(list(range(len(trip))) + [0])  # Plot tour
    X = trip[tour, 0]
    Y = trip[tour, 1]
    plt.plot(X, Y, color)


def plot_solution(pos, dist, tour_found, method, alpha=False, save_plot=False, no_MST=False):
    plot_points(pos, tour_found)
    plot_tour(pos, tour_found)
    len_sol = evaluate_solution(tour_found, dist)
    if alpha:
        plt.title(f"reco with {method} \nalpha : {np.around(alpha, 5)},"
                  f"\n solution {np.round(len_sol, 5)}")
        if save_plot:
            if not os.path.exists(f"./immagini"):
                os.mkdir(f"./immagini")
            plt.savefig(f"./immagini/alpha{np.around(alpha, 5)}{['', 'no_MST'][no_MST]}.png")
    else:
        plt.title(f"reco with {method}, solution {len_sol}")
        if save_plot:
            plt.savefig(f"{method}{['', 'no_MST'][no_MST]}.png")
    plt.show()


class SampleCreator:

    def __init__(self, dist_matrix, optimal_solution):
        n = dist_matrix.shape[0]
        self.dist = dist_matrix
        self.optimal = optimal_solution
        self.total_edges = n * (n - 1) / 2
        upper_triangularly_matrix = np.copy(dist_matrix)
        iu = np.tril_indices(n)
        upper_triangularly_matrix[iu] = np.infty
        row, col = np.unravel_index(np.argsort(upper_triangularly_matrix, axis=None), upper_triangularly_matrix.shape)
        self.positions = {i: {j: '' for j in range(n)} for i in range(n)}

        for h in range(int(self.total_edges)):
            self.positions[row[h]][col[h]] = h
            self.positions[col[h]][row[h]] = h

        self.optimal_dict = {i: {j: False for j in range(n)} for i in range(n)}
        for i, j in zip(optimal_solution, np.roll(optimal_solution, 1)):
            self.optimal_dict[i][j] = True
            self.optimal_dict[j][i] = True

    def create_samples(self, solution):
        samples = []
        for i, j in zip(solution, np.roll(solution, 1)):
            if self.optimal_dict[i][j]:
                value_ = self.positions[i][j] / self.total_edges
                samples.append(value_)
        return samples

    def save_new_data(self, data_p, data_n, solution, method):
        n = len(solution)
        for i in range(n):
            arg_i = np.argwhere(np.array(solution) == i)[0][0]
            list_P = np.argsort(self.dist[i])[1:6]
            for j in range(i + 1, n):
                if self.optimal_dict[i][j]:
                    data_p['Method'].append(method)
                    if j in [solution[arg_i - n + 1], solution[arg_i - 1]]:
                        data_p["True Positive Rate"].append(1)
                    else:
                        data_p["True Positive Rate"].append(0)
                    if j in list_P:
                        data_p['Position in the CL'].append(str(np.argwhere(list_P == j)[0][0] + 1))
                    else:
                        data_p['Position in the CL'].append(">5")
                else:
                    data_n['Method'].append(method)
                    if j in [solution[arg_i - n + 1], solution[arg_i - 1]]:
                        data_n["False Positive Rate"].append(1)
                    else:
                        data_n["False Positive Rate"].append(0)
                    if j in list_P:
                        data_n['Position in the CL'].append(str(np.argwhere(list_P == j)[0][0] + 1))
                    else:
                        data_n['Position in the CL'].append(">5")
        return data_p, data_n


class Sampler:
    def __init__(self):
        self.samples = {neig: {metric: 0 for metric in ["P", "N", "TP", "TN"]} for neig in ["first", "second"]}

    def save_from_constructor(self, P, N, TP, TN, cases):
        for i, case in enumerate(cases):
            self.samples[case]["P"] += P[i]
            self.samples[case]["N"] += N[i]
            self.samples[case]["TP"] += TP[i]
            self.samples[case]["TN"] += TN[i]


def plot_histogram(data, case):
    df = pd.DataFrame(data)
    ax = sbn.barplot(x='Position in the CL', y=case, hue='Method', data=df, order=['1', '2', '3', '4', '5', '>5'])
    with_hue(ax, df)
    change_width(ax, 0.35)
    plt.savefig(f'./data/images/{case}Methods.png')
    plt.show()


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def with_hue(plot, feature, number_of_categories, hue_categories):
    a = [p.get_height() for p in plot.patches]
    patch = [p for p in plot.patches]
    for i in range(number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j * number_of_categories + i)] * 2)
            x = patch[(j * number_of_categories + i)].get_x() + patch[
                (j * number_of_categories + i)].get_width() / 2 - 0.3
            y = patch[(j * number_of_categories + i)].get_y() + patch[
                (j * number_of_categories + i)].get_height()
            plot.annotate(percentage, (x, y), size=10)

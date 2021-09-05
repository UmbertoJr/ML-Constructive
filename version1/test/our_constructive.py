import copy
import torch
import numpy as np
from test.utils import possible_plots
from model.network import resnet_for_the_tsp
from InOut.utils import plot_cv, to_torch, plot_single_cv
from InOut.image_creator import ImageTestCreator
from test.classic_constructive import EdgeInsertion

verbose = True


class PreSelection(EdgeInsertion):

    def __init__(self, admin, prob=1e-3, method='our'):
        self.optimal_tour = admin.optimal_tour
        self.settings = admin.settings
        self.dist_matrix = admin.dist_matrix
        self.pos = admin.pos
        self.num_cit = self.dist_matrix.shape[0]
        self.k = self.settings.K
        self.cases_in_LP = self.settings.cases_in_L_P
        self.firstPhaseSolution = {str(i): [] for i in range(self.num_cit)}
        self.visits = {i: 0 for i in range(self.num_cit)}
        self.prob_to_check = prob
        self.image_creator = ImageTestCreator(self.settings, self.pos)
        self.net = resnet_for_the_tsp(admin.settings)
        self.net.to('cpu')
        self.net.load_state_dict(torch.load(f'./data/net_weights/CL_{admin.settings.cases_in_L_P}/best_diff.pth',
                                            map_location='cpu'))
        self.TP, self.TN, self.P, self.N = ([0, 0] for _ in range(4))
        self.cases = ["first", "second"]

        if method == "optimal":
            self.ML_check = self.check_EVENT_optimal
        elif "empirical" in method:
            self.ML_check = self.check_EVENT_random
        elif method == "our":
            self.ML_check = self.check_EVENT_with_net
        elif method == "yes":
            self.ML_check = self.check_yes
        elif method == "no":
            self.ML_check = self.check_no
        elif method == "first":
            self.ML_check = self.check_first
        elif method == "second":
            self.ML_check = self.check_second

        self.neighborhood = self.create_neigs()
        self.LP = self.create_LP()
        self.edges_inserted = 0

    def create_LP(self):
        len_neig = self.cases_in_LP
        LP_v = {i: {} for i in range(len_neig)}
        keys = []
        return_list = []
        for in_cl in range(len_neig):
            for node in range(self.num_cit):
                h = self.neighborhood[node][in_cl]
                if (node, h) not in keys and (h, node) not in keys:
                    LP_v[in_cl][(node, h)] = self.dist_matrix[node, h]
                    keys.append((node, h))

        for in_cl in range(len_neig):
            return_list.extend([k for k, v in sorted(LP_v[in_cl].items(), key=lambda item: item[1], reverse=False)],)
        return return_list

    def create_neigs(self):
        neigs = {}
        for i in range(self.num_cit):
            neigs[i] = np.argsort(self.dist_matrix[i])[1: self.k + 1]
        return neigs

    def condition_to_enter_sol(self, node1, node2, dict_sol):
        if self.check_if_available(node1, node2, dict_sol):
            if self.innerLoopTracker([node1, node2], dict_sol):
                return True
        return False

    def keep_middle(self, solution, level):
        if type(solution) != list and level < 10:
            # print("entrato")
            # print(level)
            solution = self.middlePhase(solution)
            self.keep_middle(solution, level + 1)
        else:
            return solution

    def solve(self):
        self.firstPhase()
        if verbose:
            plotter = possible_plots(self.pos, self.prob_to_check)
            print(self.P, self.N, self.TN, self.TP)
            print(f"inserted: {self.edges_inserted}, tot cases: {len(self.LP)} \n"
                  f"percentage inserted = {self.edges_inserted / len(self.LP)} \n"
                  f"partial solution found: {self.edges_inserted / self.num_cit} \n"
                  f"TPR0 : {self.TP[0] / self.P[0]}, FPR0 : {(self.N[0] - self.TN[0]) / self.N[0]} , "
                  f"ACC0 : {(self.TP[0] + self.TN[0])/(self.P[0] + self.N[0])} \n"
                  f"precision0 : {[self.TP[0] / (self.TP[0] + self.N[0] - self.TN[0]) if self.TP[0] + self.N[0] - self.TN[0] != 0 else 0 ][0]} \n "
                  f"TPR1 : {self.TP[1] / self.P[1]}, FPR1 : {(self.N[1] - self.TN[1]) / self.N[1]}, "
                  f"ACC1 : {(self.TP[1] + self.TN[1])/(self.P[1] + self.N[1])} \n"
                  f"precision0 : {[self.TP[1] / (self.TP[1] + self.N[1] - self.TN[1]) if self.TP[1] + self.N[1] - self.TN[1] != 0 else 0 ][0]} \n ")
        # solution = self.middlePhase(self.firstPhaseSolution)
        solution = self.secondPhase(self.firstPhaseSolution)
        # plotter.plot_current_sol(self.pos, solution)
        # plotter.plot_situation(self.firstPhaseSolution, title="first phase reconstruction")
        # plotter.plot_situation(solution)
        # self.keep_middle(solution, 1)
        # print(solution)
        # if type(solution) != list:
        # print('entrato in second phase')
        # solution = self.middlePhase(solution)
        # solution = self.secondPhase(solution)
        # print(solution)
        # solution = self.remove_crosses(solution)
        if verbose:
            plotter.plot_current_sol(self.pos, solution)
            plotter.plot_situation(self.firstPhaseSolution, title="after two opt phase reconstruction")
        # input()
        return solution

    def secondPhase(self, solution_dict):
        secondPhaseSolution = copy.deepcopy(solution_dict)
        hub = self.find_hub(dist_matrix=self.dist_matrix)
        free_cities = self.get_free_nodes(secondPhaseSolution)
        LD = self.create_LD(free_cities, hub)
        # print(free_cities, LD)
        if len(free_cities) == 2:
            # secondPhaseSolution = self.add_to_sol(free_cities[0], free_cities[1], secondPhaseSolution)
            return self.create_solution(free_cities, secondPhaseSolution, self.num_cit)
        for i, j in LD:
            if i not in secondPhaseSolution[str(j)] and j not in secondPhaseSolution[str(i)]:
                if self.condition_to_enter_sol(i, j, secondPhaseSolution):
                    secondPhaseSolution = self.add_to_sol(i, j, secondPhaseSolution)
                    if len(secondPhaseSolution[str(i)]) == 2:
                        free_cities.remove(i)
                    if len(secondPhaseSolution[str(j)]) == 2:
                        free_cities.remove(j)
                    # print(i, j, free_cities, secondPhaseSolution[str(i)], secondPhaseSolution[str(j)], hub)
                    # plotter.plot_new_selection(secondPhaseSolution, i, j)
                    if len(free_cities) == 2:
                        # print(self.get_free_nodes(secondPhaseSolution))
                        # plotter.plot_new_selection(secondPhaseSolution, i, j)
                        # print(n1, n2)
                        solution = self.create_solution(free_cities, secondPhaseSolution, self.num_cit)
                        # print(solution)
                        return solution

    def remove_crosses(self, solution):
        free_cities = self.get_free_nodes(self.firstPhaseSolution)
        while True:
            solution, improvement = self.stepModified2opt(solution, free_cities)
            # print("step")
            # print(solution)
            # print(improvement)
            # print("nuovo")
            if improvement == 0:
                return solution

    def stepModified2opt(self, solution, free_cities):
        for i in range(self.num_cit-1):
            city1 = solution[i]
            city1p = solution[i+1]
            if city1 in free_cities and city1p in free_cities:
                for j in range(i+1, self.num_cit-1):
                    city2 = solution[j]
                    city2p = solution[j+1]
                    if city2 in free_cities and city2p in free_cities:
                        old = self.dist_matrix[city1, city1p] + self.dist_matrix[city2, city2p]
                        new = self.dist_matrix[city1, city2] + self.dist_matrix[city1p, city2p]
                        if old - new > 0:
                            # print(city1, city2, i, j, old - new)
                            # print(solution)
                            new_sol = solution[:i+1] + solution[i+1: j+1][::-1] + solution[j+1:]
                            # print(new_sol)
                            return new_sol, old - new

        return solution, 0

    def create_LD(self, free_cities, hub):
        LD_v = {}
        for node_i in free_cities:
            for node_j in free_cities:
                if node_i != node_j and (node_i, node_j) not in LD_v.keys() and (node_j, node_i) not in LD_v.keys():
                    LD_v[(node_i, node_j)] = self.dist_matrix[node_i, hub] + self.dist_matrix[hub, node_j] \
                                             - self.dist_matrix[node_i, node_j]
        return [k for k, v in sorted(LD_v.items(), key=lambda item: item[1], reverse=True)]

    def firstPhase(self):
        # plotter = possible_plots(self.pos, self.prob_to_check)
        for i, j in self.LP:
            if self.condition_to_enter_sol(i, j, self.firstPhaseSolution):
                self.add_visit(i, j)
                # print("inserted", i, j)
                if self.ML_check(i, j):
                    # print("ML agrees")
                    self.firstPhaseSolution = self.add_to_sol(i, j, self.firstPhaseSolution)
                    self.edges_inserted += 1
                # plotter.plot_new_selection(self.firstPhaseSolution, i, j)
                # input()
                # if verbose:
        #     print(self.solution_shrinked)
        #     print(sum([len(self.solution_shrinked[key])==2 for key in self.solution_shrinked.keys()])/self.num_cit)
        # plotter.plot_situation(self.firstPhaseSolution)
        # plotter.create_video(self.prob_to_check)
        # input()

    def check_EVENT_optimal(self, i, j):
        ind_cur = np.argwhere(self.optimal_tour == j)
        return True if i in [self.optimal_tour[ind_cur - 1], self.optimal_tour[ind_cur + 1 - self.num_cit]] else False

    def check_EVENT_random(self, i, j):
        ret_bool = False

        if i == self.neighborhood[j][0] or j == self.neighborhood[i][0]:
            ret_bool = np.random.choice([True, False], p=[0.8854, 1 - 0.8854])
        elif i == self.neighborhood[j][1] or j == self.neighborhood[i][1]:
            ret_bool = np.random.choice([True, False], p=[0.5109, 1 - 0.5109])
        return ret_bool

    def check_EVENT_with_net(self, i, j):
        image, too_close = self.image_creator.get_image(i, j, self.firstPhaseSolution)
        # plot_single_cv(image)
        image = np.stack([image, image], axis=0)
        if too_close:
            # print('too close')
            self.update_metrics_run(i, j, [0., 1.])
            return True
        image_ = to_torch(image).to('cpu')
        self.net.eval()
        ret = self.net(image_)
        # print(ret)
        ret = ret.detach().cpu().numpy()[0]
        self.update_metrics_run(i, j, ret)
        # return True if ret[1] > self.prob_to_check else False
        if self.check_first(i, j, other=True):
            return True if ret[0] < self.prob_to_check[0] else False
        else:
            return True if ret[0] < self.prob_to_check[1] else False


    def update_metrics_run(self, i, j, ret):
        # print(self.prob_to_check, ret)
        if self.check_EVENT_optimal(i, j):
            # if ret[1] > self.prob_to_check:
            # if ret[0] < self.prob_to_check:
            if self.check_first(i, j, other=True):
                if ret[0] < self.prob_to_check[0]:
                    self.TP[0] += 1
                    self.P[0] += 1
                else:
                    self.P[0] += 1
            else:
                if ret[0] < self.prob_to_check[1]:
                    self.TP[1] += 1
                    self.P[1] += 1
                else:
                    self.P[1] += 1
            # print('true positive')
        else:
            # if ret[1] > self.prob_to_check:
            # if ret[0] < self.prob_to_check:
                # print('False Negative')
            if self.check_first(i, j, other=True):
                if ret[0] < self.prob_to_check[0]:
                    self.N[0] += 1
                else:
                    self.N[0] += 1
                    self.TN[0] += 1
            else:
                if ret[0] < self.prob_to_check[1]:
                    self.N[1] += 1
                else:
                    self.N[1] += 1
                    self.TN[1] += 1
        # print(self.TP, self.P, self.TN, self.N)

    def check_first(self, i, j, other=False):
        insert_bool = True if i == self.neighborhood[j][0] or j == self.neighborhood[i][0] else False
        if verbose and not other:
            if insert_bool:
                if self.check_EVENT_optimal(i, j):
                    # print('true positive')
                    self.TP[0] += 1
                    self.P[0] += 1
                else:
                    # print('false positive')
                    self.N[0] += 1
            else:
                if self.check_EVENT_optimal(i, j):
                    # print('false negative')
                    self.P[1] += 1
                else:
                    # print('true negative')
                    self.N[1] += 1
                    self.TN[1] += 1
        return insert_bool

    def check_second(self, i, j):
        return True if i == self.neighborhood[j][1] or j == self.neighborhood[i][1] else False

    @staticmethod
    def check_yes(*args):
        return True

    @staticmethod
    def check_no(*args):
        return False

    def add_to_sol(self, node1, node2, dict_sol):
        dict_sol[str(node1)].append(node2)
        dict_sol[str(node2)].append(node1)
        return dict_sol

    def add_visit(self, node1, node2):
        self.visits[node1] += 1
        self.visits[node2] += 1

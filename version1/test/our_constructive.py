import copy
import torch
import numpy as np
from test.utils import possible_plots
from model.network import resnet_for_the_tsp
from InOut.utils import plot_cv, to_torch
from InOut.image_creator import ImageTestCreator
from test.classic_constructive import EdgeInsertion

verbose = False


class PreSelection(EdgeInsertion):

    def __init__(self, admin, prob=0.5, method='our'):
        self.optimal_tour = admin.optimal_tour
        self.settings = admin.settings
        self.dist_matrix = admin.dist_matrix
        self.pos = admin.pos
        self.num_cit = self.dist_matrix.shape[0]
        self.k = self.settings.cases_in_L_P
        self.firstPhaseSolution = {str(i): [] for i in range(self.num_cit)}
        self.visits = {i: 0 for i in range(self.num_cit)}
        self.prob_to_check = prob
        self.image_creator = ImageTestCreator(self.settings, self.pos)
        self.net = resnet_for_the_tsp(admin.settings)
        self.net.to('cpu')
        # self.net.load_state_dict(torch.load(f'./data/net_weights/CL_2/best_model_RL_v2_PLR_new.pth',
        #                                     map_location='cpu'))
        self.net.load_state_dict(torch.load(f'./data/net_weights/CL_2/best_model_RL_v8_PLR.pth',
                                            map_location='cpu'))

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
        LP_v = {}
        for node in range(self.num_cit):
            for h in self.neighborhood[node]:
                if (node, h) not in LP_v.keys() and (h, node) not in LP_v.keys():
                    LP_v[(node, h)] = self.dist_matrix[node, h]
        return [k for k, v in sorted(LP_v.items(), key=lambda item: item[1])]

    def create_neigs(self):
        neigs = {}
        for i in range(self.num_cit):
            a, b = np.argsort(self.dist_matrix[i])[1: self.k + 1]
            neigs[i] = [a, b]
        return neigs

    def condition_to_enter_sol(self, node1, node2, dict_sol):
        if self.check_if_available(node1, node2, dict_sol):
            if self.innerLoopTracker([node1, node2], dict_sol):
                return True
        return False

    def solve(self):
        # plotter = possible_plots(self.pos, self.prob_to_check)
        self.firstPhase()
        solution = self.secondPhase()
        # plotter.plot_current_sol(self.pos, solution)
        # plotter.plot_situation(self.firstPhaseSolution, title="second phase reconstruction")
        # input()
        return solution

    def secondPhase(self):
        secondPhaseSolution = copy.deepcopy(self.firstPhaseSolution)
        hub = self.find_hub(dist_matrix=self.dist_matrix)
        free_cities = self.get_free_nodes(secondPhaseSolution)
        LD = self.create_LD(free_cities, hub)
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

    def create_LD(self, free_cities, hub):
        LD_v = {}
        for node_i in free_cities:
            for node_j in free_cities:
                if node_i != node_j and (node_i, node_j) not in LD_v.keys() and (node_j, node_i) not in LD_v.keys():
                    LD_v[(node_i, node_j)] = self.dist_matrix[node_i, hub] + self.dist_matrix[hub, node_j] \
                                             - self.dist_matrix[node_i, node_j]
        return [k for k, v in sorted(LD_v.items(), key=lambda item: - item[1])]

    def firstPhase(self):
        # plotter = possible_plots(self.pos, self.prob_to_check)
        for i, j in self.LP:
            if self.condition_to_enter_sol(i, j, self.firstPhaseSolution):
                self.add_visit(i, j)
                if self.ML_check(i, j):
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
        image, too_close = self.image_creator.get_image(i, j)
        image = np.stack([image, image], axis=0)
        if too_close:
            return True
        image_ = to_torch(image).to('cpu')
        self.net.eval()
        ret = self.net(image_)
        ret = ret.detach().cpu().numpy()[0]
        return True if ret[1] > self.prob_to_check else False

    def check_first(self, i, j):
        return True if i == self.neighborhood[j][0] or j == self.neighborhood[i][0] else False

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

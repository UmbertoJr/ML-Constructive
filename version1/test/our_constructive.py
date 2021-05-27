import torch
import numpy as np
from InOut.tools import to_generator
from test.utils import possible_plots
from model.network import resnet_for_the_tsp
from InOut.utils import plot_cv, to_torch
from InOut.image_creator import ImageTestCreator
from test.classic_constructive import EdgeInsertion

verbose = False
# model_check = ['optimal', 'random', 'network',
#                'first always', 'second always',
#                'yes', 'no'][2]


class PreSelection(EdgeInsertion):

    def __init__(self, admin, prob=0.5, method='our'):
        self.optimal_tour = admin.optimal_tour
        self.settings = admin.settings
        self.dist_matrix = admin.dist_matrix
        self.pos = admin.pos
        self.num_cit = self.dist_matrix.shape[0]
        self.solution_shrinked = {str(i): [] for i in range(self.num_cit)}
        self.visits = np.zeros(self.num_cit)
        self.prob_to_check = prob
        self.image_creator = ImageTestCreator(self.settings, self.pos)
        self.net = resnet_for_the_tsp(admin.settings)
        self.net.to('cpu')
        self.net.load_state_dict(torch.load(f'./data/net_weights/CL_2/best_model.pth',
                                            map_location='cpu'))

        if method == "optimal":
            self.check = self.check_EVENT_optimal
        elif "empirical" in method:
            self.check = self.check_EVENT_random
        elif method == "our":
            self.check = self.check_EVENT_with_net
        elif method == "yes":
            self.check = self.check_yes
        elif method == "no":
            self.check = self.check_no
        elif method == "first":
            self.check = self.check_first
        elif method == "second":
            self.check = self.check_second

        self.generator = self.mf_generator(self.dist_matrix)
        self.neighborhood = self.create_neigs()
        self.edges_inserted = 0

    def mf_generator(self, dist_matrix):
        mat = np.triu(dist_matrix, 1)
        mat[mat == 0] = 100000000
        self.mat = mat
        return to_generator(np.argsort(mat.flatten()), self.num_cit)

    @property
    def condition_to_stop(self):
        # print(f"current visited: {np.sum(self.visits >= 2)}, just one visit {np.sum(self.visits == 1)},\n"
        #       f"total cities: {self.num_cit}")
        return np.sum(self.visits >= 2) < self.num_cit

    def create_neigs(self):
        neigs = {}
        for i in range(self.num_cit):
            a, b = np.argsort(self.dist_matrix[i])[1:3]
            neigs[i] = [a, b]
        return neigs

    def condition_to_enter_sol(self, node1, node2):
        if node2 in self.neighborhood[node1] or node1 in self.neighborhood[node2]:
            self.add_visit(node1, node2)
            if self.check_if_available(node1, node2, self.solution_shrinked):
                if self.check_if_not_close([node1, node2], self.solution_shrinked):
                    return True
        return False

    def get_the_nn(self, cur, prev):
        array_ = np.copy(self.dist_matrix[cur])
        array_[cur] = 100000000
        array_[prev] = 100000000
        node_r = np.argmin(array_)
        # if verbose: print("segment considered", prev, cur, node_r)
        return node_r

    def select_neigs_for_selected_edge(self, node1, node2):
        case = 1
        neig1 = self.get_the_nn(node1, node2)
        neig2 = self.get_the_nn(node2, node1)
        if neig1 != neig2:
            case = 2
        # if verbose:print("numero vicini", case, (neig1, node1), (neig2, node2),
        #                  'probs', self.dist_matrix[node1, neig1], self.dist_matrix[node2, neig2])
        first = np.argmin([self.dist_matrix[node1, neig1], self.dist_matrix[node2, neig2]])
        return case, [[[(node2, node1), (node1, neig1)], [(node1, node2), (node2, neig2)]][first],
                      [[(node2, node1), (node1, neig1)], [(node1, node2), (node2, neig2)]][first - 1]]

    def evento(self, new_edge, old_edge):
        # print(new_edge)
        a_i, x_i = list(new_edge)
        prev_x = old_edge[1]
        caso = False
        event = self.check(a_i, x_i, prev_x)
        if event:
            self.add_to_sol(a_i, x_i)
            caso = True
            # if verbose:
            #     print("aggiunto edge", a_i, x_i)
        return caso

    def solve(self):
        plotter = possible_plots(self.pos, self.prob_to_check)
        while self.condition_to_stop:
            x_i, y_i = next(self.generator, (None, None))

            if x_i == None:
                print("generator exausted")
                assert False, 'non dovrebbe mai entrare qui'

            if self.condition_to_enter_sol(x_i, y_i):
                # plotter.plot_new_selection(self.solution_shrinked, x_i, y_i)
                situation, cases = self.select_neigs_for_selected_edge(x_i, y_i)

                # if verbose:
                #     print("ci sono vicini:", situation)
                #     print("casistica:", cases[0])
                #     print("casistica:", cases[1])
                #     possible_plots.plot_possible_previous_steps(initial_edge, situation, cases)

                for case in range(situation):
                    new_step, prev_step = cases[case]
                    positive = self.evento(new_step, prev_step)
                    # plotter.case_step(self.solution_shrinked, new_step, prev_step, positive)
                    if positive:
                        break

        # if verbose:
        #     print(self.solution_shrinked)
        #     print(sum([len(self.solution_shrinked[key])==2 for key in self.solution_shrinked.keys()])/self.num_cit)
        # plotter.plot_situation(self.solution_shrinked)
        # plotter.create_video(self.prob_to_check)
        # input()
        return self.solution_shrinked

    def check_EVENT_optimal(self, a_i, x_i, *args):
        ind_cur = np.argwhere(self.optimal_tour == x_i)
        return True if a_i in [self.optimal_tour[ind_cur - 1], self.optimal_tour[ind_cur + 1 - self.num_cit]] else False

    def check_EVENT_random(self, a_i, x_i, prev_x, *args):
        ret_bool = False
        if self.dist_matrix[a_i, x_i] < self.dist_matrix[x_i, prev_x]:
            # ret_bool = np.random.choice([True, False], p=[0.67591, 1 - 0.67591])
            ret_bool = np.random.choice([True, False], p=[0.8854, 1 - 0.8854])
        if self.dist_matrix[a_i, x_i] < self.dist_matrix[x_i, prev_x]:
            # ret_bool = np.random.choice([True, False], p=[0.17152, 1 - 0.17152])
            ret_bool = np.random.choice([True, False], p=[0.5109, 1 - 0.5109])
        return ret_bool

    def check_EVENT_with_net(self, a_i, x_i, prev_x):
        image, too_close = self.image_creator.get_image(a_i, x_i)
        image = np.stack([image, image], axis=0)
        if too_close:
            return True
        image_ = to_torch(image).to('cpu')
        self.net.train(mode=False)
        ret = self.net(image_)
        ret = ret.detach().cpu().numpy()[0]
        if verbose:
            print("probabilita rete", ret)
            # plot_cv(image)
        return True if ret[0] > self.prob_to_check else False

    def check_first(self, a_i, x_i, prev_x):
        return True if self.dist_matrix[a_i, x_i] < self.dist_matrix[x_i, prev_x] else False

    def check_second(self, a_i, x_i, prev_x):
        return True if self.dist_matrix[a_i, x_i] > self.dist_matrix[x_i, prev_x] else False

    @staticmethod
    def check_yes(*args):
        return True

    @staticmethod
    def check_no(*args):
        return False

    def add_to_sol(self, node1, node2):
        self.solution_shrinked[str(node1)].append(node2)
        self.solution_shrinked[str(node2)].append(node1)
        self.edges_inserted += 1

    def add_visit(self, node1, node2):
        self.visits[node1] += 1
        self.visits[node2] += 1

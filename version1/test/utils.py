import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from cv2 import VideoWriter, VideoWriter_fourcc

from InOut.tools import evaluate_solution, create_folder

tempo = 2
verbose = False


class OutcomeAdmin:

    def __init__(self, problem_data, methods, settings):
        self.nPoints = problem_data[0]
        self.pos = problem_data[1]
        self.dist_matrix = problem_data[2]
        self.name = problem_data[3]
        self.optimal_tour = problem_data[4]
        self.optimal_dict = self.create_dict(self.optimal_tour)
        self.len_opt = evaluate_solution(self.optimal_tour, self.dist_matrix)
        self.methods = methods
        self.sols = {}
        self.lens = {}
        self.gaps = {}
        self.accs = {}
        self.time = {}
        self.settings = settings

    def save(self, solution, method, time):
        self.time[method] = time
        self.sols[method] = solution
        self.lens[method] = evaluate_solution(self.sols[method], self.dist_matrix)
        self.gaps[method] = np.round((self.lens[method] - self.len_opt) / self.len_opt * 100., 3)
        self.accs[method] = np.round(self.compute_stats_given_sol(self.sols[method]), 3)

    def create_stats(self, sol_dict):
        tp, fn = (0. for _ in range(2))
        for key in self.optimal_dict:
            for node in self.optimal_dict[key]:
                tp += node in sol_dict[key]
                fn += node not in sol_dict[key]
        tp /= 2
        fn /= 2
        acc = tp / (fn + tp)
        return acc

    def create_dict(self, sol):
        local_dict = {str(i): [] for i in range(self.nPoints)}
        next_sol = np.roll(sol, 1)
        prev_sol = np.roll(sol, -1)
        for p_el, el, n_el in zip(prev_sol, sol, next_sol):
            local_dict[str(el)] = [p_el, n_el]
        return local_dict

    def compute_stats_given_sol(self, sol):
        sol_dict = self.create_dict(sol)
        acc = self.create_stats(sol_dict)
        return acc

    def plot_solution(self, solution, name_method):
        gap = np.round((evaluate_solution(solution, self.dist_matrix) - self.len_opt) / self.len_opt * 100., 3)
        plt.figure(figsize=(8, 8))
        plt.title(f"{self.name} solved with {name_method} solver, gap {gap}")
        plt.scatter(self.pos[:, 0], self.pos[:, 1], marker='o', c='b')
        ordered_points = self.pos[np.hstack([solution, solution[0]])]
        plt.plot(ordered_points[:, 0], ordered_points[:, 1], 'b-')
        plt.show()


class possible_plots:
    def __init__(self, pos, prob):
        self.step = 0
        self.pos = pos
        self.prob = prob
        create_folder(folder_name_to_create=f"images/prob_{prob}/", starting_folder="./data/test/")

    def plot_edge(self, x, y, c='chartreuse', style='-'):
        nodes = [x, y]
        plt.plot(self.pos[nodes, 0], self.pos[nodes, 1], marker='o', color=mcd.CSS4_COLORS[c], linestyle=style)

    def plot_new_selection(self, pre_solution, x, y):
        self.plot_situation(pre_solution)
        self.plot_edge(x, y)
        plt.title(f"selection nodes {x, y}")
        plt.savefig(f"./data/test/images/prob_{self.prob}/step{self.step}.png")
        self.step += 1

    def final_situation(self, sol):
        self.plot_situation(sol)
        plt.title("final preselection")
        plt.savefig(f"./data/test/images/prob_{self.prob}/step{self.step}.png")

    def plot_situation(self, solution_dict):
        pieces = []
        no_touch = [int(k_) for k_ in solution_dict.keys() if len(solution_dict[k_]) == 1]
        touch = [int(k_) for k_ in solution_dict.keys() if len(solution_dict[k_]) == 2]
        for key in no_touch:
            if sum([key in piece for piece in pieces]) == 0 and len(solution_dict[str(key)]) < 2:
                new_piece = possible_plots.create_piece(solution_dict, int(key))
                print(new_piece)
                pieces.append(new_piece)

        plt.figure(figsize=(8, 8))
        # ordered_points = self.pos[no_touch]
        # plt.scatter(ordered_points[:, 0], ordered_points[:, 1], marker='o', c=mcd.CSS4_COLORS['cyan'])
        for t in touch:
            print(t)
            print(solution_dict[str(t)])
        plt.scatter(self.pos[:, 0], self.pos[:, 1], marker='o', c=mcd.CSS4_COLORS['cyan'])
        plt.scatter(self.pos[touch, 0], self.pos[touch, 1], marker='o', c='b')

        for piece in pieces:
            ordered_points = self.pos[piece]
            plt.plot(ordered_points[:, 0], ordered_points[:, 1], f'b-')

        plt.savefig(f"./data/test/images/multi-frags_ML-greeedy.png")
        plt.show()

    def case_step(self, pre_solution, new_step, prev_step, case):
        x_i, x_j = new_step
        y_i, y_j = prev_step
        self.plot_situation(pre_solution)
        # self.plot_edge(x_i, x_j)
        # self.plot_edge(y_i, y_j, style='--')
        plt.title(f"the network choose {x_i, x_j}?  {case}")
        plt.savefig(f"./data/test/images/prob_{self.prob}/step{self.step}.png")
        self.step += 1

    @staticmethod
    def create_piece(solution_dict, from_city):
        end = False
        curr_c = solution_dict[str(from_city)][0]
        starting_vertex = from_city
        sol_list = [from_city, curr_c]
        while True:
            if len(solution_dict[str(curr_c)]) == 2:
                print("lista pezzo", sol_list)
                print("dict nodo", solution_dict[str(curr_c)])
                curr_c = [e for e in solution_dict[str(curr_c)] if e not in sol_list][0]
                if curr_c not in sol_list:
                    sol_list.append(curr_c)
                    print(sol_list)
            else:
                return sol_list

    @staticmethod
    def create_video(prob):
        frameSize = (800, 800)
        out = cv2.VideoWriter(f'./data/test/videos/selection_video_prob{prob}.avi',
                              VideoWriter_fourcc(*'DIVX'), 1, frameSize)
        num_steps_l = []
        path = f'./data/test/images/prob_{prob}/'
        for file in os.listdir(path):
            num_steps_l.append(int(file[4:].split('.')[0]))
        num_steps = max(num_steps_l)
        for it in range(num_steps):
            img = cv2.imread(path + f'step{it}.png')
            out.write(img)
        out.release()


    @staticmethod
    def plot_current_sol(pos, sol):
        plt.scatter(pos[:, 0], pos[:, 1], marker='o', c=mcd.CSS4_COLORS['cyan'])
        sol_p = sol + [sol[0]]
        ordered_points = pos[sol_p]
        plt.plot(ordered_points[:, 0], ordered_points[:, 1], f'b-')
        plt.show()

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from cv2 import VideoWriter, VideoWriter_fourcc

from InOut.tools import evaluate_solution, create_folder
from model import resnet_for_the_tsp
from test.tsplib_reader import EvalGenerator
import torch
from InOut import DatasetHandler
from torch.utils.data import DataLoader
from InOut.image_manager import OnlineDataSetHandler

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
        # create_folder(folder_name_to_create=f"./data/images/", starting_folder="./data/test/")

    def plot_edge(self, x, y, c='chartreuse', style='-'):
        nodes = [x, y]
        plt.plot(self.pos[nodes, 0], self.pos[nodes, 1], marker='o', color=mcd.CSS4_COLORS[c], linestyle=style)

    def plot_new_selection(self, pre_solution, x, y):
        self.plot_situation(pre_solution, title=f"added {x} {y}")
        self.plot_edge(x, y)
        plt.title(f"selection nodes {x, y}")
        # plt.savefig(f"./data/images/prob_{self.prob}/step{self.step}.png")
        self.step += 1

    def final_situation(self, sol):
        self.plot_situation(sol)
        plt.title("final preselection")
        plt.savefig(f"./data/images/prob_{self.prob}/step{self.step}.png")

    def plot_situation(self, solution_dict, title=''):
        pieces = []
        no_touch = [int(k_) for k_ in solution_dict.keys() if len(solution_dict[k_]) == 1]
        touch = [int(k_) for k_ in solution_dict.keys() if len(solution_dict[k_]) == 2]
        for key in no_touch:
            if sum([key in piece for piece in pieces]) == 0 and len(solution_dict[str(key)]) < 2:
                new_piece = possible_plots.create_piece(solution_dict, int(key))
                # print(new_piece)
                pieces.append(new_piece)

        # plt.figure(figsize=(8, 8))
        # ordered_points = self.pos[no_touch]
        # plt.scatter(ordered_points[:, 0], ordered_points[:, 1], marker='o', c=mcd.CSS4_COLORS['cyan'])
        # for t in touch:
        #     print(t)
        #     print(solution_dict[str(t)])
        plt.scatter(self.pos[:, 0], self.pos[:, 1], marker='o', c=mcd.CSS4_COLORS['cyan'])
        plt.scatter(self.pos[touch, 0], self.pos[touch, 1], marker='o', c='b')

        for piece in pieces:
            ordered_points = self.pos[piece]
            plt.plot(ordered_points[:, 0], ordered_points[:, 1], f'b-')
        for i in range(self.pos.shape[0]):
            plt.annotate(str(i), (self.pos[i, 0], self.pos[i, 1]))

        # plt.title(title)
        plt.savefig(f"./data/images/{title}.png")
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
                # print("lista pezzo", sol_list)
                # print("dict nodo", solution_dict[str(curr_c)])
                curr_c = [e for e in solution_dict[str(curr_c)] if e not in sol_list][0]
                if curr_c not in sol_list:
                    sol_list.append(curr_c)
                    # print(sol_list)
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
        sol_p = list(sol) + [sol[0]]
        print(sol_p)
        ordered_points = pos[sol_p]
        plt.plot(ordered_points[:, 0], ordered_points[:, 1], f'r-')
        for i in range(pos.shape[0]):
            plt.annotate(str(i + 1), (pos[i, 0], pos[i, 1]))
        # plt.show()


class Logger:
    @staticmethod
    def log_pred(loss, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR):
        return f'loss: {loss:.5f} ' \
               f' acc  {ACC * 100 :.2f} ' \
               f' acc bal {BAL_ACC * 100 :.2f}' \
               f' PLR {PLR :.4f}' \
               f' PLR bal {BAL_PLR :.4f}' \
               f' TPR {TPR * 100:.2f}' \
               f' FPR {FPR * 100:.2f}'


class Tester_on_eval:

    def __init__(self, settings, dir_ent, log_str_fun, cl, device):
        self.settings = settings
        self.dir_ent = dir_ent
        self.log_fun = log_str_fun
        self.df_data = {"train TPR": [], "train FPR": [], "train TNR": [], "train FNR": [],
                        "train Acc": [], "train bal Acc": [], "train PLR": [], "train bal PLR": [],
                        "eval TPR": [], "eval FPR": [], "eval TNR": [], "eval FNR": [],
                        "eval Acc": [], "eval bal Acc": [], "eval PLR": [], "eval bal PLR": [],
                        "test TPR": [], "test FPR": [], "test TNR": [], "test FNR": [],
                        "test Acc": [], "test bal Acc": [], "test PLR": [], "test bal PLR": [],
                        }
        self.iter_list = []
        self.best_bal_PLR = 0.
        self.folder_data = dir_ent.create_folder_for_train(cl)
        self.device = 'cpu'
        self.net = resnet_for_the_tsp(self.settings)

    def test(self, tpr, fnr, fpr, tnr, acc, bal_acc, plr, bal_plr, iteration_train):

        self.net.load_state_dict(torch.load(self.dir_ent.folder_train + f'checkpoint.pth',
                                            map_location='cpu'))

        def check_eval():
            generator = DatasetHandler(self.settings, path='./data/eval/')
            data_logger = DataLoader(generator, batch_size=self.settings.bs, drop_last=True)
            mht = Metrics_Handler()
            TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = (0 for i in range(8))
            self.net.eval()
            with torch.no_grad():
                # for iter, data in enumerate(data_logger):
                for _ in range(50):

                    # x, y = data["X"], data["Y"]
                    # x = x.to(self.device)
                    # y = y.to(self.device)
                    # predictions = self.net(x)

                    online_data_generator = OnlineDataSetHandler(self.settings, self.net)
                    x_online, y_online = online_data_generator.get_data()
                    predictions = self.net(x_online)

                    TP, FP, TN, FN = compute_metrics(predictions.detach(), y_online.detach())

                    TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mht.update_metrics(TP, FP, TN, FN)
                    # if iter > 500:
                    #     break
            return TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR

        def check_test():
            generator2 = EvalGenerator(self.settings)
            data_logger2 = DataLoader(generator2, batch_size=generator2.bs_test, drop_last=True)
            mht = Metrics_Handler()
            TPR_test, FNR_test, FPR_test, TNR_test, ACC_test, BAL_ACC_test, PLR_test, BAL_PLR_test = ([0, 0] for _ in
                                                                                                      range(8))
            self.net.eval()
            with torch.no_grad():
                for iter, data in enumerate(data_logger2):
                    x, y = data["X"], data["Y"]
                    # pos = data["position"]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    predictions = self.net(x)

                    TP, FP, TN, FN = compute_metrics(predictions.detach(), y.detach())

                    TPR_test, FNR_test, FPR_test, TNR_test, \
                    ACC_test, BAL_ACC_test, PLR_test, BAL_PLR_test = mht.update_metrics(TP, FP, TN, FN)
            return TPR_test, FNR_test, FPR_test, TNR_test, ACC_test, BAL_ACC_test, PLR_test, BAL_PLR_test

        TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = check_eval()
        TPR_test, FNR_test, FPR_test, TNR_test, ACC_test, BAL_ACC_test, PLR_test, BAL_PLR_test = check_test()

        self.df_data["train TPR"].append(tpr)
        self.df_data["train FNR"].append(fnr)
        self.df_data["train TNR"].append(tnr)
        self.df_data["train FPR"].append(fpr)
        self.df_data["train Acc"].append(acc)
        self.df_data["train bal Acc"].append(bal_acc)
        self.df_data["train PLR"].append(plr)
        self.df_data["train bal PLR"].append(bal_plr)

        self.df_data["eval TPR"].append(TPR)
        self.df_data["eval FNR"].append(FNR)
        self.df_data["eval TNR"].append(TNR)
        self.df_data["eval FPR"].append(FPR)
        self.df_data["eval Acc"].append(ACC)
        self.df_data["eval bal Acc"].append(BAL_ACC)
        self.df_data["eval PLR"].append(PLR)
        self.df_data["eval bal PLR"].append(BAL_PLR)

        self.df_data["test TPR"].append(TPR_test)
        self.df_data["test FNR"].append(FNR_test)
        self.df_data["test TNR"].append(TNR_test)
        self.df_data["test FPR"].append(FPR_test)
        self.df_data["test Acc"].append(ACC_test)
        self.df_data["test bal Acc"].append(BAL_ACC_test)
        self.df_data["test PLR"].append(PLR_test)
        self.df_data["test bal PLR"].append(BAL_PLR_test)

        self.iter_list.append(iteration_train)
        print()
        print(f"eval results -->   TPR : {TPR},  FPR : {FPR},  Acc : {ACC},  PLR : {PLR}, delta : {TPR - FPR}")
        print(f"eval results -->   TPR : {TPR_test},  FPR : {FPR_test}, "
              f"Acc : {ACC_test},  PLR : {PLR_test}, delta : {TPR_test - FPR_test}")
        print("\n\n\n")
        return TPR - FPR
        # return - FPR

    def save_csv(self, name):
        self.df = pd.DataFrame(data=self.df_data, index=self.iter_list)
        if not name:
            self.name_case = f'dataset_train_history'
        else:
            self.name_case = name
        create_folder(folder_name_to_create=self.folder_data)
        self.df.to_csv(f"{self.folder_data}{self.name_case}.csv")


def compute_metrics(preds, y, rl_bool=False):
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()
    TP, FP, TN, FN, CP, CN = (0 for _ in range(6))
    for i in range(preds.shape[0]):
        if not rl_bool:
            best = 1 if preds[i, 1] > 0.99 else 0
            # best = np.argsort(preds[i])[::-1][:1][0]
        else:
            best = preds[i]
        if best == y[i]:
            if y[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y[i] == 1:
                FN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


class Metrics_Handler:
    def __init__(self):
        self.CP = 0
        self.CN = 0
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def update_metrics(self, tp, fp, tn, fn):
        self.TP += tp
        self.FP += fp
        self.TN += tn
        self.FN += fn
        self.CP += tp + fn
        self.CN += tn + fp
        TPR = self.TP / self.CP
        FNR = self.FN / self.CP
        FPR = self.FP / self.CN
        TNR = self.TN / self.CN
        ACC = (self.TP + self.TN) / (self.CN + self.CP)
        BAL_ACC = (TPR + TNR) / 2
        PLR = TPR / (1 + FPR)
        # PLR = self.TP / (self.TP + self.FN + 3 * self.FP)
        BAL_PLR = self.TP / (self.FP + 1)
        return TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR

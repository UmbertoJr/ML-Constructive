import numpy as np
import torch
from collections import deque
import pandas as pd
from tqdm import tqdm
import seaborn as sbn
from torch.utils.data import DataLoader
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from InOut.tools import DirManager
from model.network import resnet_for_the_tsp
from test.utils import Logger, Tester_on_eval, Metrics_Handler, compute_metrics
from InOut.image_manager import DatasetHandler


def show_results(settings):
    training_data = {i: [] for i in range(1, 6)}
    for cl_elements in range(1, 5):
        settings.cases_in_L_P = cl_elements
        dir_ent = DirManager(settings)
        df = pd.read_csv(f"{dir_ent.folder_train}dataset_train_history.csv", index_col=0)
        df["test TPR"].plot()
        plt.title(f"cl {cl_elements}")
        plt.show()


def train_the_best_configuration(settings):
    dir_ent = DirManager(settings)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device : {device}')
    print(dir_ent.folder_train)

    model = resnet_for_the_tsp(settings)
    model = model.to(device)
    model.apply(model.weight_init)
    # model.load_state_dict(torch.load(f'./data/net_weights/CL_2/best_model_RL_v8_PLR.pth'))

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    log_str_fun = Logger.log_pred
    tester = Tester_on_eval(settings, dir_ent, log_str_fun, settings.cases_in_L_P, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)

    iteration = 0
    print(f'\n\nrunning 1 epochs for case with {settings.cases_in_L_P} neigs in L_P ...')
    mh = Metrics_Handler()
    best_list = []
    max_PLR = 0
    min_FPR = 1.
    entropy = 0
    average_plr = deque([0.5 for _ in range(100)])
    for epoch in range(1):
        generator = DatasetHandler(settings)
        data_logger = tqdm(DataLoader(generator, batch_size=settings.bs, drop_last=True))
        for data in data_logger:
            x, y = data["X"], data["Y"]
            x = x.to(device)
            y = y.to(device)

            model.train()
            optimizer.zero_grad()
            predictions1 = model(x)
            loss1 = criterion(predictions1, y)
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer.step()

            predictions2 = model(x)
            dist = Categorical(probs=predictions2)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            # entropy += dist.entropy().mean().detach()

            TP, FP, TN, FN = compute_metrics(actions.detach(), y.detach(), rl_bool=True)

            TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mh.update_metrics(TP, FP, TN, FN)

            # new_plr = (TP)/(TP + FN + 3 * FP)
            # new_plr = TPR / (1 + FPR)
            # new_plr = ACC
            # new_plr = TPR + TNR - FPR - FNR   # provare con TPR - FPR
            new_plr = - FPR
            advantage = new_plr - np.average(average_plr)
            average_plr.append(new_plr)
            average_plr.popleft()
            advantage_t = torch.FloatTensor([advantage]).to(device).detach()
            actor_loss = (log_probs * advantage_t).mean()

            optimizer2.zero_grad()
            actor_loss.backward()
            optimizer2.step()
            log_str = log_str_fun(advantage, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR)
            data_logger.set_postfix_str(log_str)

            # loss = loss1.item()
            # loss = loss / settings.bs
            # TP, FP, TN, FN, CP, CN = compute_metrics(predictions, y)

            # TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mh.update_metrics(TP, FP, TN, FN, CP, CN)
            # log_str = log_str_fun(loss1, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR)
            # data_logger.set_postfix_str(log_str)

            if iteration % 1000 == 0 and iteration != 0:
                torch.save(model.state_dict(),
                           dir_ent.folder_train + 'checkpoint.pth')
                val = tester.test(TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, iteration)
                if val > max_PLR:
                # if val < min_FPR:
                    torch.save(model.state_dict(),
                               dir_ent.folder_train + f'best_model_diff5_{val}.pth')
                    best_list.append((iteration, val))
                    max_PLR = val
                    # min_FPR = val
            iteration += 1

    tester.save_csv("looking_diff5")
    print(best_list)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
from torch.utils.data import DataLoader
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from InOut.tools import DirManager
from model.network import resnet_for_the_tsp
from InOut.image_manager import DatasetHandler, OnlineDataSetHandler
from test.utils import Logger, Tester_on_eval, Metrics_Handler, compute_metrics


def show_results(settings):
    for cl_elements in range(1, 5):
        settings.cases_in_L_P = cl_elements
        dir_ent = DirManager(settings)
        df = pd.read_csv(f"{dir_ent.folder_train}training_history.csv", index_col=0)
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
    # model.load_state_dict(torch.load(f'./data/net_weights/CL_{settings.cases_in_L_P}/best_diff.pth',
    #                                  map_location=device))
    torch.save(model.state_dict(),
               dir_ent.folder_train + 'checkpoint.pth')

    # loss function one
    criterion = torch.nn.CrossEntropyLoss()
    log_str_fun = Logger.log_pred
    tester = Tester_on_eval(settings, dir_ent, log_str_fun, settings.cases_in_L_P, device)

    # optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)

    # initialize variables
    iteration = 0
    best_list = []
    best_delta = 0
    average_delta = deque([0. for _ in range(100)])
    for epoch in range(10):
        generator = DatasetHandler(settings)
        data_logger = tqdm(DataLoader(generator, batch_size=settings.bs, drop_last=True))

        for data in data_logger:
            x, y = data["X"], data["Y"]
            x = x.to(device)
            y = y.to(device)

            if iteration <= 2000:
                model.train()
                optimizer.zero_grad()
                predictions1 = model(x)
                loss1 = criterion(predictions1, y)
                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                # loss1.backward()
                optimizer.step()

                predictions3 = model(x)
                dist = Categorical(probs=predictions3)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                TP, FP, TN, FN = compute_metrics(actions.detach(), y.detach(), rl_bool=True)

                mh_off = Metrics_Handler()
                TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, AR = mh_off.update_metrics(TP, FP, TN, FN)
                # new_plr = TPR + TNR - FPR - FNR
                # new_plr = (TP - FP) / (TP + FN)
                new_plr = AR
                advantage = new_plr - np.average(average_delta)
                average_delta.append(new_plr)
                average_delta.popleft()

                advantage_t = torch.FloatTensor([advantage]).to(device).detach()
                actor_loss = (log_probs * advantage_t).mean()

                optimizer2.zero_grad()
                actor_loss.backward()
                optimizer2.step()
            else:
                # if iteration <= 2200:
                #     best_delta = 0
                torch.save(model.state_dict(),
                           dir_ent.folder_train + 'checkpoint.pth')
                # torch.save(model.state_dict(),
                #            dir_ent.folder_train + 'checkpoint.pth')
                online_data_generator = OnlineDataSetHandler(settings, model, mode='train')
                x_online, y_online = online_data_generator.get_data()
                predictions2 = model(x_online)
                loss2 = criterion(predictions2, y_online)
                optimizer.zero_grad()
                loss2.backward(retain_graph=True)
                optimizer.step()
                #
                predictions3 = model(x_online)
                dist = Categorical(probs=predictions3)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                TP, FP, TN, FN = compute_metrics(actions.detach(), y_online.detach(), rl_bool=True)
                # TP, FP, TN, FN = compute_metrics(predictions2.detach(), y_online.detach())
                mh_online = Metrics_Handler()
                TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, AR = mh_online.update_metrics(TP, FP, TN, FN)

                # new_plr = TPR + TNR - FPR - FNR
                # new_plr = (TP - FP) / (TP + FN)
                new_plr = AR
                advantage = new_plr - np.average(average_delta)
                average_delta.append(new_plr)
                average_delta.popleft()

                advantage_t = torch.FloatTensor([advantage]).to(device).detach()
                actor_loss = (log_probs * advantage_t).mean()

                optimizer2.zero_grad()
                actor_loss.backward()
                optimizer2.step()

            log_str = log_str_fun(advantage, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR)
            data_logger.set_postfix_str(log_str)

            if iteration % 200 == 0 and iteration != 0:
                torch.save(model.state_dict(),
                           dir_ent.folder_train + 'checkpoint.pth')
                if iteration == 200:
                    torch.save(model.state_dict(),
                               dir_ent.folder_train + f'best_diff.pth')

                val = tester.test(TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, AR, iteration)
                if val > best_delta:
                    torch.save(model.state_dict(),
                               dir_ent.folder_train + f'diff_{val}.pth')
                    torch.save(model.state_dict(),
                               dir_ent.folder_train + f'best_diff.pth')
                    best_list.append((iteration, val))
                    best_delta = val

            iteration += 1

    tester.save_csv("training_history")
    print(best_list)

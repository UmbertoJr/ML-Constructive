import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from model.network import resnet_for_the_tsp
from InOut.tools import DirManager
from model.utils import compute_metrics, Metrics_Handler
from model.several_models import run_experiments
from InOut.image_manager import DatasetHandler


def train_the_net(settings):
    for cl_index in range(1, 6):
        settings.cases_in_L_P = cl_index
        dir_ent = DirManager(settings)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'device : {device}')
        print(dir_ent.folder_train)

        model = resnet_for_the_tsp(settings)
        model = model.to(device)
        model.apply(model.weight_init)

        # loss function
        criterion = torch.nn.CrossEntropyLoss()
        log_str_fun = Logger.log_pred
        tester = Tester_on_eval(settings, dir_ent, log_str_fun, cl_index, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        iteration = 0
        print(f'\n\nrunning 2 epochs for case with {settings.cases_in_L_P} neigs in L_P ...')
        mh = Metrics_Handler()
        for epoch in range(2):
            generator = DatasetHandler(settings)
            print(f'\n epoch {epoch}\n')
            data_logger = tqdm(DataLoader(generator, batch_size=settings.bs, drop_last=True))
            for data in data_logger:
                x, y = data["X"], data["Y"]
                x = x.to(device)
                y = y.to(device)

                model.train()
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, y)

                loss.backward()
                optimizer.step()
                loss = loss.item()
                loss = loss / settings.bs
                TP, FP, TN, FN, CP, CN = compute_metrics(predictions, y)

                TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mh.update_metrics(TP, FP, TN, FN, CP, CN)
                log_str = log_str_fun(loss, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR)
                data_logger.set_postfix_str(log_str)

                if iteration % 100 == 0 and iteration != 0:
                    torch.save(model.state_dict(),
                               dir_ent.folder_train + 'checkpoint.pth')
                    val = tester.test(model, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, iteration)

                iteration += 1

        tester.save_csv()


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
                        "test TPR": [], "test FPR": [], "test TNR": [], "test FNR": [],
                        "test Acc": [], "test bal Acc": [], "test PLR": [], "test bal PLR": []
                        }
        self.iter_list = []
        self.best_bal_PLR = 0.
        self.folder_data = dir_ent.create_folder_for_train(cl)
        self.device = 'cpu'
        self.net = resnet_for_the_tsp(self.settings)
        self.net.load_state_dict(torch.load(self.dir_ent.folder_train + f'checkpoint.pth',
                                            map_location='cpu'))

    def test(self, tpr, fnr, fpr, tnr, acc, bal_acc, plr, bal_plr, iteration_train):
        generator = DatasetHandler(self.settings, path='./data/eval/')
        data_logger = DataLoader(generator, batch_size=132, drop_last=True)
        mht = Metrics_Handler()
        TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = (0 for i in range(8))
        print(len(data_logger))
        for iter, data in enumerate(data_logger):
            self.net.eval()
            x, y = data["X"], data["Y"]
            x = x.to(self.device)
            y = y.to(self.device)

            predictions = self.net(x)

            TP, FP, TN, FN = compute_metrics(predictions.detach(), y.detach())

            TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mht.update_metrics(TP, FP, TN, FN)
            # print(iter)
            if iter > 100:
                break

        self.df_data["train TPR"].append(tpr)
        self.df_data["train FNR"].append(fnr)
        self.df_data["train TNR"].append(tnr)
        self.df_data["train FPR"].append(fpr)
        self.df_data["train Acc"].append(acc)
        self.df_data["train bal Acc"].append(bal_acc)
        self.df_data["train PLR"].append(plr)
        self.df_data["train bal PLR"].append(bal_plr)

        self.df_data["test TPR"].append(TPR)
        self.df_data["test FNR"].append(FNR)
        self.df_data["test TNR"].append(TNR)
        self.df_data["test FPR"].append(FPR)
        self.df_data["test Acc"].append(ACC)
        self.df_data["test bal Acc"].append(BAL_ACC)
        self.df_data["test PLR"].append(PLR)
        self.df_data["test bal PLR"].append(BAL_PLR)

        self.iter_list.append(iteration_train)
        return PLR

    def save_csv(self, name):
        self.df = pd.DataFrame(data=self.df_data, index=self.iter_list)
        if not name:
            self.name_case = f'dataset_train_history'
        else:
            self.name_case = name
        self.df.to_csv(f"{self.folder_data}{self.name_case}.csv")

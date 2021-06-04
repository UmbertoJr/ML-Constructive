import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.network import resnet_for_the_tsp
from InOut.tools import DirManager
# from model.utils import compute_metrics, Metrics_Handler
from model.several_models import run_experiments
from InOut.image_manager import DatasetHandler


# def train_the_net(settings):
#     for cl_index in range(1, 6):
#         settings.cases_in_L_P = cl_index
#         dir_ent = DirManager(settings)
#
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print(f'device : {device}')
#         print(dir_ent.folder_train)
#
#         model = resnet_for_the_tsp(settings)
#         model = model.to(device)
#         model.apply(model.weight_init)
#
#         # loss function
#         criterion = torch.nn.CrossEntropyLoss()
#         log_str_fun = Logger.log_pred
#         tester = Tester_on_eval(settings, dir_ent, log_str_fun, cl_index, device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#         iteration = 0
#         print(f'\n\nrunning 2 epochs for case with {settings.cases_in_L_P} neigs in L_P ...')
#         mh = Metrics_Handler()
#         for epoch in range(2):
#             generator = DatasetHandler(settings)
#             print(f'\n epoch {epoch}\n')
#             data_logger = tqdm(DataLoader(generator, batch_size=settings.bs, drop_last=True))
#             for data in data_logger:
#                 x, y = data["X"], data["Y"]
#                 x = x.to(device)
#                 y = y.to(device)
#
#                 model.train()
#                 optimizer.zero_grad()
#                 predictions = model(x)
#                 loss = criterion(predictions, y)
#
#                 loss.backward()
#                 optimizer.step()
#                 loss = loss.item()
#                 loss = loss / settings.bs
#                 TP, FP, TN, FN, CP, CN = compute_metrics(predictions, y)
#
#                 TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR = mh.update_metrics(TP, FP, TN, FN)
#                 log_str = log_str_fun(loss, TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR)
#                 data_logger.set_postfix_str(log_str)
#
#                 if iteration % 100 == 0 and iteration != 0:
#                     torch.save(model.state_dict(),
#                                dir_ent.folder_train + 'checkpoint.pth')
#                     val = tester.test(TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR, iteration)
#
#                 iteration += 1
#
#         tester.save_csv()
#

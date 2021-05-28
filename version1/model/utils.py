import numpy as np
from InOut.tools import print_sett


def compute_metrics(preds, y, RL=False):
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()
    TP, FP, TN, FN, CP, CN = (0. for _ in range(6))
    for i in range(preds.shape[0]):
        if not RL:
            best = np.argsort(preds[i])[::-1][:1][0]
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
        self.CP = 1
        self.CN = 1
        self.TP = 1
        self.FP = 0
        self.TN = 1
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
        TNR = self.FN / self.CN
        ACC = (self.TP + self.TN) / (self.CN + self.CP)
        BAL_ACC = (TPR + TNR) / 2
        PLR = TPR / (1 + FPR)
        # PLR = self.TP / (self.TP + self.FN + 3 * self.FP)
        BAL_PLR = self.TP / self.FP
        return TPR, FNR, FPR, TNR, ACC, BAL_ACC, PLR, BAL_PLR

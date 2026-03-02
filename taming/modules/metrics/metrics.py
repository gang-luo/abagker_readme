import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,Tuple
import torchmetrics
import numpy as np
from sklearn import metrics
from math import sqrt
from scipy import stats
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryMatthewsCorrCoef, BinaryAccuracy, BinaryPrecision, 
    BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
)

class NPMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.add_state("preds", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds = torch.cat([self.preds, preds.detach().float().reshape(-1)])
        self.targets = torch.cat([self.targets, target.detach().float().reshape(-1)])

    def compute(self) -> torch.Tensor:
        y = self.targets.cpu().numpy()
        p = self.preds.cpu().numpy()
        return torch.tensor(float(self.func(y, p)), dtype=torch.float32)



def roc_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the ROC AUC score."""
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the Precision-Recall AUC score."""
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Root Mean Squared Error."""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Mean Squared Error."""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient."""
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Spearman correlation coefficient."""
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Concordance Index."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
    
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))



class CI(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.flatten()
        target = target.flatten()
        self.y_true.append(target)
        self.y_pred.append(preds)

    def compute(self):
        # Support both list-of-tensors (from append) and tensor states
        if isinstance(self.y_true, list):
            y_true_t = torch.cat([t.detach().float().flatten() for t in self.y_true], dim=0)
        else:
            y_true_t = self.y_true.detach().float().flatten()

        if isinstance(self.y_pred, list):
            y_pred_t = torch.cat([t.detach().float().flatten() for t in self.y_pred], dim=0)
        else:
            y_pred_t = self.y_pred.detach().float().flatten()

        y_true = y_true_t.cpu().numpy()
        y_pred = y_pred_t.cpu().numpy()
        return torch.tensor(ci(y_true, y_pred), dtype=torch.float32)


class RM2(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.flatten()
        target = target.flatten()
        self.y_true.append(target)
        self.y_pred.append(preds)

    def compute(self):
        # Support both list-of-tensors (from append) and tensor states
        if isinstance(self.y_true, list):
            y_true_t = torch.cat([t.detach().float().flatten() for t in self.y_true], dim=0)
        else:
            y_true_t = self.y_true.detach().float().flatten()

        if isinstance(self.y_pred, list):
            y_pred_t = torch.cat([t.detach().float().flatten() for t in self.y_pred], dim=0)
        else:
            y_pred_t = self.y_pred.detach().float().flatten()

        y_true = y_true_t.cpu().numpy()
        y_pred = y_pred_t.cpu().numpy()
        return torch.tensor(get_rm2(y_true, y_pred), dtype=torch.float32)


class PktMultiClassEvaluator(Metric):
    """
    bindingsites Evaluator
    - update: preds_logits [B, T, 2],targets [B, T], the calue is  {0,1,2} (2 is ignore)
    - compute return  dict[metric_name -> Tensor]
    """

    full_state_update = True
    higher_is_better = True

    def __init__(self, prefix: str = "val/", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds_logits: torch.Tensor, targets: torch.Tensor):
        # preds_logits: [B, T, 2] -> [B*T, 2]
        # targets: [B, T] -> [B*T]
        logits_flat = preds_logits.detach().float().reshape(-1, 2)  # [B*T, 2]
        targets_flat = targets.detach().long().reshape(-1)          # [B*T]
        
        # filter ignore_index (the value = 2)
        valid_mask = targets_flat != 2
        if valid_mask.any():
            self.logits.append(logits_flat[valid_mask])
            self.targets.append(targets_flat[valid_mask])

    def compute(self):
        if isinstance(self.logits, list):
            logits_t = torch.cat([t for t in self.logits], dim=0)  # [N_valid, 2]
        else:
            logits_t = self.logits
        if isinstance(self.targets, list):
            targets_t = torch.cat([t for t in self.targets], dim=0)  # [N_valid]
        else:
            targets_t = self.targets

        if len(targets_t) == 0:
            return self._get_default_metrics()

        logits_cpu = logits_t.detach().to("cpu")  # [N_valid, 2]
        targets_cpu = targets_t.detach().to("cpu")  # [N_valid]

        # softmax for get positive propotis
        probs_all = torch.softmax(logits_cpu, dim=-1)  # [N_valid, 2]
        probs_pos = probs_all[:, 1]  # [N_valid] 

        # max MCC for thresholds
        best_mcc = -2.0
        best_thresh = 0.5
        thresholds = torch.arange(0.01, 1.0, 0.01)
        mcc_metric = BinaryMatthewsCorrCoef().to("cpu")

        for th in thresholds:
            pred = (probs_pos >= th).long()
            mcc = mcc_metric(pred, targets_cpu).item()
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = float(th.item())

        best_preds = (probs_pos >= best_thresh).long()

        acc = BinaryAccuracy().to("cpu")(best_preds, targets_cpu)
        prec = BinaryPrecision().to("cpu")(best_preds, targets_cpu)
        rec = BinaryRecall().to("cpu")(best_preds, targets_cpu)
        f1 = BinaryF1Score().to("cpu")(best_preds, targets_cpu)
        auroc = BinaryAUROC().to("cpu")(probs_pos, targets_cpu)
        auprc = BinaryAveragePrecision().to("cpu")(probs_pos, targets_cpu)

        return {
            f"{self.prefix}best_threshold": torch.tensor(best_thresh, dtype=torch.float32),
            f"{self.prefix}pkt_mcc": torch.tensor(best_mcc, dtype=torch.float32),
            f"{self.prefix}pkt_acc": acc.detach().to("cpu"),
            f"{self.prefix}pkt_precision": prec.detach().to("cpu"),
            f"{self.prefix}pkt_recall": rec.detach().to("cpu"),
            f"{self.prefix}pkt_f1": f1.detach().to("cpu"),
            f"{self.prefix}pkt_auroc": auroc.detach().to("cpu"),
            f"{self.prefix}pkt_auprc": auprc.detach().to("cpu"),
        }

    def _get_default_metrics(self):
        return {
            f"{self.prefix}best_threshold": torch.tensor(0.5, dtype=torch.float32),
            f"{self.prefix}pkt_mcc": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_acc": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_precision": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_recall": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_f1": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_auroc": torch.tensor(0.0, dtype=torch.float32),
            f"{self.prefix}pkt_auprc": torch.tensor(0.0, dtype=torch.float32),
        }

    def reset(self):
        super().reset()
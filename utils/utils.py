import numpy as np
import torch
from sklearn.metrics import roc_curve, jaccard_score, precision_recall_curve, auc, confusion_matrix, roc_auc_score, average_precision_score


def fpr_at_tpr(scores, labels, tpr=0.95):
    labels = labels.view(-1)
    scores = scores.cpu().view(-1)
    valid = torch.logical_and(labels<=1, labels>=0)
    fprs, tprs, _ = roc_curve(labels[valid], scores[valid])
    idx = np.argmin(np.abs(np.array(tprs)-tpr))
    return fprs[idx]


def aupr(scores, gt):
    gt = gt.view(-1)
    scores = scores.cpu().view(-1)
    valid = torch.logical_and(gt >= 0, gt <= 1)
    return average_precision_score(gt[valid], scores[valid])


def prediction_entropy(probs, eps=1e-6):
    assert len(probs.shape) == 4, "Required: probs (b, c, h, w), got shape {} instead".format(probs.shape)
    h = -(probs * (probs+eps).log()).sum(1)
    return h


def miou(logits, target, n_classes, ignore_idx=None):
    def _fast_hist(label_true, label_pred):
        mask = (label_true >= 0) & (label_true < n_classes) & (label_true != ignore_idx)
        hist = np.bincount(
            n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_classes ** 2,
        ).reshape(n_classes, n_classes)
        return hist
    cm = np.zeros((n_classes, n_classes))

    for lt, lp in zip(target.numpy(), logits.argmax(1).numpy()):
         cm += _fast_hist(lt.flatten(), lp.flatten())
    iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mean_iu = np.nanmean(iu)
    return mean_iu


class AverageValueMeter():
    def __init__(self):
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        value = float(value)
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
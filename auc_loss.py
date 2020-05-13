import torch
import torch.nn as nn


def auc_loss(y_pred, y_true):
    '''Soft version of AUC that uses Wilcoxon-Mann-Whitney U. statistic'''

    # Grab the logits of all the positive and negative examples
    pos = y_pred[y_true.view(-1, 1).bool()].view(1, -1)
    neg = y_pred[~y_true.view(-1, 1).bool()].view(-1, 1)
    gamma = 0.7
    p = 3
    difference = torch.zeros_like(pos * neg) + pos - neg - gamma
    masked = difference[difference < 0.0]
    return torch.sum(torch.pow(-masked, p))
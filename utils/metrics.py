import torch
from torch.nn.functional import one_hot

import numpy as np
from scipy.special import softmax


class NbsLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, w1=None):
        out = self.ce(input, target)
        if w1 is None:
            return out.mean()
        out = out * w1
        if self.reduction == 'mean':
            return out.mean()
        else:
            return out.sum()


class BrierLoss(torch.nn.Module):
    def __init__(self, reduction='mean', num_classes=10):
        super().__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        target_onehot = one_hot(target, self.num_classes).float()
        out = self.mse(input, target_onehot)
        return out


class CrossEntropyLossWithSoftLabel(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        loss = (-target * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Accuracy(torch.nn.Module):
    def __init__(self, reduction='mean', nlabels=5):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(.5).type_as(target)
        else:
            pred = input.argmax(1)
        acc = pred == target
        if self.reduction == 'mean':
            acc = acc.float().mean()
        elif self.reduction == 'sum':
            acc = acc.float().sum()
        return acc


class ConfusionMatrix(torch.nn.Module):
    def __init__(self, nlabels=5):
        super().__init__()
        self.nlabels = nlabels

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(.5).type_as(target)
        else:
            pred = input.argmax(1)

        cm = torch.zeros([self.nlabels, 4]).cuda()
        for l in range(self.nlabels):
            if self.nlabels == 1:
                _pred = pred.eq(1).float()
                _label = target.eq(l).float()
            else:
                _pred = pred.eq(l).float()
                _label = target.eq(l).float()

            _cm = _pred * 2 - _label
            tp = _cm.eq(1).float().sum()
            tn = _cm.eq(0).float().sum()
            fp = _cm.eq(2).float().sum()
            fn = _cm.eq(-1).float().sum()

            for j, j_ in zip(cm[l], [tp, tn, fp, fn]):
                j += j_

        return cm


# ECE
def calc_ece(logit, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    logit_softmax = torch.tensor(softmax(logit, -1))
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(logit_softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item() * 100


# NLL & Brier Score
def calc_nll_brier(logit, label, label_onehot):
    logit_softmax = softmax(logit, -1)
    brier_score = np.mean(np.sum((logit_softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    return nll.item() * 10, brier_score * 100


# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum() / len(out)


if __name__ == "__main__":
    Acc = Accuracy()
    # a = torch.rand(8, 5, 64, 256, 256).float()
    # b = torch.randint(5, [8, 64, 256, 256])
    a = torch.rand(1, 3, 5)
    b = torch.randint(3, (1, 5))
    print(a)
    print(a.argmax(1))
    print(b)
    # print(Acc(a, b))

    dice = Dice(reduction='weighted_mean', nlabels=3, weights="1,1,1")
    # dice = Dice(reduction='index', index=0)
    # dice = Dice()
    print(dice(a, b))

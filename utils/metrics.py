import torch
from torch.nn.functional import one_hot

import numpy as np
from scipy.special import softmax


class NbsLoss(torch.nn.Module):
    def __init__(self, reduction='mean',
                 base_loss=torch.nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.reduction = reduction
        self.base_loss = base_loss

    def forward(self, input, target, w=None):
        out = self.base_loss(input, target)
        if w is not None:
            out = out * w
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            return out


class BrierLoss(torch.nn.Module):
    def __init__(self, reduction='mean', num_classes=10):
        super().__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, input, target):
        target_onehot = one_hot(target, self.num_classes).float()
        out = self.mse(input.softmax(-1), target_onehot).sum(-1)
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
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


class MeanIOU(torch.nn.Module):
    def __init__(self, reduction='mean', nlabels=5):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels
        self.eps = 0.001

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(.5).type_as(target)
        else:
            pred = input.argmax(1)
        
        jccs = []
        for l in range(1, self.nlabels):
            _pred = pred.eq(l).float()
            _label = target.eq(l).float()

            _cm = _pred * 2 - _label
            dims = list(set(range(target.dim())) - set([0]))
            tp = _cm.eq(1).float().sum(dim=dims)
            tn = _cm.eq(0).float().sum(dim=dims)
            fp = _cm.eq(2).float().sum(dim=dims)
            fn = _cm.eq(-1).float().sum(dim=dims)

            jcc = (tp + self.eps) / (fn + fp + tp + self.eps)
            jccs += [jcc[:, None]]

        return torch.cat(jccs, dim=1).mean(1)


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


# class ECE(nn.Module):
#     def __init__(self, num_bins=15, is_mc=False):
#         super().__init__()
#         self.num_bins = num_bins
#         self.is_mc = is_mc

#     def forward(self, input, target):
        


# ECE
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
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


def one_hot_np(array, num=None):
    if not num:
        num = array.max() + 1
    return np.eye(num)[array]


# NLL & Brier Score
def calc_nll_brier(logit, label, num_classes):
    label_onehot = one_hot_np(label, num_classes)
    logit_softmax = softmax(logit, -1)
    brier_score = np.mean(np.sum((logit_softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    return nll.item() * 10, brier_score * 100

# NLL & Brier Score
def calc_nll_brier_mc(logit, label, num_classes):
    label_onehot = one_hot_np(label, num_classes)
    logit_softmax = softmax(logit, -1).mean(0)
    brier_score = np.mean(np.sum((logit_softmax - label_onehot) ** 2, axis=1))

    logit = logit.mean(0)
    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=-1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    return nll.item() * 10, brier_score * 100


# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum() / len(out)


def get_metrics(output, label, num_classes):
    acc = (output.argmax(1) == label).mean() * 100
    ece = calc_ece(softmax(output, -1), label)
    nll, brier = calc_nll_brier(output, label, num_classes)
    return acc, ece, nll, brier


def get_metrics_mc(output, label, num_classes):
    acc = (output.mean(0).argmax(-1) == label).mean() * 100
    ece = calc_ece(softmax(output, -1).mean(0), label)
    nll, brier = calc_nll_brier_mc(output, label, num_classes)
    return acc, ece, nll, brier


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

    dice = MeanIOU(reduction='mean', nlabels=3)
    # dice = Dice(reduction='index', index=0)
    # dice = Dice()
    print(dice(a, b).numpy())

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import softmax
from torch.distributions.exponential import Exponential
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score



def multi_calibration_curve(logits, labels, bins=10, is_softmax=False):
    bin_boundaries = np.linspace(0., 1., bins+1, dtype=float)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if is_softmax:
        softmaxes = logits
    else:
        softmaxes = softmax(logits, 1)
    confidences = softmaxes.max(1)
    predictions = softmaxes.argmax(1)
    accuracies = predictions == labels
    ece = np.zeros(1)
    acc = np.zeros(10)
    conf = np.zeros(10)
    pnt = np.zeros(10)
    cnt = np.zeros(10)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        pnt[i] += (bin_lower + bin_upper) / 2
        cnt[i] += prop_in_bin
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
            acc[i] += acc_in_bin
            conf[i] += avg_conf_in_bin

    return pnt, acc, conf, cnt, ece


def plot_multiclass_calibration_curve(probs, labels, bins=10,
                                      title=None, is_softmax=False):
    fsize = 20
    midpnts, accs, cfds, cnt, ece = multi_calibration_curve(probs, labels,
                                                            bins, is_softmax)
    title = 'Reliability Diagram' if title is None else title
    plt.rcParams['font.family'] = ['Times New Roman']

    fig, ax = plt.subplots(2, 1,
                           gridspec_kw={
                                'height_ratios': [5, 2]}, figsize=(10, 10))

    ax[0].bar(midpnts, accs, width=1.0/float(bins),
              align='center', lw=1, ec='#000000', fc='#2233aa',
              alpha=1, label='Outputs', zorder=0)
    ax[0].scatter(midpnts, accs, lw=2, ec='black', fc="#ffffff", zorder=2)
    ax[0].plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2,
               alpha=.7, color='gray', label='Perfectly calibrated', zorder=1)

    ax[0].set_title(f'{title} (ECE = {ece[0] * 100:.2f}%)\n', fontsize=fsize)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel('Accuracy\n', fontsize=fsize)
    ax[0].set_xticks(midpnts)  # , rotation=-45)
    ax[0].legend(loc='upper left', fontsize=fsize)
    plt.tight_layout()

    ax[1].bar(midpnts, cnt * 100, width=1.0/float(bins),
              align='center', lw=1, ec='#000000', fc='#2233aa',
              alpha=1, label='Model', zorder=0)
    ax[1].set_xlim(0.0, 1.0)
#     ax[1].set_ylim(0, 10)
    ax[1].set_xticks(midpnts)
    ax[1].set_xlabel('\nConfidence', fontsize=fsize)
    ax[1].set_ylabel('% of Samples', fontsize=fsize)
    # ax[1].set_yscale('log')
#     ax[1].set_yticks([0, 10, 40])
    # ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
#     ax[1].get_yaxis().set_tick_params(which='minor', size=0)
#     ax[1].get_yaxis().set_tick_params(which='minor', width=0)

    plt.show()
    return midpnts, accs, cfds, cnt


@torch.no_grad()
def infer(loader, model, num_bs, num_classes, fac, with_acc=False):
    model.eval()
    a_test_ = Exponential(torch.ones([1, 400]))
    a_test = a_test_.sample((num_bs,))
    acc = 0.
    outputs = np.zeros([num_bs, len(loader.dataset), num_classes + 1])
    beg = 0
    for i, (img, label) in enumerate(loader):
        index = list(range(beg, beg + img.size(0)))
        beg = beg + img.size(0)
        label = label.numpy().reshape(-1, 1)
        for _ in range(num_bs):
            w_test = a_test[_].repeat_interleave(img.shape[0], dim=0)
            output = model(img, w_test, fac).cpu().numpy()
            outputs[_, index] = np.concatenate([output, label], axis=1)

    if with_acc:
        pred = outputs.sum(0)[:, :-1].argmax(1)
        label = outputs[0][:, -1]
        acc = pred == label
        print(f"[Test] acc : {acc.mean()}")
    return outputs


def predictive_mean(x):
    return softmax(x, axis=-1).mean(0)


def predictive_entropy(x):
    return entropy(predictive_mean(x), axis=-1)


def expected_entropy(x):
    return entropy(softmax(x, axis=-1), axis=-1).mean(0)


def mutual_information(x):
    return predictive_entropy(x) - expected_entropy(x)


def predictive_std(x):
    return softmax(x, axis=-1).std(0).sum(1)


def histograms(mean_id, mean_od, std_id, std_od, bald_id, bald_od, label):
    plt.rcParams['font.family'] = ['Times New Roman']

    fig, ax = plt.subplots(1, 3,
                         gridspec_kw={
                             'width_ratios': [.3, .3, .3]}, figsize=(20, 4))

    beg = min(mean_id.min(), mean_od.min())
    end = max(mean_id.max(), mean_od.max())
    ax[0].hist(mean_id.max(1), range=(beg, end), bins=200, alpha=0.5, label='In-dist.')
    ax[0].hist(mean_od.max(1), range=(beg, end), bins=200, alpha=0.5, label='Out-dis.')
    ax[0].legend()
    mean = np.r_[mean_id, mean_od]
    tpr, fpr, ths = roc_curve(label, mean.max(1))
    if auc(fpr, tpr) < .5:
        fpr, tpr = tpr, fpr
    ax[0].set_ylabel('Count')
    ax[0].set_title(f'Predictive mean (AUC={auc(fpr, tpr):.4})')

    beg = min(std_id.min(), std_od.min())
    end = max(std_id.max(), std_od.max())
    ax[1].hist(std_id, range=(beg, end), bins=200, alpha=0.5, label='In-dist.')
    ax[1].hist(std_od, range=(beg, end), bins=200, alpha=0.5, label='Out-dis.')
    ax[1].legend()
    std = np.r_[std_id, std_od]
    fpr, tpr, ths = roc_curve(label, std)
    if auc(fpr, tpr) < .5:
        fpr, tpr = tpr, fpr
    ax[1].set_title(f'Predictive std (AUC={auc(fpr, tpr):.4})')

    beg = min(bald_id.min(), bald_od.min())
    end = max(bald_id.max(), bald_od.max())
    ax[2].hist(bald_id, range=(beg, end), bins=200, alpha=0.5, label='In-dist.')
    ax[2].hist(bald_od, range=(beg, end), bins=200, alpha=0.5, label='Out-dis.')
    ax[2].legend()
    bald = np.r_[bald_id, bald_od]
    fpr, tpr, ths = roc_curve(label, bald)
    if auc(fpr, tpr) < .5:
        fpr, tpr = tpr, fpr
    ax[2].set_title(f'Mutual Information (AUC={auc(fpr, tpr):.4})')
    plt.show()

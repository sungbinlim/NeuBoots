import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


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

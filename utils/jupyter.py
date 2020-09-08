import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
def infer(loader, model, num_bs, num_classes, fac, with_acc=False, seed=0, is_mlp=False):
    torch.manual_seed(seed)
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
            # w_test = a_test_.sample((img.shape[0],))
            if is_mlp:
                img = img.view(img.shape[0], -1)
            output = model(img, w_test, fac).cpu().numpy()
            outputs[_, index] = np.concatenate([output, label], axis=1)

    if with_acc:
        pred = outputs.sum(0)[:, :-1].argmax(1)
        label = outputs[0][:, -1]
        acc = pred == label
        print(f"[Test] acc : {acc.mean()}")
    return outputs


def odin_infer(loader, model, num_bs, num_classes, fac, with_acc=False, seed=0, T=1000, eps=0.0014):
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.manual_seed(seed)
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
            img_ = img.cuda()
            img_.requires_grad = True
            output = model(img_, w_test, fac)

            output = output / T
            pseudo_label = output.argmax(-1).cuda()
            loss = loss_fn(output, pseudo_label)
            loss.backward()

            gradient = torch.ge(img_.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
 
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

            img_new = torch.add(img_.data, -eps, gradient)
            output_new = model(img_new, w_test, fac).cpu().detach().numpy()
            outputs[_, index] = np.concatenate([output_new, label], axis=1)
    
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
    ax[0].hist(mean_id.max(1), range=(beg, end), bins=200, alpha=0.5, label='Before ADV attack', color='dodgerblue')
    ax[0].hist(mean_od.max(1), range=(beg, end), bins=200, alpha=0.5, label='After ADV attack', color='orangered')
    ax[0].legend()
    mean = np.r_[mean_id, mean_od]
    tpr, fpr, ths = roc_curve(label, mean.max(1))
    if auc(fpr, tpr) < .5:
        fpr, tpr = tpr, fpr
    ax[0].set_ylabel('Count')
    ax[0].set_title(f'Predictive mean (AUC={auc(fpr, tpr):.4})')

    beg = min(std_id.min(), std_od.min())
    end = max(std_id.max(), std_od.max())
    ax[1].hist(std_id, range=(beg, end), bins=200, alpha=0.5, label='Before ADV attack', color='dodgerblue')
    ax[1].hist(std_od, range=(beg, end), bins=200, alpha=0.5, label='After ADV attack', color='orangered')
    ax[1].legend()
    std = np.r_[std_id, std_od]
    if std_id.shape[0] == std_od.shape[0]:
        fpr, tpr, ths = roc_curve(label, std)

        if auc(fpr, tpr) < .5:
            fpr, tpr = tpr, fpr
        score = auc(fpr, tpr)
    else:
        score = average_precision_score(label, std)
    ax[1].set_title(f'Predictive std (AUC={score:.4})')

    beg = min(bald_id.min(), bald_od.min())
    end = max(bald_id.max(), bald_od.max())
    ax[2].hist(bald_id, range=(beg, end), bins=200, alpha=0.5, label='Before ADV attack', color='dodgerblue')
    ax[2].hist(bald_od, range=(beg, end), bins=200, alpha=0.5, label='After ADV attack', color='orangered')
    ax[2].legend()
    bald = np.r_[bald_id, bald_od]
    if bald_id.shape[0] == bald_od.shape[0]:
        fpr, tpr, ths = roc_curve(label, bald)

        if auc(fpr, tpr) < .5:
            fpr, tpr = tpr, fpr
        score = auc(fpr, tpr)
    else:
        score = average_precision_score(label, bald)
    ax[2].set_title(f'Mutual Information (AUC={score:.4})')
    plt.show()


def hist_all(mean_id, mean_od0, mean_od1, mean_od2,
             std_id, std_od0, std_od1, std_od2,
             bald_id, bald_od0, bald_od1, bald_od2):
    plt.rcParams['font.family'] = ['Times New Roman']

    fig, ax = plt.subplots(1, 3,
                        gridspec_kw={
                            'width_ratios': [.3, .3, .3]}, figsize=(20, 4))

    beg = min(mean_id.min(), mean_od0.min(), mean_od1.min(), mean_od2.min())
    end = max(mean_id.max(), mean_od0.max(), mean_od1.max(), mean_od2.max())
    ax[0].hist(mean_id.max(1), range=(beg, end), bins=200, alpha=0.5, label='In-dist.(CIFAR-100)', color='dodgerblue')
    ax[0].hist(mean_od0.max(1), range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(Imagenet)', color='mediumpurple')
    ax[0].hist(mean_od1.max(1), range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(LSUN)', color='lawngreen')
    ax[0].hist(mean_od2.max(1), range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(SVHN 33%)', color='orangered')
    ax[0].legend()
    ax[0].set_ylabel('Count')
    ax[0].set_title(f'Predictive means')

    beg = min(std_id.min(), std_od0.min(), std_od1.min(), std_od2.min())
    end = max(std_id.max(), std_od0.max(), std_od1.max(), std_od2.max())
    ax[1].hist(std_id, range=(beg, end), bins=200, alpha=0.5, label='In-dist.(CIFAR-100)', color='dodgerblue')
    ax[1].hist(std_od0, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(Imagenet)', color='mediumpurple')
    ax[1].hist(std_od1, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(LSUN)', color='lawngreen')
    ax[1].hist(std_od2, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(SVHN 33%)', color='orangered')
    ax[1].legend()
    ax[1].set_title(f'Predictive stds')

    beg = min(bald_id.min(), bald_od0.min(), bald_od1.min(), bald_od2.min())
    end = max(bald_id.max(), bald_od0.max(), bald_od1.max(), bald_od2.max())
    ax[2].hist(bald_id, range=(beg, end), bins=200, alpha=0.5, label='In-dist.(CIFAR-100)', color='dodgerblue')
    ax[2].hist(bald_od0, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(Imagenet)', color='mediumpurple')
    ax[2].hist(bald_od1, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(LSUN)', color='lawngreen')
    ax[2].hist(bald_od2, range=(beg, end), bins=200, alpha=0.5, label='Out-dist.(SVHN 33%)', color='orangered')
    ax[2].legend()
    ax[2].set_title(f'Mutual Informations')
    plt.show()


def fgsm(model, img, target, step_size=0.1, train_mode=False, mode=None, w=torch.ones([1, 400]), fac=1):
    loss_fn = torch.nn.CrossEntropyLoss()
    img = img.cuda()
    img.requires_grad = True
    model.eval()
    model.zero_grad()
    output = model(img, w, fac)
    pred = output.data.cpu().argmax(1).squeeze()
    # corr = pred.eq(target)
    loss = loss_fn(output, target.cuda())
    loss.backward()
    grad_sign = img.grad.sign()
    img_new = img + step_size * grad_sign
    output_new = model(img_new, w, fac)
    pred_new = output_new.data.cpu().argmax(1).squeeze()
    return img_new, pred, pred_new


@torch.no_grad()
def infer_a_sample(img, model, num_classes, num_bs, fac):
    model.eval()
    a_test = Exponential(torch.ones([1, 400]))
    w_test = a_test.sample((num_bs,))
    output = np.zeros([num_bs, num_classes])
    for _ in range(num_bs):
        o = model(img, w_test[_], fac).cpu().numpy()
        output[_] = o
    return output
    

def odin_infer_a_sample(img, model, num_classes, num_bs, fac, T=1000, eps=0.0001):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    img = img.cuda()
    img.requires_grad = True
    model.zero_grad()
    a_test = Exponential(torch.ones([1, 400]))
    w_test = a_test.sample((num_bs,))
    outputs = np.zeros([num_bs, num_classes])
    for _ in range(num_bs):
        # w_test = a_test[_].repeat_interleave(img.shape[0], dim=0)
        img_ = img.cuda()
        img_.requires_grad = True
        output = model(img_, w_test[_], fac)

        output = output / T
        pseudo_label = output.argmax(-1).cuda()
        loss = loss_fn(output, pseudo_label)
        loss.backward()

        gradient = torch.ge(img_.grad.data, 0)
        # gradient = (gradient.float() - 0.5) * 2

        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                            #  gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                            #  gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                            #  gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        img_new = torch.add(img_.data, -eps, gradient)
        output_new = model(img_new, w_test[_], fac).cpu().detach().numpy()
        outputs[_] = output_new
    return outputs


def save_fgsm(path, model, img, target, step_size=0.1, fac=1):
    w = torch.ones([1, 400])
    loss_fn = torch.nn.CrossEntropyLoss()
    img = img.cuda()
    img.requires_grad = True
    model.eval()
    model.zero_grad()
    output = torch.cat([model(img, w, fac) for _ in range(5)], 0).sum(0, keepdim=True)
    pred = output.data.cpu().argmax(1).squeeze()
    # pred = output.data.cpu().argmax()
    if pred == target:
        loss = loss_fn(output, target.cuda())
        loss.backward()
        grad_sign = img.grad.sign()
        img_new = img + step_size * grad_sign
        new_img = img_new[0].cpu().detach().numpy().transpose(1, 2, 0)
        new_img[..., 0] *= 0.2023
        new_img[..., 1] *= 0.1994
        new_img[..., 2] *= 0.2010
        new_img[..., 0] += 0.4914
        new_img[..., 1] += 0.4822
        new_img[..., 2] += 0.4465
        new_img *= 255
        new_img = new_img.clip(0, 255)

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), new_img)
import torch
from torch.nn.functional import one_hot
from torch.distributions.exponential import Exponential

import numpy as np
from scipy.special import softmax

from runner.cnn_runner import CnnClsfier
from utils.jupyter import *


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()
        m.p = 0.2


class NbsCnnClsfier(CnnClsfier):
    def __init__(self, args, loader, model, optim, lr_schdlr, loss):
        self.n_a = args.n_a  # n_a: # of sub-group
        self.epoch_th = args.epoch_th
        self.n_test = loader.n_test
        self.group_indices = loader.groups
        self.alpha = torch.ones([1, self.n_a])
        self.num_bs = args.num_bs
        self.is_nbs = args.is_nbs
        self.num_classes = args.num_classes
        self.dropout_rate = args.dropout_rate
        self.schdlr_type = args.scheduler
        super().__init__(args, loader, model, optim, lr_schdlr, loss)
        self.save_kwargs['alpha'] = self.alpha

    def _update_weight(self):
        if self.epoch > self.epoch_th:
            self.alpha = Exponential(torch.ones([1, self.n_a])).sample()

    def _calc_loss(self, img, label, idx):
        self.G.train()
        n0 = img.size(0)    # n0: batch_size
        u_is = []
        for i in idx:
            u_i = np.where(self.group_indices == i.item())[0][0]
            u_is += [u_i]

        if self.is_nbs:
            w = self.alpha[0, u_is].cuda()
        else:
            w = None

        output = self.G(img, self.alpha.repeat_interleave(n0, 0))
        loss = self.loss(output, label.cuda(), w)
        return loss

    @torch.no_grad()
    def _valid_a_batch(self, img, label):
        self._update_weight()
        self.G.eval()
        w_test = torch.zeros([img.size(0), self.n_a]).cuda()
        output = self.G(img, w_test)
        pred = output.argmax(1).cpu()
        return (pred == label).numpy()

    @torch.no_grad()
    def test(self):
        self.load('model.pth')
        loader = self.loader.load('test')
        acc = 0.
        beg = 0
        outputs = np.zeros([self.num_bs, self.n_test, self.num_classes + 1])
        self.G.eval()
        if self.dropout_rate > 0.:
            self.G.apply(apply_dropout)
        for img, label in loader:
            index = list(range(beg, beg + img.size(0)))
            beg = beg + img.size(0)
            label = label.numpy().reshape(1, -1, 1).repeat(self.num_bs, axis=0)
            output = self.G(img.cuda(), self.num_bs).cpu().numpy()
            outputs[:, index] = np.concatenate([output, label], axis=2)

        pred = outputs.sum(0)[:, :-1].argmax(1)
        label = outputs[0][:, -1]
        acc = pred == label
        ece = calc_ece(softmax(outputs[:, :, :-1], -1).mean(0), label)
        nll, brier = calc_nll_brier(softmax(outputs[:, :, :-1], -1).mean(0),
                                    outputs[:, :, :-1].mean(0),
                                    label,
                                    one_hot(torch.from_numpy(label.astype(int)),
                                            self.num_classes).numpy())
        print(f"[Test] acc : {acc.mean()}")
        self.logger.write(f"[Test] acc : {acc.mean()}, ece : {ece}, nll : {nll}, brier : {brier}")
        np.save(f'{self.save_path}/output.npy', outputs)

import torch
from torch.distributions.exponential import Exponential

import numpy as np
from random import sample
from math import ceil
from itertools import cycle

from runner.cnn_runner import CnnClsfier


class GbsCnnClsfier(CnnClsfier):
    def __init__(self, args, loader, model, optim, lr_schdlr, loss):
        self.n_a = loader.n_a
        self.V = args.v
        self.sub_size = loader.sub_size
        self.n_test = loader.n_test
        self.n_b = loader.n_b
        self.nsub = int(self.sub_size * self.n_b)
        self.a_sample = Exponential(torch.ones([1, self.V]))
        self.a_test = Exponential(torch.ones([1, self.n_a]))
        self.A = torch.eye(self.sub_size).repeat_interleave(self.n_b, 0).t()
        self.alpha = torch.ones([self.nsub, self.n_a])
        self.k0 = args.k0
        self.fac1 = args.fac1
        self.num_bs = args.num_bs
        self.is_gbs = args.is_gbs
        self.num_classes = args.num_classes
        self.schdlr_type = args.scheduler
        self.model_type = args.model
        self.beg_train = 0
        super().__init__(args, loader, model, optim, lr_schdlr, loss)
        self.save_kwargs['alpha'] = self.alpha

    def _get_weight(self, batch):
        idx_sampled = sample(range(self.n_a), self.sub_size)
        ind_a = sample(range(self.nsub), ceil(batch / self.nsub * 2 * self.V))
        for k in range(ceil(batch / self.nsub * 2 * self.V)):
            ind_b = sample(range(self.n_a), self.V)
            self.alpha[ind_a[k], ind_b] = self.a_sample.sample()

        w1 = self.alpha[:, idx_sampled] @ self.A
        return w1.t().cuda()

    def _get_indices(self, start, end):
        indices = []
        for _, idx in enumerate(cycle(range(self.nsub))):
            if start <= _ < end:
                indices += [idx]
            if _ >= end:
                break
        return indices

    def _calc_loss(self, img, label):
        self.G.train()
        batch = img.size(0)
        end = self.beg_train + batch
        indices = self._get_indices(self.beg_train, end)
        self.beg_train = end % self.nsub

        if self.is_gbs:
            w1 = self._get_weight(batch)[indices]
        else:
            w1 = None

        output = self.G(img, self.alpha[indices], self.fac1)
        loss = self.loss(output, label.cuda(), w1)
        return loss

    @torch.no_grad()
    def _valid_a_batch(self, img, label):
        self.G.eval()
        w_test = torch.ones([img.size(0), self.n_a]).cuda()
        output = self._infer_a_batch(img, w_test, self.fac1)
        pred = output.argmax(1).cpu()
        return (pred == label).numpy()

    @torch.no_grad()
    def test(self):
        self.G.eval()
        self.load('best.pth')
        a_test = self.a_test.sample((self.num_bs,))
        loader = self.loader.load('test')
        acc = 0.
        beg = 0
        outputs = np.zeros([self.num_bs, self.n_test, self.num_classes + 1])
        for i, (img, label) in enumerate(loader):
            index = list(range(beg, beg + img.size(0)))
            beg = beg + img.size(0)
            label = label.numpy().reshape(-1, 1)
            for _ in range(self.num_bs):
                w_test = a_test[_].repeat_interleave(img.size(0), dim=0)
                if self.model_type == 'mlp':
                    img = img.view(img.shape[0], -1)
                output = self.G(img, w_test, self.fac1).cpu().numpy()
                outputs[_, index] = np.concatenate([output, label], axis=1)

        pred = outputs.sum(0)[:, :-1].argmax(1)
        label = outputs[0][:, -1]
        acc = pred == label
        self.logger.write(f"[Test] acc : {acc.mean()}")
        print(f"[Test] acc : {acc.mean()}")

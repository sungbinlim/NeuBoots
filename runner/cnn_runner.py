import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.optim import lr_scheduler, Adam
from torch.distributions.exponential import Exponential
import numpy as np
from pathlib import Path
import shutil
import math
from tqdm import tqdm
from random import sample
from runner.base import BaseRunner


class GbsCnnClsfier(BaseRunner):
    def __init__(self, loader, save_path, num_epoch, model, optim, lr_schdlr, loss, k0):
        super().__init__(loader, save_path, num_epoch, model, optim, lr_schdlr)
        n_train = loader.n_train
        n_a = loader.n_a
        sub_size = loader.sub_size
        self.n_test = loader.n_test
        self.n_b = loader.n_b
        self.loss = loss
        self.a_sample = Exponential(torch.ones([1, sub_size]))
        self.A = torch.eye(sub_size).repeat_interleave(self.n_b, dim=0)
        self.A_total = torch.ones([n_a, 1])
        self.alpha = torch.ones([1, n_a])
        self.k0 = k0
        self.a0 = 30.0 / math.sqrt(n_train)
        self.inc1 = 10.0 / math.sqrt(n_train * loader.p)
        self.fac1 = 1.0
        self.w_test = torch.ones([self.n_test, n_a]).cuda()

    def _get_weights(self, index):
        idx_sampled = (index // self.n_b).unique()
        self.alpha[:, idx_sampled] = self.a_sample.sample()
        w0 = self.A_total @ self.alpha
        w1 = self.A @ self.alpha[:, idx_sampled].t()
        return w0.cuda(), w1.cuda()

    def train(self):
        print("Start to train")
        losses = []
        for epoch in range(self.epoch, self.num_epoch):
            self.G.train()
            loss = 0
            for m in range(self.k0):
                loader = self.loader.load('train')
                for i, (img, label, index) in enumerate(loader):
                    w0, w1 = self._get_weights(index)
                    label = F.one_hot(label, 10).cuda()
                    output = self.G(img, w0, self.a0, self.fac1)
                    _loss = self.loss(output, label, w1) / self.k0
                    loss += _loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses += [loss.item()]
            print(f"Train {epoch} loss : {loss.item()}")

            if epoch % 50 == 20:
                self.fac1 += self.inc1

            if epoch % 100 == 0:
                self.val(epoch)

    def val(self, epoch):
        self.G.eval()
        with torch.no_grad():
            loader = self.loader.load('val')
            acc = 0.
            for i, (img, label, index) in enumerate(loader):
                index = index - 50000
                output = self.G(img, self.w_test[index], self.a0, self.fac1)
                pred = output.argmax(1).cpu()
                _acc = (pred == label).sum().float()
                acc += _acc
            acc /= self.n_test
            self.save(epoch, acc)
            self.lr_schler.step(acc)
            print(f"Val {epoch} acc : {acc}")

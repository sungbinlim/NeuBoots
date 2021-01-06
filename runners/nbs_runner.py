import torch
from torch.nn.functional import one_hot
from torch.distributions.exponential import Exponential

import h5py
import numpy as np
from tqdm import tqdm

from utils.metrics import calc_ece, calc_nll_brier
from runners.base_runner import gather_tensor
from runners.cnn_runner import CnnRunner


class NbsRunner(CnnRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch,
                 loss_with_weight, val_metric, test_metric, logger,
                 model_path, rank, epoch_th, num_mc):
        self.num_mc = num_mc
        self.n_a = loader.n_a
        self.epoch_th = epoch_th
        self.group_indices = loader.groups
        self.alpha = torch.ones([1, self.n_a])
        super().__init__(loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                         val_metric, test_metric, logger, model_path, rank)
        self.save_kwargs['alpha'] = self.alpha

    def _update_weight(self):
        if self.epoch > self.epoch_th:
            self.alpha = Exponential(torch.ones([1, self.n_a])).sample()

    def _calc_loss(self, img, label, idx):
        n0 = img.size(0)
        u_is = []
        for i in idx:
            u_i = np.where(self.group_indices == i.item())[0][0]
            u_is += [u_i]

        w = self.alpha[0, u_is].cuda()

        output = self.model(img.cuda(non_blocking=True),
                            self.alpha.repeat_interleave(n0, 0))
        label = label.cuda(non_blocking=True)
        loss_ = 0
        for loss, w in self.loss_with_weight:
            _loss = w * loss(output, label, w)
            loss_ += _loss
        return loss_

    @torch.no_grad()
    def _valid_a_batch(self, img, label, _, with_output=False):
        self._update_weight()
        self.model.eval()
        output = self.model(img.cuda(non_blocking=True), self.num_mc)
        label = label.cuda(non_blocking=True)
        result = self.val_metric(output.mean(0), label)
        if with_output:
            result = [result, output]
        return result

    def test(self):
        self.load('model.pth')
        loader = self.loader.load('test')
        if self.rank == 0:
            t_iter = tqdm(loader, total=self.loader.len)
        else:
            t_iter = loader

        outputs = []
        labels = []
        self.model.eval()
        for img, label, index in t_iter:
            _, output = self._valid_a_batch(img, label, with_output=True)
            outputs += [gather_tensor(output).cpu().numpy()]
            labels += [gather_tensor(label).cpu().numpy()]
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs, axis=1)
        acc = (outputs.mean(0).argmax(-1) == labels).mean() * 100
        ece = calc_ece(outputs.mean(0), labels)
        nll, brier = calc_nll_brier(outputs.mean(0), labels, one_hot(torch.from_numpy(labels.astype(int)),
                                    self.model.module.classifer.out_features).numpy())
        log = f"[Test] ACC: {acc:.2f}, ECE : {ece:.2f}, "
        log += f"NLL : {nll:.2f}, Brier : {brier:.2f}"
        self.log(log, 'info')
        with h5py.File(f"{self.model_path}/output.h5", 'w') as h:
            h.create_dataset('output', data=outputs)
            h.create_dataset('label', data=labels)

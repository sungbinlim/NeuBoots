import torch
from torch.nn.functional import one_hot
from torch.distributions.exponential import Exponential
from scipy.special import softmax

import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.metrics import calc_ece, calc_nll_brier, calc_nll_brier_mc
from runners.base_runner import gather_tensor
from runners.cnn_runner import CnnRunner


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()


class McdRunner(CnnRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch,
                 loss_with_weight, val_metric, test_metric, logger,
                 model_path, rank, num_mc, adv_training):
        self.num_mc = num_mc
        super().__init__(loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                         val_metric, test_metric, logger, model_path, rank, adv_training)

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        self.model.eval()
        self.model.apply(apply_dropout)
        if label.dim() == 3:
            output = torch.zeros([self.num_mc, img.size(0), self.num_classes,
                                img.size(-2), img.size(-1)]).cuda()
        else:
            output = torch.zeros([self.num_mc, img.size(0), self.num_classes]).cuda()
        for i in range(self.num_mc):
            output[i] += self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        result = self.val_metric(output.mean(0), label)
        if with_output:
            result = [result, output]
        return result

    def test(self, is_seg):
        self.load('model.pth')
        loader = self.loader.load('test')
        if self.rank == 0:
            t_iter = tqdm(loader, total=self.loader.len)
        else:
            t_iter = loader

        outputs = []
        labels = []
        metrics = []
        self.model.eval()
        self.model.apply(apply_dropout)
        for img, label in t_iter:
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            outputs += [gather_tensor(output).cpu().numpy()]
            labels += [gather_tensor(label).cpu().numpy()]
            metrics += [gather_tensor(_metric).cpu().numpy()]
        if is_seg:
            met = np.concatenate(metrics).mean()
            self.log(f"[Test] MeanIOU: {met:.2f}", 'info')
            save_path = Path(self.model_path) / 'infer'
            save_path.mkdir(parents=True, exist_ok=True)
            index = 0
            for out, label in zip(outputs, labels):
                for i in range(label.shape[0]):
                    l = label[i]
                    o = out[:, i]

                    with h5py.File(f"{save_path}/{index}.h5", 'w') as h:
                        h.create_dataset('output', data=o)
                        h.create_dataset('label', data=l)
                    index += 1
        else:
            labels = np.concatenate(labels)
            outputs = np.concatenate(outputs, axis=1)
            acc = (outputs.mean(0).argmax(-1) == labels).mean() * 100
            ece = calc_ece(softmax(outputs, -1).mean(0), labels)
            nll, brier = calc_nll_brier_mc(outputs, labels, self.num_classes)
            log = f"[Test] ACC: {acc:.2f}, ECE : {ece:.2f}, "
            log += f"NLL : {nll:.2f}, Brier : {brier:.2f}"
            self.log(log, 'info')
            with h5py.File(f"{self.model_path}/output.h5", 'w') as h:
                h.create_dataset('output', data=outputs)
                h.create_dataset('label', data=labels)

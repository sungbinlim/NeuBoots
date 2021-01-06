import torch
from torch.nn.functional import one_hot

import h5py
import shutil
import numpy as np
from tqdm import tqdm

from utils.metrics import calc_ece, calc_nll_brier, VATLoss
from runners.base_runner import BaseRunner, reduce_tensor, gather_tensor


class CnnRunner(BaseRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                 val_metric, test_metric, logger, model_path, rank, adv_training):
        self.num_epoch = num_epoch
        self.epoch = 0
        self.loss_with_weight = loss_with_weight
        self.adv_training = adv_training
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.best_score = 0.
        self.save_kwargs = {}
        self.world_size = torch.distributed.get_world_size()
        super().__init__(loader, model, logger, model_path, rank)
        self.load()

    def _calc_loss(self, img, label):
        self.model.train()
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        loss_ = 0
        for loss, w in self.loss_with_weight:
            _loss = w * loss(output, label)
            loss_ += _loss
        return loss_

    def fgsm(self, img, label):
        step_size = 0.01
        loss_fn = torch.nn.CrossEntropyLoss()
        img = img.cuda()
        img.requires_grad = True
        self.model.eval()
        self.model.zero_grad()
        output = self.model(img)
        loss = loss_fn(output, label.cuda())
        loss.backward()
        grad_sign = img.grad.sign()
        img_new = img + step_size * grad_sign
        return img_new.cpu().detach()

    def _train_a_batch(self, batch):
        with torch.autograd.set_detect_anomaly(True):
            loss = self._calc_loss(*batch)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.adv_training:
                img_new = self.fgsm(*batch)
                loss = self._calc_loss(img_new, *batch[1:])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            _loss = reduce_tensor(loss, True).item()
            return _loss

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        result = self.val_metric(output, label)
        if with_output:
            result = [result, output]
        return result

    def train(self):
        self.log("Start to train", 'debug')
        for epoch in range(self.epoch, self.num_epoch):
            self.model.train()
            loader = self.loader.load("train")
            if self.rank == 0:
                t_iter = tqdm(loader, total=self.loader.len,
                              desc=f"[Train {epoch}]")
            else:
                t_iter = loader
            losses = 0
            for i, batch in enumerate(t_iter):
                loss = self._train_a_batch(batch)
                losses += loss
                if self.rank == 0:
                    t_iter.set_postfix(loss=f"{loss:.4} / {losses/(i+1):.4}")

            self.log(f"[Train] epoch:{epoch} loss:{losses/(i+1)}", 'info')
            self.lr_scheduler.step()
            self.val(epoch)

    def val(self, epoch):
        loader = self.loader.load('val')
        v_iter = loader

        metrics = []
        self.model.eval()
        for batch in v_iter:
            _metric = self._valid_a_batch(*batch, with_output=False)
            metrics += [gather_tensor(_metric).cpu().numpy()]
        acc = np.concatenate(metrics).mean()
        self.log(f"[Val] {epoch} Score: {acc}", 'info')
        if self.rank == 0:
            self.save(epoch, acc, **self.save_kwargs)

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
        for img, label in t_iter:
            _, output = self._valid_a_batch(img, label, with_output=True)
            outputs += [gather_tensor(output).cpu().numpy()]
            labels += [gather_tensor(label).cpu().numpy()]
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs, axis=0)
        acc = (outputs.argmax(1) == labels).mean() * 100
        ece = calc_ece(outputs, labels)
        nll, brier = calc_nll_brier(outputs, labels, one_hot(torch.from_numpy(labels.astype(int)),
                                    self.model.module.fc.out_features).numpy())
        log = f"[Test] ACC: {acc:.2f}, ECE : {ece:.2f}, "
        log += f"NLL : {nll:.2f}, Brier : {brier:.2f}"
        self.log(log, 'info')
        with h5py.File(f"{self.model_path}/output.h5", 'w') as h:
            h.create_dataset('output', data=outputs)
            h.create_dataset('label', data=labels)

    def save(self, epoch, metric, file_name="model", **kwargs):
        torch.save({"epoch": epoch,
                    "param": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    "best": self.best_score,
                    "lr_schdlr": self.lr_scheduler.state_dict(),
                    **kwargs}, f"{self.model_path}/{file_name}.pth")

        cond = metric >= self.best_score
        if cond:
            self.log(f"{self.best_score} -------------------> {metric}", 'debug')
            self.best_score = metric
            shutil.copy2(f"{self.model_path}/{file_name}.pth",
                         f"{self.model_path}/best.pth")
            self.log(f"Model has saved at {epoch} epoch.", 'debug')

    def load(self, file_name="model.pth"):
        self.log(self.model_path, 'debug')
        if (self.model_path / file_name).exists():
            self.log(f"Loading {self.model_path} File", 'debug')
            ckpoint = torch.load(f"{self.model_path}/{file_name}", map_location='cpu')

            for key, value in ckpoint.items():
                if key == 'param':
                    self.model.load_state_dict(value)
                elif key == 'optimizer':
                    self.optim.load_state_dict(value)
                elif key == 'lr_schdlr':
                    self.lr_scheduler.load_state_dict(value)
                elif key == 'epoch':
                    self.epoch = value + 1
                elif key == 'best':
                    self.best_score = value
                else:
                    self.__dict__[key] = value

            self.log(f"Model Type : {file_name}, epoch : {self.epoch}", 'debug')
        else:
            self.log("Failed to load, not existing file", 'debug')

    def get_lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]['lr']

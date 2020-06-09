import torch

import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

from runner.base_runner import BaseRunner


class CnnClsfier(BaseRunner):
    def __init__(self, args, loader, model, optim, lr_schdlr, loss):
        self.num_epoch = args.num_epoch
        self.epoch = 0
        self.loss = loss
        self.optim = optim
        self.lr_schdlr = lr_schdlr
        self.best_metric = 0
        self.save_kwargs = {}
        super().__init__(args, loader, model)

    def _infer_a_batch(self, img):
        return self.G(img)

    def _calc_loss(self, img, label):
        output = self._infer_a_batch(img)
        loss = self.loss(output, label.cuda())
        return loss

    def _train_a_batch(self, img, label):
        self.G.train()
        loss = self._calc_loss(img, label)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_schdlr.step()
        return loss.item()

    @torch.no_grad()
    def _valid_a_batch(self, img, label):
        self.G.eval()
        output = self._infer_a_batch(img)
        pred = output.argmax(1).cpu()
        return (pred == label).numpy()

    def train(self):
        print("Start to train")
        for epoch in range(self.epoch, self.num_epoch):
            self.G.train()
            loader = self.loader.load("train")
            t_iter = tqdm(loader, total=self.loader.len,
                          desc=f"[Train {epoch}]")
            losses = 0
            for i, (img, label) in enumerate(t_iter):
                loss = self._train_a_batch(img, label)
                losses += loss
                t_iter.set_postfix(loss=f"{loss:.4} / {losses/(i+1):.4}")

            self.logger.write(f"[Train] epoch:{epoch} loss:{losses/i}")
            self.val(epoch)

    def val(self, epoch):
        loader = self.loader.load('val')
        acc = []
        for i, (img, label) in enumerate(loader):
            _acc = self._valid_a_batch(img, label)
            acc += [_acc]
        acc = np.concatenate(acc).mean()
        self.save(epoch, acc, **self.save_kwargs)
        self.logger.write(f"[Val] {epoch} acc : {acc}")

    def test(self):
        self.load('best.pth')
        loader = self.loader.load('test')
        acc = []
        for i, (img, label) in enumerate(loader):
            _acc = self._valid_a_batch(img, label)
            acc += [_acc]
        acc = np.concatenate(acc).mean()
        self.logger.write(f"[Test] acc : {acc}")

    def save(self, epoch, metric, file_name="model", **kwargs):
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({"epoch": epoch,
                    "param": self.G.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    "best": self.best_metric,
                    "lr_schdlr": self.lr_schdlr.state_dict(),
                    **kwargs}, f"{save_path}/{file_name}.pth")

        if metric >= self.best_metric:
            print(f"{self.best_metric} -------------------> {metric}")
            self.best_metric = metric
            shutil.copy2(f"{save_path}/{file_name}.pth",
                         f"{save_path}/best.pth")
            print(f"Model has saved at {epoch} epoch.")

    def load(self, file_name="model.pth"):
        save_path = Path(self.save_path)
        print(save_path)
        if (save_path / file_name).exists():
            print(f"Load {save_path} File")
            ckpoint = torch.load(f"{save_path}/{file_name}")

            for key, value in ckpoint.items():
                if key == 'param':
                    self.G.load_state_dict(value)
                elif key == 'optimizer':
                    self.optim.load_state_dict(value)
                elif key == 'lr_schdlr':
                    self.lr_schdlr.load_state_dict(value)
                elif key == 'epoch':
                    self.epoch = value
                elif key == 'score':
                    self.best_metric = value
                else:
                    self.__dict__[key] = value

            print(f"Load Model Type : {file_name}, epoch : {self.epoch}")
        else:
            print("Load Failed, not exists file")

    def get_lr(self):
        return self.lr_schdlr.optimizer.param_groups[0]['lr']

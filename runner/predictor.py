import torch
from torch.distributions.exponential import Exponential

import numpy as np
from pathlib import Path
from scipy.special import softmax

from runner.base_runner import BaseRunner


class Predictor(BaseRunner):
    def __init__(self, args, loader, model):
        super().__init__(args, loader, model)
        self.args = args

    @torch.no_grad()
    def _infer_a_batch(self, img):
        self.G.eval()
        output = self.G(img).cpu().numpy()
        return output

    @torch.no_grad()
    def _infer_a_batch_a_bs(self, img, w):
        self.G.eval()
        output = self.G(img, w, self.args.fac1)
        return output.cpu().numpy()

    def _infer_a_batch_odin(self, img, temp=1000, eps=0.0014, w=None):
        self.G.eval()
        criteria = torch.nn.CrossEntropyLoss()
        img_ = img.cuda()
        img_.requires_grad = True
        if w is not None:
            output = self.G(img_, w, self.args.fac1)
        else:
            output = self.G(img_)

        output = output / temp
        pseudo_label = output.argmax(-1).cuda()
        loss = criteria(output, pseudo_label)
        loss.backward()

        gradient = torch.ge(img_.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.tensor([0]).cuda(),
                             gradient.index_select(1, torch.tensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.tensor([1]).cuda(),
                             gradient.index_select(1, torch.tensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.tensor([2]).cuda(),
                             gradient.index_select(1, torch.tensor([2]).cuda()) / (0.2010))

        img_new = torch.add(img_.data, -eps, gradient)
        if w is not None:
            output_new = self.G(img_new, w, self.args.fac1)
        else:
            output_new = self.G(img_new)
        return output_new.cpu().detach().numpy()

    def _infer(self, with_acc=False):
        outputs = np.zeros([len(self.loader.dataset), self.args.num_classes + 1])
        beg = 0
        for i, (img, label) in enumerate(self.loader):
            end = beg + img.size(0)
            output = self._infer_a_batch(img)
            label = label.numpy().reshape(-1, 1)
            outputs[beg: end] = np.concatenate([output, label], axis=1)
            beg = end

        if with_acc:
            pred = outputs.argmax(1)
            label = outputs[:, -1]
            acc = (pred == label).mean()
            print(f"[Test] acc : {acc}")
        return outputs

    def _infer_gbs(self, with_acc=False, seed=0):
        torch.manual_seed(seed)
        a_test_ = Exponential(torch.ones([1, self.args.n_a]))
        a_test = a_test_.sample([self.args.num_bs, ])
        outputs = np.zeros([self.args.num_bs, len(self.loader.dataset), self.args.num_classes + 1])
        beg = 0
        for i, (img, label) in enumerate(self.loader):
            end = beg + img.size(0)
            label = label.numpy().reshape(-1, 1)
            for _ in range(self.args.num_bs):
                w_test = a_test[_].repeat_interleave(img.size(0), dim=0)
                output = self._infer_a_batch_a_bs(img, w_test)
                outputs[_, beg: end] = np.concatenate([output, label], axis=1)
            beg = end

        if with_acc:
            pred = outputs.sum(0)[:, :-1].argmax(1)
            label = outputs[0][:, -1]
            acc = (pred == label).mean()
            print(f"[Test] acc : {acc}")
        return outputs

    def _infer_odin(self, with_acc=False, is_gbs=False):
        outputs = np.zeros([len(self.loader.dataset), self.args.num_classes + 1])
        beg = 0
        for i, (img, label) in enumerate(self.loader):
            end = beg + img.size(0)
            output = self._infer_a_batch_odin(img, self.args.temp, self.args.eps)
            label = label.numpy().reshape(-1, 1)
            outputs[beg: end] = np.concatenate([output, label], axis=1)
            beg = end

        if with_acc:
            pred = outputs.argmax(1)
            label = outputs[:, -1]
            acc = (pred == label).mean()
            print(f"[Test] acc : {acc}")
        return outputs

    def _infer_gbs_odin(self, with_acc=False, seed=0):
        torch.manual_seed(seed)
        a_test_ = Exponential(torch.ones([1, self.args.n_a]))
        a_test = a_test_.sample([self.args.num_bs, ])
        outputs = np.zeros([self.args.num_bs, len(self.loader.dataset), self.args.num_classes + 1])
        beg = 0
        for i, (img, label) in enumerate(self.loader):
            end = beg + img.size(0)
            label = label.numpy().reshape(-1, 1)
            for _ in range(self.args.num_bs):
                w_test = a_test[_].repeat_interleave(img.size(0), dim=0)
                output = self._infer_a_batch_odin(img, self.args.temp, self.args.eps, w_test)
                outputs[_, beg: end] = np.concatenate([output, label], axis=1)
            beg = end

        if with_acc:
            pred = outputs.sum(0)[:, :-1].argmax(1)
            label = outputs[0][:, -1]
            acc = (pred == label).mean()
            print(f"[Test] acc : {acc}")
        return outputs

    def infer(self, is_gbs, is_odin, with_acc=False, seed=0):
        if is_gbs:
            if is_odin:
                self.output = self._infer_gbs_odin(with_acc, seed)
            else:
                self.output = self._infer_gbs(with_acc, seed)
        else:
            if is_odin:
                self.output = self._infer_odin(with_acc)
            else:
                self.output = self._infer(with_acc)
        return self.output

    @staticmethod
    def predictive_mean(x, temp):
        x = softmax(x / temp, axis=-1)
        if x.ndim == 3:
            x = x.mean(0)
        return x

    def load(self):
        file_name = "model.pth"
        save_path = Path(self.save_path)
        print(save_path)
        if (save_path / file_name).exists():
            print(f"Load {save_path} File")
            ckpoint = torch.load(f"{save_path}/{file_name}")
            self.G.load_state_dict(ckpoint['param'])
            print(f"Load Model Type : {file_name}")
        else:
            print("Load Failed, not exists file")

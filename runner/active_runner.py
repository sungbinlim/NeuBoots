import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions.exponential import Exponential
from torch.utils.data.sampler import SubsetRandomSampler

from utils.jupyter import infer, predictive_std
from models.gbsnet import D
from models import _get_model
from data.data_loader import Dataset
from utils.preprocessing import get_transform


class ActiveRunner(object):
    def __init__(self, model_type, num_groups, num_query, num_epoch):
        self.model_type = model_type
        self.num_groups = num_groups
        self.num_query = num_query
        self.num_epoch = num_epoch

        self.dataset = Dataset(
            CIFAR10(root='.cifar10', train=True, download=True,
                    transform=get_transform(32, 4, 16)['train'])
        )
        self.testset = Dataset(
            CIFAR10(root='.cifar10', train=False, download=True,
                    transform=get_transform(32, 4, 20)['test'])
        )
        self.test_loader = DataLoader(self.testset, batch_size=1024,
                                      num_workers=4, pin_memory=True)
        self.indice = list(range(len(self.dataset)))
        random.Random(0).shuffle(self.indice)

    def _init_for_training(self):
        torch.cuda.empty_cache()
        for module in ('model', 'optim', 'sched', 'loss'):
            if hasattr(self, module):
                del self.__dict__[module]
        self.model = _get_model(self.model_type,
                                self.num_groups,
                                10, True, 0.0).cuda()
        self.optim = SGD(self.model.parameters(),
                         lr=0.1, weight_decay=0.0005,
                         nesterov=True, momentum=0.9)
        self.sched = CosineAnnealingLR(self.optim, self.num_epoch, 0)
        self.loss_fn = D

    def train_coreset(self):
        indice_coreset = np.array(self.indice[:self.num_query])
        self._init_for_training()
        self._train_a_query(indice_coreset)
        test_res, acc = infer(self.test_loader, self.model,
                              100, 10, True, False)
        self._save(acc, 'coreset')
        self.trained_indice = indice_coreset.tolist()

    def train_next_query(self, save_name):
        next_indice = self._sample_next_query_using_uncertainty()
        indices = np.concatenate([self.trained_indice, next_indice])
        self._init_for_training()
        self._train_a_query(indices)
        test_res, acc = infer(self.test_loader, self.model,
                              100, 10, True, False)
        self._save(acc, save_name)
        self.trained_indice = indices.tolist()

    def _sample_next_query_using_uncertainty(self):
        target_indice = list(set(self.indice) - set(self.trained_indice))
        sampler = SubsetRandomSampler(target_indice)
        loader = DataLoader(self.dataset, batch_size=2048, sampler=sampler,
                            num_workers=4, pin_memory=True)
        target_res, indices = infer(loader, self.model, 100, 10, False, True)
        uncertainties = predictive_std(target_res[..., :-1])
        pairs = sorted(
            [[uncertainties[i], indices[i]] for i in range(len(target_indice))],
            key=lambda x: x[0])
        return np.array(pairs)[:self.num_query, 1]

    def _train_a_query(self, indice):
        indexer = np.stack(
            [np.array(indice[i::self.num_groups])
             for i in range(self.num_groups)])
        sampler = SubsetRandomSampler(indice)
        loader = DataLoader(self.dataset, batch_size=128, sampler=sampler,
                            num_workers=4, pin_memory=True)
        alpha_generator = Exponential(torch.ones([1, self.num_groups]))
        t_iter = tqdm(range(self.num_epoch),
                      total=self.num_epoch,
                      desc="Training")
        for t in t_iter:
            alpha = alpha_generator.sample().cuda()
            for img, label, index in loader:
                self.model.train()
                n0 = img.size(0)
                u_is = []
                for i in index:
                    u_i = np.where(indexer == i.item())[0][0]
                    u_is += [u_i]
                w = alpha[0, u_is].cuda()
                output = self.model(img.cuda(), alpha.repeat_interleave(n0, 0))
                loss = self.loss_fn(output, label.cuda(), w)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            self.sched.step()

    def _save(self, acc, file_name):
        save_path = Path('active')
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({'param': self.model.state_dict(),
                    'score': acc}, f"{save_path}/{file_name}.pth")

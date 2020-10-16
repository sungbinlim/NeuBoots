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

from utils.jupyter import *
from models.nbsnet import D
from models import _get_model
from data.data_loader import Dataset
from utils.preprocessing import get_transform


class ActiveRunner(object):
    def __init__(self, data_type, model_type, num_groups, num_query,
                 num_epoch, sampling_type='nbs', save_name='0'):
        self.data_type = data_type
        self.model_type = model_type
        self.num_groups = num_groups
        self.num_query = num_query
        self.num_epoch = num_epoch
        self.sampling_type = sampling_type
        self.save_name = save_name

        if data_type == 'cifar10':
            self.dataset = Dataset(
                CIFAR10(root='.cifar10', train=True, download=True,
                        transform=get_transform(32, 4, 16)['train'])
            )
            self.testset = Dataset(
                CIFAR10(root='.cifar10', train=False, download=True,
                        transform=get_transform(32, 4, 16)['test'])
            )
        else:
            self.dataset = Dataset(
                CIFAR100(root='.cifar100', train=True, download=True,
                         transform=get_transform(32, 4, 8)['train'])
            )
            self.testset = Dataset(
                CIFAR100(root='.cifar100', train=False, download=True,
                         transform=get_transform(32, 4, 8)['test'])
            )
        self.test_loader = DataLoader(self.testset, batch_size=6144,
                                      num_workers=4, pin_memory=True)
        self.indice = list(range(len(self.dataset)))
        random.Random(0).shuffle(self.indice)

    def _init_for_training(self):
        torch.cuda.empty_cache()
        for module in ('model', 'optim', 'sched', 'loss'):
            if hasattr(self, module):
                del self.__dict__[module]
        
        is_nbs = False
        drop_rate = 0.0
        is_fa = False

        if self.sampling_type in ['nbs', 'nbs_fa']:
            is_nbs = True

        if self.sampling_type == 'nbs_fa':
            is_fa = True
        
        if self.sampling_type == 'mcd':
            drop_rate = 0.2
        else:
            drop_rate = 0.0

        if self.data_type == 'cifar10':
            self.num_classes = 10
        else:
            self.num_classes = 100
        self.model = _get_model(self.model_type,
                                self.num_groups,
                                self.num_classes,
                                is_nbs, drop_rate,
                                is_fa).cuda()
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
                              100, self.num_classes, True, False)
        self._save(acc, 'coreset')
        self.trained_indice = indice_coreset.tolist()

    def train_next_query(self, save_name):
        if self.sampling_type != 'random':
            next_indice = self._sample_next_query_using_uncertainty()
        else:
            next_indice = self._sample_next_query_randomly()
        indices = np.concatenate([self.trained_indice, next_indice])
        self._init_for_training()
        self._train_a_query(indices)
        if self.sampling_type == 'mcd':
            is_mcd = True
        else:
            is_mcd = False
        test_res, acc = infer(self.test_loader, self.model,
                              100, self.num_classes, True, False, is_mcd)
        self._save(acc, save_name)
        self.trained_indice = indices.tolist()

    def _sample_next_query_using_uncertainty(self):
        target_indice = list(set(self.indice) - set(self.trained_indice))
        sampler = SubsetRandomSampler(target_indice)
        loader = DataLoader(self.dataset, batch_size=6144, sampler=sampler,
                            num_workers=4, pin_memory=True)
        if self.sampling_type == 'mcd':
            is_mcd = True
        else:
            is_mcd = False
        target_res, indices = infer(loader, self.model, 100, self.num_classes, False, True, is_mcd)
        uncertainties = predictive_std(target_res[..., :-1])
        pairs = sorted(
            [[uncertainties[i], indices[i]] for i in range(len(target_indice))],
            key=lambda x: x[0], reverse=True)
        return np.array(pairs)[:self.num_query, 1].astype(int)

    def _sample_next_query_randomly(self):
        beg = len(self.trained_indice)
        target_indice = self.indice[beg: beg + self.num_query]
        return target_indice

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
        save_path = Path(
            f'active/{self.data_type}/{self.save_name}/{self.sampling_type}')
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({'param': self.model.state_dict(),
                    'score': acc}, f"{save_path}/{file_name}.pth")

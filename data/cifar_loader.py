from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler

from math import ceil
from sklearn.model_selection import StratifiedShuffleSplit

from data.block_sampler import BlockSampler, BlockSubsetSampler
from utils.preprocessing import transform_train, transform_test


class CifarLoader(object):
    def __init__(self, cifar_type, batch_size, n_a, sub_size, cpus, seed=0):
        self.sub_size = sub_size
        self.n_a = n_a
        if cifar_type == 'cifar10':
            self.trainset = CIFAR10(root='.cifar10', train=True, download=True,
                                    transform=transform_train)
            self.testset = CIFAR10(root='.cifar10', train=False, download=True,
                                   transform=transform_test)
        elif cifar_type == 'cifar100':
            self.trainset = CIFAR100(root='.cifar100', train=True,
                                     download=True, transform=transform_train)
            self.testset = CIFAR100(root='.cifar100', train=False,
                                    download=True, transform=transform_test)
        self.p = self.trainset[0][0].nelement()
        train_targets = [label for img, label in self.trainset]
        splitter = StratifiedShuffleSplit(1, test_size=0.2, random_state=0)
        indices = range(len(self.trainset))
        self.indices = next(splitter.split(indices, train_targets))

        self.n_train = len(self.indices[0])
        self.n_val = len(self.indices[1])
        self.n_test = len(self.testset)
        self.n_b = self.n_train // n_a
        self.batch_size = batch_size
        self.cpus = cpus

    def load(self, dataset):
        _f = {'train': lambda: self._train(),
              'val': lambda: self._val(),
              'test': lambda: self._test()}
        try:  # lambda for lazy evaluation
            loader = _f[dataset]()
            return loader
        except:
            raise ValueError('Dataset should be one of [train, val, test]')

    def _train(self):
        self.len = ceil(self.n_train / self.batch_size)
        # sampler = BlockSampler(self.indices[:50000], self.n_a)
        sampler = SubsetRandomSampler(self.indices[0])
        loader = DataLoader(self.trainset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _val(self):
        sampler = SubsetRandomSampler(self.indices[1])
        loader = DataLoader(self.trainset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _test(self):
        loader = DataLoader(self.testset, batch_size=self.batch_size,
                            num_workers=self.cpus, pin_memory=True)
        return loader

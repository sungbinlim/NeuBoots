from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, STL10, VisionDataset

import numpy as np
from math import ceil
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from utils.preprocessing import get_transform, MnistNorm


def _get_split_indices(trainset, p, seed):
    train_targets = [label for img, label, idx in trainset]
    splitter = StratifiedShuffleSplit(1, test_size=p, random_state=seed)
    indices = range(len(trainset))
    return next(splitter.split(indices, train_targets))


def _get_kfolded_indices(valid_indices, trainset, num_k, seed):
    train_targets = [label for img, label, idx in trainset]
    valid_targets = [train_targets[i] for i in valid_indices]
    splitter = StratifiedKFold(num_k, True, seed)
    mask_iter = splitter._iter_test_masks(valid_indices, valid_targets)
    kfolded_indices = [np.array(valid_indices[np.nonzero(m)])
                       for m in mask_iter]
    base_len = len(kfolded_indices[0])
    for i, k in enumerate(kfolded_indices):
        if len(k) < base_len:
            kfolded_indices[i] = np.pad(k, (0, base_len - len(k)),
                                        mode='edge')[None, ...]
        else:
            kfolded_indices[i] = k[None, ...]
    return np.concatenate(kfolded_indices, 0)


class Dataset(VisionDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx

    def __len__(self):
        return len(self.dataset)


class NbsDataLoader(object):
    def __init__(self, dataset, batch_size, n_a, sub_size, cpus, seed=0):
        self.dataset = self._get_dataset(dataset)
        self.split_indices = _get_split_indices(self.dataset['train'],
                                                0.1, seed)
        self.p = self.dataset['train'][0][0].nelement()
        self.sub_size = sub_size
        self.n_a = n_a
        self.n_train = len(self.split_indices[0])
        self.n_val = len(self.split_indices[1])
        self.groups = _get_kfolded_indices(self.split_indices[0],
                                           self.dataset['train'],
                                           n_a, seed)
        self.n_test = len(self.dataset['test'])
        self.n_b = self.n_train // n_a
        self.batch_size = batch_size
        self.cpus = cpus

    def load(self, phase):
        _f = {'train': lambda: self._train(),
              'val': lambda: self._val(),
              'test': lambda: self._test()}
        try:
            loader = _f[phase]()
            return loader
        except KeyError:
            raise ValueError('Dataset should be one of [train, val, test]')

    def _train(self):
        self.len = ceil(self.n_train / self.batch_size)
        sampler = SubsetRandomSampler(self.split_indices[0])
        loader = DataLoader(self.dataset['train'], batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _val(self):
        sampler = SubsetRandomSampler(self.split_indices[1])
        loader = DataLoader(self.dataset['train'], batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _test(self):
        loader = DataLoader(self.dataset['test'], batch_size=self.batch_size,
                            num_workers=self.cpus, pin_memory=True)
        return loader

    def _get_dataset(self, dataset):
        _d = {'cifar10': lambda: self._load_cifar10(),
              'cifar100': lambda: self._load_cifar100(),
              'mnist': lambda: self._load_mnist(),
              'svhn': lambda: self._load_svhn(),
              'svhn_extra': lambda: self._load_svhn(use_extra=True),
              'stl': lambda: self._load_stl()}
        try:
            _dataset = _d[dataset]()
            return _dataset
        except KeyError:
            raise ValueError(
                "Dataset should be one of [mnist, cifar10"
                ", cifar100, svhn, svhn_extra, stl]")

    def _load_mnist(self):
        trainset = Dataset(MNIST(root='.mnist', train=True, download=True,
                           transform=MnistNorm()))
        testset = MNIST(root='.mnist', train=False, download=True,
                        transform=MnistNorm())
        return {'train': trainset, 'test': testset}

    def _load_cifar10(self):
        trainset = Dataset(CIFAR10(root='.cifar10', train=True, download=True,
                           transform=get_transform(32, 4, 16)['train']))
        testset = CIFAR10(root='.cifar10', train=False, download=True,
                          transform=get_transform(32, 4, 16)['test'])
        return {'train': trainset, 'test': testset}

    def _load_cifar100(self):
        trainset = Dataset(CIFAR100(root='.cifar100', train=True, download=True,
                           transform=get_transform(32, 4, 8)['train']))
        testset = CIFAR100(root='.cifar100', train=False, download=True,
                           transform=get_transform(32, 4, 8)['test'])
        return {'train': trainset, 'test': testset}

    def _load_svhn(self, use_extra=False):
        trainset = Dataset(SVHN(root='.svhn', split='train', download=True,
                           transform=get_transform(32, 4, 20)['train']))
        testset = SVHN(root='.svhn', split='test', download=True,
                       transform=get_transform(32, 4, 20)['test'])
        if not use_extra:
            return {'train': trainset, 'test': testset}

        extraset = Dataset(SVHN(root='.svhn', split='extra', download=True,
                           transform=get_transform(32, 4, 20)['train']))
        return {'train': trainset + extraset, 'test': testset}

    def _load_stl(self):
        trainset = Dataset(STL10(root='.stl', split='train', download=True,
                           transform=get_transform(96, 12, 32, True)['train']))
        testset = STL10(root='.stl', split='test', download=True,
                        transform=get_transform(96, 12, 32, True)['test'])
        return {'train': trainset, 'test': testset}

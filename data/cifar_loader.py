from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
from math import ceil
from sklearn.model_selection import StratifiedShuffleSplit


from data.block_sampler import BlockSampler, BlockSubsetSampler


class _CIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


# TODO : should be generalized
class Cifar10Loader(object):
    def __init__(self, batch_size, n_a, sub_size, cpus, seed=0):
        self.sub_size = sub_size
        self.n_a = n_a
        self.trainset = _CIFAR10(root='.cifar10', train=True, download=True,
                                 transform=transforms.ToTensor())
        self.testset = _CIFAR10(root='.cifar10', train=False, download=True,
                                transform=transforms.ToTensor())
        self.p = self.trainset[0][0].nelement()
        train_targets = [label for img, label, idx in self.trainset]
        splitter = StratifiedShuffleSplit(1, 0.2, random_state=0)
        indices = range(len(self.trainset))
        self.indices = next(splitter.split(indices, train_targets))

        self.n_train = len(self.indices[0])
        self.n_val = len(self.indices[1])
        self.n_test = len(self.testset)
        self.n_b = self.n_train // n_a
        self.batch_size = batch_size
        self.cpus = cpus


    def load(self, dataset):
        _f = {'train': self._train(),
              'val': self._val(),
              'test': self._test()}
        try:
            loader = _f[dataset]
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

    def len(self, phase):
        _l = {'train': 50000,
              'val': 10000,
              'test': 10000}
        try:
            return _l[phase]
        except:
            raise ValueError('Phase should be one of [train, val, test]')

import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MnistLoader(object):
    def __init__(self, batch_size, cpus, seed=0):
        self.batch_size = batch_size
        self.cpus = cpus
        self.indices = list(range(60000))
        np.random.seed(seed)
        np.random.shuffle(self.indices)

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
        dataset = MNIST(root='.mnist', train=True, download=True,
                        transform=transforms.ToTensor())
        sampler = SubsetRandomSampler(self.indices[:50000])
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _val(self):
        dataset = MNIST(root='.mnist', train=True, download=True,
                        transform=transforms.ToTensor())
        sampler = SubsetRandomSampler(self.indices[50000:])
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _test(self):
        dataset = MNIST(root='.mnist', train=False, download=True,
                        transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=self.batch_size,
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
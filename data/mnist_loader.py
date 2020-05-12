from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
from math import ceil

from data.block_sampler import BlockSampler, BlockSubsetSampler


class _MNIST(MNIST):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.repeat_interleave(3, dim=0) / 255, target, index


# TODO : should be generalized
class MnistLoader(object):
    def __init__(self, batch_size, n_a, sub_size, cpus, seed=0):
        self.sub_size = sub_size
        self.n_a = n_a
        self.n_train = 50000
        self.n_val = 10000
        self.n_test = 10000
        self.indices = list(range(60000))
        self.n_b = self.n_train // n_a
        self.batch_size = batch_size
        self.cpus = cpus
        self.p = next(iter(self.load('train')))[0][0].nelement()
        # np.random.seed(seed)
        # np.random.shuffle(self.indices)

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
        dataset = _MNIST(root='.mnist', train=True, download=True,
                         transform=transforms.ToTensor())
        self.len = ceil(self.n_train / self.batch_size)
        sampler = BlockSampler(self.indices[:50000], self.n_a)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _val(self):
        dataset = _MNIST(root='.mnist', train=True, download=True,
                         transform=transforms.ToTensor())
        sampler = SubsetRandomSampler(self.indices[50000:])
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _test(self):
        dataset = _MNIST(root='.mnist', train=False, download=True,
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

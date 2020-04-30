import torch
from torch.nn import DataParallel
from torch.optim import lr_scheduler, Adam
import numpy as np
from pathlib import Path
from loader import MnistLoader
from model import ConvNet, D


loader = MnistLoader(128, 8)
print(loader.load('train'))
print(loader.len('train'))

# number of training set
n_train = loader.len('train')
n_test = loader.len('test')
p = next(iter(loader.load('train')))[0][0].nelement()

n_a = 500
n_b = n_train / n_a


class GBS_Runner(object):
    def __init__(self, loader, save_path, num_epoch, lr_init, hidden_size, n_a, p):
        self.loader = loader
        self.num_epoch = num_epoch
        self.G = DataParallel(ConvNet(hidden_size, n_a, p)).cuda()
        self.optim = Adam(self.G.parameters())
        self.lr_schler = lr_scheduler.ReduceLROnPlateau(self.optim, lr=lr_init)
        self.load()
        self.loss = D
        self.save_path = save_path

    def save(self, epoch, metric, filename="modles"):
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exists_ok=True)

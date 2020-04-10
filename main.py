import torch
from torch.nn import DataParallel
from torch.optim import lr_scheduler, Adam
from torch.distributions.exponential import Exponential
import numpy as np
import math
from pathlib import Path
from data.mnist_loader import MnistLoader
import shutil
from tqdm import tqdm
from random import sample
from runner.cnn_runner import GbsCnnClsfier
from model.cnn import ConvNet, D


loader = MnistLoader(500, 500, 8, 5)
p = loader.p
n_a = loader.n_a

hidden_size = p // 2 if p // 2 >= 100 else 100
lr_init = 0.01 / math.sqrt(loader.n_train)
print(hidden_size, lr_init)

model = DataParallel(ConvNet(hidden_size, n_a, p)).cuda()
optim = Adam(model.parameters(), lr=lr_init)
lr_schdlr = lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.2, 30)
loss = D

runner = GbsCnnClsfier(loader, 'outs/mnist', 25000, model, optim, lr_schdlr, loss, 5)
runner.train()

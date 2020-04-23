from torch.nn import DataParallel
from torch.optim import lr_scheduler, Adam, RMSprop
import math
import os
from data.mnist_loader import MnistLoader
from runner.cnn_runner import GbsCnnClsfier
from models.cnn import ConvNet, D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sub_size = 1
n_a = 300
V = 5
cpus = 4
loader = MnistLoader(n_a, sub_size, cpus, 5)
p = loader.p

hidden_size = p if p  >= 100 else 100
lr_init = 0.0001
print(hidden_size, lr_init)

model = DataParallel(ConvNet(hidden_size, n_a, p)).cuda()
#optim = Adam(model.parameters(), lr=lr_init)
#lr_schdlr = lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.2, 30)
optim = RMSprop(model.parameters(), lr= lr_init, alpha=0.99, eps=1e-08)
lr_schdlr = lr_scheduler.CyclicLR(optim, base_lr=0.000001, max_lr=0.00001, step_size_up = 1000)

runner = GbsCnnClsfier(loader, 'outs/mnist', 1000, model, optim, lr_schdlr, D, 5, V)
runner.train()

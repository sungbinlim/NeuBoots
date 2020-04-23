from torch.nn import DataParallel
from torch.optim import lr_scheduler, Adam
import math
import os
from data.mnist_loader import MnistLoader
from runner.cnn_runner import GbsCnnClsfier
from model.cnn import ConvNet, D


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 500
n_a = 500
cpus = 8
loader = MnistLoader(batch_size, n_a, 8, 5)
p = loader.p

hidden_size = p // 2 if p // 2 >= 100 else 100
lr_init = 0.01 / math.sqrt(loader.n_train)
print(hidden_size, lr_init)

model = DataParallel(ConvNet(hidden_size, n_a, p)).cuda()
optim = Adam(model.parameters(), lr=lr_init)
lr_schdlr = lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.2, 30)

runner = GbsCnnClsfier(loader, 'outs/mnist', 25000, model, optim, lr_schdlr, D, 5)
runner.train()

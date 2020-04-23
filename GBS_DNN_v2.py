import os
import torch
import torchvision
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import numpy as np
from random import sample
from torch.autograd import Variable
import sys
# Device configuration
gpu_ind = int(r.gpu_ind)
if torch.cuda.is_available():
  device = torch.device('cuda', gpu_ind)
else:
  device = torch.device('cpu')
"r.cpu_ind" in dir(os)
try:
   r.cpu_ind
except:
    print("No cpu_ind")
else:
    cpu_ind = int(r.cpu_ind)
    if cpu_ind == 1:
      device = torch.device('cpu')
n = int(r.n)
n_test = int(r.n_test)
p = int(r.p)

n_a = int(r.n_a)
n_b = int(r.n_b)

X = r.X
y1 = r.y
y0 = r.y0
X_test = r.X_test
y1_test = r.y_test
y0_test = r.y0_test

X = torch.from_numpy(X)
y0 = torch.from_numpy(y0)
y = torch.zeros(n)
for i in range(n):
  y[i] = y1[i]
  
X_test = torch.from_numpy(X_test)
y0_test = torch.from_numpy(y0_test)
y_test = torch.zeros(n_test)
for i in range(n_test):
  y_test[i] = y1_test[i]

X = X.to(device, dtype = torch.float)
y = y.to(device, dtype = torch.float)
y0 = y0.to(device, dtype = torch.float)
X_test = X_test.to(device, dtype = torch.float)
y_test = y_test.to(device, dtype = torch.float)
y0_test = y0_test.to(device, dtype = torch.float)

X_rand = torch.from_numpy(r.X_rand)
X_rand = X_rand.to(device, dtype = torch.float)
X_cpu = X.cpu()
y_cpu = y.cpu()

# Image processing

class ConvNet(nn.Module):
  def __init__(self, hidden_size, num_classes = 10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1,32, kernel_size=7, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride=2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(32,64, kernel_size = 7, stride=1, padding=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.Lrelu = nn.ReLU()#nn.LeakyReLU(0.01)
    self.tanh = nn.Tanh()
    self.fc1 = nn.Linear(5*5*64 + n_a, hidden_size)
    self.fc2 = nn.Linear(hidden_size + n_a, hidden_size)
    self.fc3 = nn.Linear(hidden_size + n_a, hidden_size)
    self.fc_out = nn.Linear(hidden_size + n_a, num_classes)
    self.soft = nn.Softmax(dim=1)
    self.bn1 = nn.BatchNorm1d(5*5*64)
    self.bn2 = nn.BatchNorm1d(hidden_size)
    self.bn3 = nn.BatchNorm1d(hidden_size)
    self.bn4 = nn.BatchNorm1d(hidden_size)
    self.drop = nn.Dropout(p=0.5)
    
  def forward(self, x, w):
    out = self.layer1(x)
    out = self.layer2(out) 
    out = out.reshape(out.size(0), -1)
    out = self.bn1(out)
    out2 = 5.0*torch.exp(-1.0*w)
    out0 = torch.cat([out2, out],dim=1)
    out0 = self.Lrelu(self.fc1(out0))
    out0 = self.bn2(out0)
    out0 = torch.cat([out2, out0],dim=1)
    out0 = self.Lrelu(self.fc2(out0)) 
    out0 = self.bn3(out0)
    out0 = torch.cat([out2, out0],dim=1)
    out0 = self.Lrelu(self.fc3(out0))
    out0 = self.bn4(out0)
    out0 = torch.cat([out2,  out0],dim=1)
    out0 = self.fc_out(out0)
    out0 = self.soft(out0)
    return out0

def D(Prob, y1, w1):
    out = -1.0*y1*torch.log(Prob)
    out = out.sum(1).reshape(nsub,1)
    out = out*w1
    out = torch.sum(out)
    return out

def Schedule(it0, lag1, lag):
    s = (1.0 + np.cos(3.14*(it0 - lag1*np.floor(it0/lag))/lag ))
    return s
#############################################
lam = 1.0
V = 5
lr0 = 0.0001
hidden_size = int(p)
num_classes = int(r.d)
sub_size = 1 # the number of subgroups among total n_a subgroups at one iteration   
nsub = int(sub_size*n_b) # the number of images used in one iteration
num_it = 6000

ones = torch.ones(2,2).to(device)
generator_CNN = ConvNet(hidden_size, num_classes).to(device)
optimizer = torch.optim.RMSprop(generator_CNN.parameters(), lr= 0.0005, alpha=0.99, eps=1e-08)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.00001, step_size_up = 1000)
a_sample = torch.distributions.exponential.Exponential(torch.ones(1, V))
a_test = torch.distributions.exponential.Exponential(torch.ones(1, n_a))
J = 1
it = 0
LOSS = torch.zeros(num_it).to(device)
alpha = torch.ones(nsub, n_a).to(device)
w_test = torch.ones(n_test,n_a).to(device)
w_train = torch.ones(n,n_a)
loss0 = 0.0
A = torch.zeros(sub_size*n_b, sub_size)
for i in range(sub_size):
  ind = range(i*n_b,(i+1)*n_b)
  A[ind,i] = 1
A = A.t().to(device)
s_sample= torch.distributions.exponential.Exponential(torch.ones(1))
a_many_sample = torch.distributions.exponential.Exponential(torch.ones(1,n_a))
A_many = torch.ones(n_test, 1).to(device)
A_many_train = torch.ones(n, 1).to(device)
THETA_test = torch.zeros(10, n_test, num_classes).to(device)
ones_test = torch.ones(n_test,1).to(device)

while J == 1:
    scheduler.step()
    loss = 0.0
    ind_samp = sample(range(n_a), sub_size)
    IND = []
    for k in range(sub_size):
      a = (ind_samp[k])*n_b
      b = (ind_samp[k]+1)*n_b
      IND += [o for o in range(a,b)]
    X1 = X[IND,:,:,:]
    y1 = y0[IND,:]
    ind_a = sample(range(nsub), V)
    for k in range(V):
      ind_b = sample(range(n_a), V)
      alpha[ind_a[k], ind_b] = a_sample.sample().to(device)
    w1 = torch.matmul(alpha[:, ind_samp], A).t()
    Prob = generator_CNN(X1, alpha)
    loss1 = D(Prob, y1, w1)/nsub
    loss += loss1
    loss0 += loss1.item()/nsub
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    LOSS[it] = loss.item()
    it += 1
    if it > (num_it-1):
      J = 0
    if(it+1) % 200==0:
      for param_group in optimizer.param_groups:
        lr1 = param_group["lr"] 
      with torch.no_grad():
        print('MNIST-CNN [{}/{}], Loss: {:.4f}, nsub: {}, lr: {:.6f}'
          .format(it+1, num_it, loss0/100, nsub,  lr1))
        print('n: {}, p: {},, hidden: {},  n_a: {}'
          .format(n, p, hidden_size, n_a))
        generator_CNN.eval()
        w_test = a_test.sample().to(device)
        w_test = torch.matmul(ones_test,w_test)
        Prob_test = generator_CNN(X_test, w_test).to(device)
        _, predicted = torch.max(Prob_test,1)
        predicted = predicted.to(dtype=torch.int)
        correct_test = (predicted == y_test.to(dtype=torch.int)).sum()
        v_test = 0.0
        a_many_sample = torch.distributions.exponential.Exponential(torch.ones(1,n_a))
        for i in range(10):
          alpha_many = a_many_sample.sample().to(device)
          w0_many =  torch.matmul(A_many, alpha_many)
          THETA_test[i,:,:] = generator_CNN(X_test, w0_many).to(device)
          v_test = torch.sqrt(THETA_test.var(0)).mean()
        print('Accuracy of the model on the 10,000 test images: {} %, sd: {:.5f}'.format(100.0 * float(correct_test) / n_test, v_test))
        v_train = 0.0
        generator_CNN.train()
      loss0 = 0.0
      sys.stdout.flush()
        
torch.save(generator_CNN.state_dict(),'GBS_DNN_MNIST.ckpt')
      
  




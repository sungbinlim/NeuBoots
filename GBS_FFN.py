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
p = int(r.p)
sig0 = r.sig0
X = r.X
y = r.y

X = torch.from_numpy(X)
y = torch.from_numpy(y)

X = X.to(device, dtype = torch.float)
y = y.to(device, dtype = torch.float)

fac1 = 1.0
class Net(nn.Module):
  def __init__(self, n,  hidden_size):
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.Lrelu2 = nn.LeakyReLU(0.01)
    self.tanh = nn.Tanh()
    self.fc1 = nn.Linear(k, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size+n)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc_out = nn.Linear(hidden_size+n, 1)
    self.fc_V = nn.Linear(n+p, k)
    self.bn1 = nn.BatchNorm1d(k)
    self.bn2 = nn.BatchNorm1d(hidden_size)
    self.bn3 = nn.BatchNorm1d(hidden_size+n)
    self.drop1 = nn.Dropout(p=0.5)
    self.drop2 = nn.Dropout(p=0.5)
    self.drop3 = nn.Dropout(p=0.5)
    
    
  def forward(self, w1, X1):
    out1 = fac1*torch.exp(-1.0*w1)
    #out1 = torch.log(w1)
    out2 = torch.cat([X1, out1],dim=1)
    #out2 = self.bn0(out2)
    out = self.relu(self.fc_V(out2))
    #out = self.drop1(out)
    out = self.bn1(out)
    #out = torch.cat([a0*out1, out],dim=1)
    out = self.relu(self.fc1(out))
    #out = self.drop2(out)
    out = self.bn2(out)
    #out = torch.cat([a0*out1, out],dim=1)
    out = self.relu(self.fc2(out))
    #out = self.drop3(out)
    out = self.bn3(out)
    #out = torch.cat([a0*out1, out],dim=1)
    out = self.fc_out(out)
    return out

def D(Theta1, y, w1):
    out =  0.5*((y - Theta1)**2)*w1.t()/sig0
    return out
    
def Schedule(it0, lag1, lag):
    s = (1.0 + np.cos(3.14*(it0 - lag1*np.floor(it0/lag))/lag ))
    return s

#############################################
K0 = 10
inc1 = 0.005 # 1.0 work good. Let's try 2.0
a0 = 1.0#0/n_a#0.01
lr0 = 0.01/np.sqrt(n)

k = 300
hidden_size = 500

nsub = int(sub_size*n_b)
epoch = 2000#int(5000.0*float(n)/nsub)
num_it = epoch
ones = torch.ones(2,2).to(device)
n1 = float(n)
generator_CNN = Net(n, hidden_size).to(device)
#optimizer = torch.optim.Adam(generator_CNN.parameters(), lr= lr0)
optimizer = torch.optim.SGD(generator_CNN.parameters(), lr= lr0, momentum=0.9)

#lag for changing learning rate
lag = 1000
lag1 = float(lag)
a_sample = torch.distributions.exponential.Exponential(torch.ones(1,n))

J = 1
it = 0
it1 = 0
it0 = 0.0
LOSS = 10000.0*torch.zeros(num_it).to(device)
loss0 = 0.0
X1 = X
y1 = y
ones = torch.ones(n,1).to(device)
while J == 1:
    lr = lr0*Schedule(it0, lag1, lag)/((it0+1.0)**0.4) + 0.1**12
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    loss = 0.0
    for m in range(K0):
      a = torch.rand(1)
      if a.item() < 0.9:
        w1 = a_sample.sample().to(device)
      else:
        w1 = torch.ones(1,n).to(device)
      alpha = torch.matmul(ones,w1)
      Theta1 = generator_CNN(alpha, X)
      loss1 = D(Theta1, y, w1).sum()/K0/n
      loss += loss1
      loss0 += loss1.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    LOSS[it] = loss.item()
    if it % 50 == 20:
        fac1 += inc1
    it0 = it0 + 1.0
    it += 1
    if it > (num_it-1):
      J = 0
    if(it+1) % 100==0:
      print('GBS-Nonpara [{}/{}], Loss: {:.4f}, nsub: {}, lr: {:.6f}'
          .format(it+1, num_it, loss0/100, nsub, lr))
      print('n: {}, p: {},  var_fact: {:.3f}, hidden: {}'
          .format(n, p, fac1, hidden_size))
      loss0 = 0.0
      sys.stdout.flush()
      
##################################
N = 2000
#generator_CNN.eval()
###################################
#generator_CNN.train()
#device = 'cuda'
THETA = torch.zeros(N,n).to(device)
THETA0 = torch.zeros(N,n).to(device)
with torch.no_grad():
  generator_CNN = generator_CNN.to(device)
  for i in range(N):
    w1 = a_sample.sample().to(device)
    alpha = torch.matmul(ones,w1)
    THETA[i,:] = generator_CNN(alpha, X ).to(device)[:,0]
    alpha = torch.ones(n,n).to(device)
    THETA0[i,:] = generator_CNN(alpha, X ).to(device)[:,0]
    
    if (i+1) % 100 == 0:
      print(device)
      print(i+1)
Theta = THETA.cpu().detach().numpy()
Theta0 = THETA0.cpu().detach().numpy()
LOSS0 = LOSS.cpu()
LOSS0 = LOSS0.detach().numpy()
del(THETA)
###########################


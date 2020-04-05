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
w = r.w

X = r.X
y = r.y

X = torch.from_numpy(X)
y = torch.from_numpy(y)
w = torch.from_numpy(w)

X = X.to(device, dtype = torch.float)
y = y.to(device, dtype = torch.float)
w = w.to(device, dtype = torch.float)
fac1 = 1.0

class Net(nn.Module):
  def __init__(self, n_a,  hidden_size):
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.Lrelu2 = nn.LeakyReLU(0.01)
    self.tanh = nn.Tanh()
    self.fc1 = nn.Linear(k+p, int(hidden_size))
    self.fc2 = nn.Linear(int(hidden_size+p), int(hidden_size))
    self.fc3 = nn.Linear(int(hidden_size), int(hidden_size))
    self.fc_out = nn.Linear(hidden_size+p, 1)
    self.fc_V = nn.Linear(p, k)
    self.fc_U = nn.Linear(k, hidden_size)
    self.bn0 = nn.BatchNorm1d(n_a*2+p)
    self.bn1 = nn.BatchNorm1d(k)
    self.bn2 = nn.BatchNorm1d(hidden_size)
    self.bn3 = nn.BatchNorm1d(hidden_size+2*n_a)
    self.drop1 = nn.Dropout(p=0.5)
    self.drop2 = nn.Dropout(p=0.5)
    self.drop3 = nn.Dropout(p=0.5)
    self.drop4 = nn.Dropout(p=0.5)
    
    
  def forward(self, X1):
    out = self.relu(self.fc_V(X1))
    out = self.drop1(out)
    #out = self.bn1(out)
    out = torch.cat([a0*X1, out],dim=1)
    out = self.relu(self.fc1(out))
    out = self.drop2(out)
    #out = self.bn2(out)
    out = torch.cat([a0*X1, out],dim=1)
    out = self.tanh(self.fc2(out))
    out = self.drop3(out)
    
    #out = self.bn3(out)
    out = torch.cat([a0*X1, out],dim=1)
    out = self.fc_out(out)
#    out = self.soft(out)
    return out

def D(Theta1, y1, w1):
    #out = -1.0*y1*torch.log(Prob) - (1.0-y1)*torch.log(1-Prob)
    #out *= w1
    #out = torch.sum(out)
    out =  0.5*((y1 - Theta1)**2)*w1/sig0
    return out
    
def Schedule(it0, lag1, lag):
    s = (1.0 + np.cos(3.14*(it0 - lag1*np.floor(it0/lag))/lag ))
    return s

#############################################
inc1 = 0.0#05 # 1.0 work good. Let's try 2.0
a0 = 1.0 #5.0/n_a#0.01
lr0 = 0.01/np.sqrt(n)
k = 200
hidden_size = 500

epoch = 2000#int(5000.0*float(n)/nsub)
num_it = epoch
generator_CNN = Net(n_a, hidden_size).to(device)
optimizer = torch.optim.Adam(generator_CNN.parameters(), lr= lr0)
lag = 1000
lag1 = float(lag)

J = 1
it = 0
it1 = 0
it0 = 0.0
LOSS = 10000.0*torch.zeros(num_it).to(device)
loss0 = 0.0
while J == 1:
    lam0 = 0.1*lam*it0**0.5/(it0**0.5+500.0)
    if it0 > 10000:
      lam0 = lam
    else:
      if it0 > 5000:
        lam0 = 0.1*lam*it0**0.5/(it0**0.5+1000.0)
      else:
        lam0 = 0.0
    lr = lr0*Schedule(it0, lag1, lag)/((it0+1.0)**0.4) + 0.1**12
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    loss = 0.0
    Theta1 = generator_CNN(X)
    loss = D(Theta1, y, w).sum()/n
    loss0 += loss.item()
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
    if(it+1) % 500==0:
      print('GBS-Nonpara [{}/{}], Loss: {:.4f}, nsub: {}, lr: {:.6f}'
          .format(it+1, num_it, loss0/500, nsub, lr))
      print('n: {}, p: {},  var_fact: {:.3f}, hidden: {},  n_a: {}, lam0: {:.2f}'
          .format(n, p, fac1, hidden_size, n_a, lam0))
      loss0 = 0.0
      sys.stdout.flush()
      
##################################
N = 2000
generator_CNN.eval()
###################################
theta = generator_CNN(X).to(device)
theta = theta.cpu().detach().numpy()

#generator_CNN.train()
#device = 'cuda'
###########################


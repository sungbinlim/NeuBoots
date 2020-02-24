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
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

# n = sample size
# p = # of predictors
# n_a = # of subgroups
# n_b = # of samples in each subgroup
fac1 = 1.0

class Net(nn.Module):
  def __init__(self, n_a,  hidden_size, ones):
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.Lrelu2 = nn.LeakyReLU(0.01)
    self.fc1 = nn.Linear(k+2*n_a, int(hidden_size))
    self.fc2 = nn.Linear(int(hidden_size+2*n_a), int(hidden_size+2*n_a))
    self.fc3 = nn.Linear(int(hidden_size), int(hidden_size))
    self.fc_out = nn.Linear(hidden_size+4*n_a, p)
    self.fc_V = nn.Linear(n_a*2, k)
    self.fc_U = nn.Linear(k, hidden_size)
  
  def forward(self, w1, ones):
    out1 = torch.exp(-1.0*w1)
    out2 = fac1*torch.cat([out1, 1.0 - out1],dim=1)
    out = self.Lrelu2(self.fc_V(out2))
    out = torch.cat([a0*out2, out],dim=1)
    out = self.Lrelu2(self.fc1(out))
    out = torch.cat([a0*out2, out],dim=1)
    out = self.Lrelu2(self.fc2(out))
    #out = self.Lrelu2(self.fc3(out))
    out = torch.cat([a0*out2, out],dim=1)
    out = self.fc_out(out)

    ################### Linear transformation
    #out = self.fc0(w1)
    return out

def D(y1,X1, Theta):
    c = torch.matmul(X1,Theta.t())
    out =  -0.5*(y1 - c)**2
    return out

def Schedule(it0, lag1, lag):
    s = (1.0 + np.cos(3.14*(it0 - lag1*np.floor(it0/lag))/lag ))
    return s

#############################################
K0 = 100
inc1 = 1.0/np.sqrt(n*p) 
a0 = 10.0/np.sqrt(n) 
lr0 = 0.01/np.sqrt(n) # learning rate

if n > 5000:
  sub_size = int(5000.0*float(n_a)/float(n))
else:
  sub_size = n_a
if sub_size == 0:
  sub_size = 1
hidden_size = int(p/4)
if hidden_size < 100:
  hidden_size = 100
k = int(n_a*4)
nsub = int(sub_size*n_b)
epoch = int(3000.0*float(n)/nsub)
if epoch < 10000:
  epoch = 10000
if epoch > 35000:
  epoch = 35000
num_it = epoch

ones = torch.ones(2,2).to(device)
n_a = int(r.n_a)
n1 = float(n)
generator_CNN = Net(int(n_a), int(hidden_size), ones).to(device)
optimizer = torch.optim.Adam(generator_CNN.parameters(), lr= lr0)

##########
lag = 1000
lag1 = float(lag)
a_sample = torch.distributions.exponential.Exponential(torch.ones(K0,sub_size))
J = 1
it = 0
it0 = 0.0
LOSS = 10000.0*torch.zeros(num_it).to(device)
alpha = torch.ones(K0,n_a).to(device)

A = torch.zeros(sub_size*n_b,sub_size)
for i in range(sub_size):
  ind = range(i*n_b,(i+1)*n_b)
  A[ind,i] = 1
A = A.t().to(device)

if sub_size == n_a:
  X1 = X.to(device)
  y1 = y.reshape(n,1).to(device)
else:
  if sub_size < n_a:
      ind_samp = sample(range(n_a),sub_size)
      IND = []
      for k in range(sub_size):
        a = (ind_samp[k])*n_b
        b = (ind_samp[k]+1)*n_b
        IND += [o for o in range(a,b)]
      X1 = X[IND,:].to(device)
      y1 = y[IND].reshape(sub_size*n_b,1).to(device)

      alpha[:,ind_samp] = a_sample.sample().to(device)
      w1 = torch.matmul(alpha[:,ind_samp],A)

  else:
      alpha = a_sample.sample().to(device)
      w1 = torch.matmul(alpha,A)

while J == 1:
    n = int(r.n)
    lr = lr0*Schedule(it0, lag1, lag)/((it0+1.0)**0.1) + 0.1**12
    if it0 > (0.8*float(num_it)):
      lr = 0.05*lr0/((it0+1.0)**0.1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if sub_size < n_a:
      ind_samp = sample(range(n_a),sub_size)
      IND = []
      for k in range(sub_size):
        a = (ind_samp[k])*n_b
        b = (ind_samp[k]+1)*n_b
        IND += [o for o in range(a,b)]
      X1 = X[IND,:].to(device)
      y1 = y[IND].reshape(sub_size*n_b,1).to(device)

      alpha[:,ind_samp] = a_sample.sample().to(device)
      w1 = torch.matmul(alpha[:,ind_samp],A)#alpha[:,ind_samp].repeat(1,n_b).to(device)

    else:
      alpha = a_sample.sample().to(device)
      w1 = torch.matmul(alpha,A)

    if it % 50 == 20:
      fac1 += inc1

    Theta = generator_CNN(alpha,ones)
    loss1 = D(y1, X1, Theta).to(device)
    loss_log = loss1*(w1.t())
    loss2 = (-1.0*loss_log.sum(0))/nsub
    loss = torch.mean(loss2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    LOSS[it] = loss.item()
    it0 = it0 + 1.0
    it += 1
    if it > (num_it-1):
      J = 0
    if(it+1) % 100==0:
      print('LinReg (Shallow) [{}/{}], Loss: {:.4f}, nsub: {}, lr: {:.6f}'
          .format(it+1, num_it, loss2.sum()/K0, nsub,  lr))
      print('n: {}, p: {},  var_fact: {:.3f}, hidden: {}, K0: {}, n_a: {}, cor: {}'
          .format(n, p, fac1 , hidden_size, K0, n_a, int(r.T1)))
      sys.stdout.flush()
##################################
N = 5000
a_many_sample = torch.distributions.exponential.Exponential(torch.ones(N,n_a))
if n<40000:
  with torch.no_grad():
    alpha = a_many_sample.sample().to(device)*1.2
    Theta1 = generator_CNN(alpha, ones)
else:
  with torch.no_grad():
    alpha = a_many_sample.sample().to(device)
    Theta1 = generator_CNN(alpha, ones).to('cpu')

Alpha = alpha.cpu().detach().numpy()
Theta = Theta1.cpu().detach().numpy()
LOSS0 = LOSS.cpu()
LOSS0 = LOSS0.detach().numpy()
del(alpha)
del(Theta1)



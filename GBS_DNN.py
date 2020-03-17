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

# Image processing
#fac0 = 1.0
fac1 = 1.0

class ConvNet(nn.Module):
  def __init__(self, hidden_size, num_classes = 10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1,16, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride=2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(16,32, kernel_size = 5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc1 = nn.Linear(7*7*32 + 2*n_a, hidden_size)
    self.Lrelu = nn.LeakyReLU(0.01)
    self.fc2 = nn.Linear(hidden_size+ 2*n_a, hidden_size)
    self.fc3 = nn.Linear(hidden_size+ 2*n_a, hidden_size)
    self.fc4 = nn.Linear(hidden_size+ 2*n_a, hidden_size)
    self.fc5 = nn.Linear(hidden_size+ 2*n_a, hidden_size)
    #self.fc2 = nn.Linear(hidden_size, hidden_size)
    #self.fc3 = nn.Linear(hidden_size, hidden_size)
    #self.fc4 = nn.Linear(hidden_size, hidden_size)
    #self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.fc_out = nn.Linear(hidden_size+ 2*n_a, num_classes)
    self.soft = nn.Softmax(dim=1)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.bn2 = nn.BatchNorm1d(hidden_size)
    self.bn3 = nn.BatchNorm1d(hidden_size)
    self.bn4 = nn.BatchNorm1d(hidden_size)
        
  def forward(self, x, w):
    out = self.layer1(x)
    out = self.layer2(out) 
    out = out.reshape(out.size(0), -1)
    out1 = torch.exp(-1.0*w)
    out2 = fac1*torch.cat([out1, 1.0 - out1],dim=1)
    out0 = torch.cat([out2, out],dim=1)
    out0 = self.Lrelu(self.fc1(out0))
    out0 = self.bn1(out0)
    out0 = torch.cat([a0*out2, out0],dim=1)
    out0 = self.Lrelu(self.fc2(out0))
    out0 = self.bn2(out0)
    out0 = torch.cat([a0*out2, out0],dim=1)
    out0 = self.Lrelu(self.fc3(out0))
    #out0 = self.Lrelu(self.fc4(out0))
    #out0 = torch.cat([a0*out2, out0],dim=1)
    #out0 = self.Lrelu(self.fc5(out0))
    #out0 = torch.cat([a0*out2, out0],dim=1)
    out0 = self.bn4(out0)
    out0 = torch.cat([a0*out2, out0],dim=1)
    out0 = self.fc_out(out0)
    out0 = self.soft(out0)
    return out0

def D(Prob, y1, w1):
    out = -1.0*y1*torch.log(Prob)
    out *= w1
    out = torch.sum(out)
    return out

def Schedule(it0, lag1, lag):
    s = (1.0 + np.cos(3.14*(it0 - lag1*np.floor(it0/lag))/lag ))
    return s

#############################################
K0 = 5
inc1 = 10.0/np.sqrt(n*p) # 1.0 work good. Let's try 2.0
a0 = 30.0/np.sqrt(n) #5.0/n_a#0.01
lr0 = 0.01/np.sqrt(n)

num_classes = int(r.d)
if n > 500:
  sub_size = int(500.0*float(n_a)/float(n))
else:
  sub_size = n_a
if sub_size == 0:
  sub_size = 1
hidden_size = int(p/2)  
if hidden_size < 100:
  hidden_size = 100
nsub = int(sub_size*n_b)
#epoch = 50*int(np.sqrt(float(p*n_a))) + 5000
epoch = 25000#int(5000.0*float(n)/nsub)
num_it = epoch
ones = torch.ones(2,2).to(device)
n_a = int(r.n_a)
n1 = float(n)
generator_CNN = ConvNet(hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(generator_CNN.parameters(), lr= lr0)

#lag for changing learning rate
lag = 1000
lag1 = float(lag)
a_sample = torch.distributions.exponential.Exponential(torch.ones(sub_size))
J = 1
it = 0
it1 = 0
it0 = 0.0
LOSS = 10000.0*torch.zeros(num_it).to(device)
alpha = torch.ones(n_a,1).to(device)
w_test = torch.ones(n_test,n_a).to(device)
loss0 = 0.0
A = torch.zeros(sub_size*n_b, sub_size)
A_total = torch.ones(sub_size*n_b, 1).to(device)
for i in range(sub_size):
  ind = range(i*n_b,(i+1)*n_b)
  A[ind,i] = 1
A = A.to(device)

while J == 1:
    lr = lr0*Schedule(it0, lag1, lag)/((it0+1.0)**0.3) + 0.1**12
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    loss = 0.0
    for m in range(K0):
      if sub_size < n_a:
        ind_samp = sample(range(n_a),sub_size)
        IND = []
        for k in range(sub_size):
          a = (ind_samp[k])*n_b
          b = (ind_samp[k]+1)*n_b
          IND += [o for o in range(a,b)]
        X1 = X[IND,:,:,:]
        y1 = y0[IND,:]
        alpha[ind_samp, 0] = a_sample.sample().to(device)
        w1 = torch.matmul(A, alpha[ind_samp,:])  
      else:
        alpha = a_sample.sample().to(device)
        w1 = torch.matmul(A, alpha)  
    
      w0 = torch.matmul(A_total,alpha.t())
      Prob = generator_CNN(X1, w0)
      loss1 = D(Prob, y1, w1)/K0
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
      print('MNIST-CNN [{}/{}], Loss: {:.4f}, nsub: {}, lr: {:.6f}'
          .format(it+1, num_it, loss0/100, nsub,  lr))
      print('n: {}, p: {},  var_fact: {:.3f}, hidden: {},  n_a: {}'
          .format(n, p, fac1, hidden_size, n_a))
      loss0 = 0.0
      with torch.no_grad():
        Prob_test = generator_CNN(X_test, w_test).to('cpu')
        _, predicted = torch.max(Prob_test,1)
        predicted = predicted.to('cpu')
        correct = (predicted == y_test.to('cpu', dtype=torch.int)).sum()
      print('Accuracy of the model on the 10,000 test images: {} %'.format(100.0 * float(correct) / n_test))
      sys.stdout.flush()
      
torch.save(generator_CNN.state_dict(),'GBS_DNN_MNIST.ckpt')
      
      #fac0*fac1*fac[0,0].item()
##################################
N = 2000
generator_CNN.eval()
###################################
#generator_CNN.train()
device = 'cuda'
A_many = torch.ones(n_test, 1).to(device)
a_many_sample = torch.distributions.exponential.Exponential(torch.ones(1,n_a))
X_test = X_test.to(device)
THETA = torch.zeros(N,n_test,num_classes).to(device)
with torch.no_grad():
  generator_CNN = generator_CNN.to(device)
  for i in range(N):
    alpha_many = a_many_sample.sample().to(device)
    w0_many =  torch.matmul(A_many, alpha_many)
    THETA[i,:,:] = generator_CNN(X_test, w0_many ).to(device)
    if (i+1) % 100 == 0:
      print(device)
      print(i+1)
Theta = THETA.cpu().detach().numpy()
LOSS0 = LOSS.cpu()
LOSS0 = LOSS0.detach().numpy()
del(THETA)
###########################
n_rand = int(r.n_rand)
A_many = torch.ones(n_rand, 1).to(device)
a_many_sample = torch.distributions.exponential.Exponential(torch.ones(1,n_a))
X_rand = torch.from_numpy(r.X_rand)
X_rand = X_rand.to(device, dtype = torch.float)
THETA_rand = torch.zeros(N, n_rand, num_classes).to(device)
THETA_rand_f = torch.zeros(N, n_rand, num_classes).to(device)
w0_ones = torch.ones(n_rand, n_a).to(device)
with torch.no_grad():
  generator_CNN = generator_CNN.to(device)
  for i in range(N):
    alpha_many = a_many_sample.sample().to(device)
    w0_many =  torch.matmul(A_many, alpha_many)
    THETA_rand[i,:,:] = generator_CNN(X_rand, w0_many ).to(device)
    THETA_rand_f[i,:,:] = generator_CNN(X_rand, w0_ones ).to(device)
    
    if (i+1) % 100 == 0:
      print(device)
      print(i+1)
Theta_rand = THETA_rand.cpu().detach().numpy()
Theta_rand_f = THETA_rand_f.cpu().detach().numpy()
del(THETA_rand)
del(THETA_rand_f)


###########################
n_fake = int(r.n_fake)
A_many = torch.ones(n_fake, 1).to(device)
a_many_sample = torch.distributions.exponential.Exponential(torch.ones(1,n_a))
X_fake = torch.from_numpy(r.X_fake)
X_fake = X_fake.to(device, dtype = torch.float)
THETA_fake = torch.zeros(N, n_fake, num_classes).to(device)
THETA_fake_f = torch.zeros(N, n_fake, num_classes).to(device)
w0_ones = torch.ones(n_fake, n_a).to(device)
with torch.no_grad():
  generator_CNN = generator_CNN.to(device)
  for i in range(N):
    alpha_many = a_many_sample.sample().to(device)
    w0_many =  torch.matmul(A_many, alpha_many)
    THETA_fake[i,:,:] = generator_CNN(X_fake, w0_many ).to(device)
    THETA_fake_f[i,:,:] = generator_CNN(X_fake, w0_ones ).to(device)
    
    if (i+1) % 100 == 0:
      print(device)
      print(i+1)
Theta_fake = THETA_fake.cpu().detach().numpy()
Theta_fake_f = THETA_fake_f.cpu().detach().numpy()
del(THETA_fake)
del(THETA_fake_f)








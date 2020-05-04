library(reticulate)
library(dslabs)
mnist <- read_mnist()
X1 = mnist$train$images
X1 = X1/255
n = nrow(X1)
X = array(X1, dim = c(n,1,28,28))
for(i in 1:n){
  X[i,1,,] = matrix(X1[i,],28,28)[,28:1]
}
X_test1 = mnist$test$images
X_test1 = X_test1/255
n_test = nrow(X_test1)  
X_test = array(X_test1, dim = c(n_test,1,28,28))
for(i in 1:n_test){
  X_test[i,1,,] = matrix(X_test1[i,],28,28)[,28:1]
}
y = mnist$train$labels
y_test = mnist$test$labels

d = 10
n = nrow(X)
p = length(X[1,1,,1])^2
y0 = matrix(0,n,10)
for(i in 1:n){
  y0[i,y[i]+1] = 1
}
y0_test = matrix(0,n_test,10)
for(i in 1:n_test){
  y0_test[i,y_test[i]+1] = 1
}
source('~/Dropbox/Paper work/GBS/GSB-DNN/Fake_numbers.R')
n_rand = 100
X_rand = array(0, c(n_rand,1,28,28))
for( i in 1:n_rand){
  ind = rbinom(28^2,1,0.1)
  b = ind*rbeta(28^2,1,0.3)
  #b = matrix(rbeta(28^2,1,1),28,28)
  X_rand[i,1,,] = b   
}
#################################################
gpu_ind = 0
cpu_ind = 0
n_a = 500
n_b = n/n_a
pmt =proc.time()[3]
file = '~/Dropbox/Paper work/GBS/GSB-DNN/GBS_DNN.py'
reticulate::source_python(file,envir = NULL,convert = FALSE)
print(proc.time()[3]-pmt)

Theta_BT = py$Theta
par(mfrow=c(1,2), mai=c(0.4,0.4,0.1,0.1))
i = base::sample(1:n_test,1)
boxplot(Theta_BT[,i,], axes = F,lwd=0.7, cex=0.5)
qt = function(x) quantile(x, 0.05)
L = apply(Theta_BT[,i,],2,qt)
points(1:10, L, col="red", pch = 4, cex = 1.5, lwd =1.5)
axis(2)
axis(1, at=1:10, label=0:9)
text(1,0.95,y_test[i], col="red", cex = 2)
image(X_test[i,1,,], col = gray(seq(0, 1, 0.05)))

#########################################
Theta_fake = py$Theta_fake
Theta_fake_f = py$Theta_fake_f

par(mfrow=c(1,2), mai=c(0.4,0.4,0.1,0.1))
i = base::sample(1:n_fake,1)
boxplot(Theta_fake[,i,], axes = F,lwd=0.5, cex=0.3)
axis(2)
axis(1, at=1:10, label=0:9)
qt = function(x) quantile(x, 0.05)
L = apply(Theta_fake[,i,],2,qt)
points(1:10, L, col="red", pch = 4, cex = 1.5, lwd =1.5)
image(X_fake[i,1,,], col = gray(seq(0, 1, 0.05)))

#########################################

Theta_rand = py$Theta_rand
Theta_rand_f = py$Theta_rand_f

par(mfrow=c(1,2), mai=c(0.4,0.4,0.1,0.1))
i = base::sample(1:n_rand,1)
boxplot(Theta_rand[,i,], axes = F,lwd=0.7, cex=0.5)
axis(2)
axis(1, at=1:10, label=0:9)
qt = function(x) quantile(x, 0.01)
L = apply(Theta_rand[,i,],2,qt)
points(1:10, L, col="red", pch = 4, cex = 1.5, lwd =1.5)
#theta_f = Theta_rand_f[1,i,]
#points(1:10, theta_f, col="blue", pch = 4, cex = 1.5, lwd =1.5)
image(X_rand[i,1,,], col = gray(seq(0, 1, 0.05)))



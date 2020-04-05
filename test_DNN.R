library(reticulate)
gpu_ind = 0
n = 300
p = 1
sig0 = 0.2
X = matrix(rnorm(n*p),n,p)
X[,1] = sort(X[,1])
ix = sort.int(X[,1], index.return = T)$ix
mu = sin(X[,1]*3.14/2)
prob = 1/(1+exp(-1*mu))
y0 = mu + rnorm(n)*sqrt(sig0)
y = y0
y = matrix(y,n,1)
Sig =  matrix(0,n,n)
for(i in 1:n){
  for(k in i:n){
    Sig[i,k] = exp( - sum(abs(X[i,] - X[k,])^2 )/2 )
  }
}
Sig = Sig + t(Sig)
diag(Sig) = 1
Sig_inv = solve(Sig+ diag(n)*0.00000001) 
###########################
lam = 1
S = solve( lam*solve(Sig + diag(n)*0.0000001)*sig0 + diag(n) )
ch = chol(S)
y_hat = S%*%y0
N = 1000
mu_save1 = mu_save2 = matrix(0,n,N)
mu_GP = matrix(0,n,N)
for(i in 1:N){
  mu_GP[,i] = y_hat + t(ch)%*%rnorm(n)*sqrt(sig0)
  if(i %% 100 == 0) print(i)
}
L_GP = U_GP = rep(0,n)
for(i in 1:n){
  L_GP[i] = quantile(mu_GP[i,], 0.025)
  U_GP[i] = quantile(mu_GP[i,], 0.975)
}

par(mfrow=c(1,1), mai=c(0.4,0.4,0.1,0.1))
plot(X[ix,1], y_hat[ix], type="l",lwd=2, ylim=c(-2,2), xlim=c(-2.5,2.65))
points(X[ix,1], y0[ix], lwd=0.3, cex = 0.7)
lines(X[ix,1], L_GP[ix], col="blue")
lines(X[ix,1], U_GP[ix], col="blue")
lines(X[ix,1], mu[ix], col="red", lty=2,lwd=2)

print(s)
###################################
pmt =proc.time()[3]
#file = '~/Dropbox/Paper work/GBS/sim/nonpara/GPU_nonpara_optim.py'
#file = '~/Dropbox/Paper work/GBS/sim/nonpara/GBS_nonpara_v2.py'
file = '~/Dropbox/Paper work/GBS/GSB-DNN/FFN.py'
N = 200
MU_FFN = matrix(0,N,n)
for(i in 1:N){
  w = matrix(rexp(n,1),n,1)
  reticulate::source_python(file,envir = NULL,convert = FALSE)
  MU_FFN[i,] = t(py$theta)
  print(i)
}
mu_FFN = apply(MU_FFN,2,mean)
L_FFN =  U_FFN = rep(0,n)
for(i in 1:n){
  L_FFN[i] = quantile(MU_FFN[,i], 0.025)
  U_FFN[i] = quantile(MU_FFN[,i], 0.975)
}

file = '~/Dropbox/Paper work/GBS/GSB-DNN/GBS_FFN.py'
reticulate::source_python(file,envir = NULL,convert = FALSE)
print(proc.time()[3]-pmt)
mu_GBS = t(py$Theta)
mu_GBS0 = t(py$Theta0)
MU_GBS = apply(mu_GBS, 1, mean)
MU_GBS0 = apply(mu_GBS0, 1, mean)
L_GBS =  U_GBS = rep(0,n)
L_GBS0 =  U_GBS0 = rep(0,n)
for(i in 1:n){
  L_GBS[i] = quantile(mu_GBS[i,], 0.025)
  U_GBS[i] = quantile(mu_GBS[i,], 0.975)
  L_GBS0[i] = quantile(mu_GBS0[i,], 0.025)
  U_GBS0[i] = quantile(mu_GBS0[i,], 0.975)
}

par(mfrow=c(1,3), mai=c(0.4,0.4,0.1,0.1))
xlim = c(-2.8,3)
plot(X[ix,1], MU_GBS[ix], type="l",lwd=2, ylim=c(-2,2), xlim=xlim)
points(X[ix,1], y0[ix], lwd=0.3, cex = 1)
lines(X[ix,1], L_GBS[ix], col="blue",lwd=1.5)
lines(X[ix,1], U_GBS[ix], col="blue",lwd=1.5)
lines(X[ix,1], mu[ix], col="red", lty=2,lwd=2)
#plot(X[ix,1], MU_GBS0[ix], type="l",lwd=2, ylim=c(-2,2), xlim=xlim)
#points(X[ix,1], y0[ix], lwd=0.3, cex = 1)
#lines(X[ix,1], L_GBS0[ix], col="blue",lwd=1.5)
#lines(X[ix,1], U_GBS0[ix], col="blue",lwd=1.5)
#lines(X[ix,1], mu[ix], col="red", lty=2,lwd=2)
plot(X[ix,1], mu_FFN, type="l",lwd=2, ylim=c(-2,2), xlim=xlim)
points(X[ix,1], y0[ix], lwd=0.3, cex = 1)
lines(X[ix,1], L_FFN[ix], col="blue",lwd=1.5)
lines(X[ix,1], U_FFN[ix], col="blue",lwd=1.5)
lines(X[ix,1], mu[ix], col="red", lty=2,lwd=2)
plot(X[ix,1], y_hat[ix], type="l",lwd=2, ylim=c(-2,2), xlim=xlim)
points(X[ix,1], y0[ix], lwd=0.3, cex = 1)
lines(X[ix,1], L_GP[ix], col="blue",lwd=1.5)
lines(X[ix,1], U_GP[ix], col="blue",lwd=1.5)
lines(X[ix,1], mu[ix], col="red", lty=2,lwd=2)







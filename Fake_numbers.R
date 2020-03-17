n_fake = 500
X_fake = array(0,dim=c(n_fake,1,28,28))
for(i in 1:n_fake){
  ind = base::sample(1:n_test,2)
  for(k in 1:1){
    X_fake[i,1,,] = X_fake[i,1,,] + X_test[ind[k],1,,]/4
  }
  ind = base::sample(1:n_test,2)
  for(k in 1:2){
    X_fake[i,1,,] = X_fake[i,1,,] + t(X_test[ind[k],1,,])/4
  }
  ind = base::sample(1:n_test,2)
  for(k in 1:1){
    X_fake[i,1,,] = X_fake[i,1,,] + X_test[ind[k],1,28:1,]/4
  }
}
par(mfrow=c(2,2), mai=c(0.4,0.4,0.1,0.1))
for(k in 1:4){
  i = base::sample(1:n_fake,1)
  image(X_fake[i,1,,], col = gray(seq(0, 1, 0.05)))
}


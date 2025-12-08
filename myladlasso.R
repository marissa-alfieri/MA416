myladlasso = function(Xk, Y, lam=0, method='ols', seedx=1234){
  ### (C) December 2025, Matthew N. Moore
  lam = abs(lam)
  n = length(Y)
  Xk = as.matrix(Xk)
  p = dim(Xk)[2]
  set.seed(seedx)
  
  f = function(bhat){
    res = Y - as.vector(cbind(1, Xk)%*%bhat)
    
    if(method == 'ols'){
      return(sum(res^2))
    }
    if(method == 'ridge'){
      return(sum(res^2) + lam*sum(bhat^2))
    }
    if(method == 'lad'){
      return(sum(abs(res)))
    }
    if(method == 'lasso'){
      return(sum(res^2) + lam*sum(abs(bhat)))
    }
    if(method == 'ladlasso'){
      return(sum(abs(res)) + lam*sum(abs(bhat)))
    }
  }
  
  bhat00 = rep(0, p+1)
  So = optim(bhat00, f, method ='Nelder-Mead',
             control=list(reltol=1e-10))
  bhat = So$par
  
  return(bhat)
}
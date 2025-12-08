myladlasso_fixed = function(Xk, Y, lam=0, method='ols', seedx=1234){
  ### Fixed version - does NOT penalize intercept
  lam = abs(lam)
  n = length(Y)
  Xk = as.matrix(Xk)
  p = dim(Xk)[2]
  set.seed(seedx)

  f = function(bhat){
    res = Y - as.vector(cbind(1, Xk)%*%bhat)
    bhat_no_intercept = bhat[-1]  # Exclude intercept from penalty

    if(method == 'ols'){
      return(sum(res^2))
    }
    if(method == 'ridge'){
      return(sum(res^2) + lam*sum(bhat_no_intercept^2))  # Don't penalize intercept
    }
    if(method == 'lad'){
      return(sum(abs(res)))
    }
    if(method == 'lasso'){
      return(sum(res^2) + lam*sum(abs(bhat_no_intercept)))  # Don't penalize intercept
    }
    if(method == 'ladlasso'){
      return(sum(abs(res)) + lam*sum(abs(bhat_no_intercept)))  # Don't penalize intercept
    }
  }

  bhat00 = rep(0, p+1)
  So = optim(bhat00, f, method ='Nelder-Mead',
             control=list(reltol=1e-10, maxit=10000))
  bhat = So$par

  return(bhat)
}

# M07.4 - Ridge Regression

data = read.csv("CompanyPayroll_9.25.csv")

Y = data$Efficiency_Factor_Obs
X1 = data$Hourly_Wage
X = cbind(1, X1)

p = ncol(X) - 1   
n = length(Y)


# question 1 
beta_hat = solve(t(X) %*% X) %*% (t(X) %*% Y)
Y_hat = X %*% beta_hat
res = Y - Y_hat
SSE = sum(res^2)
SST = sum((Y - mean(Y))^2)
SSR = SST - SSE

F_stat = (SSR / p) / (SSE / (n - p - 1))
p_value = 1 - pf(F_stat, p, n - p - 1)
p_value


# question 2
SSM_1 = SST - SSE
RMSE_1 = sqrt(SSE / n)

X = cbind(1, X1, X1^2)
p = ncol(X) - 1
beta = solve(t(X) %*% X) %*% t(X) %*% Y
Y_hat = X %*% beta
res = Y - Y_hat
SSE = sum(res^2)
SSM_2 = SST - SSE
RMSE_2 = sqrt(SSE / n)

X = cbind(1, X1, X1^2, X1^3)
p = ncol(X) - 1
beta = solve(t(X) %*% X) %*% t(X) %*% Y
Y_hat = X %*% beta
res = Y - Y_hat
SSE = sum(res^2)
SSM_3 = SST - SSE
RMSE_3 = sqrt(SSE / n)


# question 3
X = cbind(1, X1, X1^2, X1^3)
p = ncol(X) - 1

I = diag(4)
L_vals = c(0, 5000, 100000, 400000, 900000)
p_vals = numeric(length(L_vals))

for (i in 1:length(L_vals)) {
  L = L_vals[i]
  
  beta_ridge = solve(t(X) %*% X + L * I) %*% (t(X) %*% Y)
  
  Y_hat = X %*% beta_ridge
  res   = Y - Y_hat
  
  SSE = sum(res^2)
  SSR = SST - SSE
  
  F_stat = (SSR / p) / (SSE / (n - p - 1))
  p_vals[i] = 1 - pf(F_stat, p, n - p - 1)
}

round(data.frame(L = L_vals, p_value = p_vals), 4)


# question 4 - Ridge regression with unscaled data
X1 = data$Hourly_Wage
X = cbind(1, X1, X1^2, X1^3)
p = ncol(X) - 1

P = diag(c(0, 1, 1, 1))
L_vals = c(0, 5000, 100000, 400000, 900000)
p_vals = numeric(length(L_vals))

for (i in 1:length(L_vals)) {
  L = L_vals[i]

  # ridge coefficients: (X'X + L P)^(-1) X'Y
  beta_ridge = solve(t(X) %*% X + L * P) %*% (t(X) %*% Y)

  Y_hat = X %*% beta_ridge
  res   = Y - Y_hat

  SSE = sum(res^2)
  SSR = SST - SSE

  F_stat = (SSR / p) / (SSE / (n - p - 1))
  p_vals[i] = 1 - pf(F_stat, p, n - p - 1)
}

round(data.frame(L = L_vals, p_value = p_vals), 4)


# CORRECT ANSWER for Canvas Question 3
# Ridge with UNSCALED data and FULL identity matrix
# Using the literal formula: beta_r = (X'X + lambda*I)^(-1) X'Y
X1 = data$Hourly_Wage
X = cbind(1, X1, X1^2, X1^3)
p = ncol(X) - 1

# Full identity matrix (penalizes ALL coefficients including intercept)
I = diag(4)
L_vals = c(0, 5000, 100000, 400000, 900000)
p_vals = numeric(length(L_vals))

for (i in 1:length(L_vals)) {
  L = L_vals[i]

  # ridge coefficients: (X'X + L*I)^(-1) X'Y
  beta_ridge = solve(t(X) %*% X + L * I) %*% (t(X) %*% Y)

  Y_hat = X %*% beta_ridge
  res   = Y - Y_hat

  SSE = sum(res^2)
  SSR = SST - SSE

  F_stat = (SSR / p) / (SSE / (n - p - 1))
  p_vals[i] = 1 - pf(F_stat, p, n - p - 1)
}

cat("\n*** CORRECT ANSWER FOR CANVAS ***\n")
round(data.frame(L = L_vals, p_value = p_vals), 4)


















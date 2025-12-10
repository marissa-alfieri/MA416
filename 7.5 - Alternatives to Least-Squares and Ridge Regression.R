# M07.5 - Alternatives to Least-Squares and Ridge Regression

source("myladlasso.R")
data = read.csv("HousingData_9.25.csv")

Y = data$price_usd
X1 = data$living_space_ft2
X2 = data$home_age
X3 = data$distance_city_mi
X4 = data$crime_rate_index

Xk = cbind(X1, X2, X3, X4)

n = length(Y)
p = ncol(Xk)

SST = sum((Y - mean(Y))^2)
X = cbind(1, Xk)

# Least Squares
b_ols = myladlasso(Xk, Y, lam = 0, method = "ols")
Yhat = X %*% b_ols
res = Y - Yhat
SSE = sum(res^2)
RMSE_LS = sqrt(SSE / (n - p - 1))
SSR = SST - SSE
F_LS = (SSR / p) / (SSE / (n - p - 1))

# Ridge 
b_ridge = myladlasso(Xk, Y, lam = 15, method = "ridge")
Yhat = X %*% b_ridge
res = Y - Yhat
SSE = sum(res^2)
RMSE_RIDGE = sqrt(SSE / (n - p - 1))
SSR = SST - SSE
F_RIDGE = (SSR / p) / (SSE / (n - p - 1))

# LAD
b_lad = myladlasso(Xk, Y, lam = 0, method = "lad")
Yhat = X %*% b_lad
res = Y - Yhat
SSE = sum(res^2)
RMSE_LAD = sqrt(SSE / (n - p - 1))
SSR = SST - SSE
F_LAD = (SSR / p) / (SSE / (n - p - 1))


# Lasso
b_lasso = myladlasso(Xk, Y, lam = 15, method = "lasso")
Yhat = X %*% b_lasso
res = Y - Yhat
SSE = sum(res^2)
RMSE_LASSO = sqrt(SSE / (n - p - 1))
SSR = SST - SSE
F_LASSO = (SSR / p) / (SSE / (n - p - 1))


# LAD-Lasso 
b_ladlasso = myladlasso(Xk, Y, lam = 15, method = "ladlasso")
Yhat = X %*% b_ladlasso
res = Y - Yhat
SSE = sum(res^2)
RMSE_LADLASSO = sqrt(SSE / (n - p - 1))
SSR = SST - SSE
F_LADLASSO = (SSR / p) / (SSE / (n - p - 1))

results = data.frame(
  Model = c("Least-Squares", "Ridge", "LAD", "Lasso", "LAD-Lasso"),
  RMSE = c(RMSE_LS, RMSE_RIDGE, RMSE_LAD, RMSE_LASSO, RMSE_LADLASSO),
  F_Statistic = c(F_LS, F_RIDGE, F_LAD, F_LASSO, F_LADLASSO)
)

results$RMSE = round(results$RMSE, 2)
results$F_Statistic = round(results$F_Statistic, 2)

results


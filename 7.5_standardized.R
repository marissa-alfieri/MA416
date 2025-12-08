# M07.5 - With STANDARDIZED predictors

source("myladlasso.R")
data = read.csv("HousingData_9.25.csv")

Y = data$price_usd
X1 = data$living_space_ft2
X2 = data$home_age
X3 = data$distance_city_mi
X4 = data$crime_rate_index

# STANDARDIZE the predictors (critical for Ridge/Lasso!)
X1_scaled = scale(X1)
X2_scaled = scale(X2)
X3_scaled = scale(X3)
X4_scaled = scale(X4)

Xk_scaled = cbind(X1_scaled, X2_scaled, X3_scaled, X4_scaled)

n = length(Y)
p = ncol(Xk_scaled)
SST = sum((Y - mean(Y))^2)
X_scaled = cbind(1, Xk_scaled)

# Least Squares
b_ols = myladlasso(Xk_scaled, Y, lam = 0, method = "ols")
Yhat = X_scaled %*% b_ols
res = Y - Yhat
SSE = sum(res^2)
RMSE_LS = sqrt(SSE / n)
SSR = SST - SSE
F_LS = (SSR / p) / (SSE / (n - p - 1))

# Ridge
b_ridge = myladlasso(Xk_scaled, Y, lam = 15, method = "ridge")
Yhat = X_scaled %*% b_ridge
res = Y - Yhat
SSE = sum(res^2)
RMSE_RIDGE = sqrt(SSE / n)
SSR = SST - SSE
F_RIDGE = (SSR / p) / (SSE / (n - p - 1))

# LAD
b_lad = myladlasso(Xk_scaled, Y, lam = 0, method = "lad")
Yhat = X_scaled %*% b_lad
res = Y - Yhat
SSE = sum(res^2)
RMSE_LAD = sqrt(SSE / n)
SSR = SST - SSE
F_LAD = (SSR / p) / (SSE / (n - p - 1))

# Lasso
b_lasso = myladlasso(Xk_scaled, Y, lam = 15, method = "lasso")
Yhat = X_scaled %*% b_lasso
res = Y - Yhat
SSE = sum(res^2)
RMSE_LASSO = sqrt(SSE / n)
SSR = SST - SSE
F_LASSO = (SSR / p) / (SSE / (n - p - 1))

# LAD-Lasso
b_ladlasso = myladlasso(Xk_scaled, Y, lam = 15, method = "ladlasso")
Yhat = X_scaled %*% b_ladlasso
res = Y - Yhat
SSE = sum(res^2)
RMSE_LADLASSO = sqrt(SSE / n)
SSR = SST - SSE
F_LADLASSO = (SSR / p) / (SSE / (n - p - 1))

results = data.frame(
  Model = c("Least-Squares", "Ridge", "LAD", "Lasso", "LAD-Lasso"),
  RMSE = c(RMSE_LS, RMSE_RIDGE, RMSE_LAD, RMSE_LASSO, RMSE_LADLASSO),
  F_Statistic = c(F_LS, F_RIDGE, F_LAD, F_LASSO, F_LADLASSO)
)

results$RMSE = round(results$RMSE, 2)
results$F_Statistic = round(results$F_Statistic, 2)

cat("\n*** WITH STANDARDIZED PREDICTORS ***\n")
print(results)

# Verify coefficients are now different
cat("\nVerifying coefficients are different:\n")
cat("OLS == Ridge:", all.equal(b_ols, b_ridge), "\n")
cat("OLS == Lasso:", all.equal(b_ols, b_lasso), "\n")

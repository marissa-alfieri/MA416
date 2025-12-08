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

# Test different lambda values
lambda_values = c(15, 150, 1500, 15000, 150000)

cat("\nTesting different lambda values for Ridge:\n")
cat("==========================================\n")

for (lam in lambda_values) {
  b_ridge = myladlasso(Xk, Y, lam = lam, method = "ridge")
  Yhat = X %*% b_ridge
  res = Y - Yhat
  SSE = sum(res^2)
  RMSE = sqrt(SSE / n)
  SSR = SST - SSE
  F_stat = (SSR / p) / (SSE / (n - p - 1))

  cat(sprintf("Lambda = %7d: RMSE = %8.2f, F-stat = %6.2f\n",
              lam, RMSE, F_stat))
}

cat("\nTesting different lambda values for Lasso:\n")
cat("==========================================\n")

for (lam in lambda_values) {
  b_lasso = myladlasso(Xk, Y, lam = lam, method = "lasso")
  Yhat = X %*% b_lasso
  res = Y - Yhat
  SSE = sum(res^2)
  RMSE = sqrt(SSE / n)
  SSR = SST - SSE
  F_stat = (SSR / p) / (SSE / (n - p - 1))

  cat(sprintf("Lambda = %7d: RMSE = %8.2f, F-stat = %6.2f\n",
              lam, RMSE, F_stat))
}

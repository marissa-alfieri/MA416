# Verify ridge regression calculation
data = read.csv("CompanyPayroll_9.25.csv")

Y = data$Efficiency_Factor_Obs
X1 = data$Hourly_Wage
n = length(Y)

SST = sum((Y - mean(Y))^2)

# Use SCALED data (as in question 3)
Z = scale(X1)
X = cbind(1, Z, Z^2, Z^3)
p = ncol(X) - 1

cat("n =", n, "\n")
cat("p =", p, "\n")
cat("SST =", SST, "\n\n")

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

  cat("L =", L, "\n")
  cat("  SSE =", SSE, "\n")
  cat("  SSR =", SSR, "\n")
  cat("  F-stat =", F_stat, "\n")
  cat("  p-value =", p_vals[i], "\n")
  cat("  p-value (4 dec) =", round(p_vals[i], 4), "\n\n")
}

cat("\nFinal Results:\n")
print(round(data.frame(L = L_vals, p_value = p_vals), 4))

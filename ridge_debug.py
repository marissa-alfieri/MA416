import numpy as np
import pandas as pd
from scipy.stats import f

# Read the data
data = pd.read_csv("CompanyPayroll_9.25.csv")

# Extract variables
Y = data['Efficiency_Factor_Obs'].values
X1 = data[' Hourly_Wage '].values

# SCALE the data
Z = (X1 - np.mean(X1)) / np.std(X1, ddof=1)

# Create design matrix with cubic terms (SCALED)
X = np.column_stack([np.ones(len(X1)), Z, Z**2, Z**3])

p = X.shape[1] - 1  # number of predictors (excluding intercept)
n = len(Y)

print(f"n = {n}")
print(f"p = {p}")
print(f"Mean of Y: {np.mean(Y):.6f}")
print(f"Mean of X1: {np.mean(X1):.6f}")
print(f"Std of X1: {np.std(X1, ddof=1):.6f}")
print()

# Calculate SST (total sum of squares)
SST = np.sum((Y - np.mean(Y))**2)
print(f"SST = {SST:.6f}")
print()

# Penalty matrix (don't penalize intercept)
P = np.diag([0, 1, 1, 1])

# Ridge penalization terms
L_vals = [0, 5000, 100000, 400000, 900000]

for L in L_vals:
    # Ridge coefficients: (X'X + L*P)^(-1) X'Y
    XtX = X.T @ X
    beta_ridge = np.linalg.solve(XtX + L * P, X.T @ Y)

    print(f"\n--- L = {L} ---")
    print(f"Beta coefficients: {beta_ridge}")

    # Predictions and residuals
    Y_hat = X @ beta_ridge
    res = Y - Y_hat

    # Sum of squares
    SSE = np.sum(res**2)
    SSR = SST - SSE

    print(f"SSE = {SSE:.6f}")
    print(f"SSR = {SSR:.6f}")
    print(f"SSR/SST = {SSR/SST:.6f}")

    # F-statistic and p-value
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    p_value = 1 - f.cdf(F_stat, p, n - p - 1)

    print(f"F-statistic = {F_stat:.6f}")
    print(f"p-value = {p_value:.10f}")
    print(f"p-value (4 decimals) = {p_value:.4f}")

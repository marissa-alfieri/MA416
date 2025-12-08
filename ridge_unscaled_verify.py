import numpy as np
import pandas as pd
from scipy.stats import f

# Read the data
data = pd.read_csv("CompanyPayroll_9.25.csv")

# Extract variables
Y = data['Efficiency_Factor_Obs'].values
X1 = data[' Hourly_Wage '].values

# Create design matrix with cubic terms (UNSCALED - as in user's code)
X = np.column_stack([np.ones(len(X1)), X1, X1**2, X1**3])

p = X.shape[1] - 1
n = len(Y)

print(f"Using UNSCALED data")
print(f"n = {n}, p = {p}\n")

# Calculate SST
SST = np.sum((Y - np.mean(Y))**2)
print(f"SST = {SST:.6f}\n")

# Penalty matrix (don't penalize intercept)
P = np.diag([0, 1, 1, 1])

# Ridge penalization terms
L_vals = [0, 5000, 100000, 400000, 900000]

print("Results:")
print("=" * 50)

for L in L_vals:
    # Ridge coefficients: (X'X + L*P)^(-1) X'Y
    XtX = X.T @ X
    beta_ridge = np.linalg.solve(XtX + L * P, X.T @ Y)

    # Predictions and residuals
    Y_hat = X @ beta_ridge
    res = Y - Y_hat

    # Sum of squares
    SSE = np.sum(res**2)
    SSR = SST - SSE

    # F-statistic and p-value
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    p_value = 1 - f.cdf(F_stat, p, n - p - 1)

    print(f"L = {L:>7}: p-value = {p_value:.4f}")

print("\nDetailed view:")
for L in L_vals:
    XtX = X.T @ X
    beta_ridge = np.linalg.solve(XtX + L * P, X.T @ Y)
    Y_hat = X @ beta_ridge
    res = Y - Y_hat
    SSE = np.sum(res**2)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    p_value = 1 - f.cdf(F_stat, p, n - p - 1)

    print(f"\nL = {L}")
    print(f"  Betas: {beta_ridge}")
    print(f"  SSE: {SSE:.10f}")
    print(f"  SSR: {SSR:.10f}")
    print(f"  F-stat: {F_stat:.10f}")
    print(f"  p-value: {p_value:.10f}")
    print(f"  p-value (4 dec): {p_value:.4f}")

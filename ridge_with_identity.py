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

p = X.shape[1] - 1
n = len(Y)

# Calculate SST
SST = np.sum((Y - np.mean(Y))**2)

# USE IDENTITY MATRIX (as shown in the problem formula)
I = np.eye(4)  # Full identity matrix - penalizes ALL coefficients

L_vals = [0, 5000, 100000, 400000, 900000]

print("Ridge Regression with FULL Identity Matrix (λ·I)")
print("=" * 60)

for L in L_vals:
    # Ridge coefficients: (X'X + λ·I)^(-1) X'Y
    XtX = X.T @ X
    beta_ridge = np.linalg.solve(XtX + L * I, X.T @ Y)

    # Predictions and residuals
    Y_hat = X @ beta_ridge
    res = Y - Y_hat

    # Sum of squares
    SSE = np.sum(res**2)
    SSR = SST - SSE

    # F-statistic and p-value
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    p_value = 1 - f.cdf(F_stat, p, n - p - 1)

    print(f"L = {L:>7}: p-value = {p_value:.4f} (exact: {p_value:.10f})")

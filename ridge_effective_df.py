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

# Penalty matrix (don't penalize intercept)
P = np.diag([0, 1, 1, 1])

# Ridge penalization terms
L_vals = [0, 5000, 100000, 400000, 900000]

print("Ridge Regression with Effective Degrees of Freedom")
print("=" * 60)

for L in L_vals:
    # Ridge coefficients
    XtX = X.T @ X
    ridge_matrix = np.linalg.inv(XtX + L * P)
    beta_ridge = ridge_matrix @ (X.T @ Y)

    # Effective degrees of freedom (trace of hat matrix)
    # H = X(X'X + λP)^(-1)X'
    H = X @ ridge_matrix @ X.T
    df_model = np.trace(H) - 1  # subtract 1 for intercept
    df_resid = n - np.trace(H)

    # Predictions and residuals
    Y_hat = X @ beta_ridge
    res = Y - Y_hat

    # Sum of squares
    SSE = np.sum(res**2)
    SSR = SST - SSE

    # F-statistic with effective df
    F_stat_eff = (SSR / df_model) / (SSE / df_resid)
    p_value_eff = 1 - f.cdf(F_stat_eff, df_model, df_resid)

    # Traditional F-statistic
    F_stat_trad = (SSR / p) / (SSE / (n - p - 1))
    p_value_trad = 1 - f.cdf(F_stat_trad, p, n - p - 1)

    print(f"\nL = {L}")
    print(f"  Effective df_model = {df_model:.4f}, df_resid = {df_resid:.4f}")
    print(f"  Traditional: p-value = {p_value_trad:.4f}")
    print(f"  Effective:   p-value = {p_value_eff:.4f}")

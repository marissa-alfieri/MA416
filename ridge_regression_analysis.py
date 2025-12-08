import numpy as np
import pandas as pd
from scipy.stats import f

# Read the data
data = pd.read_csv("CompanyPayroll_9.25.csv")

# Extract variables
Y = data['Efficiency_Factor_Obs'].values
X1 = data[' Hourly_Wage '].values  # Note: column name has spaces

# Create design matrix with cubic terms (unscaled)
X = np.column_stack([np.ones(len(X1)), X1, X1**2, X1**3])

p = X.shape[1] - 1  # number of predictors (excluding intercept)
n = len(Y)

# Calculate SST (total sum of squares)
SST = np.sum((Y - np.mean(Y))**2)

# Penalty matrix (don't penalize intercept)
P = np.diag([0, 1, 1, 1])

# Ridge penalization terms
L_vals = [0, 5000, 100000, 400000, 900000]
p_vals = []

print("Ridge Regression P-values for Cubic Model")
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

    p_vals.append(p_value)
    print(f"L = {L:>7}: p-value = {p_value:.4f}")

# Create results dataframe
results = pd.DataFrame({
    'Penalization Term (L)': L_vals,
    'P-value': [f"{p:.4f}" for p in p_vals]
})

print("\n" + "=" * 50)
print("\nResults Summary:")
print(results.to_string(index=False))

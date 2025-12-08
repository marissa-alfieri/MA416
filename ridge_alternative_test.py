import numpy as np
import pandas as pd
from scipy.stats import f, chi2

# Read the data
data = pd.read_csv("CompanyPayroll_9.25.csv")

Y = data['Efficiency_Factor_Obs'].values
X1 = data[' Hourly_Wage '].values
Z = (X1 - np.mean(X1)) / np.std(X1, ddof=1)
X = np.column_stack([np.ones(len(X1)), Z, Z**2, Z**3])

p = X.shape[1] - 1
n = len(Y)
SST = np.sum((Y - np.mean(Y))**2)
P = np.diag([0, 1, 1, 1])

L_vals = [0, 5000, 100000, 400000, 900000]

print("Comparing different p-value calculations:")
print("=" * 80)

for L in L_vals:
    XtX = X.T @ X
    beta_ridge = np.linalg.solve(XtX + L * P, X.T @ Y)
    Y_hat = X @ beta_ridge
    res = Y - Y_hat

    SSE = np.sum(res**2)
    SSR = SST - SSE

    # Method 1: Traditional F-test (what we've been using)
    F_stat_trad = (SSR / p) / (SSE / (n - p - 1))
    p_trad = 1 - f.cdf(F_stat_trad, p, n - p - 1)

    # Method 2: Test against intercept-only model
    Y_mean = np.mean(Y)
    SSE_null = np.sum((Y - Y_mean)**2)
    F_stat_alt = ((SSE_null - SSE) / p) / (SSE / (n - p - 1))
    p_alt = 1 - f.cdf(F_stat_alt, p, n - p - 1)

    # Method 3: Likelihood ratio test (chi-square)
    LR = n * np.log(SSE_null / SSE)
    p_lr = 1 - chi2.cdf(LR, p)

    print(f"\nL = {L}:")
    print(f"  Method 1 (current): {p_trad:.4f}")
    print(f"  Method 2 (vs null): {p_alt:.4f}")
    print(f"  Method 3 (LR test): {p_lr:.4f}")

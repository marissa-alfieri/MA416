import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Read the data
data = pd.read_csv("HousingData_9.25.csv")

# Extract variables
Y = data['price_usd'].values
X1 = data['living_space_ft2'].values
X2 = data['home_age'].values
X3 = data['distance_city_mi'].values
X4 = data['crime_rate_index'].values

# Create design matrix
Xk = np.column_stack([X1, X2, X3, X4])
X = np.column_stack([np.ones(len(Y)), Xk])

n = len(Y)
p = Xk.shape[1]
SST = np.sum((Y - np.mean(Y))**2)

print("Testing different penalty approaches:")
print("=" * 80)

# Function to optimize - NOT penalizing intercept
def optimize_model_no_intercept_penalty(method, lam=0):
    """
    Optimize coefficients WITHOUT penalizing the intercept
    """
    def objective(beta):
        res = Y - X @ beta
        beta_no_intercept = beta[1:]  # Exclude intercept from penalty

        if method == 'ols':
            return np.sum(res**2)
        elif method == 'ridge':
            return np.sum(res**2) + lam * np.sum(beta_no_intercept**2)
        elif method == 'lad':
            return np.sum(np.abs(res))
        elif method == 'lasso':
            return np.sum(res**2) + lam * np.sum(np.abs(beta_no_intercept))
        elif method == 'ladlasso':
            return np.sum(np.abs(res)) + lam * np.sum(np.abs(beta_no_intercept))

    beta0 = np.zeros(p + 1)
    result = minimize(objective, beta0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    return result.x

# Also try matching R exactly - penalizing ALL coefficients
def optimize_model_all_penalty(method, lam=0):
    """
    Optimize coefficients penalizing ALL (including intercept) - matches R code
    """
    def objective(beta):
        res = Y - X @ beta

        if method == 'ols':
            return np.sum(res**2)
        elif method == 'ridge':
            return np.sum(res**2) + lam * np.sum(beta**2)  # Penalize ALL
        elif method == 'lad':
            return np.sum(np.abs(res))
        elif method == 'lasso':
            return np.sum(res**2) + lam * np.sum(np.abs(beta))  # Penalize ALL
        elif method == 'ladlasso':
            return np.sum(np.abs(res)) + lam * np.sum(np.abs(beta))  # Penalize ALL

    beta0 = np.zeros(p + 1)
    result = minimize(objective, beta0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    return result.x

models = [
    ('Least-Squares', 'ols', 0),
    ('Ridge', 'ridge', 15),
    ('LAD', 'lad', 0),
    ('Lasso', 'lasso', 15),
    ('LAD-Lasso', 'ladlasso', 15)
]

print("\nApproach 1: NOT penalizing intercept")
print("-" * 80)
for model_name, method, lam in models:
    beta = optimize_model_no_intercept_penalty(method, lam)
    Y_hat = X @ beta
    res = Y - Y_hat
    SSE = np.sum(res**2)
    RMSE = np.sqrt(SSE / n)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    print(f"{model_name:15} | RMSE: {RMSE:10.2f} | F-stat: {F_stat:10.2f}")

print("\nApproach 2: Penalizing ALL coefficients (including intercept) - matches R")
print("-" * 80)
for model_name, method, lam in models:
    beta = optimize_model_all_penalty(method, lam)
    Y_hat = X @ beta
    res = Y - Y_hat
    SSE = np.sum(res**2)
    RMSE = np.sqrt(SSE / n)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))
    print(f"{model_name:15} | RMSE: {RMSE:10.2f} | F-stat: {F_stat:10.2f}")

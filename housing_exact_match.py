import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Match R's seed
np.random.seed(1234)

data = pd.read_csv("HousingData_9.25.csv")

Y = data['price_usd'].values
X1 = data['living_space_ft2'].values
X2 = data['home_age'].values
X3 = data['distance_city_mi'].values
X4 = data['crime_rate_index'].values

Xk = np.column_stack([X1, X2, X3, X4])
X = np.column_stack([np.ones(len(Y)), Xk])

n = len(Y)
p = Xk.shape[1]
SST = np.sum((Y - np.mean(Y))**2)

def myladlasso_exact(method, lam=0):
    """
    Exactly match professor's function - penalizes ALL coefficients
    """
    def objective(bhat):
        res = Y - (X @ bhat)

        if method == 'ols':
            return np.sum(res**2)
        elif method == 'ridge':
            return np.sum(res**2) + lam * np.sum(bhat**2)  # Penalize ALL
        elif method == 'lad':
            return np.sum(np.abs(res))
        elif method == 'lasso':
            return np.sum(res**2) + lam * np.sum(np.abs(bhat))  # Penalize ALL
        elif method == 'ladlasso':
            return np.sum(np.abs(res)) + lam * np.sum(np.abs(bhat))  # Penalize ALL

    bhat0 = np.zeros(p + 1)

    # Try to match R's optim with Nelder-Mead
    result = minimize(objective, bhat0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True, 'maxiter': 100000})

    return result.x

print("Matching Professor's myladlasso Function (penalizes ALL coefficients)")
print("=" * 80)

models = [
    ('Least-Squares', 'ols', 0),
    ('Ridge', 'ridge', 15),
    ('LAD', 'lad', 0),
    ('Lasso', 'lasso', 15),
    ('LAD-Lasso', 'ladlasso', 15)
]

for model_name, method, lam in models:
    bhat = myladlasso_exact(method, lam)
    Yhat = X @ bhat
    res = Y - Yhat

    SSE = np.sum(res**2)
    RMSE = np.sqrt(SSE / n)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))

    print(f"{model_name:15} | RMSE: {RMSE:>10.2f} | F-stat: {F_stat:>10.2f}")
    print(f"  Coefficients: {bhat}")
    print(f"  Convergence: {result.success}, Iterations: {result.nit}")
    print()

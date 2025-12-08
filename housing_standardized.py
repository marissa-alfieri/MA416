import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(1234)

data = pd.read_csv("HousingData_9.25.csv")

Y = data['price_usd'].values
X1 = data['living_space_ft2'].values
X2 = data['home_age'].values
X3 = data['distance_city_mi'].values
X4 = data['crime_rate_index'].values

# STANDARDIZE predictors (mean=0, std=1)
X1_scaled = (X1 - np.mean(X1)) / np.std(X1, ddof=1)
X2_scaled = (X2 - np.mean(X2)) / np.std(X2, ddof=1)
X3_scaled = (X3 - np.mean(X3)) / np.std(X3, ddof=1)
X4_scaled = (X4 - np.mean(X4)) / np.std(X4, ddof=1)

Xk_scaled = np.column_stack([X1_scaled, X2_scaled, X3_scaled, X4_scaled])
X_scaled = np.column_stack([np.ones(len(Y)), Xk_scaled])

n = len(Y)
p = Xk_scaled.shape[1]
SST = np.sum((Y - np.mean(Y))**2)

print("Housing Regression with STANDARDIZED Predictors")
print("=" * 80)

def myladlasso(method, lam=0):
    def objective(bhat):
        res = Y - (X_scaled @ bhat)

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
    result = minimize(objective, bhat0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100000})
    return result.x

models = [
    ('Least-Squares', 'ols', 0),
    ('Ridge', 'ridge', 15),
    ('LAD', 'lad', 0),
    ('Lasso', 'lasso', 15),
    ('LAD-Lasso', 'ladlasso', 15)
]

results = []

for model_name, method, lam in models:
    bhat = myladlasso(method, lam)
    Yhat = X_scaled @ bhat
    res = Y - Yhat

    SSE = np.sum(res**2)
    RMSE = np.sqrt(SSE / n)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))

    results.append({
        'Model': model_name,
        'RMSE': round(RMSE, 2),
        'F_Statistic': round(F_stat, 2)
    })

    print(f"{model_name:15} | RMSE: {RMSE:>10.2f} | F-stat: {F_stat:>10.2f}")

print("\n" + "=" * 80)
print("\nFinal Results:")
df = pd.DataFrame(results)
print(df.to_string(index=False))

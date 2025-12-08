import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Set seed for reproducibility
np.random.seed(1234)

# Read data
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

def myladlasso(method, lam=0):
    """
    Match R's myladlasso function - NOT penalizing intercept
    """
    def objective(bhat):
        res = Y - (X @ bhat)
        bhat_no_intercept = bhat[1:]  # Exclude intercept from penalty

        if method == 'ols':
            return np.sum(res**2)
        elif method == 'ridge':
            return np.sum(res**2) + lam * np.sum(bhat_no_intercept**2)
        elif method == 'lad':
            return np.sum(np.abs(res))
        elif method == 'lasso':
            return np.sum(res**2) + lam * np.sum(np.abs(bhat_no_intercept))
        elif method == 'ladlasso':
            return np.sum(np.abs(res)) + lam * np.sum(np.abs(bhat_no_intercept))

    # Start from zeros like R
    bhat0 = np.zeros(p + 1)

    # Use same method as R with tight tolerance
    result = minimize(objective, bhat0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 50000})

    return result.x

print("Housing Regression Models - Fixed Version")
print("=" * 80)
print("(Not penalizing intercept - standard practice)")
print()

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
    Yhat = X @ bhat
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
print("\nFinal Results Table:")
df = pd.DataFrame(results)
print(df.to_string(index=False))

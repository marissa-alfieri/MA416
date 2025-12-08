import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import f as f_dist

# Read the data
data = pd.read_csv("HousingData_9.25.csv")

# Extract variables (predictors and response)
Y = data['price_usd'].values
X1 = data['living_space_ft2'].values
X2 = data['home_age'].values
X3 = data['distance_city_mi'].values
X4 = data['crime_rate_index'].values

# Create design matrix
Xk = np.column_stack([X1, X2, X3, X4])
X = np.column_stack([np.ones(len(Y)), Xk])  # Add intercept

n = len(Y)
p = Xk.shape[1]  # number of predictors (excluding intercept)

# Calculate SST
SST = np.sum((Y - np.mean(Y))**2)

print("Housing Price Regression Models Comparison")
print("=" * 80)

# Function to optimize different models
def optimize_model(method, lam=0):
    """
    Optimize coefficients for different regression methods
    method: 'ols', 'ridge', 'lad', 'lasso', 'ladlasso'
    lam: penalty parameter (lambda)
    """
    def objective(beta):
        res = Y - X @ beta

        if method == 'ols':
            return np.sum(res**2)
        elif method == 'ridge':
            return np.sum(res**2) + lam * np.sum(beta**2)
        elif method == 'lad':
            return np.sum(np.abs(res))
        elif method == 'lasso':
            return np.sum(res**2) + lam * np.sum(np.abs(beta))
        elif method == 'ladlasso':
            return np.sum(np.abs(res)) + lam * np.sum(np.abs(beta))

    # Initial guess
    beta0 = np.zeros(p + 1)

    # Optimize
    result = minimize(objective, beta0, method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-10})

    return result.x

# Calculate metrics for each model
models = [
    ('Least-Squares', 'ols', 0),
    ('Ridge', 'ridge', 15),
    ('LAD', 'lad', 0),
    ('Lasso', 'lasso', 15),
    ('LAD-Lasso', 'ladlasso', 15)
]

results = []

for model_name, method, lam in models:
    print(f"\n{model_name} (λ={lam}):")

    # Get optimized coefficients
    beta = optimize_model(method, lam)

    # Predictions
    Y_hat = X @ beta
    res = Y - Y_hat

    # Calculate metrics
    SSE = np.sum(res**2)
    RMSE = np.sqrt(SSE / n)
    SSR = SST - SSE
    F_stat = (SSR / p) / (SSE / (n - p - 1))

    print(f"  Coefficients: {beta}")
    print(f"  SSE: {SSE:.2f}")
    print(f"  SSR: {SSR:.2f}")
    print(f"  RMSE: {RMSE:.2f}")
    print(f"  F-statistic: {F_stat:.2f}")

    results.append({
        'Model': model_name,
        'RMSE': round(RMSE, 2),
        'F_Statistic': round(F_stat, 2)
    })

print("\n" + "=" * 80)
print("\nFINAL RESULTS (rounded to 2 decimal places):")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n\nFormatted for Canvas:")
print("-" * 80)
for r in results:
    print(f"{r['Model']:15} | RMSE: {r['RMSE']:>10} | F-stat: {r['F_Statistic']:>10}")

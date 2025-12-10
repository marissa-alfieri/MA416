import csv
import math

# Read the data
stock_prices = []
with open("StockMarketB_9.25.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        stock_prices.append(float(row[1]))

y = stock_prices
n = len(y)
print(f"Dataset has {n} observations\n")

# ============================================================================
# Question 8: Piecewise linear regression subsets
# ============================================================================
print("="*70)
print("Question 8: Piecewise Linear Regression Subsets")
print("="*70)
print("Based on visual inspection of the plot, looking for distinct linear trends:")
print("Answer: 5 subsets")
print()

# ============================================================================
# Question 9: Polynomial Regression
# ============================================================================
print("="*70)
print("Question 9: Polynomial Regression")
print("="*70)

# Create time indices
T = list(range(1, n + 1))

# Standardize time predictor
T_mean = sum(T) / len(T)
T_var = sum((t - T_mean)**2 for t in T) / (len(T) - 1)
T_std = math.sqrt(T_var)
T_standardized = [(t - T_mean) / T_std for t in T]

def fit_polynomial(x, y, degree):
    """Fit polynomial using least squares - manual implementation"""
    n_obs = len(x)

    # Create design matrix X (include intercept)
    X = []
    for i in range(n_obs):
        row = [1.0] + [x[i]**deg for deg in range(1, degree + 1)]
        X.append(row)

    # Solve normal equations: X'X * beta = X'y
    # Using Gaussian elimination
    k = degree + 1  # Include intercept

    # Build X'X
    XtX = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            for row in range(n_obs):
                XtX[i][j] += X[row][i] * X[row][j]

    # Build X'y
    Xty = [0.0 for _ in range(k)]
    for i in range(k):
        for row in range(n_obs):
            Xty[i] += X[row][i] * y[row]

    # Solve using Gaussian elimination
    # Forward elimination
    for i in range(k):
        # Find pivot
        max_row = i
        for row in range(i + 1, k):
            if abs(XtX[row][i]) > abs(XtX[max_row][i]):
                max_row = row

        XtX[i], XtX[max_row] = XtX[max_row], XtX[i]
        Xty[i], Xty[max_row] = Xty[max_row], Xty[i]

        # Eliminate column
        for row in range(i + 1, k):
            if XtX[i][i] != 0:
                factor = XtX[row][i] / XtX[i][i]
                for col in range(i, k):
                    XtX[row][col] -= factor * XtX[i][col]
                Xty[row] -= factor * Xty[i]

    # Back substitution
    coeffs = [0.0 for _ in range(k)]
    for i in range(k - 1, -1, -1):
        coeffs[i] = Xty[i]
        for j in range(i + 1, k):
            coeffs[i] -= XtX[i][j] * coeffs[j]
        if XtX[i][i] != 0:
            coeffs[i] /= XtX[i][i]

    # Make predictions
    y_pred = []
    for row in range(n_obs):
        pred = sum(coeffs[j] * X[row][j] for j in range(k))
        y_pred.append(pred)

    return coeffs, y_pred

# Try different degrees
print("Testing different polynomial degrees:")
for deg in range(2, 7):
    coeffs, y_pred = fit_polynomial(T_standardized, y, deg)

    # Calculate R-squared
    y_mean = sum(y) / len(y)
    ss_tot = sum((y[i] - y_mean)**2 for i in range(n))
    ss_res = sum((y[i] - y_pred[i])**2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Degree {deg} - R-squared: {r_squared:.6f}")

# Use degree 5 (minimum degree necessary - big improvement from 4 to 5)
deg = 5
coeffs, y_pred = fit_polynomial(T_standardized, y, deg)

# a) Calculate SSE
SSE = sum((y[i] - y_pred[i])**2 for i in range(n))
print(f"\nQuestion 9a - SSE: {SSE:.4f}")

# b) Calculate RMSE
MSE = SSE / n
RMSE = math.sqrt(MSE)
print(f"Question 9b - RMSE: {RMSE:.4f}")
print()

# ============================================================================
# Question 10: MA{5} Weights
# ============================================================================
print("="*70)
print("Question 10: MA{5} Weights")
print("="*70)
print("For MA{5}, we use 5 points centered around the target.")
print("Each point gets equal weight.")
print("Weight for each point (w_j): 1/5")
print()

# ============================================================================
# Question 11: MA{5} Approximations
# ============================================================================
print("="*70)
print("Question 11: MA{5} Approximations")
print("="*70)

def ma5_approx(data, t):
    """Calculate MA{5} approximation at time t (1-indexed)"""
    if t < 3 or t > len(data) - 2:
        return None
    # t is 1-indexed, convert to 0-indexed
    idx = t - 1
    return sum(data[idx-2:idx+3]) / 5.0

# a) t = 7
yma_7 = ma5_approx(y, 7)
print(f"Question 11a - Yma at t=7: {yma_7:.4f}")

# b) t = 23
yma_23 = ma5_approx(y, 23)
print(f"Question 11b - Yma at t=23: {yma_23:.4f}")

# c) t = 79
yma_79 = ma5_approx(y, 79)
print(f"Question 11c - Yma at t=79: {yma_79:.4f}")
print()

# ============================================================================
# Question 12: EMA{3, 0.32} Weights at t=35
# ============================================================================
print("="*70)
print("Question 12: EMA{3, 0.32} Weights at t=35")
print("="*70)

a = 0.32
w = 3

# Calculate unnormalized weights
w0_unnorm = a
w1_unnorm = (1 - a) * w0_unnorm
w2_unnorm = (1 - a) * w1_unnorm
w3_unnorm = (1 - a) * w2_unnorm

# Normalize
sum_weights = w0_unnorm + w1_unnorm + w2_unnorm + w3_unnorm
w0_ema = w0_unnorm / sum_weights
w1_ema = w1_unnorm / sum_weights
w2_ema = w2_unnorm / sum_weights
w3_ema = w3_unnorm / sum_weights

print(f"Question 12a - w0 (at t=35): {w0_ema:.4f}")
print(f"Question 12b - w1 (at t=34): {w1_ema:.4f}")
print(f"Question 12c - w2 (at t=33): {w2_ema:.4f}")
print(f"Question 12d - w3 (at t=32): {w3_ema:.4f}")
print(f"Sum of weights (verification): {w0_ema + w1_ema + w2_ema + w3_ema:.6f}")
print()

# ============================================================================
# Question 13: EMA{3, 0.32} Approximation at t=35
# ============================================================================
print("="*70)
print("Question 13: EMA{3, 0.32} Approximation at t=35")
print("="*70)

t_target = 35  # 1-indexed
# Convert to 0-indexed for array access
idx = t_target - 1

ema_approx = (w0_ema * y[idx] +
              w1_ema * y[idx - 1] +
              w2_ema * y[idx - 2] +
              w3_ema * y[idx - 3])

print(f"Question 13 - EMA{{3, 0.32}} approximation at t=35: {ema_approx:.4f}")
print()

print("="*70)
print("SUMMARY OF ALL ANSWERS")
print("="*70)
print(f"Q8:  Number of piecewise linear subsets: 5")
print(f"Q9a: SSE = {SSE:.4f}")
print(f"Q9b: RMSE = {RMSE:.4f}")
print(f"Q10: MA{{5}} weight = 1/5")
print(f"Q11a: Yma at t=7 = {yma_7:.4f}")
print(f"Q11b: Yma at t=23 = {yma_23:.4f}")
print(f"Q11c: Yma at t=79 = {yma_79:.4f}")
print(f"Q12a: w0 = {w0_ema:.4f}")
print(f"Q12b: w1 = {w1_ema:.4f}")
print(f"Q12c: w2 = {w2_ema:.4f}")
print(f"Q12d: w3 = {w3_ema:.4f}")
print(f"Q13:  EMA approximation at t=35 = {ema_approx:.4f}")

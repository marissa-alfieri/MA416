#!/usr/bin/env python3
"""
M08.1 - Binary Logistic Regression using IRLS
"""

import csv
import math

# Load data
data = []
with open('HumanZombie_9.25.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        y = 1 if row['ZombieOrHuman'] == 'zombie' else 0
        speed = float(row['Speed'])
        awareness = float(row['AwarenessLevel'])
        data.append([1.0, speed, awareness, y])  # 1.0 for intercept

n = len(data)
print(f"Loaded {n} observations")

# Matrix operations
def transpose(matrix):
    """Transpose a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_mult(A, B):
    """Multiply two matrices"""
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_vector_mult(A, v):
    """Multiply matrix by vector"""
    result = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j] * v[j]
    return result

def sigmoid(z):
    """Sigmoid function with numerical stability"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

def matrix_inverse_3x3(matrix):
    """Invert a 3x3 matrix"""
    a = matrix[0][0]; b = matrix[0][1]; c = matrix[0][2]
    d = matrix[1][0]; e = matrix[1][1]; f = matrix[1][2]
    g = matrix[2][0]; h = matrix[2][1]; i = matrix[2][2]

    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular")

    inv = [
        [(e*i - f*h)/det, (c*h - b*i)/det, (b*f - c*e)/det],
        [(f*g - d*i)/det, (a*i - c*g)/det, (c*d - a*f)/det],
        [(d*h - e*g)/det, (b*g - a*h)/det, (a*e - b*d)/det]
    ]
    return inv

# IRLS algorithm
beta = [0.0, 0.0, 0.0]  # Initialize coefficients

for iteration in range(20):  # IRLS typically converges in < 10 iterations
    # Calculate predicted probabilities
    probabilities = []
    for row in data:
        z = beta[0] * row[0] + beta[1] * row[1] + beta[2] * row[2]
        p = sigmoid(z)
        probabilities.append(p)

    # Build weighted design matrix and response
    X = [[row[0], row[1], row[2]] for row in data]
    X_T = transpose(X)

    # Build diagonal weight matrix W and adjusted response z
    z_values = []
    weights = []
    for i in range(n):
        w = probabilities[i] * (1 - probabilities[i])
        w = max(w, 1e-10)  # Avoid division by zero
        weights.append(w)

        # Working response
        eta = beta[0] * data[i][0] + beta[1] * data[i][1] + beta[2] * data[i][2]
        z_val = eta + (data[i][3] - probabilities[i]) / w
        z_values.append(z_val)

    # Calculate X^T W X
    XTW = [[0 for _ in range(n)] for _ in range(3)]
    for i in range(3):
        for j in range(n):
            XTW[i][j] = X_T[i][j] * weights[j]

    XTWX = matrix_mult(XTW, X)

    # Calculate X^T W z
    XTWz = [0, 0, 0]
    for i in range(3):
        for j in range(n):
            XTWz[i] += XTW[i][j] * z_values[j]

    # Solve for new beta: beta = (X^T W X)^{-1} X^T W z
    try:
        XTWX_inv = matrix_inverse_3x3(XTWX)
        beta_new = matrix_vector_mult(XTWX_inv, XTWz)

        # Check convergence
        diff = sum((beta_new[i] - beta[i])**2 for i in range(3))
        beta = beta_new

        if diff < 1e-8:
            print(f"Converged after {iteration + 1} iterations")
            break
    except ValueError:
        print("Warning: Matrix inversion failed")
        break

print("\n=== Question 5: Coefficients ===")
print(f"a) Intercept = {beta[0]:.4f}")
print(f"b) Gradient of log-odds for speed = {beta[1]:.4f}")
print(f"c) Gradient of log-odds for awareness level = {beta[2]:.4f}")

# Question 6: Log-likelihood
ll = 0.0
for row in data:
    z = beta[0] * row[0] + beta[1] * row[1] + beta[2] * row[2]
    p = sigmoid(z)
    p = max(min(p, 1 - 1e-15), 1e-15)
    ll += row[3] * math.log(p) + (1 - row[3]) * math.log(1 - p)

print("\n=== Question 6: Log-Likelihood ===")
print(f"Log-likelihood = {ll:.4f}")

# Question 7: RMSE
squared_errors = []
for row in data:
    z = beta[0] * row[0] + beta[1] * row[1] + beta[2] * row[2]
    p = sigmoid(z)
    squared_errors.append((row[3] - p) ** 2)
rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

print("\n=== Question 7: RMSE ===")
print(f"RMSE = {rmse:.4f}")

# Question 8: McFadden pseudo R²
# Null model: just the proportion of zombies
prop_zombies = sum(row[3] for row in data) / n
ll_null = 0.0
p_null = max(min(prop_zombies, 1 - 1e-15), 1e-15)
for row in data:
    ll_null += row[3] * math.log(p_null) + (1 - row[3]) * math.log(1 - p_null)

mcfadden_r2 = 1 - (ll / ll_null)

print("\n=== Question 8: McFadden Pseudo R² ===")
print(f"McFadden R² = {mcfadden_r2:.4f}")

# Question 9: Predicted probabilities
def predict(beta, speed, awareness):
    z = beta[0] + beta[1] * speed + beta[2] * awareness
    return sigmoid(z)

pred_a = predict(beta, 70, 33)
pred_b = predict(beta, 16, 16)

print("\n=== Question 9: Predicted Probabilities ===")
print(f"a) Speed = 70, Awareness Level = 33, Probability = {pred_a:.4f}")
print(f"b) Speed = 16, Awareness Level = 16, Probability = {pred_b:.4f}")

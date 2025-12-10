import csv

# Read data
stock_prices = []
with open("StockMarketB_9.25.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        stock_prices.append(float(row[1]))

y = stock_prices
a = 0.32
w = 3
t = 35

print("="*70)
print("CORRECT EMA WEIGHTS (Standard Formula)")
print("="*70)

print("\nFormula: w_j = a × (1-a)^j, then normalize")
print(f"where a = {a}, so (1-a) = {1-a}")
print()

# Calculate weights using standard EMA formula
w0_unnorm = a * (1 - a)**0  # j=0
w1_unnorm = a * (1 - a)**1  # j=1
w2_unnorm = a * (1 - a)**2  # j=2
w3_unnorm = a * (1 - a)**3  # j=3

print("Step 1 - Unnormalized weights:")
print(f"w0 = {a} × {(1-a)**0} = {w0_unnorm:.6f}")
print(f"w1 = {a} × {(1-a)**1:.4f} = {w1_unnorm:.6f}")
print(f"w2 = {a} × {(1-a)**2:.4f} = {w2_unnorm:.6f}")
print(f"w3 = {a} × {(1-a)**3:.4f} = {w3_unnorm:.6f}")

sum_weights = w0_unnorm + w1_unnorm + w2_unnorm + w3_unnorm
print(f"\nSum = {sum_weights:.6f}")

print("\nStep 2 - Normalize (divide by sum):")
w0 = w0_unnorm / sum_weights
w1 = w1_unnorm / sum_weights
w2 = w2_unnorm / sum_weights
w3 = w3_unnorm / sum_weights

print(f"w0 = {w0_unnorm:.6f} / {sum_weights:.6f} = {w0:.4f}")
print(f"w1 = {w1_unnorm:.6f} / {sum_weights:.6f} = {w1:.4f}")
print(f"w2 = {w2_unnorm:.6f} / {sum_weights:.6f} = {w2:.4f}")
print(f"w3 = {w3_unnorm:.6f} / {sum_weights:.6f} = {w3:.4f}")
print(f"\nVerify sum = {w0+w1+w2+w3:.6f}")

print("\n" + "="*70)
print("CORRECT ANSWERS FOR QUESTION 12:")
print("="*70)
print(f"a) t = 35, w0 = {w0:.4f}")
print(f"b) t = 34, w1 = {w1:.4f}")
print(f"c) t = 33, w2 = {w2:.4f}")
print(f"d) t = 32, w3 = {w3:.4f}")

# Calculate EMA approximation
idx = t - 1
ema_value = w0 * y[idx] + w1 * y[idx-1] + w2 * y[idx-2] + w3 * y[idx-3]

print("\n" + "="*70)
print("QUESTION 13 - EMA APPROXIMATION:")
print("="*70)
print(f"EMA at t=35 = {w0:.4f}×{y[idx]:.4f} + {w1:.4f}×{y[idx-1]:.4f}")
print(f"            + {w2:.4f}×{y[idx-2]:.4f} + {w3:.4f}×{y[idx-3]:.4f}")
print(f"            = {ema_value:.4f}")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print("What you submitted (a^j normalized):      What should be submitted (a×(1-a)^j normalized):")
print(f"w0 = 0.6871                               w0 = {w0:.4f}")
print(f"w1 = 0.2199                               w1 = {w1:.4f}")
print(f"w2 = 0.0704                               w2 = {w2:.4f}")
print(f"w3 = 0.0226                               w3 = {w3:.4f}")

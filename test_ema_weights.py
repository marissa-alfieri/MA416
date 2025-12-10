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
print("EMA{3, 0.32} Weight Interpretations")
print("="*70)

print("\nInterpretation 1: w_j = a*(1-a)^j (NORMALIZED)")
print("This is what was submitted:")
w0 = 0.32
w1 = 0.32 * 0.68
w2 = 0.32 * 0.68**2
w3 = 0.32 * 0.68**3
sum_w = w0 + w1 + w2 + w3
print(f"w0 = {w0/sum_w:.4f}")
print(f"w1 = {w1/sum_w:.4f}")
print(f"w2 = {w2/sum_w:.4f}")
print(f"w3 = {w3/sum_w:.4f}")

print("\nInterpretation 2: w_j = a*(1-a)^j (UNNORMALIZED)")
print(f"w0 = {w0:.4f}")
print(f"w1 = {w1:.4f}")
print(f"w2 = {w2:.4f}")
print(f"w3 = {w3:.4f}")

print("\nInterpretation 3: w_j = a^j")
w0_v3 = a**0
w1_v3 = a**1
w2_v3 = a**2
w3_v3 = a**3
print(f"w0 = {w0_v3:.4f}")
print(f"w1 = {w1_v3:.4f}")
print(f"w2 = {w2_v3:.4f}")
print(f"w3 = {w3_v3:.4f}")

print("\nInterpretation 4: w_j = a^j (NORMALIZED)")
sum_v3 = w0_v3 + w1_v3 + w2_v3 + w3_v3
print(f"w0 = {w0_v3/sum_v3:.4f}")
print(f"w1 = {w1_v3/sum_v3:.4f}")
print(f"w2 = {w2_v3/sum_v3:.4f}")
print(f"w3 = {w3_v3/sum_v3:.4f}")

print("\nInterpretation 5: w_j = (1-a)^j")
w0_v5 = (1-a)**0
w1_v5 = (1-a)**1
w2_v5 = (1-a)**2
w3_v5 = (1-a)**3
print(f"w0 = {w0_v5:.4f}")
print(f"w1 = {w1_v5:.4f}")
print(f"w2 = {w2_v5:.4f}")
print(f"w3 = {w3_v5:.4f}")

print("\nInterpretation 6: w_j = (1-a)^j (NORMALIZED)")
sum_v5 = w0_v5 + w1_v5 + w2_v5 + w3_v5
print(f"w0 = {w0_v5/sum_v5:.4f}")
print(f"w1 = {w1_v5/sum_v5:.4f}")
print(f"w2 = {w2_v5/sum_v5:.4f}")
print(f"w3 = {w3_v5/sum_v5:.4f}")

# Calculate EMA approximations for each interpretation
print("\n" + "="*70)
print("Corresponding EMA approximations at t=35:")
print("="*70)

idx = t - 1

# Current (normalized a*(1-a)^j)
w0_n = 0.4070
w1_n = 0.2768
w2_n = 0.1882
w3_n = 0.1280
ema_current = w0_n * y[idx] + w1_n * y[idx-1] + w2_n * y[idx-2] + w3_n * y[idx-3]
print(f"Current submitted (normalized a*(1-a)^j): {ema_current:.4f}")

# Unnormalized a*(1-a)^j
ema_unnorm = w0 * y[idx] + w1 * y[idx-1] + w2 * y[idx-2] + w3 * y[idx-3]
print(f"Unnormalized a*(1-a)^j: {ema_unnorm:.4f}")

# a^j
ema_v3 = w0_v3 * y[idx] + w1_v3 * y[idx-1] + w2_v3 * y[idx-2] + w3_v3 * y[idx-3]
print(f"a^j: {ema_v3:.4f}")

# a^j normalized
ema_v3_norm = (w0_v3/sum_v3) * y[idx] + (w1_v3/sum_v3) * y[idx-1] + (w2_v3/sum_v3) * y[idx-2] + (w3_v3/sum_v3) * y[idx-3]
print(f"a^j (normalized): {ema_v3_norm:.4f}")

# (1-a)^j
ema_v5 = w0_v5 * y[idx] + w1_v5 * y[idx-1] + w2_v5 * y[idx-2] + w3_v5 * y[idx-3]
print(f"(1-a)^j: {ema_v5:.4f}")

# (1-a)^j normalized
ema_v5_norm = (w0_v5/sum_v5) * y[idx] + (w1_v5/sum_v5) * y[idx-1] + (w2_v5/sum_v5) * y[idx-2] + (w3_v5/sum_v5) * y[idx-3]
print(f"(1-a)^j (normalized): {ema_v5_norm:.4f}")

print("\n" + "="*70)
print("Data values used:")
print(f"y[35] = {y[idx]:.4f}")
print(f"y[34] = {y[idx-1]:.4f}")
print(f"y[33] = {y[idx-2]:.4f}")
print(f"y[32] = {y[idx-3]:.4f}")

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

print("="*70)
print("EXPLORING DIFFERENT INTERPRETATIONS")
print("="*70)

# ============================================================================
# MA{5} Exploration
# ============================================================================
print("\n" + "="*70)
print("MA{5} Model - Different Interpretations")
print("="*70)

print("\nInterpretation 1: MA{5} uses 5 points (centered)")
print("Points: t-2, t-1, t, t+1, t+2")
print("Weight: 1/5")

def ma5_v1(data, t):
    """MA{5} with 5 points centered"""
    idx = t - 1
    return sum(data[idx-2:idx+3]) / 5.0

print(f"t=7:  {ma5_v1(y, 7):.4f}")
print(f"t=23: {ma5_v1(y, 23):.4f}")
print(f"t=79: {ma5_v1(y, 79):.4f}")

print("\nInterpretation 2: MA{5} uses 6 points (w+1) - backward looking")
print("Points: t, t-1, t-2, t-3, t-4, t-5")
print("Weight: 1/6")

def ma5_v2(data, t):
    """MA{5} with 6 points, backward looking"""
    idx = t - 1
    return sum(data[idx-5:idx+1]) / 6.0

print(f"t=7:  {ma5_v2(y, 7):.4f}")
print(f"t=23: {ma5_v2(y, 23):.4f}")
print(f"t=79: {ma5_v2(y, 79):.4f}")

print("\nInterpretation 3: MA{5} uses 6 points (w+1) - forward looking")
print("Points: t, t+1, t+2, t+3, t+4, t+5")
print("Weight: 1/6")

def ma5_v3(data, t):
    """MA{5} with 6 points, forward looking"""
    idx = t - 1
    if idx + 6 > len(data):
        return None
    return sum(data[idx:idx+6]) / 6.0

print(f"t=7:  {ma5_v3(y, 7):.4f}")
print(f"t=23: {ma5_v3(y, 23):.4f}")
print(f"t=79: {ma5_v3(y, 79):.4f}")

print("\nInterpretation 4: MA{5} uses 6 points (w+1) - centered (t-2 to t+3)")
print("Points: t-2, t-1, t, t+1, t+2, t+3")
print("Weight: 1/6")

def ma5_v4(data, t):
    """MA{5} with 6 points, centered at t-0.5"""
    idx = t - 1
    if idx + 4 > len(data):
        return None
    return sum(data[idx-2:idx+4]) / 6.0

print(f"t=7:  {ma5_v4(y, 7):.4f}")
print(f"t=23: {ma5_v4(y, 23):.4f}")
print(f"t=79: {ma5_v4(y, 79):.4f}")

print("\nInterpretation 5: MA{5} uses 6 points (w+1) - centered (t-3 to t+2)")
print("Points: t-3, t-2, t-1, t, t+1, t+2")
print("Weight: 1/6")

def ma5_v5(data, t):
    """MA{5} with 6 points, centered at t+0.5"""
    idx = t - 1
    return sum(data[idx-3:idx+3]) / 6.0

print(f"t=7:  {ma5_v5(y, 7):.4f}")
print(f"t=23: {ma5_v5(y, 23):.4f}")
print(f"t=79: {ma5_v5(y, 79):.4f}")

# ============================================================================
# EMA Exploration
# ============================================================================
print("\n" + "="*70)
print("EMA{3, 0.32} - Different Interpretations")
print("="*70)

a = 0.32

print("\nInterpretation 1: Standard exponential weights (unnormalized)")
print("w_j = a * (1-a)^j")

w0 = a
w1 = a * (1 - a)
w2 = a * (1 - a)**2
w3 = a * (1 - a)**3

print(f"w0 = {w0:.4f}")
print(f"w1 = {w1:.4f}")
print(f"w2 = {w2:.4f}")
print(f"w3 = {w3:.4f}")
print(f"Sum = {w0+w1+w2+w3:.4f}")

t = 35
idx = t - 1
ema1 = w0*y[idx] + w1*y[idx-1] + w2*y[idx-2] + w3*y[idx-3]
print(f"EMA at t=35: {ema1:.4f}")

print("\nInterpretation 2: Exponential weights (normalized)")
sum_w = w0 + w1 + w2 + w3
w0_norm = w0 / sum_w
w1_norm = w1 / sum_w
w2_norm = w2 / sum_w
w3_norm = w3 / sum_w

print(f"w0 = {w0_norm:.4f}")
print(f"w1 = {w1_norm:.4f}")
print(f"w2 = {w2_norm:.4f}")
print(f"w3 = {w3_norm:.4f}")
print(f"Sum = {w0_norm+w1_norm+w2_norm+w3_norm:.4f}")

ema2 = w0_norm*y[idx] + w1_norm*y[idx-1] + w2_norm*y[idx-2] + w3_norm*y[idx-3]
print(f"EMA at t=35: {ema2:.4f}")

print("\nInterpretation 3: Different weight formula w_j = (1-a)^j * a")
w0_v3 = a
w1_v3 = (1-a) * a
w2_v3 = (1-a)**2 * a
w3_v3 = (1-a)**3 * a

print(f"w0 = {w0_v3:.4f}")
print(f"w1 = {w1_v3:.4f}")
print(f"w2 = {w2_v3:.4f}")
print(f"w3 = {w3_v3:.4f}")
print(f"Sum = {w0_v3+w1_v3+w2_v3+w3_v3:.4f}")

ema3 = w0_v3*y[idx] + w1_v3*y[idx-1] + w2_v3*y[idx-2] + w3_v3*y[idx-3]
print(f"EMA at t=35: {ema3:.4f}")

# Let me also check raw values for debugging
print("\n" + "="*70)
print("Data values for verification:")
print("="*70)
for t_check in [7, 23, 35, 79]:
    print(f"\nAround t={t_check}:")
    idx_check = t_check - 1
    for offset in range(-5, 6):
        pos = idx_check + offset
        if 0 <= pos < len(y):
            print(f"  t={pos+1} (row {pos+1}): {y[pos]:.7f}")

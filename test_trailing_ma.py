import csv

# Read the data
stock_prices = []
with open("StockMarketB_9.25.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        stock_prices.append(float(row[1]))

y = stock_prices

print("Testing TRAILING MA{5} (uses current + 4 previous points)")
print("="*70)

def ma5_trailing(data, t):
    """MA{5} trailing: uses t-4, t-3, t-2, t-1, t"""
    idx = t - 1  # Convert to 0-indexed
    return sum(data[idx-4:idx+1]) / 5.0

# Test the three time points
for t in [7, 23, 79]:
    result = ma5_trailing(y, t)
    print(f"t={t}: {result:.4f}")

    # Show which values are used
    idx = t - 1
    print(f"  Using: t-4={t-4} ({y[idx-4]:.4f}), t-3={t-3} ({y[idx-3]:.4f}), " +
          f"t-2={t-2} ({y[idx-2]:.4f}), t-1={t-1} ({y[idx-1]:.4f}), t={t} ({y[idx]:.4f})")
    print()

print("\nAlternatively, testing FORWARD MA{5} (uses current + 4 future points)")
print("="*70)

def ma5_forward(data, t):
    """MA{5} forward: uses t, t+1, t+2, t+3, t+4"""
    idx = t - 1  # Convert to 0-indexed
    return sum(data[idx:idx+5]) / 5.0

for t in [7, 23, 79]:
    result = ma5_forward(y, t)
    print(f"t={t}: {result:.4f}")

    # Show which values are used
    idx = t - 1
    print(f"  Using: t={t} ({y[idx]:.4f}), t+1={t+1} ({y[idx+1]:.4f}), " +
          f"t+2={t+2} ({y[idx+2]:.4f}), t+3={t+3} ({y[idx+3]:.4f}), t+4={t+4} ({y[idx+4]:.4f})")
    print()

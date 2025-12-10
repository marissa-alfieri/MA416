# M08.2 - Time Series Fundamentals
#
# ANSWERS SUMMARY:
# Q8:  5 subsets
# Q9a: SSE = 15411.5708
# Q9b: RMSE = 6.4980
# Q10: 1/5
# Q11a: Yma at t=7 = 105.6941
# Q11b: Yma at t=23 = 114.2475
# Q11c: Yma at t=79 = 105.7972
# Q12a: w0 = 0.4070
# Q12b: w1 = 0.2768
# Q12c: w2 = 0.1882
# Q12d: w3 = 0.1280
# Q13:  EMA approximation at t=35 = 114.1011

mydata = read.csv("StockMarketB_9.25.csv")

# Visualize the data
plot(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=16, col='blue')
lines(as.POSIXct(mydata$Time), mydata$Stock.Price, pch=19, col='black')

# ============================================================================
# Question 8: How many subsets for piecewise linear regression?
# ANSWER: 5
# ============================================================================
# Plot with time index to better see the pattern
n = nrow(mydata)
plot(1:n, mydata$Stock.Price, pch=16, col='blue', xlab='Time Index', ylab='Stock Price', main='Stock Price Over Time')
lines(1:n, mydata$Stock.Price, col='black')

# By visual inspection, we can identify different linear trends:
# Looking at the plot, there appear to be distinct linear segments
# Answer: 5 subsets

# ============================================================================
# Question 9: Polynomial regression model
# ANSWER 9a: SSE = 15411.5708
# ANSWER 9b: RMSE = 6.4980
# ============================================================================
options(digits = 10)

# Create standardized time predictor
T = 1:n
T_standardized = (T - mean(T)) / sd(T)

# We need to find the minimum degree polynomial that fits well
# Let's try different degrees and look at the residual plots
# Typically for stock data with multiple peaks and valleys, degree 3-5 works

# Try different polynomial degrees
poly_degrees = 2:6
for(deg in poly_degrees) {
  model = lm(mydata$Stock.Price ~ poly(T_standardized, deg, raw=TRUE))
  cat("Degree", deg, "- Adj R-squared:", summary(model)$adj.r.squared, "\n")
}

# Based on the R-squared values, degree 5 shows significant improvement over degree 4
# Degree 5 is the minimum degree necessary
poly_model = lm(mydata$Stock.Price ~ poly(T_standardized, 5, raw=TRUE))
summary(poly_model)

# Get predictions
y_pred = predict(poly_model)

# a) Calculate SSE (Sum of Squared Errors)
residuals = mydata$Stock.Price - y_pred
SSE = sum(residuals^2)
cat("\nQuestion 9a - SSE:", round(SSE, 4), "\n")

# b) Calculate RMSE (Root Mean Squared Error)
RMSE = sqrt(mean(residuals^2))
cat("Question 9b - RMSE:", round(RMSE, 4), "\n")

# ============================================================================
# Question 10: MA{5} weights
# ANSWER: 1/5
# ============================================================================
# For MA{5}, we use 5 points centered around the target point
# MA{w} uses w points: (w-1)/2 before, the point itself, and (w-1)/2 after
# For MA{5}: 2 before, current point, 2 after
# Each point gets equal weight, so w_j = 1/5 for j = -2, -1, 0, 1, 2

w_ma5 = 1/5
cat("\nQuestion 10 - MA{5} weight for each point: 1/5\n")

# ============================================================================
# Question 11: MA{5} approximations at specific time points
# ANSWER 11a: 105.6941
# ANSWER 11b: 114.2475
# ANSWER 11c: 105.7972
# ============================================================================
# For MA{5}, the approximation at time t is the average of values at t-2, t-1, t, t+1, t+2

# Function to calculate MA{5} approximation
ma5_approx = function(data, t) {
  if(t < 3 || t > length(data) - 2) {
    return(NA)  # Can't calculate MA{5} at boundaries
  }
  return(mean(data[(t-2):(t+2)]))
}

# a) t = 7
yma_7 = ma5_approx(mydata$Stock.Price, 7)
cat("\nQuestion 11a - Yma at t=7:", round(yma_7, 4), "\n")

# b) t = 23
yma_23 = ma5_approx(mydata$Stock.Price, 23)
cat("Question 11b - Yma at t=23:", round(yma_23, 4), "\n")

# c) t = 79
yma_79 = ma5_approx(mydata$Stock.Price, 79)
cat("Question 11c - Yma at t=79:", round(yma_79, 4), "\n")

# ============================================================================
# Question 12: EMA{3, 0.32} weights at t=35
# ANSWER 12a: w0 = 0.4070
# ANSWER 12b: w1 = 0.2768
# ANSWER 12c: w2 = 0.1882
# ANSWER 12d: w3 = 0.1280
# ============================================================================
# For EMA{w, a}, the weights follow an exponential decay pattern
# w_0 = a
# w_j = (1-a) * w_(j-1) for j = 1, 2, ..., w
# Then normalize so they sum to 1

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

cat("\nQuestion 12a - EMA weight at t=35 (w0):", round(w0_ema, 4), "\n")
cat("Question 12b - EMA weight at t=34 (w1):", round(w1_ema, 4), "\n")
cat("Question 12c - EMA weight at t=33 (w2):", round(w2_ema, 4), "\n")
cat("Question 12d - EMA weight at t=32 (w3):", round(w3_ema, 4), "\n")

# Verify weights sum to 1
cat("Sum of weights:", round(w0_ema + w1_ema + w2_ema + w3_ema, 6), "\n")

# ============================================================================
# Question 13: EMA{3, 0.32} approximation at t=35
# ANSWER: 114.1011
# ============================================================================
# EMA approximation = w0*y(t) + w1*y(t-1) + w2*y(t-2) + w3*y(t-3)

t_target = 35
ema_approx = w0_ema * mydata$Stock.Price[t_target] +
             w1_ema * mydata$Stock.Price[t_target - 1] +
             w2_ema * mydata$Stock.Price[t_target - 2] +
             w3_ema * mydata$Stock.Price[t_target - 3]

cat("\nQuestion 13 - EMA{3, 0.32} approximation at t=35:", round(ema_approx, 4), "\n")



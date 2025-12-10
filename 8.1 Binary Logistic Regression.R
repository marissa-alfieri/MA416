# M08.1 - Binary Logistic Regression

mydata = read.csv("HumanZombie_9.25.csv")

# Convert response to binary (1 = zombie, 0 = human)
mydata$Y = ifelse(mydata$ZombieOrHuman == "zombie", 1, 0)

# Question 5: Fit logistic regression model
model = glm(Y ~ Speed + AwarenessLevel, data = mydata, family = binomial)
summary(model)

# Get coefficients (to 4 decimal places)
coef_intercept = round(coef(model)[1], 4)
coef_speed = round(coef(model)[2], 4)
coef_awareness = round(coef(model)[3], 4)

cat("\n=== Question 5: Coefficients ===\n")
cat("a) Intercept =", coef_intercept, "\n")
cat("b) Gradient of log-odds for speed =", coef_speed, "\n")
cat("c) Gradient of log-odds for awareness level =", coef_awareness, "\n")

# Question 6: Log-likelihood value
loglik = round(logLik(model)[1], 4)
cat("\n=== Question 6: Log-Likelihood ===\n")
cat("Log-likelihood =", loglik, "\n")

# Question 7: RMSE
predictions = predict(model, type = "response")
rmse = round(sqrt(mean((mydata$Y - predictions)^2)), 4)
cat("\n=== Question 7: RMSE ===\n")
cat("RMSE =", rmse, "\n")

# Question 8: McFadden pseudo R²
# Null model (intercept only)
null_model = glm(Y ~ 1, data = mydata, family = binomial)
loglik_null = logLik(null_model)[1]
loglik_model = logLik(model)[1]
mcfadden_r2 = round(1 - (loglik_model / loglik_null), 4)
cat("\n=== Question 8: McFadden Pseudo R² ===\n")
cat("McFadden R² =", mcfadden_r2, "\n")

# Question 9: Predicted probabilities
# a) Speed = 70, Awareness Level = 33
pred_a = predict(model, newdata = data.frame(Speed = 70, AwarenessLevel = 33), type = "response")
pred_a = round(pred_a, 4)

# b) Speed = 16, Awareness Level = 16
pred_b = predict(model, newdata = data.frame(Speed = 16, AwarenessLevel = 16), type = "response")
pred_b = round(pred_b, 4)

cat("\n=== Question 9: Predicted Probabilities ===\n")
cat("a) Speed = 70, Awareness Level = 33, Probability =", pred_a, "\n")
cat("b) Speed = 16, Awareness Level = 16, Probability =", pred_b, "\n")

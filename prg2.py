import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# ----------------------------
# 1️⃣Load the dataset
# ----------------------------
url ="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())
# We will predict MEDV (Median value of owner-occupied homes)
# using RM (average number of rooms per dwelling)
X = df[['rm']] # Feature must be 2D
y = df['medv'] # Target
# ----------------------------
# 2️⃣Train the Linear Regression model
# ----------------------------
model = LinearRegression()
model.fit(X, y)
print("\nIntercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
# Predictions
y_pred = model.predict(X)
# Model performance
print("\nMean Squared Error:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))
# ----------------------------
# 3️⃣Visualization
# ----------------------------
# Regression Line Plot
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Median home value (MEDV)")
plt.title("Simple Linear Regression: RM vs MEDV")
plt.legend()
plt.show()
# Residuals Plot
residuals = y - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Predicted MEDV")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.show()
# filepath: d:\_\Code Alpha\CodeAlpha_Car_Price_Prediction\car_price_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('car data.csv')

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Check the shape and info
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

df['Car_Age'] = 2025 - df['Year']

df = df.drop(['Car_Name', 'Year'], axis=1)

# 3. Convert categorical columns to numeric using get_dummies
df = pd.get_dummies(df, drop_first=True)

# 4. Show the processed data
print("\nProcessed DataFrame:")
print(df.head())

# 5. Split the data into features (X) and target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Store results
results = {}

# Linear Regression (already trained)
results['Linear Regression'] = {
    'rmse': rmse,
    'r2': r2
}

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_pred) ** 0.5
dt_r2 = r2_score(y_test, dt_pred)
results['Decision Tree'] = {
    'rmse': dt_rmse,
    'r2': dt_r2
}

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)
results['Random Forest'] = {
    'rmse': rf_rmse,
    'r2': rf_r2
}

# Visualize RMSE
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(results.keys(), [v['rmse'] for v in results.values()], color=['blue','green','orange'])
plt.title('RMSE Comparison')
plt.ylabel('RMSE')

# Visualize R2 Score
plt.subplot(1,2,2)
plt.bar(results.keys(), [v['r2'] for v in results.values()], color=['blue','green','orange'])
plt.title('R² Score Comparison')
plt.ylabel('R² Score')

plt.tight_layout()
plt.show()

# Print all results
print("\nModel Comparison:")
for name, scores in results.items():
    print(f"{name}: RMSE={scores['rmse']:.2f}, R²={scores['r2']:.2f}")
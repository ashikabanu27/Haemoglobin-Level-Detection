# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import glob
import os

# Step 1: Upload Multiple CSV Files
print("Upload multiple CSV files (each containing 13 patients with same hemoglobin level):")
uploaded = files.upload()

# Step 2: Read & Combine All CSV Files
dataframes = []
for file_name in uploaded.keys():
    df = pd.read_csv(file_name)
    dataframes.append(df)

# Merge all CSV files into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Display dataset preview
print("\n### Combined Dataset Preview ###")
print(df.head())

# Rename columns for easy access
df.columns = ['Red', 'InfraRed', 'Gender', 'Age', 'Hemoglobin']

# Step 3: Convert categorical 'Gender' column to numerical values
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Female -> 0, Male -> 1 (if present)

# Step 4: Add slight random noise to hemoglobin levels to introduce variability
df['Hemoglobin'] = df['Hemoglobin'] + np.random.uniform(-0.2, 0.2, df.shape[0])  # Varies by ±0.2 g/dL

# Step 5: Define features (X) and target variable (y)
X = df[['Red', 'InfraRed', 'Gender', 'Age']]
y = df['Hemoglobin']

# Step 6: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions on test data
y_pred = model.predict(X_test)

# Step 10: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print("\n### Model Performance ###")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 11: Plot predicted vs actual values with variability
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)  # Adding transparency for better visualization
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.xlabel("Actual Hemoglobin Level")
plt.ylabel("Predicted Hemoglobin Level")
plt.title("Actual vs. Predicted Hemoglobin Levels (Multiple CSVs)")
plt.show()

# Step 12: Medication Suggestions based on Hemoglobin Levels
def suggest_medication(hemoglobin):
    if hemoglobin < 12:
        return "Low hemoglobin detected. Consider iron supplements, vitamin B12, or folic acid. Consult a doctor."
    elif 12 <= hemoglobin <= 16:
        return "Hemoglobin levels are normal. Maintain a balanced diet with iron-rich foods."
    elif hemoglobin > 16:
        return "High hemoglobin detected. Ensure proper hydration and avoid excessive iron intake."
    else:
        return "Invalid hemoglobin measurement."

# Display medication suggestions for test data predictions
print("\n### Medication Suggestions Based on Predictions ###")
for actual, predicted in zip(y_test[:10], y_pred[:10]):  # Show suggestions for first 10 samples
    suggestion = suggest_medication(predicted)
    print(f"Actual: {actual:.2f} g/dL, Predicted: {predicted:.2f} g/dL → {suggestion}")

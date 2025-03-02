import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Upload Multiple CSV Files
st.title("Hemoglobin Level Monitoring from PPG Sensor")
st.sidebar.header("Upload CSV Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple CSV files (each containing 13 patients with the same hemoglobin level):",
    type="csv", accept_multiple_files=True
)

if uploaded_files:
    dataframes = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(dataframes, ignore_index=True)  # Combine all CSV files into one dataframe
    st.write("### Combined Data from All CSVs")
    st.write(df.head())  # Display first few rows

    # Step 2: Data Preprocessing
    df.dropna(inplace=True)  # Remove missing values
    df["Gender"] = df["Gender"].astype("category").cat.codes  # Encode gender (Female=0, Male=1)

    # Step 3: Feature Selection
    X = df[["Red (a.u)", "Infra Red (a.u)", "Gender", "Age (year)"]]
    y = df["Hemoglobin (g/dL)"]

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Step 6: Display Model Performance
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Step 7: Scatter Plot of Predicted vs Actual Hemoglobin Levels
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.xlabel("Actual Hemoglobin Level")
    plt.ylabel("Predicted Hemoglobin Level")
    plt.title("Predicted vs Actual Hemoglobin Levels")
    st.pyplot(fig)

    # Step 8: Hemoglobin Level Analysis & Medication Suggestions
    st.write("### Medication Suggestions Based on Hemoglobin Level")
    for hemoglobin in sorted(df["Hemoglobin (g/dL)"].unique()):
        if hemoglobin < 11.0:
            medication = "Iron supplements, Vitamin C, and folic acid."
        elif 11.0 <= hemoglobin <= 15.0:
            medication = "Maintain a balanced diet with iron-rich foods."
        else:
            medication = "Consult a doctor for further testing."
        st.write(f"- **Hemoglobin: {hemoglobin} g/dL** → {medication}")

else:
    st.warning("Please upload CSV files to proceed.")

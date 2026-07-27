# ==========================================================
# FL_4_global_evaluation.py
# Evaluate Federated Global BiLSTM Model
# ==========================================================

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# Configuration
# ==========================================================

LOOK_BACK = 24

# ==========================================================
# Load Data
# ==========================================================

data = pd.read_csv("normalized_data.csv")

dataset = data[["Solar", "Wind"]].values

# ==========================================================
# Create Sequences
# ==========================================================

X = []
Y = []

for i in range(len(dataset) - LOOK_BACK):

    X.append(dataset[i:i+LOOK_BACK])
    Y.append(dataset[i+LOOK_BACK])

X = np.array(X)
Y = np.array(Y)

# ==========================================================
# Train/Test Split
# ==========================================================

split = int(0.8 * len(X))

X_test = X[split:]
Y_test = Y[split:]

# ==========================================================
# Load Scaler
# ==========================================================

scaler = joblib.load("scaler.save")

# ==========================================================
# Load Federated Global Model
# ==========================================================

model = load_model("FL_global_model.keras")

print("="*60)
print("Evaluating Federated Global Model")
print("="*60)

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(X_test, verbose=0)

# ==========================================================
# Convert Back to Original Scale
# ==========================================================

actual = scaler.inverse_transform(Y_test)
prediction = scaler.inverse_transform(prediction)

# ==========================================================
# Solar Metrics
# ==========================================================

solar_actual = actual[:,0]
solar_pred = prediction[:,0]

solar_mae = mean_absolute_error(solar_actual, solar_pred)
solar_rmse = np.sqrt(mean_squared_error(solar_actual, solar_pred))
solar_r2 = r2_score(solar_actual, solar_pred)

solar_mape = np.mean(
    np.abs(
        (solar_actual - solar_pred) /
        (solar_actual + 1e-8)
    )
) * 100

# ==========================================================
# Wind Metrics
# ==========================================================

wind_actual = actual[:,1]
wind_pred = prediction[:,1]

wind_mae = mean_absolute_error(wind_actual, wind_pred)
wind_rmse = np.sqrt(mean_squared_error(wind_actual, wind_pred))
wind_r2 = r2_score(wind_actual, wind_pred)

wind_mape = np.mean(
    np.abs(
        (wind_actual - wind_pred) /
        (wind_actual + 1e-8)
    )
) * 100

# ==========================================================
# Average Metrics
# ==========================================================

avg_mae = (solar_mae + wind_mae) / 2
avg_rmse = (solar_rmse + wind_rmse) / 2
avg_mape = (solar_mape + wind_mape) / 2
avg_r2 = (solar_r2 + wind_r2) / 2

# ==========================================================
# Performance Matrix
# ==========================================================

results = pd.DataFrame({

    "Parameter":[
        "Solar",
        "Wind",
        "Average"
    ],

    "MAE":[
        solar_mae,
        wind_mae,
        avg_mae
    ],

    "RMSE":[
        solar_rmse,
        wind_rmse,
        avg_rmse
    ],

    "MAPE":[
        solar_mape,
        wind_mape,
        avg_mape
    ],

    "R2 Score":[
        solar_r2,
        wind_r2,
        avg_r2
    ]

})

# ==========================================================
# Save Results
# ==========================================================

results.to_csv(
    "FL_performance_matrix.csv",
    index=False
)

with open("FL_evaluation_results.txt","w") as f:

    f.write("Federated Learning Global Model\n")
    f.write("="*60+"\n\n")

    f.write(results.to_string(index=False))

print("\nPerformance Matrix\n")
print(results)

print("\nResults Saved Successfully")

print("FL_performance_matrix.csv")
print("FL_evaluation_results.txt")
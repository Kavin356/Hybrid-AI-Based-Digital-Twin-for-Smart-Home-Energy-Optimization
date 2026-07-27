# ==========================================================
# FL_6_visualization.py
# Visualization for Federated Learning BiLSTM
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

# ==========================================================
# Configuration
# ==========================================================

LOOK_BACK = 24

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("normalized_data.csv")

dataset = data[["Solar","Wind"]].values

# ==========================================================
# Create Sequences
# ==========================================================

X = []
Y = []

for i in range(len(dataset)-LOOK_BACK):

    X.append(dataset[i:i+LOOK_BACK])
    Y.append(dataset[i+LOOK_BACK])

X = np.array(X)
Y = np.array(Y)

split = int(0.8*len(X))

X_test = X[split:]
Y_test = Y[split:]

# ==========================================================
# Load Model
# ==========================================================

model = load_model("FL_global_model.keras")

scaler = joblib.load("scaler.save")

prediction = model.predict(X_test, verbose=0)

actual = scaler.inverse_transform(Y_test)
prediction = scaler.inverse_transform(prediction)

# ==========================================================
# Plot 1 : Solar Prediction
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(actual[:,0],label="Actual Solar")

plt.plot(prediction[:,0],label="Predicted Solar")

plt.xlabel("Samples")

plt.ylabel("Solar Generation")

plt.title("Federated Learning - Solar Prediction")

plt.legend()

plt.grid(True)

plt.savefig("FL_solar_prediction.png")

plt.close()

# ==========================================================
# Plot 2 : Wind Prediction
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(actual[:,1],label="Actual Wind")

plt.plot(prediction[:,1],label="Predicted Wind")

plt.xlabel("Samples")

plt.ylabel("Wind Generation")

plt.title("Federated Learning - Wind Prediction")

plt.legend()

plt.grid(True)

plt.savefig("FL_wind_prediction.png")

plt.close()

# ==========================================================
# Load Forecast File
# ==========================================================

forecast = pd.read_csv("FL_forecast_24hours.csv")

# ==========================================================
# Plot 3 : Solar Forecast
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(

    forecast["Hour"],

    forecast["Forecast_Solar"],

    marker='o'

)

plt.xlabel("Hour")

plt.ylabel("Solar Forecast")

plt.title("24 Hour Solar Forecast")

plt.grid(True)

plt.savefig("FL_solar_forecast.png")

plt.close()

# ==========================================================
# Plot 4 : Wind Forecast
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(

    forecast["Hour"],

    forecast["Forecast_Wind"],

    marker='o'

)

plt.xlabel("Hour")

plt.ylabel("Wind Forecast")

plt.title("24 Hour Wind Forecast")

plt.grid(True)

plt.savefig("FL_wind_forecast.png")

plt.close()

# ==========================================================
# Plot 5 : Training Loss
# ==========================================================

loss = pd.read_csv("FL_Local_Models/client_1_loss.csv")

plt.figure(figsize=(10,5))

plt.plot(

    loss["Epoch"],

    loss["Training Loss"],

    label="Training Loss"

)

plt.plot(

    loss["Epoch"],

    loss["Validation Loss"],

    label="Validation Loss"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("BiLSTM Training Loss")

plt.legend()

plt.grid(True)

plt.savefig("FL_training_loss.png")

plt.close()

print("="*60)
print("Visualization Completed Successfully")
print("="*60)

print("\nGenerated Figures")

print("---------------------------")

print("FL_training_loss.png")

print("FL_solar_prediction.png")

print("FL_wind_prediction.png")

print("FL_solar_forecast.png")

print("FL_wind_forecast.png")
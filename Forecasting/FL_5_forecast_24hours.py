# ==========================================================
# FL_5_forecast_24hours.py
# 24-Hour Forecast using Federated Global BiLSTM Model
# ==========================================================

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# ==========================================================
# Configuration
# ==========================================================

LOOK_BACK = 24
FORECAST_HOURS = 24

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("normalized_data.csv")

dataset = data[["Solar", "Wind"]].values

# ==========================================================
# Load Scaler
# ==========================================================

scaler = joblib.load("scaler.save")

# ==========================================================
# Load Federated Global Model
# ==========================================================

model = load_model("FL_global_model.keras")

print("=" * 60)
print("24-Hour Federated Learning Forecast")
print("=" * 60)

# ==========================================================
# Initial Input Sequence
# ==========================================================

current_sequence = dataset[-LOOK_BACK:].copy()

forecast = []

# ==========================================================
# Recursive Forecast
# ==========================================================

for hour in range(FORECAST_HOURS):

    input_data = current_sequence.reshape(1, LOOK_BACK, 2)

    prediction = model.predict(input_data, verbose=0)

    forecast.append(prediction[0])

    current_sequence = np.vstack([
        current_sequence[1:],
        prediction[0]
    ])

# ==========================================================
# Convert to Original Scale
# ==========================================================

forecast = np.array(forecast)

forecast_original = scaler.inverse_transform(forecast)

# ==========================================================
# Save Forecast
# ==========================================================

forecast_df = pd.DataFrame({

    "Hour": np.arange(1, FORECAST_HOURS + 1),

    "Forecast_Solar": forecast_original[:,0],

    "Forecast_Wind": forecast_original[:,1]

})

forecast_df.to_csv(

    "FL_forecast_24hours.csv",

    index=False

)

print("\nForecast Completed Successfully\n")

print(forecast_df)

print("\nSaved As")

print("FL_forecast_24hours.csv")
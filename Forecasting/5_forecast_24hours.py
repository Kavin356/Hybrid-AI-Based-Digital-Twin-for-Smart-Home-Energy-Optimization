import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------------
# Load Trained Model
# ---------------------------------
# Change to bilstm_model.keras if you saved in Keras format
model = load_model("bilstm_model.h5")

print("Model Loaded Successfully!")

# ---------------------------------
# Load Processed Data
# ---------------------------------
data = pd.read_csv("processed_data.csv")

print("\nColumns in processed_data.csv:")
print(data.columns)

# ---------------------------------
# Select Features
# ---------------------------------
values = data[['Solar', 'Wind']].values

print("\nDataset Shape:", values.shape)

# ---------------------------------
# Last 24 Hours
# ---------------------------------
window_size = 24

current_input = values[-window_size:]

forecast = []

# ---------------------------------
# Predict Next 24 Hours
# ---------------------------------
print("\nGenerating 24-Hour Forecast...\n")

for hour in range(24):

    prediction = model.predict(
        current_input.reshape(1, window_size, 2),
        verbose=0
    )

    prediction = prediction[0]

    forecast.append(prediction)

    # Update input window
    current_input = np.vstack(
        (current_input[1:], prediction)
    )

forecast = np.array(forecast)

# ---------------------------------
# Create Forecast Table
# ---------------------------------
forecast_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    "Forecast_Solar": forecast[:, 0],
    "Forecast_Wind": forecast[:, 1]
})

print(forecast_df)

# ---------------------------------
# Save Forecast
# ---------------------------------
forecast_df.to_csv(
    "24_hour_forecast.csv",
    index=False
)

print("\nForecast saved successfully as:")
print("24_hour_forecast.csv")

# ---------------------------------
# Plot Solar Forecast
# ---------------------------------
plt.figure(figsize=(10,5))

plt.plot(
    forecast_df["Hour"],
    forecast_df["Forecast_Solar"],
    marker='o',
    linewidth=2,
    label="Solar Forecast"
)

plt.title("Next 24-Hour Solar Forecast")
plt.xlabel("Hour")
plt.ylabel("Normalized Solar Generation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Plot Wind Forecast
# ---------------------------------
plt.figure(figsize=(10,5))

plt.plot(
    forecast_df["Hour"],
    forecast_df["Forecast_Wind"],
    marker='o',
    linewidth=2,
    label="Wind Forecast"
)

plt.title("Next 24-Hour Wind Forecast")
plt.xlabel("Hour")
plt.ylabel("Normalized Wind Generation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# Combined Plot
# ---------------------------------
plt.figure(figsize=(12,5))

plt.plot(
    forecast_df["Hour"],
    forecast_df["Forecast_Solar"],
    marker='o',
    label="Solar"
)

plt.plot(
    forecast_df["Hour"],
    forecast_df["Forecast_Wind"],
    marker='s',
    label="Wind"
)

plt.title("24-Hour Renewable Energy Forecast")
plt.xlabel("Hour")
plt.ylabel("Normalized Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nForecasting Completed Successfully!")
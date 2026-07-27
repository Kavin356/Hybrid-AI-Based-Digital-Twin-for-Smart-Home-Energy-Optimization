import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# -------------------------------
# Load Test Data
# -------------------------------
X = np.load("X.npy")
Y = np.load("Y.npy")

# 80-20 Split (same as training)
split = int(len(X) * 0.8)

X_test = X[split:]
Y_test = Y[split:]

print("Testing Samples :", X_test.shape)

# -------------------------------
# Load Trained Model
# -------------------------------
model = load_model("bilstm_model.h5")

print("\nModel Loaded Successfully!")

# -------------------------------
# Prediction
# -------------------------------
Y_pred = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------

mse = mean_squared_error(Y_test, Y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Squared Error  : {mse:.6f}")
print(f"Root Mean Squared Error : {rmse:.6f}")
print(f"Mean Absolute Error : {mae:.6f}")
print(f"R² Score            : {r2:.6f}")

# -------------------------------
# Solar Prediction Plot
# -------------------------------

plt.figure(figsize=(12,5))

plt.plot(Y_test[:,0], label="Actual Solar")
plt.plot(Y_pred[:,0], label="Predicted Solar")

plt.title("Solar Power Forecast")
plt.xlabel("Samples")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Wind Prediction Plot
# -------------------------------

plt.figure(figsize=(12,5))

plt.plot(Y_test[:,1], label="Actual Wind")
plt.plot(Y_pred[:,1], label="Predicted Wind")

plt.title("Wind Power Forecast")
plt.xlabel("Samples")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("BiLSTM_SolarWind.keras")

# Load scaler
scaler = joblib.load("scaler.save")

# Load dataset
data = pd.read_csv("processed_data.csv")

data = data[['Solar', 'Wind']]

data = data.interpolate()

scaled = scaler.transform(data)

window = 24

current = scaled[-window:]

forecast = []

for i in range(24):

    pred = model.predict(
        current.reshape(1,24,2),
        verbose=0
    )

    forecast.append(pred[0])

    current = np.vstack([current[1:],pred])

forecast = np.array(forecast)

forecast = scaler.inverse_transform(forecast)

result = pd.DataFrame({

    "Hour":range(1,25),

    "Forecast Solar":forecast[:,0],

    "Forecast Wind":forecast[:,1]

})

result.to_csv("24hour_forecast.csv",index=False)

print(result)
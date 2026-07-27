import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("time_series_60min_singleindex.csv")

# Keep only required columns
df = df[['utc_timestamp',
         'DE_solar_generation_actual',
         'DE_wind_generation_actual']]

# Rename columns
df.columns = ['Time', 'Solar', 'Wind']

# Convert timestamp
df['Time'] = pd.to_datetime(df['Time'])

# Sort data
df = df.sort_values('Time')

# Fill missing values
df['Solar'] = df['Solar'].interpolate()
df['Wind'] = df['Wind'].interpolate()

# Remove remaining NaNs
df = df.dropna()

# Normalization
scaler = MinMaxScaler()

df[['Solar','Wind']] = scaler.fit_transform(df[['Solar','Wind']])

# Save scaler
joblib.dump(scaler,'scaler.save')

# Save processed data
df.to_csv("processed_data.csv",index=False)

print(df.head())
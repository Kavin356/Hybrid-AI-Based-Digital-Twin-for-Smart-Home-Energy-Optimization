import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

# ======================================
# Load Dataset
# ======================================

data = pd.read_csv("processed_data.csv")

# Keep only required columns
data = data[['Solar', 'Wind']]

# Fill missing values
data = data.interpolate()

# ======================================
# Normalize
# ======================================

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data)

joblib.dump(scaler,"scaler.save")

# ======================================
# Create Sliding Window
# ======================================

window = 24

X = []
Y = []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    Y.append(scaled[i])

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

# ======================================
# Train/Test Split
# ======================================

split = int(0.8*len(X))

X_train = X[:split]
X_test  = X[split:]

Y_train = Y[:split]
Y_test  = Y[split:]

# ======================================
# Build BiLSTM
# ======================================

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        ),
        input_shape=(24,2)
    )
)

model.add(Dropout(0.2))

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))

model.add(Dense(2))

model.compile(

    optimizer=Adam(0.001),

    loss='mse',

    metrics=['mae']

)

model.summary()

# ======================================
# Train
# ======================================

early = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

history = model.fit(

    X_train,

    Y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=32,

    callbacks=[early],

    verbose=1

)

# ======================================
# Save
# ======================================

model.save("BiLSTM_SolarWind.keras")

print("Model Saved Successfully")
# ==========================================================
# FL_2_client_training.py
# Local BiLSTM Training for Federated Learning Clients
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# Configuration
# ==========================================================

LOOK_BACK = 24
CLIENT_FOLDER = "FL_Clients"
MODEL_FOLDER = "FL_Local_Models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ==========================================================
# Load Scaler
# ==========================================================

scaler = joblib.load("scaler.save")

# ==========================================================
# Function to Create Sequences
# ==========================================================

def create_dataset(dataset, look_back):

    X = []
    Y = []

    for i in range(len(dataset) - look_back):

        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])

    return np.array(X), np.array(Y)

# ==========================================================
# Get Client Files
# ==========================================================

client_files = sorted([
    f for f in os.listdir(CLIENT_FOLDER)
    if f.endswith(".csv")
])

print("="*60)
print("Federated Learning - Local Client Training")
print("="*60)

# ==========================================================
# Train Each Client
# ==========================================================

for client in client_files:

    print("\nTraining", client)

    # ------------------------------------------

    data = pd.read_csv(
        os.path.join(CLIENT_FOLDER, client)
    )

    dataset = data[["Solar", "Wind"]].values

    X, Y = create_dataset(dataset, LOOK_BACK)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    Y_train = Y[:split]

    X_val = X[split:]
    Y_val = Y[split:]

    # ------------------------------------------
    # Build BiLSTM Model
    # ------------------------------------------

    model = Sequential()

    model.add(
        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            ),
            input_shape=(LOOK_BACK, 2)
        )
    )

    model.add(Dropout(0.2))

    model.add(
        Bidirectional(
            LSTM(64)
        )
    )

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(2))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # ------------------------------------------

    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ------------------------------------------

    history = model.fit(

        X_train,
        Y_train,

        validation_data=(X_val, Y_val),

        epochs=30,

        batch_size=32,

        verbose=1,

        callbacks=[early]

    )

    # ------------------------------------------
    # Save Model
    # ------------------------------------------

    model_name = client.replace(".csv", ".keras")

    model.save(
        os.path.join(
            MODEL_FOLDER,
            model_name
        )
    )

    # ------------------------------------------
    # Save Training Loss
    # ------------------------------------------

    loss_df = pd.DataFrame({

        "Epoch":
            range(
                1,
                len(history.history["loss"]) + 1
            ),

        "Training Loss":
            history.history["loss"],

        "Validation Loss":
            history.history["val_loss"]

    })

    loss_name = client.replace(".csv", "_loss.csv")

    loss_df.to_csv(

        os.path.join(
            MODEL_FOLDER,
            loss_name
        ),

        index=False

    )

    print(client, "Training Completed")

print("\n======================================")
print("All Client Models Successfully Trained")
print("======================================")
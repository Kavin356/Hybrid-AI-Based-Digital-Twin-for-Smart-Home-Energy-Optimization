# ==========================================================
# FL_3_federated_server.py
# Federated Averaging (FedAvg)
# ==========================================================

import os
import numpy as np

from tensorflow.keras.models import load_model

# ==========================================================
# Configuration
# ==========================================================

MODEL_FOLDER = "FL_Local_Models"

GLOBAL_MODEL_NAME = "FL_global_model.keras"

# ==========================================================
# Load Client Models
# ==========================================================

client_models = []

model_files = sorted([
    f for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".keras")
])

print("="*60)
print("Federated Learning Server")
print("="*60)

print("\nLoading Client Models...")

for model_name in model_files:

    path = os.path.join(
        MODEL_FOLDER,
        model_name
    )

    model = load_model(path)

    client_models.append(model)

    print(model_name, "Loaded")

# ==========================================================
# Extract Weights
# ==========================================================

client_weights = []

for model in client_models:

    client_weights.append(
        model.get_weights()
    )

print("\nTotal Clients :", len(client_weights))

# ==========================================================
# Federated Averaging (FedAvg)
# ==========================================================

print("\nApplying Federated Averaging...")

average_weights = []

for weights in zip(*client_weights):

    average_weights.append(
        np.mean(weights, axis=0)
    )

# ==========================================================
# Create Global Model
# ==========================================================

global_model = load_model(
    os.path.join(
        MODEL_FOLDER,
        model_files[0]
    )
)

global_model.set_weights(
    average_weights
)

# ==========================================================
# Save Global Model
# ==========================================================

global_model.save(
    GLOBAL_MODEL_NAME
)

print("\nGlobal Model Saved Successfully")

print("\nSaved As :")

print(GLOBAL_MODEL_NAME)

print("\n==========================================")
print("Federated Learning Completed Successfully")
print("==========================================")
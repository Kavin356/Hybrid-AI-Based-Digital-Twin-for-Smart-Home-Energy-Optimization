# ==========================================================
# FL_1_split_clients.py
# Split Dataset into Federated Learning Clients
# ==========================================================

import pandas as pd
import numpy as np
import os

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("normalized_data.csv")

print("="*60)
print("Federated Learning - Client Dataset Generation")
print("="*60)

print("\nTotal Samples :", len(data))

# ==========================================================
# Number of Clients
# ==========================================================

NUM_CLIENTS = 3

# ==========================================================
# Create Folder
# ==========================================================

CLIENT_FOLDER = "FL_Clients"

os.makedirs(CLIENT_FOLDER, exist_ok=True)

# ==========================================================
# Split Dataset
# ==========================================================

client_size = len(data) // NUM_CLIENTS

for i in range(NUM_CLIENTS):

    start = i * client_size

    if i == NUM_CLIENTS - 1:
        end = len(data)
    else:
        end = (i + 1) * client_size

    client_data = data.iloc[start:end]

    filename = os.path.join(
        CLIENT_FOLDER,
        f"client_{i+1}.csv"
    )

    client_data.to_csv(
        filename,
        index=False
    )

    print(f"\nClient {i+1}")
    print("----------------------------")
    print("Samples :", len(client_data))
    print("Saved :", filename)

print("\n====================================")
print("Dataset Successfully Split")
print("====================================")

print("\nGenerated Files:")

for i in range(NUM_CLIENTS):
    print(f"client_{i+1}.csv")
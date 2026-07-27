import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Optimized Schedule
# -----------------------------
data = pd.read_csv("optimized_energy_schedule.csv")

# -----------------------------
# Plot 1
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(data["Hour"], data["Solar"],
         marker='o', linewidth=2, label="Solar")

plt.plot(data["Hour"], data["Wind"],
         marker='s', linewidth=2, label="Wind")

plt.title("Solar and Wind Forecast")
plt.xlabel("Hour")
plt.ylabel("Normalized Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(data["Hour"], data["Load"],
         linewidth=2, label="Load")

plt.plot(data["Hour"], data["Renewable"],
         linewidth=2, label="Renewable")

plt.title("Load vs Renewable Generation")
plt.xlabel("Hour")
plt.ylabel("Normalized Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3
# -----------------------------
plt.figure(figsize=(12,6))

plt.bar(data["Hour"],
        data["Battery"],
        label="Battery")

plt.title("Battery Usage Schedule")
plt.xlabel("Hour")
plt.ylabel("Battery Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 4
# -----------------------------
plt.figure(figsize=(12,6))

plt.bar(data["Hour"],
        data["Grid"],
        color="red",
        label="Grid Import")

plt.title("Grid Energy Consumption")
plt.xlabel("Hour")
plt.ylabel("Grid Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 5
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(data["Hour"], data["Load"],
         linewidth=2, label="Load")

plt.plot(data["Hour"], data["Renewable"],
         linewidth=2, label="Renewable")

plt.plot(data["Hour"], data["Battery"],
         linewidth=2, label="Battery")

plt.plot(data["Hour"], data["Grid"],
         linewidth=2, label="Grid")

plt.title("Complete Smart Home Energy Management")
plt.xlabel("Hour")
plt.ylabel("Normalized Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nVisualization Completed Successfully!")
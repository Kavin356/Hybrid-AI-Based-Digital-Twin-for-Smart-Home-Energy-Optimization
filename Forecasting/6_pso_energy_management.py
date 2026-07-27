import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps

# -----------------------------
# Load Forecast
# -----------------------------
forecast = pd.read_csv("24_hour_forecast.csv")

solar = forecast["Forecast_Solar"].values
wind = forecast["Forecast_Wind"].values

renewable = solar + wind

# -----------------------------
# Example Load Demand
# (Replace with actual smart home load if available)
# -----------------------------
load = np.array([
    0.60,0.58,0.55,0.54,0.56,0.62,
    0.70,0.82,0.90,0.95,0.98,1.00,
    1.05,1.02,0.98,0.95,0.92,0.96,
    1.00,1.08,1.02,0.90,0.80,0.68
])

# -----------------------------
# Objective Function
# -----------------------------
def objective(x):

    cost = []

    for particle in x:

        battery = particle

        grid = np.maximum(load - renewable - battery, 0)

        battery_penalty = np.sum(np.square(battery))

        grid_cost = np.sum(grid)

        total_cost = grid_cost + 0.2 * battery_penalty

        cost.append(total_cost)

    return np.array(cost)

# -----------------------------
# PSO Parameters
# -----------------------------
options = {
    'c1':1.5,
    'c2':1.5,
    'w':0.7
}

lower = np.zeros(24)
upper = np.ones(24) * 0.5

optimizer = ps.single.GlobalBestPSO(
    n_particles=40,
    dimensions=24,
    options=options,
    bounds=(lower, upper)
)

# -----------------------------
# Run PSO
# -----------------------------
best_cost, best_battery = optimizer.optimize(
    objective,
    iters=100
)

print("\nBest Cost :", best_cost)

# -----------------------------
# Energy Allocation
# -----------------------------
grid = np.maximum(
    load - renewable - best_battery,
    0
)

result = pd.DataFrame({

    "Hour":np.arange(1,25),

    "Solar":solar,

    "Wind":wind,

    "Renewable":renewable,

    "Load":load,

    "Battery":best_battery,

    "Grid":grid
})

print(result)

result.to_csv(
    "optimized_energy_schedule.csv",
    index=False
)

print("\nEnergy Schedule Saved!")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(load,label="Load",linewidth=2)

plt.plot(renewable,label="Renewable",linewidth=2)

plt.plot(best_battery,label="Battery",linewidth=2)

plt.plot(grid,label="Grid",linewidth=2)

plt.xlabel("Hour")

plt.ylabel("Normalized Energy")

plt.title("PSO Optimized Smart Home Energy Management")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()
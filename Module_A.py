import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
file_path = "ess_test_data_electrical_v2.csv"
df = pd.read_csv(file_path)

# Standardize column names (lowercase, strip whitespace, replace spaces with underscores)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 1. Datetime Parsing & Time Delta Calculation
# Adjust 'timestamp' if your time column uses a different name (e.g., 'time', 'date_time')
time_col = [col for col in df.columns if "time" in col or "date" in col]
if time_col:
    df[time_col[0]] = pd.to_datetime(df[time_col[0]])
    df = df.sort_values(by=time_col[0]).reset_index(drop=True)
    # Calculate elapsed time in hours for numerical integration
    df["time_diff_hours"] = (
        df[time_col[0]].diff().dt.total_seconds().fillna(0) / 3600.0
    )
else:
    # Fallback if time is given in seconds
    df["time_diff_hours"] = 1.0 / 3600.0

# 2. Identify Core Electrical Parameters
# Flexible column mapping for standard ESS test metrics
v_col = [col for col in df.columns if "vol" in col or col == "v"][0]
i_col = [col for col in df.columns if "cur" in col or col == "i"][0]
temp_col = [col for col in df.columns if "temp" in col]
soc_col = [col for col in df.columns if "soc" in col]

# 3. Electrical Computations & Feature Engineering
# Power in Watts (P = V * I)
df["power_w"] = df[v_col] * df[i_col]
df["power_kw"] = df["power_w"] / 1000.0

# Cumulative Ah (Capacity Integration: Q = ∫ I dt)
df["ampere_hours"] = (df[i_col] * df["time_diff_hours"]).cumsum()

# Cumulative Wh (Energy Integration: E = ∫ P dt)
df["watt_hours"] = (df["power_w"] * df["time_diff_hours"]).cumsum()
df["kwh"] = df["watt_hours"] / 1000.0

# 4. Summary Metrics & State Classification
df["operating_mode"] = np.select(
    [df[i_col] > 0.05, df[i_col] < -0.05],
    ["Charge", "Discharge"],
    default="Idle",
)

charge_energy_kwh = (
    df[df[i_col] > 0]["power_w"] * df[df[i_col] > 0]["time_diff_hours"]
).sum() / 1000.0
discharge_energy_kwh = (
    abs(df[df[i_col] < 0]["power_w"] * df[df[i_col] < 0]["time_diff_hours"])
).sum() / 1000.0

rte = (
    (discharge_energy_kwh / charge_energy_kwh) * 100
    if charge_energy_kwh > 0
    else 0.0
)

# Output Summary Statistics
print(f"Data Records: {len(df)}")
print(
    f"Voltage Range: {df[v_col].min():.2f} V - {df[v_col].max():.2f} V"
)
print(
    f"Current Range: {df[i_col].min():.2f} A - {df[i_col].max():.2f} A"
)
print(f"Total Charge Energy: {charge_energy_kwh:.3f} kWh")
print(f"Total Discharge Energy: {discharge_energy_kwh:.3f} kWh")
print(f"Estimated Round-Trip Efficiency (RTE): {rte:.2f}%")

# 5. Visualization Dashboard
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Voltage & Current Profile
ax1_twin = axes[0].twinx()
axes[0].plot(df.index, df[v_col], color="tab:blue", label="Voltage (V)")
ax1_twin.plot(df.index, df[i_col], color="tab:orange", label="Current (A)")
axes[0].set_ylabel("Voltage (V)", color="tab:blue")
ax1_twin.set_ylabel("Current (A)", color="tab:orange")
axes[0].set_title("ESS Electrical Performance Overview")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Power & Energy Profile
axes[1].plot(df.index, df["power_kw"], color="tab:green", label="Power (kW)")
axes[1].set_ylabel("Power (kW)")
axes[1].grid(True, linestyle="--", alpha=0.6)

# SOC / Temperature Profile
if soc_col:
    axes[2].plot(df.index, df[soc_col[0]], color="tab:red", label="SOC (%)")
    axes[2].set_ylabel("SOC (%)")
elif temp_col:
    axes[2].plot(
        df.index, df[temp_col[0]], color="tab:purple", label="Temp (°C)"
    )
    axes[2].set_ylabel("Temperature (°C)")
else:
    axes[2].plot(df.index, df["kwh"], color="tab:purple", label="Energy (kWh)")
    axes[2].set_ylabel("Cumulative Energy (kWh)")

axes[2].set_xlabel("Sample Index / Time Step")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

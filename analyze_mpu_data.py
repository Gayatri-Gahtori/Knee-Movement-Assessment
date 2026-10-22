
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# KNEE MOVEMENT PROJECT - SENSOR DATA ANALYSIS
# ==========================================

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically select an existing recording
possible_files = [
    "mpu6050_test_100hz.csv",
    "mpu6050_data.csv"
]

INPUT_FILE = next(
    (f for f in possible_files if os.path.exists(f)),
    None
)

if INPUT_FILE is None:
    raise FileNotFoundError(
        "No sensor CSV found. Keep this script in the "
        "same folder as mpu6050_test_100hz.csv or mpu6050_data.csv."
    )

print("=" * 55)
print("KNEE MOVEMENT - SENSOR DATA ANALYSIS")
print("=" * 55)
print("Input file:", INPUT_FILE)

# 1. Load and validate
df = pd.read_csv(INPUT_FILE)
required = [
    "timestamp_s",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required].copy()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().sort_values("timestamp_s")
df = df.drop_duplicates(subset="timestamp_s")
df = df.reset_index(drop=True)

if len(df) < 10:
    raise ValueError("Too few valid sensor samples for analysis.")

# 2. Time and sampling rate
time_values = df["timestamp_s"].to_numpy()
duration = time_values[-1] - time_values[0]
dt = np.diff(time_values)
dt = dt[dt > 0]

sampling_rate = 1 / np.median(dt) if len(dt) else np.nan

print(f"Valid samples: {len(df)}")
print(f"Recording duration: {duration:.2f} seconds")
print(f"Estimated sampling rate: {sampling_rate:.2f} Hz")

# 3. Derive movement signals
accel_cols = ["accel_x", "accel_y", "accel_z"]
gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

df["accel_magnitude"] = np.sqrt(
    (df[accel_cols] ** 2).sum(axis=1)
)

df["gyro_magnitude"] = np.sqrt(
    (df[gyro_cols] ** 2).sum(axis=1)
)

# Gravity is included in acceleration magnitude.
# This is a movement indicator, NOT knee angle.
df["accel_dynamic_indicator"] = np.abs(
    df["accel_magnitude"] - 9.80665
)

# 4. Flag unusual acceleration readings for inspection
df["high_accel_flag"] = (
    df["accel_magnitude"] > 15
).astype(int)

print(
    "Samples flagged for high acceleration:",
    int(df["high_accel_flag"].sum())
)

# 5. Export cleaned data with derived signals
clean_path = os.path.join(
    OUTPUT_DIR, "cleaned_mpu_data.csv"
)
df.to_csv(clean_path, index=False)

# 6. Extract window-based features
# Use 2-second windows; no overlapping windows.
feature_rows = []
window_seconds = 2.0

start_time = time_values[0]
window_id = 0

while start_time + window_seconds <= time_values[-1]:
    end_time = start_time + window_seconds

    window = df[
        (df["timestamp_s"] >= start_time) &
        (df["timestamp_s"] < end_time)
    ]

    if len(window) >= 5:
        row = {
            "window_id": window_id,
            "start_s": start_time,
            "end_s": end_time,
            "sample_count": len(window)
        }

        signal_cols = [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_magnitude", "gyro_magnitude",
            "accel_dynamic_indicator"
        ]

        for col in signal_cols:
            row[f"{col}_mean"] = window[col].mean()
            row[f"{col}_std"] = window[col].std()
            row[f"{col}_min"] = window[col].min()
            row[f"{col}_max"] = window[col].max()

        row["high_accel_fraction"] = (
            window["high_accel_flag"].mean()
        )

        feature_rows.append(row)
        window_id += 1

    start_time = end_time

features = pd.DataFrame(feature_rows)

feature_path = os.path.join(
    OUTPUT_DIR, "movement_features.csv"
)

if not features.empty:
    features.to_csv(feature_path, index=False)
    print("Feature windows generated:", len(features))
else:
    print("No complete 2-second windows found.")
    print("Check recording duration and timestamps.")

# 7. Plot acceleration
plt.figure(figsize=(12, 6))

for col in accel_cols:
    plt.plot(
        df["timestamp_s"], df[col],
        label=col, linewidth=0.8
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("MPU6050 Accelerometer Signals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "accelerometer.png"),
    dpi=150
)
plt.close()

# 8. Plot gyroscope
plt.figure(figsize=(12, 6))

for col in gyro_cols:
    plt.plot(
        df["timestamp_s"], df[col],
        label=col, linewidth=0.8
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("MPU6050 Gyroscope Signals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "gyroscope.png"),
    dpi=150
)
plt.close()

# 9. Plot movement indicators
plt.figure(figsize=(12, 5))
plt.plot(
    df["timestamp_s"],
    df["accel_dynamic_indicator"],
    label="Acceleration deviation from gravity",
    linewidth=0.8
)
plt.plot(
    df["timestamp_s"],
    df["gyro_magnitude"],
    label="Gyroscope magnitude",
    linewidth=0.8
)
plt.xlabel("Time (seconds)")
plt.title("Movement Indicators")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "movement_indicators.png"),
    dpi=150
)
plt.close()

# 10. Save a text summary
summary_path = os.path.join(
    OUTPUT_DIR, "analysis_report.txt"
)

with open(summary_path, "w", encoding="utf-8") as report:
    report.write("KNEE MOVEMENT PROJECT - DATA REPORT\n")
    report.write("=" * 40 + "\n")
    report.write(f"Input file: {INPUT_FILE}\n")
    report.write(f"Valid samples: {len(df)}\n")
    report.write(f"Duration: {duration:.3f} seconds\n")
    report.write(f"Estimated sampling rate: {sampling_rate:.3f} Hz\n")
    report.write(
        f"High acceleration flags: {int(df['high_accel_flag'].sum())}\n"
    )
    report.write(
        f"Feature windows: {len(features)}\n"
    )
    report.write(
        "\nAcceleration summary:\n"
        + df[accel_cols].describe().to_string()
    )
    report.write(
        "\n\nGyroscope summary:\n"
        + df[gyro_cols].describe().to_string()
    )
    report.write(
        "\n\nNote: This report does not calculate knee angle "
        "or diagnose abnormal gait."
    )

print("\nAnalysis complete!")
print("Generated files:")
for name in os.listdir(OUTPUT_DIR):
    print(" -", os.path.join(OUTPUT_DIR, name))

print("\nNext: inspect the plots and feature table.")

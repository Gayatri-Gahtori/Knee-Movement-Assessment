
import pandas as pd

df = pd.read_csv("mpu6050_test_100hz.csv")

print("Total samples:", len(df))
print("First timestamp:", df["timestamp_s"].iloc[0])
print("Last timestamp:", df["timestamp_s"].iloc[-1])

duration = df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0]
print("Recording duration:", round(duration, 2), "seconds")

if duration > 0:
    print("Average sampling rate:", round((len(df) - 1) / duration, 2), "Hz")

print("\nColumn names:")
print(df.columns.tolist())

print("\nAcceleration summary:")
print(df[["accel_x", "accel_y", "accel_z"]].describe())

import pandas as pd

df = pd.read_csv("mpu6050_test_100hz.csv")

axes = ["accel_x", "accel_y", "accel_z"]

# Magnitude of the measured acceleration vector
df["accel_magnitude"] = (
    df["accel_x"]**2
    + df["accel_y"]**2
    + df["accel_z"]**2
) ** 0.5

print("Total samples:", len(df))
print("\nAcceleration magnitude summary:")
print(df["accel_magnitude"].describe().round(3))

# Find unusually large acceleration magnitudes
extreme = df[df["accel_magnitude"] > 15]

print("\nSamples with magnitude above 15 m/s²:")
print(extreme[["timestamp_s"] + axes + ["accel_magnitude"]].head(20))
print("Total such samples:", len(extreme))


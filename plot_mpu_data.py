
import pandas as pd
import matplotlib.pyplot as plt

# Load the collected sensor data
df = pd.read_csv("mpu6050_data.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 readings:")
print(df.head())

# Plot accelerometer readings
plt.figure(figsize=(10, 5))
for axis in ["accel_x", "accel_y", "accel_z"]:
    plt.plot(df["timestamp_s"], df[axis], label=axis)

plt.title("MPU6050 Accelerometer Readings")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot gyroscope readings
plt.figure(figsize=(10, 5))
for axis in ["gyro_x", "gyro_y", "gyro_z"]:
    plt.plot(df["timestamp_s"], df[axis], label=axis)

plt.title("MPU6050 Gyroscope Readings")
plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

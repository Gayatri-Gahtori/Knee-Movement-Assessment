
import serial
import csv
import time

PORT = "COM7"
BAUD_RATE = 115200
OUTPUT_FILE = "stationary_2.csv"

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()

    print("Connected to ESP32!")
    print("Collecting MPU6050 data.")
    print("Keep the sensor still initially.")
    print("Press Ctrl+C to stop.\n")

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp_s",
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ])

        start_time = time.time()
        samples = 0

        while True:
            line = ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if not line:
                continue

            values = line.split(",")

            # Expected Arduino output: 6 comma-separated numbers
            if len(values) != 6:
                continue

            try:
                numbers = [float(v) for v in values]
            except ValueError:
                continue

            timestamp = time.time() - start_time
            writer.writerow([timestamp] + numbers)
            file.flush()

            samples += 1
            if samples % 20 == 0:
                print(
                    f"Samples: {samples} | "
                    f"Latest reading: {numbers}"
                )

except KeyboardInterrupt:
    print("\nCollection stopped by user.")

except serial.SerialException as error:
    print("Serial connection error:", error)
    print("Close Arduino Serial Monitor and check COM port.")

finally:
    try:
        ser.close()
    except (NameError, AttributeError):
        pass

print("Check stationary_2.csv in the current folder.")

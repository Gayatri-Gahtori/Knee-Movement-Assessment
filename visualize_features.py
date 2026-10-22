
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = "analysis_output"
features_path = os.path.join(folder, "movement_features.csv")
clean_path = os.path.join(folder, "cleaned_mpu_data.csv")

features = pd.read_csv(features_path)
data = pd.read_csv(clean_path)

# Exclude IDs and timestamps from movement-feature comparisons
feature_cols = [
    c for c in features.columns
    if c not in [
        "window_id", "start_s", "end_s", "sample_count"
    ]
]

# Rank windows by movement variability
features["movement_score"] = (
    features["accel_dynamic_indicator_std"].fillna(0)
    + features["gyro_magnitude_std"].fillna(0)
)

ranked = features.sort_values(
    "movement_score", ascending=False
)

print("\nTOP 10 WINDOWS WITH HIGHEST MOVEMENT VARIABILITY")
print(
    ranked[
        [
            "window_id", "start_s", "end_s",
            "accel_dynamic_indicator_std",
            "gyro_magnitude_std", "movement_score"
        ]
    ].head(10).round(4).to_string(index=False)
)

# Save ranked windows for later inspection
ranked.to_csv(
    os.path.join(folder, "ranked_movement_windows.csv"),
    index=False
)

# Plot movement scores across windows
plt.figure(figsize=(12, 5))
plt.plot(
    features["start_s"],
    features["movement_score"],
    marker="o"
)
plt.xlabel("Window start time (seconds)")
plt.ylabel("Movement variability score")
plt.title("Movement Variability Across Recording")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(folder, "movement_score.png"),
    dpi=150
)
plt.close()

# Plot the raw signals for the five highest-score windows
top_windows = ranked.head(5)

fig, axes = plt.subplots(
    len(top_windows), 1,
    figsize=(12, 12)
)

for ax, (_, row) in zip(axes, top_windows.iterrows()):
    start = row["start_s"]
    end = row["end_s"]

    segment = data[
        (data["timestamp_s"] >= start)
        & (data["timestamp_s"] < end)
    ]

    ax.plot(
        segment["timestamp_s"],
        segment["accel_magnitude"],
        label="Acceleration magnitude"
    )
    ax.plot(
        segment["timestamp_s"],
        segment["gyro_magnitude"],
        label="Gyroscope magnitude"
    )
    ax.set_title(
        f"Window {int(row['window_id'])}: "
        f"{start:.1f}–{end:.1f} s"
    )
    ax.set_xlabel("Time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(folder, "top_movement_windows.png"),
    dpi=150
)
plt.close()

print("\nSaved:")
print("- analysis_output/ranked_movement_windows.csv")
print("- analysis_output/movement_score.png")
print("- analysis_output/top_movement_windows.png")
print("\nVisualization analysis complete.")
print(
    "Note: high variability may reflect movement, "
    "sensor handling, or noise; it is not a diagnosis."
)

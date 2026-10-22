
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --------------------------------------------------
# 1. File locations
# --------------------------------------------------
FILES = {
    "stationary_1.csv": "stationary",
    "moving_1.csv": "moving",
    "stationary_2.csv": "stationary",
}

OUTPUT_DIR = Path("ml_output")
OUTPUT_DIR.mkdir(exist_ok=True)

required_columns = [
    "timestamp_s",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
]

# --------------------------------------------------
# 2. Read recordings and extract 2-second windows
# --------------------------------------------------
all_windows = []

for filename, label in FILES.items():
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {filename}. Put it in the project folder."
        )

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    df = df[required_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().sort_values("timestamp_s").reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"{filename} contains too few valid samples.")

    # Make timestamps start at zero for each recording
    df["relative_s"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    df["accel_magnitude"] = np.sqrt(
        df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2
    )
    df["gyro_magnitude"] = np.sqrt(
        df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2
    )

    df["window_id"] = (df["relative_s"] // 2).astype(int)

    for window_id, window in df.groupby("window_id"):
        if len(window) < 10:
            continue

        features = {}
        for col in [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_magnitude", "gyro_magnitude",
        ]:
            features[f"{col}_mean"] = window[col].mean()
            features[f"{col}_std"] = window[col].std()
            features[f"{col}_min"] = window[col].min()
            features[f"{col}_max"] = window[col].max()

        features["label"] = label
        features["recording"] = filename
        features["window_id"] = int(window_id)
        features["start_s"] = window["relative_s"].min()
        features["end_s"] = window["relative_s"].max()

        all_windows.append(features)

dataset = pd.DataFrame(all_windows)

if dataset.empty:
    raise ValueError("No usable windows were generated.")

dataset.to_csv(OUTPUT_DIR / "labelled_windows.csv", index=False)

print("\nWindows created:", len(dataset))
print("\nWindows per class:")
print(dataset["label"].value_counts())
print("\nWindows per recording:")
print(dataset.groupby(["recording", "label"]).size())

# --------------------------------------------------
# 3. Check that both classes have enough windows
# --------------------------------------------------
if dataset["label"].nunique() < 2:
    raise ValueError("Both stationary and moving recordings are required.")

counts = dataset["label"].value_counts()
if counts.min() < 2:
    raise ValueError("Collect more data: each class needs at least 2 windows.")

# --------------------------------------------------
# 4. Train a preliminary classifier
# --------------------------------------------------
metadata = ["label", "recording", "window_id", "start_s", "end_s"]
X = dataset.drop(columns=metadata)
y = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nPRELIMINARY MODEL RESULTS")
print("\nConfusion matrix (label order: moving, stationary):")
print(confusion_matrix(
    y_test, predictions,
    labels=["moving", "stationary"]
))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model and feature names
joblib.dump(
    {"model": model, "feature_columns": list(X.columns)},
    OUTPUT_DIR / "movement_classifier.joblib"
)

print("\nSaved files:")
print("- ml_output/labelled_windows.csv")
print("- ml_output/movement_classifier.joblib")
print("\nThis is a preliminary model, not a validated gait classifier.")

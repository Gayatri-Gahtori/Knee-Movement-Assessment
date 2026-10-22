
import pandas as pd
from pathlib import Path

input_file = Path("analysis_output/movement_features.csv")
output_file = Path("analysis_output/ml_dataset_review.csv")

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Show the most useful signals for manually reviewing movement
review_columns = [
    "window_id",
    "start_s",
    "end_s",
    "accel_dynamic_indicator_std",
    "gyro_magnitude_std",
    "movement_score"
]

# Some files may not contain movement_score, so select available columns
review_columns = [c for c in review_columns if c in df.columns]
review = df[review_columns].copy()

# Add a blank label column for manual annotation
review["label"] = ""

review.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("\nFirst 10 windows:")
print(review.head(10).to_string(index=False))
print("\nNext: inspect the CSV and label windows only when you know")
print("whether the sensor was stationary or moving during that interval.")

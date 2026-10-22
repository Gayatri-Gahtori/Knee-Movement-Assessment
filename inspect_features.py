
import pandas as pd
import os

file_path = os.path.join(
    "analysis_output",
    "movement_features.csv"
)

df = pd.read_csv(file_path)

print("=" * 50)
print("MOVEMENT FEATURE INSPECTION")
print("=" * 50)

print("\nDataset shape:", df.shape)

print("\nFeature columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum().to_string())

print("\nFeature summary:")
print(df.describe().round(3).to_string())

print("\nVariation across windows:")
numeric = df.select_dtypes(include="number")
variation = numeric.std().sort_values(ascending=False)
print(variation.head(15).round(4).to_string())

print("\nInspection complete.")

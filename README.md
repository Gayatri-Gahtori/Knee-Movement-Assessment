# Knee Movement Assessment — Dashboard Starter

A lightweight local dashboard for the ESP32 + MPU6050 prototype.

## Current scope
- Visualize raw accelerometer and gyroscope CSV recordings.
- Show signal magnitudes and labelled feature-window counts.
- Load the preliminary movement classifier and display predictions.
- Explain completed work and remaining work for knee-angle estimation.

**Important:** The current model predicts general sensor movement (`moving` / `stationary`). It does not yet estimate knee angle, diagnose osteoarthritis, or classify abnormal gait.

## Setup
Copy `app.py`, `requirements.txt`, and `.gitignore` into your existing `Knee_Movement_Project` folder. Keep your existing files.

Expected layout:
```text
Knee_Movement_Project/
  app.py
  requirements.txt
  stationary_1.csv
  moving_1.csv
  stationary_2.csv
  ml_output/
    labelled_windows.csv
    movement_classifier.joblib
```
Raw sensor CSV columns expected: `timestamp_s, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`.

In VS Code terminal:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Share with teammates using GitHub
Commit the Python scripts, dashboard, README, and requirements file. Do not commit passwords, API keys, or large private datasets. For a lightweight repository, share only a short sample CSV if needed. Teammates can clone the repo, install requirements, and run the commands above.

## Validation
The earlier 100% score came from a small random split of windows from the same recordings. Validate on separate sessions; nearby windows from one recording are correlated and can inflate results.

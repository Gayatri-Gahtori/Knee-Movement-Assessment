from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Knee Movement Assessment", page_icon="🦵", layout="wide")
ROOT = Path(__file__).resolve().parent
ML_DIR = ROOT / "ml_output"
FEATURES_FILE = ML_DIR / "labelled_windows.csv"
MODEL_FILE = ML_DIR / "movement_classifier.joblib"

st.title("Knee Movement Assessment — Prototype Dashboard")
st.caption("ESP32 + MPU6050 | Sensor data → signal features → preliminary ML classifier")

with st.sidebar:
    st.header("About this prototype")
    st.warning("Current model predicts general sensor movement (moving/stationary). It does not yet estimate knee angle, diagnose a condition, or classify abnormal gait.")

raw_files = sorted(p for p in ROOT.glob("*.csv") if p.name not in {"labelled_windows.csv", "ml_dataset_review.csv"})
a, b, c = st.columns(3)
a.metric("Raw CSV files found", len(raw_files))
b.metric("ML window file", "Found" if FEATURES_FILE.exists() else "Not found")
c.metric("Saved model", "Found" if MODEL_FILE.exists() else "Not found")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Raw sensor readings", "ML predictions", "Review explanation"])

with tab1:
    st.subheader("Project pipeline")
    st.markdown("**MPU6050 → ESP32 → CSV → signal features → ML prediction**")
    st.markdown("- Accelerometer: acceleration, including gravity.\n- Gyroscope: angular velocity.\n- Window features: summaries of short sensor intervals.\n- Current model: moving vs stationary.")
    if FEATURES_FILE.exists():
        features = pd.read_csv(FEATURES_FILE)
        st.metric("Feature windows", len(features))
        if "label" in features.columns:
            counts = features["label"].value_counts().rename_axis("Class").reset_index(name="Windows")
            st.plotly_chart(px.bar(counts, x="Class", y="Windows", title="Labelled windows by class"), use_container_width=True)
        st.info("The earlier 100% score used a random split of windows from the same recordings. It is preliminary, not proof of generalization.")

with tab2:
    st.subheader("Inspect raw sensor data")
    if not raw_files:
        st.info("Copy one or more raw sensor CSV files next to app.py. Expected columns: timestamp_s, accel_x/y/z, gyro_x/y/z.")
    else:
        selected = st.selectbox("Choose recording", raw_files, format_func=lambda p: p.name)
        try:
            raw = pd.read_csv(selected)
            expected = ["timestamp_s", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
            if not set(expected).issubset(raw.columns):
                st.error("This CSV does not contain the expected timestamp and six sensor columns.")
                st.write("Columns found:", raw.columns.tolist())
            else:
                raw = raw[expected].apply(pd.to_numeric, errors="coerce").dropna()
                raw["accel_magnitude"] = np.sqrt(raw.accel_x**2 + raw.accel_y**2 + raw.accel_z**2)
                raw["gyro_magnitude"] = np.sqrt(raw.gyro_x**2 + raw.gyro_y**2 + raw.gyro_z**2)
                duration = raw.timestamp_s.max() - raw.timestamp_s.min()
                st.caption(f"{len(raw):,} valid samples | {duration:.1f} seconds")
                signals = st.multiselect("Signals to plot", expected[1:], default=["accel_x", "accel_y", "accel_z"])
                if signals:
                    plot_df = raw[["timestamp_s"] + signals].melt(id_vars="timestamp_s", var_name="Signal", value_name="Value")
                    st.plotly_chart(px.line(plot_df, x="timestamp_s", y="Value", color="Signal", title="Raw sensor signals"), use_container_width=True)
                left, right = st.columns(2)
                left.plotly_chart(px.line(raw, x="timestamp_s", y="accel_magnitude", title="Acceleration magnitude"), use_container_width=True)
                right.plotly_chart(px.line(raw, x="timestamp_s", y="gyro_magnitude", title="Gyroscope magnitude"), use_container_width=True)
                with st.expander("View first 100 rows"):
                    st.dataframe(raw.head(100), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

with tab3:
    st.subheader("Window-level model predictions")
    if not FEATURES_FILE.exists() or not MODEL_FILE.exists():
        st.info("Place ml_output/labelled_windows.csv and ml_output/movement_classifier.joblib in the expected folder.")
    else:
        try:
            windows = pd.read_csv(FEATURES_FILE)
            bundle = joblib.load(MODEL_FILE)
            model = bundle["model"]
            cols = bundle["feature_columns"]
            if not all(col in windows.columns for col in cols):
                st.error("Feature columns do not match the saved model.")
            else:
                windows["predicted_label"] = model.predict(windows[cols])
                counts = windows["predicted_label"].value_counts().rename_axis("Prediction").reset_index(name="Windows")
                st.plotly_chart(px.bar(counts, x="Prediction", y="Windows", title="Predicted class counts"), use_container_width=True)
                if "recording" in windows.columns:
                    choice = st.selectbox("Filter recording", ["All recordings"] + sorted(windows.recording.dropna().unique().tolist()))
                    shown = windows if choice == "All recordings" else windows[windows.recording == choice]
                else:
                    shown = windows
                display_cols = [x for x in ["recording", "window_id", "start_s", "end_s", "label", "predicted_label"] if x in shown.columns]
                st.dataframe(shown[display_cols], use_container_width=True)
                st.caption("These predictions are on the existing dataset and are for inspection only, not an independent accuracy test.")
        except Exception as exc:
            st.error(f"Could not load model/data: {exc}")

with tab4:
    st.subheader("Simple explanation for your review")
    st.markdown("""
**Completed**
1. Read acceleration and gyroscope values from the MPU6050 via ESP32.
2. Saved readings to CSV.
3. Summarized short windows into numerical features.
4. Trained a preliminary moving/stationary classifier.
5. Built this dashboard to inspect signals and predictions.

**Still needed for the actual knee-movement problem**
- Securely mount two sensors, one on thigh and one on shank.
- Calibrate and estimate each segment's orientation.
- Estimate relative thigh–shank angle and define interpretable movement patterns.
- Validate against reference-angle data and separate recording sessions.
""")
    st.warning("This is a research prototype, not a medical diagnostic tool. MPU6050 has no magnetometer, and gyroscope integration can drift.")

st.divider()
st.caption("Local dashboard prototype; this app does not upload data anywhere.")

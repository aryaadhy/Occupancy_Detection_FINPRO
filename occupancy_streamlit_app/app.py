import streamlit as st
import os
import pandas as pd
import pickle
from feature_engineering import create_features

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Room Occupancy Detection",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Smart Room Occupancy Prediction")
st.markdown(
    "Predict whether a room is **occupied or not** based on environmental sensor data."
)

st.divider()

# ==============================
# Load Model
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "occupancy_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()  # stop app if model not found

# ==============================
# Input Form
# ==============================
st.subheader("📊 Input Sensor Values")

col1, col2 = st.columns(2)

with col1:
    Light = st.number_input("💡 Light (Lux)", value=418)
    CO2 = st.number_input("🌫 CO2 (ppm)", value=680)
    hour = st.slider("⏰ Hour of Day", 0, 23)

with col2:
    Temperature = st.number_input("🌡 Temperature (°C)", value=23)
    HumidityRatio = st.number_input(
        "💧 Humidity Ratio", value=0.00480, format="%.5f"
    )

st.divider()

predict_button = st.button("🔍 Predict Occupancy")

# ==============================
# Prediction Logic
# ==============================
if predict_button:
    input_data = pd.DataFrame({
        "Light": [Light],
        "CO2": [CO2],
        "hour": [hour],
        "Temperature": [Temperature],
        "HumidityRatio": [HumidityRatio]
    })

    # Feature engineering
    try:
        input_data = create_features(input_data)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

    features = [
        "Light_Temp_ratio",
        "CO2_Humidity_ratio",
        "hour",
        "Temperature",
        "HumidityRatio",
        "hour_cos",
        "hour_sin"
    ]

    try:
        prediction = model.predict(input_data[features])
    except KeyError as e:
        st.error(f"Missing features in input data: {e}")
        st.stop()

    st.divider()
    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:
        st.success("🟢 Room is OCCUPIED")
        st.metric(label="Occupancy Status", value="Occupied")
        st.info(
            "💡 Energy systems such as lighting or HVAC **should remain active**."
        )
    else:
        st.error("🔴 Room is NOT OCCUPIED")
        st.metric(label="Occupancy Status", value="Not Occupied")
        st.warning(
            "⚡ Energy saving opportunity detected. Systems like **lighting or AC can be turned off**."
        )
import streamlit as st
import pandas as pd
import pickle
from feature_engineering import create_features

# Page config
st.set_page_config(
    page_title="Room Occupancy Detection",
    page_icon="🏢",
    layout="centered"
)

# Load model
model = pickle.load(open("models/occupancy_model.pkl","rb"))

st.title("🏢 Smart Room Occupancy Prediction")
st.markdown("Predict whether a room is **occupied or not** based on environmental sensor data.")

st.divider()

st.subheader("📊 Input Sensor Values")

col1, col2 = st.columns(2)

with col1:
    Light = st.number_input(
        "💡 Light (Lux)",
        value=418
    )

    CO2 = st.number_input(
        "🌫 CO2 (ppm)", 
        value=680
    )

    hour = st.slider(
        "⏰ Hour of Day",
        0,23
    )

with col2:
    Temperature = st.number_input(
        "🌡 Temperature (°C)",
        value=23
    )

    HumidityRatio = st.number_input(
        "💧 Humidity Ratio",
        value=0.00480,
        format="%.5f"
    )

st.divider()

predict_button = st.button("🔍 Predict Occupancy")

if predict_button:

    input_data = pd.DataFrame({
        "Light":[Light],
        "CO2":[CO2],
        "hour":[hour],
        "Temperature":[Temperature],
        "HumidityRatio":[HumidityRatio]
    })

    # feature engineering
    input_data = create_features(input_data)

    features = [
        "Light_Temp_ratio",
        "CO2_Humidity_ratio",
        "hour",
        "Temperature",
        "HumidityRatio",
        "hour_cos",
        "hour_sin"
    ]

    prediction = model.predict(input_data[features])

    st.divider()
    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:

        st.success("🟢 Room is OCCUPIED")

        st.metric(
            label="Occupancy Status",
            value="Occupied"
        )

        st.info(
            "💡 Energy systems such as lighting or HVAC **should remain active**."
        )

    else:

        st.error("🔴 Room is NOT OCCUPIED")

        st.metric(
            label="Occupancy Status",
            value="Not Occupied"
        )

        st.warning(
            "⚡ Energy saving opportunity detected. Systems like **lighting or AC can be turned off**."
        )
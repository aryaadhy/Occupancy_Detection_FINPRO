import streamlit as st
import pandas as pd
import pickle
from feature_engineering import create_features

st.write("Step 1: App started")
st.write("Step 2: Import success")

# Load model
model = pickle.load(open("models/occupancy_model.pkl","rb"))

st.write("Step 3: Model loaded")

st.title("Room Occupancy Prediction")

st.write("Input sensor values")

Light = st.number_input("Light")
CO2 = st.number_input("CO2")
hour = st.slider("Hour",0,23)
Temperature = st.number_input("Temperature")
HumidityRatio = st.number_input(
    "Humidity Ratio",
    value=0.00480,
    format="%.5f"
)

st.write("Step 4: UI start and Input loaded")

if st.button("Predict"):

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

    if prediction[0] == 1:
        st.success("Room is Occupied")
    else:
        st.success("Room is Not Occupied")
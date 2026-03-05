import numpy as np
import pandas as pd

def create_features(df):

    # Cyclical encoding
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # Ratio Features
    df["Light_Temp_ratio"] = df["Light"] / df["Temperature"]
    df["CO2_Humidity_ratio"] = df["CO2"] / df["HumidityRatio"]

    return df
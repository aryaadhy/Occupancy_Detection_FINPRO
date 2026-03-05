import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier

# ======================
# Load Dataset

df = pd.read_csv("https://raw.githubusercontent.com/aryaadhy/Occupancy_Detection_FINPRO/main/dataset/datatraining.csv")
df.drop_duplicates(keep='first', inplace=True)
df['date'] = pd.to_datetime(df['date'])


# ======================
# Feature Engineering
# ======================

df["hour"] = df["date"].dt.hour
df.set_index('date', inplace=True)

# Cyclical encoding
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

# Ratio Features
df["Light_Temp_ratio"] = df["Light"] / df["Temperature"]
df["CO2_Humidity_ratio"] = df["CO2"] / df["Humidity"]

# ======================
# Feature Selection
# ======================

features = [
    "Light_Temp_ratio",
    "CO2_Humidity_ratio",
    "hour",
    "Temperature",
    "HumidityRatio",
    "hour_cos",
    "hour_sin"
]

X = df[features]
y = df["Occupancy"]

# ======================
# Train Model
# ======================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=5
)

model.fit(X, y)

# ======================
# Save Model
# ======================

with open("models/occupancy_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model training complete")
print("Model saved to models/occupancy_model.pkl")
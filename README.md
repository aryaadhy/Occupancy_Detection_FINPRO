# 🏢 Smart Room Occupancy Detection
### Machine Learning for Energy Optimization

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

---

# 📌 Project Overview

Buildings often waste electricity because **lighting and HVAC systems remain active even when rooms are empty**.

This project develops a **Machine Learning model that detects room occupancy using environmental sensor data** and enables **smart energy management systems**.

The goal is to allow buildings to automatically:

- 💡 Turn off lights when rooms are empty  
- ❄️ Reduce HVAC activity  
- 🌱 Improve energy efficiency  
- 💰 Reduce operational costs  

---

# 🚀 Live Demo

Try the deployed application online:

👉 **Streamlit App**  
https://occupancydetectionfinpro-8c84xbtlf66sjmzuhtfvtm.streamlit.app

The application allows users to input sensor values and receive **real-time occupancy predictions**.

---

# 📊 Dataset

The dataset contains indoor environmental sensor measurements.

| Feature | Description |
|------|------|
| Temperature | Room temperature (°C) |
| Humidity | Relative humidity |
| Light | Light intensity (Lux) |
| CO2 | Carbon dioxide concentration |
| HumidityRatio | Absolute humidity |
| Occupancy | Target variable (0 = Empty, 1 = Occupied) |

Multiple datasets were used to ensure **model stability and generalization**.

---

# ⚙️ Feature Engineering

Several feature engineering techniques were applied to improve model performance.

### Cyclical Time Encoding

```python
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
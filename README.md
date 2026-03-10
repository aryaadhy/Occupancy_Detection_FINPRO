# 🏢 Smart Room Occupancy Detection
### Machine Learning for Energy Optimization

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Buildings often waste electricity because **lighting and HVAC systems remain active even when rooms are empty**.

This project develops a **Machine Learning model to detect room occupancy using environmental sensor data**, enabling smart systems to automatically optimize energy consumption.

By detecting occupancy in real-time, buildings can:

- 💡 Turn off lights in empty rooms  
- ❄️ Reduce HVAC usage  
- 🌱 Improve energy efficiency  
- 💰 Reduce operational costs  

---

## 🎯 Project Objective

Build a **predictive model that can accurately detect room occupancy** using environmental sensors.

The model is designed to support **smart building automation systems** that adjust energy usage based on real-time occupancy.

---

## 📊 Dataset

The dataset contains environmental measurements from indoor sensors.

| Feature | Description |
|------|------|
| Temperature | Room temperature (°C) |
| Humidity | Relative humidity |
| Light | Light intensity (Lux) |
| CO2 | Carbon dioxide concentration |
| HumidityRatio | Absolute humidity |
| Occupancy | Target variable (0 = empty, 1 = occupied) |

The model was evaluated using **multiple test datasets** to ensure stability and generalization.

---

## ⚙️ Feature Engineering

To improve predictive performance, several feature engineering techniques were applied.

### 1️⃣ Cyclical Time Encoding

Human activity follows daily patterns. Time features were transformed using cyclical encoding.

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
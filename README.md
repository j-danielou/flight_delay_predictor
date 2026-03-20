# Flight Departure Risk Predictor (NYC)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Classification-yellow.svg)

> A Machine Learning-based decision support tool designed to predict flight departure delays from New York (JFK, EWR, LGA) based on local weather conditions and temporal factors.

**[TEST THE LIVE APPLICATION (STREAMLIT CLOUD)](https://flight-delay-predictor-nyc.streamlit.app/)** 
---

## Project Context

Flight delays cost the aviation industry billions of dollars annually. This project leverages the well-known `nycflights13` dataset (over 330,000 flights) combined with local weather reports to identify the underlying risk factors for departure delays (>15 minutes).

**Key Achievements:**
- Robust database merging (Data Engineering) with strict prevention of data leakage.
- Temporal Feature Engineering (Creation of *Time of Day*, *Day of Week* features).
- Training and hyperparameter tuning of an **XGBoost** model via `RandomizedSearchCV`.
- End-to-End deployment using a REST API and an Interactive Web Interface.

---

## Model & Performance

The champion model is an optimized **XGBoost Classifier** (`n_estimators=250`, `max_depth=8`, `learning_rate=0.2`). 
It was specifically selected and tuned for its ability to generalize well without overfitting on a highly imbalanced dataset.

* **F1-Score (Test Set):** 0.53
* **Recall:** 65%
* **Precision:** 45%

**Top 3 Feature Importances (Model Explainability):**
1. `time_of_day_Morning`: Morning flights are the #1 indicator of punctuality (absence of the operational "domino effect").
2. `precip` & `visib`: Stormy weather and low visibility (fog) are the primary environmental disruptors.
3. `carrier_EV`: Smaller regional carriers are statistically more prone to delays during heavy Air Traffic Control regulations compared to major airlines.

---

## MLOps Architecture

This repository demonstrates proficiency in two distinct deployment architectures:

1. **Cloud Architecture (Monolith) - `app_cloud.py`**: The model inference is integrated directly within Streamlit for seamless, serverless hosting (optimized for Streamlit Community Cloud).
2. **Microservices Architecture - `app.py` + `api.py`**: A clean separation of Front-End and Back-End via FastAPI. This approach is highly scalable and represents enterprise-grade production standards.

```text
flight_delay_predictor/
│
├── data/                   # (Git Ignored) Raw and processed datasets
├── models/                 # Serialized XGBoost model and features (.joblib)
├── src/
│   ├── feature_engineering.py # Data cleaning and joins
│   ├── train_xgboost.py       # Model training and optimization script
│   ├── train_model.py         # Baseline Model training script  
│   ├── api.py                 # FastAPI Server (Back-end)
│   ├── app.py                 # Streamlit Interface (Microservice)
│   └── app_cloud.py           # Streamlit Interface (Monolith for Cloud hosting)
│
├── requirements.txt        # Project dependencies
└── README.md               # This file

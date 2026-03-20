from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Flight Delay Predictor API",
    description="Une IA qui prédit les retards de vols grâce à la météo et au calendrier",
    version="1.0"
)

print("Chargement du modèle XGBoost et de ses colonnes")
model = joblib.load('models/flight_xgboost_model.joblib')
model_columns = joblib.load('models/model_features.joblib')

class FlightTicket(BaseModel):
    month: int
    day_of_week: int  # 0 = Monday, 6 = Sunday
    time_of_day: str  # 'Morning', 'Afternoon', 'Evening', 'Night'
    origin: str       # 'JFK', 'EWR', 'LGA'
    carrier: str      # 'DL', 'UA', 'AA', etc.
    temp: float
    wind_speed: float
    precip: float
    visib: float

@app.post("/predict")
def predict_delay(ticket: FlightTicket):

    df_input = pd.DataFrame([ticket.model_dump()])
    
    df_encoded = pd.get_dummies(df_input)
    
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(df_final)[0]
    probabilite = model.predict_proba(df_final)[0][1] 
    
    return {
        "is_delayed": int(prediction),
        "delay_probability": round(float(probabilite) * 100, 2),
        "message": "High Risk of Delay!" if prediction == 1 else "Flight is likely On Time"
    }
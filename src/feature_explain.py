import joblib
import pandas as pd

print("Analyse XGboost\n")
model = joblib.load('models/flight_xgboost_model.joblib')
features = joblib.load('models/model_features.joblib')

importance_df = pd.DataFrame({
    'Variable': features,
    'Importance (%)': model.feature_importances_ * 100
})

importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)

print("TOP CAUSES DE RETARD :\n")
print(importance_df.head(10).to_string(index=False))
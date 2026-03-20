import pandas as pd
import time
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_optimized_xgboost():
    print("Démarrage de l'entraînement XGBoost avec Optimisation")
    start_time = time.time()

    df = pd.read_csv('data/processed/model_dataset.csv')
    features = ['month', 'day_of_week', 'time_of_day', 'origin', 'carrier', 'temp', 'wind_speed', 'precip', 'visib']
    X = df[features]
    y = df['is_delayed']

    X_encoded = pd.get_dummies(X, columns=['origin', 'carrier', 'time_of_day'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    ratio_retard = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    print(f"Ratio de déséquilibre calculé : {ratio_retard:.2f}")

    xgb = XGBClassifier(
        scale_pos_weight=ratio_retard, 
        random_state=42, 
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],      
        'max_depth': [5, 8, 12, 14, 16, 18],              
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]     
    }

    print("GridSearchCV en cours.")
    
    random_search = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_grid, 
        n_iter=15, 
        cv=4, 
        scoring='f1',
        verbose=1, 
        random_state=42
    )

    random_search.fit(X_train, y_train)

    duree = (time.time() - start_time) / 60
    print(f"\nOptimisation terminée en {duree:.1f} minutes.")
    print(f"Les meilleurs paramètres sont : {random_search.best_params_}")

    best_model = random_search.best_estimator_

    print("\nÉVALUATION SUR LE JEU D'ENTRAÎNEMENT")
    y_train_pred = best_model.predict(X_train)
    print(classification_report(y_train, y_train_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    print("\nÉVALUATION SUR LE JEU DE TEST")
    y_test_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    print("\nEntraînement terminé.\n")

    y_pred = best_model.predict(X_test)
    
    print("\nRésultats\n")
    print(classification_report(y_test, y_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/flight_xgboost_model.joblib')
    joblib.dump(list(X_encoded.columns), 'models/model_features.joblib')
    print("Modèle XGBoost sauvegardé")

if __name__ == "__main__":
    train_optimized_xgboost()
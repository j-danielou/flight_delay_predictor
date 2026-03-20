import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_flight_model():
    print("Démarrage de l'entraînement")

    df = pd.read_csv('data/processed/model_dataset.csv')

    features = ['month', 'hour', 'origin', 'carrier', 'temp', 'wind_speed', 'precip', 'visib']
    X = df[features]
    y = df['is_delayed']

    print("Transformation des données textuelles")
    X_encoded = pd.get_dummies(X, columns=['origin', 'carrier'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print(f"données train : {len(X_train)} vols et données test : {len(X_test)} vols.")

    print("Entraînement en cours")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        class_weight='balanced',
        random_state=42,
        n_jobs=-1 
    )
    model.fit(X_train, y_train)

    # print("\nÉVALUATION SUR LE JEU D'ENTRAÎNEMENT")
    # y_train_pred = model.predict(X_train)
    # print(classification_report(y_train, y_train_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    # print("\nÉVALUATION SUR LE JEU DE TEST")
    # y_test_pred = model.predict(X_test)
    # print(classification_report(y_test, y_test_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    # print("\nEntraînement terminé.\n")
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred, target_names=["À l'heure (0)", "En retard (1)"]))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/flight_rf_model.joblib')
    
    joblib.dump(list(X_encoded.columns), 'models/model_features.joblib')
    
    print("\n model sauvegardé dans le dossier models/")

if __name__ == "__main__":
    train_flight_model()
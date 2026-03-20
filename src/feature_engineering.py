import pandas as pd
import os

def build_dataset():

    print("Chargement des fichiers CSV")
    flights = pd.read_csv('data/raw/flights.csv')
    weather = pd.read_csv('data/raw/weather.csv')

    flights['is_delayed'] = (flights['arr_delay'] > 15).astype(int)

    flights['hour'] = (flights['sched_dep_time'] // 100).astype(int)
    
    print("Création des nouvelles variables")
    
    flights['date'] = pd.to_datetime(flights[['year', 'month', 'day']])
    flights['day_of_week'] = flights['date'].dt.dayofweek 
    
    def categorize_time(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        elif 18 <= hour <= 23: return 'Evening'
        else: return 'Night'
        
    flights['time_of_day'] = flights['hour'].apply(categorize_time)

    print("Fusion des vols avec les données météorologiques")
    dataset = pd.merge(
        flights, weather, 
        how='inner', 
        on=['origin', 'year', 'month', 'day', 'hour']
    )

    colonnes_a_garder = [
        'month', 'day_of_week', 'time_of_day', 
        'origin', 'carrier',                   
        'temp', 'wind_speed', 'precip', 'visib', 
        'is_delayed'                           
    ]
    
    dataset_clean = dataset[colonnes_a_garder].dropna()

    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/model_dataset.csv'
    dataset_clean.to_csv(output_path, index=False)

    print(f"Le jeu de données final contient {len(dataset_clean)} lignes.")
    print(f"Taux de vols en retard (>15 min) : {dataset_clean['is_delayed'].mean() * 100:.2f}%")

if __name__ == "__main__":
    build_dataset()
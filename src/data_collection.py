import pandas as pd
from nycflights13 import flights, weather
import os

def fetch_aviation_data():
    print("Lancement du téléchargement des données de vols et météo")

    df_flights = flights
    df_weather = weather

    print(f"{len(df_flights)} vols récupérés.")
    print(f"{len(df_weather)} relevés météo récupérés.")

    print("Nettoyage des données de vols")
    df_flights_clean = df_flights.dropna(subset=['arr_time', 'arr_delay'])
    print(f"Après nettoyage des vols annulés, il reste {len(df_flights_clean)} vols effectifs.")

    os.makedirs('data/raw', exist_ok=True)

    flights_path = 'data/raw/flights.csv'
    weather_path = 'data/raw/weather.csv'
    
    df_flights_clean.to_csv(flights_path, index=False)
    df_weather.to_csv(weather_path, index=False)
    
    print(f"Fichiers sauvegardés avec succès dans le dossier data/raw/")
    
    print("\nAperçu des vols")
    print(df_flights_clean[['year', 'month', 'day', 'carrier', 'origin', 'dest', 'arr_delay']].head())

if __name__ == "__main__":
    fetch_aviation_data()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_data_ingestion_global.py
-------------------------------------------------
Script unificado de ingesta de datos para múltiples ligas.
Compatible con ejecución local y GitHub Actions.
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

# ==============================================================
# 🔑 Carga de API Key (desde entorno o archivo config/credentials.json)
# ==============================================================
def load_api_key():
    api_key = os.getenv("FOOTBALL_API_KEY")

    if not api_key and os.path.exists("config/credentials.json"):
        try:
            with open("config/credentials.json", "r") as f:
                creds = json.load(f)
                api_key = creds.get("api_football_key")
        except Exception as e:
            print(f"⚠️ Error leyendo credentials.json: {e}")

    if not api_key:
        print("❌ ERROR: No se encontró FOOTBALL_API_KEY ni credentials.json")
        print("   Usa export FOOTBALL_API_KEY='TU_API_KEY_AQUI'")
        exit(1)

    return api_key


FOOTBALL_API_KEY = load_api_key()
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# ==============================================================
# 🌍 Ligas objetivo
# ==============================================================

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    61: "Ligue 1",
    135: "Serie A",
    78: "Bundesliga",
    262: "Liga MX",
}

SEASON = 2024
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# ⚙️ Función para consultar API
# ==============================================================

def get_api_data(url: str, params: dict) -> dict:
    """Consulta la API con manejo de errores y reintentos."""
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⏳ Límite de peticiones alcanzado. Esperando 60s...")
                time.sleep(60)
            else:
                print(f"⚠️ Error HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Intento {attempt}/{retries} fallido: {e}")
            time.sleep(5)

    print("❌ Error persistente: no se pudo obtener datos.")
    return {}


# ==============================================================
# 📊 Descarga y almacenamiento
# ==============================================================

def fetch_fixtures(league_id: int, league_name: str):
    """Descarga fixtures para una liga específica."""
    url = "https://v3.football.api-sports.io/v3/fixtures"
    params = {"league": league_id, "season": SEASON}

    print(f"🏟️ Descargando {league_name} temporada {SEASON}...")
    data = get_api_data(url, params)
    if not data or "response" not in data:
        print(f"⚠️ No se encontraron datos para {league_name}.")
        return

    df = pd.json_normalize(data["response"])
    file_path = os.path.join(OUTPUT_DIR, f"{league_name.lower().replace(' ', '_')}_{SEASON}.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ {league_name} guardada en: {file_path}")


# ==============================================================
# 🚀 Ejecución principal
# ==============================================================

def main():
    print(f"\n🚀 Iniciando ingesta global de datos para {len(LEAGUES)} ligas...")
    print(f"📅 Temporada: {SEASON}\n")

    for league_id, league_name in LEAGUES.items():
        fetch_fixtures(league_id, league_name)
        time.sleep(3)  # Delay para evitar rate limit

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ Ingesta completada correctamente a las {timestamp}.\n")
    print(f"Archivos disponibles en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
import os
import json
import requests
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# === CONFIGURACIÓN ===
CONFIG_PATH = "config/credentials.json"
RAW_DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
LOG_PATH = "data/logs/ingestion.log"

# Crear directorios si no existen
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === CARGAR CREDENCIALES ===
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

API_KEY = config["api_football_key"]
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# === FUNCIONES PRINCIPALES ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def fetch_fixtures(league_id: int, season: int):
    """Descarga los partidos de una liga en una temporada."""
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    return data.get("response", [])

def save_json(data, path):
    """Guarda un diccionario como JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def process_fixtures(fixtures):
    """Extrae solo campos útiles del JSON."""
    processed = []
    for match in fixtures:
        fixture = match.get("fixture", {})
        league = match.get("league", {})
        teams = match.get("teams", {})
        goals = match.get("goals", {})
        processed.append({
            "fixture_id": fixture.get("id"),
            "date": fixture.get("date"),
            "league": league.get("name"),
            "season": league.get("season"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "goals_home": goals.get("home"),
            "goals_away": goals.get("away"),
            "status": fixture.get("status", {}).get("short")
        })
    return processed

if __name__ == "__main__":
    try:
        LEAGUE_ID = 140  # La Liga
        SEASON = 2023

        logging.info(f"Descargando datos de LaLiga {SEASON}...")
        fixtures = fetch_fixtures(LEAGUE_ID, SEASON)

        if not fixtures:
            raise ValueError("No se obtuvieron datos desde la API.")

        # Guardar datos crudos
        raw_filename = f"fixtures_league_{LEAGUE_ID}_{SEASON}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        raw_path = os.path.join(RAW_DATA_PATH, raw_filename)
        save_json(fixtures, raw_path)
        logging.info(f"Datos crudos guardados en {raw_path}")

        # Procesar y guardar versión limpia
        processed_data = process_fixtures(fixtures)
        processed_filename = f"processed_league_{LEAGUE_ID}_{SEASON}.json"
        processed_path = os.path.join(PROCESSED_PATH, processed_filename)
        save_json(processed_data, processed_path)
        logging.info(f"Datos procesados guardados en {processed_path}")

        print(f"✅ Guardados {len(fixtures)} partidos de LaLiga {SEASON}")
        print(f"🗃 Crudos: {raw_path}")
        print(f"🧹 Procesados: {processed_path}")

    except Exception as e:
        logging.error(f"Error durante la ingesta: {e}")
        print(f"❌ Error: {e}")

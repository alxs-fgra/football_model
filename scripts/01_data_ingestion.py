import os
import json
import requests
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuración
CONFIG_PATH = "config/credentials.json"
RAW_DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
LOG_PATH = "data/logs/ingestion.log"

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
API_KEY = config["api_football_key"]
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

with open('config/leagues.json', 'r') as f:
    leagues = json.load(f)['leagues']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def fetch_fixtures(league_id: int, season: int):
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    logging.info(f"API response for league {league_id}, season {season}: {data.get('results', 'N/A')} fixtures")
    fixtures = data.get("response", [])
    if not fixtures:
        logging.warning(f"No fixtures para league {league_id}, season {season} (temporada parcial o limitación de plan)")
    return fixtures

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def process_fixtures(fixtures):
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
    seasons = range(2015, 2025)  # Histórico para backtesting
    total_fixtures = 0
    for league in leagues:
        league_id = league['id']
        league_name = league['name']
        for season in seasons:
            try:
                logging.info(f"Descargando datos de {league_name} {season}...")
                fixtures = fetch_fixtures(league_id, season)
                raw_filename = f"fixtures_league_{league_id}_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                raw_path = os.path.join(RAW_DATA_PATH, raw_filename)
                save_json(fixtures, raw_path)
                logging.info(f"Datos crudos guardados en {raw_path}")
                processed_data = process_fixtures(fixtures)
                processed_filename = f"processed_league_{league_id}_{season}.json"
                processed_path = os.path.join(PROCESSED_PATH, processed_filename)
                save_json(processed_data, processed_path)
                logging.info(f"Datos procesados guardados en {processed_path}")
                total_fixtures += len(fixtures)
                print(f"✅ Guardados {len(fixtures)} partidos de {league_name} {season}")
                print(f"🗃 Crudos: {raw_path}")
                print(f"🧹 Procesados: {processed_path}")
            except Exception as e:
                logging.error(f"Error en {league_name} {season}: {e}")
                print(f"❌ Error en {league_name} {season}: {e}")
    print(f"📊 Total partidos descargados: {total_fixtures}")
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
LOG_PATH = "data/logs/ingestion_2024_ft.log"

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

LEAGUES = [
    {"id": 140, "name": "La Liga"},
    {"id": 39, "name": "Premier League"},
    {"id": 78, "name": "Bundesliga"}
]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def fetch_fixtures(league_id: int, season: int):
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&status=FT&from=2024-08-01&to=2025-10-22"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    logging.info(f"API response for league {league_id}, season {season}: {data.get('results', 'N/A')} fixtures")
    fixtures = data.get("response", [])
    if not fixtures:
        logging.warning(f"No fixtures para league {league_id}, season {season} (temporada parcial)")
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

def convert_to_jsonl(processed_path):
    jsonl_path = processed_path.replace(".json", ".jsonl")
    with open(processed_path, "r") as infile, open(jsonl_path, "w") as outfile:
        data = json.load(infile)
        for record in data:
            json.dump(record, outfile)
            outfile.write("\n")
    return jsonl_path

if __name__ == "__main__":
    total_matches = 0
    for league in LEAGUES:
        try:
            logging.info(f"Descargando {league['name']} 2024/25...")
            fixtures = fetch_fixtures(league["id"], 2025)  # Cambiar a season=2025
            if not fixtures:
                print(f"⚠️ No se obtuvieron partidos para {league['name']} 2024/25")
                continue
            raw_filename = f"fixtures_league_{league['id']}_2024_25_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            raw_path = os.path.join(RAW_DATA_PATH, raw_filename)
            save_json(fixtures, raw_path)
            processed = process_fixtures(fixtures)
            processed_filename = f"processed_league_{league['id']}_2024_25.json"
            processed_path = os.path.join(PROCESSED_PATH, processed_filename)
            save_json(processed, processed_path)
            jsonl_path = convert_to_jsonl(processed_path)
            print(f"✅ Guardados {len(fixtures)} partidos de {league['name']} 2024/25")
            print(f"🗃 Crudos: {raw_path}")
            print(f"🧹 Procesados: {processed_path}")
            print(f"📦 JSONL listo para BigQuery: {jsonl_path}")
            total_matches += len(fixtures)
        except Exception as e:
            logging.error(f"Error en {league['name']} 2024/25: {e}")
            print(f"❌ Error en {league['name']} 2024/25: {e}")
    print(f"📊 Total de partidos descargados: {total_matches}")
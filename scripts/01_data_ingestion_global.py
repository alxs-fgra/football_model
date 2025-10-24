import os
import requests
import pandas as pd
from datetime import datetime

# ==============================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================

API_BASE_URL = "https://v3.football.api-sports.io/fixtures"
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY
}

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    61: "Ligue 1",
    135: "Serie A",
    78: "Bundesliga",
    262: "Liga MX"
}

CURRENT_SEASON = 2024
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def fetch_fixtures(league_id: int, season: int):
    """Descarga los fixtures para una liga y temporada dadas."""
    params = {"league": league_id, "season": season}
    try:
        response = requests.get(API_BASE_URL, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return data["response"]
        else:
            print(f"⚠️ Error HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error al obtener datos para la liga {league_id}: {e}")
    return None


def normalize_fixtures(raw_data: list):
    """Convierte el JSON en un DataFrame tabular."""
    if not raw_data:
        return pd.DataFrame()

    records = []
    for fixture in raw_data:
        try:
            f = fixture["fixture"]
            teams = fixture["teams"]
            goals = fixture.get("goals", {})
            league = fixture.get("league", {})

            records.append({
                "fixture_id": f["id"],
                "date": f["date"],
                "venue": f.get("venue", {}).get("name", ""),
                "league_id": league.get("id"),
                "league_name": league.get("name", ""),
                "home_team": teams["home"]["name"],
                "away_team": teams["away"]["name"],
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": f["status"]["short"]
            })
        except Exception:
            continue
    return pd.DataFrame(records)


# ==============================================================
# PROCESO PRINCIPAL
# ==============================================================

def main():
    print("\n🚀 Iniciando ingesta global de datos para", len(LEAGUES), "ligas...")
    print(f"📅 Temporada: {CURRENT_SEASON}\n")

    for league_id, league_name in LEAGUES.items():
        print(f"🏟️ Descargando {league_name} temporada {CURRENT_SEASON}...")

        raw = fetch_fixtures(league_id, CURRENT_SEASON)
        if raw:
            df = normalize_fixtures(raw)
            if not df.empty:
                output_path = f"{DATA_DIR}/league_{league_id}_{CURRENT_SEASON}.csv"
                df.to_csv(output_path, index=False)
                print(f"✅ {league_name} guardada en: {output_path}")
            else:
                print(f"⚠️ Sin datos válidos para {league_name}.")
        else:
            print(f"⚠️ No se pudieron obtener datos para {league_name}.")

    print(f"\n✅ Ingesta completada correctamente a las {datetime.now().strftime('%Y%m%d_%H%M%S')}.")
    print(f"\nArchivos disponibles en: {DATA_DIR}")


if __name__ == "__main__":
    main()
import os
import json
import pandas as pd
import glob
from datetime import datetime

# ==========================================
#  CONFIGURACIÓN GENERAL
# ==========================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_PATH = "logs/feature_engineering.log"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - INFO - {message}"
    print(line)
    with open(LOG_PATH, "a") as log:
        log.write(line + "\n")

# ==========================================
#  PROCESAMIENTO DE LIGA
# ==========================================
def process_league_data(league_id):
    log_message(f"Iniciando feature engineering para la liga {league_id}...")
    processed_files = []

    for filename in sorted(os.listdir(RAW_DIR)):
        if f"league_{league_id}" in filename and filename.endswith(".json"):
            raw_path = os.path.join(RAW_DIR, filename)
            try:
                with open(raw_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                log_message(f"Error al leer {filename}, JSON inválido.")
                continue

            matches_raw = data if isinstance(data, list) else data.get("response", [])
            matches = []
            for match in matches_raw:
                info = match.get("fixture", {})
                teams = match.get("teams", {})
                goals = match.get("goals", {})

                matches.append({
                    "fixture_id": info.get("id"),
                    "date": info.get("date"),
                    "league_id": match.get("league", {}).get("id"),
                    "season": match.get("league", {}).get("season"),
                    "home_team": teams.get("home", {}).get("name"),
                    "away_team": teams.get("away", {}).get("name"),
                    "home_goals": goals.get("home"),
                    "away_goals": goals.get("away"),
                    "winner": (
                        "draw" if goals.get("home") == goals.get("away")
                        else "home" if goals.get("home") > goals.get("away")
                        else "away"
                    ),
                })

            df = pd.DataFrame(matches)
            df["total_goals"] = df["home_goals"] + df["away_goals"]
            df["goal_diff"] = (df["home_goals"] - df["away_goals"]).abs()
            df["is_draw"] = (df["home_goals"] == df["away_goals"]).astype(int)

            processed_name = f"processed_fixtures_league_{league_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            processed_path = os.path.join(PROCESSED_DIR, processed_name)
            df.to_json(processed_path, orient="records", lines=True)
            log_message(f"✅ Procesado: {processed_name} → {len(df)} partidos")
            processed_files.append(processed_path)

    log_message(f"🏁 Finalizado liga {league_id}: {len(processed_files)} archivos procesados")
    return processed_files

# ==========================================
#  GENERAR CSV FINAL CONSOLIDADO
# ==========================================
def generate_final_csv():
    log_message("🧠 Generando CSV final consolidado...")
    all_dfs = []
    league_names = {140: "la_liga", 39: "premier_league", 78: "bundesliga"}

    for league_id, name in league_names.items():
        pattern = os.path.join(PROCESSED_DIR, f"processed_fixtures_league_{league_id}_*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        df_list = [pd.read_json(f, lines=True) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        df["league"] = name
        all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(PROCESSED_DIR, "features_all_leagues_2015_2024.csv")
        final_df.to_csv(csv_path, index=False)
        log_message(f"✅ CSV FINAL GENERADO: {csv_path} → {len(final_df)} partidos")
    else:
        log_message("⚠️ No se generó CSV final: no hay datos procesados.")

# ==========================================
#  MAIN
# ==========================================
if __name__ == "__main__":
    log_message("🚀 Iniciando feature engineering global...")
    leagues = [140, 39, 78]
    all_processed = []
    for league in leagues:
        all_processed.extend(process_league_data(league))
    generate_final_csv()
    log_message("✅ Feature engineering completado exitosamente.")

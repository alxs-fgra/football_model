import os
import json
import pandas as pd
from datetime import datetime

# ==========================================
#  ⚙️ CONFIGURACIÓN GENERAL
# ==========================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_PATH = "logs/feature_engineering.log"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def log_message(message):
    """Simple logger con timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - INFO - {message}"
    print(line)
    with open(LOG_PATH, "a") as log:
        log.write(line + "\n")


# ==========================================
#  🧩 FUNCIÓN PRINCIPAL DE FEATURE ENGINEERING
# ==========================================
def process_league_data(league_id):
    """
    Procesa todos los años disponibles para una liga dada (usando los archivos descargados en 01_data_ingestion.py).
    """
    log_message(f"Iniciando feature engineering para la liga {league_id}...")
    processed_files = []

    for filename in sorted(os.listdir(RAW_DIR)):
        if f"league_{league_id}" in filename and filename.endswith(".json"):
            raw_path = os.path.join(RAW_DIR, filename)
            try:
                with open(raw_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                log_message(f"⚠️ Error al leer {filename}, archivo JSON inválido.")
                continue

            matches = []
            for match in data.get("response", []):
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

            # Feature engineering adicional
            df["total_goals"] = df["home_goals"] + df["away_goals"]
            df["goal_diff"] = (df["home_goals"] - df["away_goals"]).abs()
            df["is_draw"] = (df["home_goals"] == df["away_goals"]).astype(int)

            # Guardar CSV procesado
            processed_name = f"processed_{filename.replace('.json', '.jsonl')}"
            processed_path = os.path.join(PROCESSED_DIR, processed_name)
            df.to_json(processed_path, orient="records", lines=True)

            log_message(f"✅ Procesado: {processed_name} con {len(df)} partidos.")
            processed_files.append(processed_path)

    log_message(f"🏁 Finalizado feature engineering para liga {league_id}. Total de archivos procesados: {len(processed_files)}")
    return processed_files


# ==========================================
#  ☁️ CARGA OPCIONAL A BIGQUERY (si existen credenciales)
# ==========================================
def upload_to_bigquery(processed_files):
    """Carga los archivos procesados a BigQuery solo si existen credenciales GCP."""
    GCP_CREDENTIALS = "config/football-prediction-2025-c55b44ba599d.json"

    if not os.path.exists(GCP_CREDENTIALS):
        log_message("⚠️ No se encontraron credenciales de BigQuery. Saltando carga remota...")
        return

    try:
        from google.cloud import bigquery
        client = bigquery.Client.from_service_account_json(GCP_CREDENTIALS)
        log_message("✅ Conectado a BigQuery correctamente.")

        dataset_id = "football_data"
        table_id = "matches"

        for file_path in processed_files:
            df = pd.read_json(file_path, lines=True)
            client.load_table_from_dataframe(df, f"{dataset_id}.{table_id}")
            log_message(f"📤 Subido {os.path.basename(file_path)} a BigQuery.")

    except Exception as e:
        log_message(f"⚠️ Error al conectar con BigQuery: {e}")


# ==========================================
#  🚀 MAIN
# ==========================================
if __name__ == "__main__":
    log_message("🧠 Iniciando feature engineering global...")

    # Procesar ligas principales
    leagues = [140, 78, 135]  # Ejemplo: La Liga, Bundesliga, Serie A
    all_processed = []
    for league in leagues:
        all_processed.extend(process_league_data(league))

    # Intentar subir a BigQuery solo si aplica
    upload_to_bigquery(all_processed)

    log_message("✅ Feature engineering completado exitosamente.")

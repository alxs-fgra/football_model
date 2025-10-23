import os
import json
import glob
import pandas as pd
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
#  FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ==========================================
def process_league_data(league_id):
    log_message(f"Iniciando feature engineering para liga {league_id}...")
    processed_rows = []

    # Buscar archivos JSON de la liga
    json_files = sorted(glob.glob(os.path.join(RAW_DIR, f"fixtures_league_{league_id}_*.json")))
    if not json_files:
        log_message(f"⚠️ No se encontraron archivos para la liga {league_id}")
        return pd.DataFrame()

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "response" in data:
                data = data["response"]
        except Exception as e:
            log_message(f"⚠️ Error leyendo {file}: {e}")
            continue

        # Convertir fixtures a DataFrame
        matches = []
        for m in data:
            fixture = m.get("fixture", {})
            league = m.get("league", {})
            teams = m.get("teams", {})
            goals = m.get("goals", {})

            if not fixture or not teams:
                continue

            matches.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "season": league.get("season"),
                "league_id": league.get("id"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away")
            })

        df = pd.DataFrame(matches)
        if df.empty:
            log_message(f"⚠️ No se pudieron generar filas válidas para {os.path.basename(file)}")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # ==========================================
        #  FEATURE ENGINEERING SIN DATA LEAKAGE
        # ==========================================

        # --- Diferencia de goles ---
        df["goal_diff"] = df["home_goals"] - df["away_goals"]

        # --- Resultados categóricos ---
        df["winner"] = df["goal_diff"].apply(lambda x: "home" if x > 0 else "away" if x < 0 else "draw")

        # --- Puntos obtenidos por partido ---
        df["home_points"] = df["winner"].map({"home": 3, "draw": 1, "away": 0})
        df["away_points"] = df["winner"].map({"away": 3, "draw": 1, "home": 0})

        # --- Estadísticas previas (sin incluir el partido actual) ---
        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            # Filtrar partidos previos por equipo
            df[f"{prefix}_avg_goals_last5"] = (
                df.groupby(team_col)[f"{prefix}_goals"]
                .apply(lambda x: x.shift(1).rolling(5, closed="left").mean())
                .reset_index(level=0, drop=True)
            )
            df[f"{prefix}_avg_conceded_last5"] = (
                df.groupby(team_col)[f"{'away' if prefix == 'home' else 'home'}_goals"]
                .apply(lambda x: x.shift(1).rolling(5, closed="left").mean())
                .reset_index(level=0, drop=True)
            )
            df[f"{prefix}_form_points"] = (
                df.groupby(team_col)[f"{prefix}_points"]
                .apply(lambda x: x.shift(1).rolling(5, closed="left").sum())
                .reset_index(level=0, drop=True)
            )
            df[f"{prefix}_goal_diff_last5"] = (
                df.groupby(team_col)["goal_diff"]
                .apply(lambda x: x.shift(1).rolling(5, closed="left").mean())
                .reset_index(level=0, drop=True)
            )

        # --- Progreso de temporada ---
        df["match_index"] = df.groupby("season").cumcount() + 1
        df["season_progress"] = df["match_index"] / df.groupby("season")["match_index"].transform("max")

        processed_rows.append(df)

        log_message(f"✅ Procesado {len(df)} partidos para liga {league_id}: {os.path.basename(file)}")

    if not processed_rows:
        log_message(f"⚠️ No se generaron datos válidos para liga {league_id}")
        return pd.DataFrame()

    # Unir todos los archivos procesados de esta liga
    df_all = pd.concat(processed_rows, ignore_index=True)
    processed_path = os.path.join(PROCESSED_DIR, f"processed_fixtures_league_{league_id}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl")
    df_all.to_json(processed_path, orient="records", lines=True)
    log_message(f"💾 Guardado: {processed_path} → {len(df_all)} partidos")
    return df_all


# ==========================================
#  GENERAR CSV FINAL CONSOLIDADO
# ==========================================
def generate_final_csv():
    log_message("Generando CSV final consolidado...")
    league_dfs = []
    league_names = {140: "la_liga", 39: "premier_league", 78: "bundesliga"}

    for league_id in league_names:
        pattern = os.path.join(PROCESSED_DIR, f"processed_fixtures_league_{league_id}_*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            continue

        df_list = [pd.read_json(f, lines=True) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        df["league"] = league_names[league_id]
        league_dfs.append(df)

    if not league_dfs:
        log_message("⚠️ No hay datos para generar el CSV final.")
        return

    final_df = pd.concat(league_dfs, ignore_index=True)
    csv_path = os.path.join(PROCESSED_DIR, "features_all_leagues_2015_2024.csv")
    final_df.to_csv(csv_path, index=False)
    log_message(f"💾 CSV FINAL: {csv_path} → {len(final_df)} partidos totales")


# ==========================================
#  MAIN
# ==========================================
if __name__ == "__main__":
    log_message("🚀 Iniciando feature engineering optimizado...")
    leagues = [140, 39, 78]
    all_processed = []
    for league in leagues:
        df_league = process_league_data(league)
        if not df_league.empty:
            all_processed.append(df_league)

    if all_processed:
        generate_final_csv()
        log_message("✅ Feature engineering completado con éxito.")
    else:
        log_message("⚠️ No se generaron datos procesados válidos.")
import os
import sys
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from google.cloud import bigquery

# ---------------- CONFIGURACIÓN ---------------- #
# Ruta a las credenciales de GCP
CREDENTIALS_PATH = '/Users/B-yond/Documents/Sports Machine Learning/football_model/config/football-prediction-2025-c554bb4a599d.json'

# Crear logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Permitir pasar liga como argumento (por defecto La Liga)
LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "La Liga"
OUTPUT_PATH = f"data/processed/features_{LEAGUE.lower().replace(' ', '_')}_2015_2023.csv"

# ---------------- FUNCIONES AUXILIARES ---------------- #

def load_data_from_bq(league: str) -> pd.DataFrame:
    """Carga datos de BigQuery para una liga específica."""
    try:
        log.info(f"Conectando a BigQuery y extrayendo datos para {league}...")
        client = bigquery.Client.from_service_account_json(CREDENTIALS_PATH)
        query = f"""
        SELECT fixture_id, date, league, season, home_team, away_team,
               CAST(goals_home AS INT64) AS goals_home,
               CAST(goals_away AS INT64) AS goals_away,
               status
        FROM `football_ds.fixtures`
        WHERE league = '{league}'
          AND season BETWEEN 2015 AND 2023
          AND status = 'FT'
        """
        df = client.query(query).to_dataframe()
        log.info(f"✅ Datos cargados: {df.shape[0]} filas")
        return df
    except Exception as e:
        log.error(f"❌ Error al cargar datos desde BigQuery: {e}")
        sys.exit(1)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza los datos básicos."""
    df = df.dropna(subset=['goals_home', 'goals_away']).copy()
    df = df.drop_duplicates(subset=['fixture_id'])
    df['goals_home'] = df['goals_home'].astype(int)
    df['goals_away'] = df['goals_away'].astype(int)
    df['total_goals'] = df['goals_home'] + df['goals_away']
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df

def add_avg_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios históricos (últimos 3 años)."""
    log.info("⚙️ Calculando promedios de goles (2021–2023)...")
    recent = df[df['season'] >= 2021]
    home_avg = recent.groupby('home_team')['goals_home'].mean().rename('avg_goals_home')
    away_avg = recent.groupby('away_team')['goals_away'].mean().rename('avg_goals_away')
    df = df.merge(home_avg, left_on='home_team', right_index=True, how='left')
    df = df.merge(away_avg, left_on='away_team', right_index=True, how='left')
    return df

def calculate_form(team, date, df, n=5):
    """Calcula puntos promedio últimos N partidos."""
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    team_matches = team_matches[team_matches['date'] < date].sort_values('date', ascending=False).head(n)
    if team_matches.empty:
        return np.nan
    points = 0
    for _, r in team_matches.iterrows():
        if r['home_team'] == team:
            points += 3 if r['goals_home'] > r['goals_away'] else (1 if r['goals_home'] == r['goals_away'] else 0)
        else:
            points += 3 if r['goals_away'] > r['goals_home'] else (1 if r['goals_away'] == r['goals_home'] else 0)
    return points / len(team_matches)

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega forma reciente (últimos 5 partidos)."""
    log.info("⚙️ Calculando forma reciente (últimos 5 partidos)...")
    df['home_form'] = df.apply(lambda x: calculate_form(x['home_team'], x['date'], df), axis=1)
    df['away_form'] = df.apply(lambda x: calculate_form(x['away_team'], x['date'], df), axis=1)
    return df

@lru_cache(maxsize=5000)
def head_to_head(home, away, date, df, n=5):
    """Promedio de goles en los últimos N enfrentamientos directos."""
    h2h = df[((df['home_team'] == home) & (df['away_team'] == away)) |
             ((df['home_team'] == away) & (df['away_team'] == home))]
    h2h = h2h[h2h['date'] < date].sort_values('date', ascending=False).head(n)
    return h2h['total_goals'].mean() if not h2h.empty else np.nan

def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega promedio head-to-head."""
    log.info("⚙️ Calculando head-to-head promedio (últimos 5 partidos)...")
    df['h2h_avg_goals'] = df.apply(lambda x: head_to_head(x['home_team'], x['away_team'], x['date'], df), axis=1)
    return df

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    log.info(f"🚀 Iniciando feature engineering para {LEAGUE}...")

    df = load_data_from_bq(LEAGUE)
    df = clean_data(df)
    df = add_avg_goals(df)
    df = add_form_features(df)
    df = add_h2h(df)
    df['is_home'] = 1

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"✅ Features generadas y guardadas en {OUTPUT_PATH}")
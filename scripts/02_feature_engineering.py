import os
import pandas as pd
from google.cloud import bigquery

# === CONFIGURACIÓN ===
CREDENTIALS_PATH = '/Users/B-yond/Documents/Sports Machine Learning/football_model/config/football-prediction-2025-c554bb4a599d.json'

# Cliente BigQuery
client = bigquery.Client.from_service_account_json(CREDENTIALS_PATH)

# === 1. CARGA DE DATOS ===
print("📥 Cargando datos históricos de LaLiga 2015–2023...")
query = """
SELECT fixture_id, date, league, season, home_team, away_team,
       CAST(goals_home AS INT64) AS goals_home,
       CAST(goals_away AS INT64) AS goals_away,
       status
FROM `football_ds.fixtures`
WHERE league = 'La Liga' AND season BETWEEN 2015 AND 2023 AND status = 'FT'
"""
df = client.query(query).to_dataframe()
print(f"✅ Datos cargados: {df.shape[0]} filas")

# === 2. LIMPIEZA ===
df = df.dropna(subset=['goals_home', 'goals_away']).copy()
df['goals_home'] = df['goals_home'].astype(int)
df['goals_away'] = df['goals_away'].astype(int)
df['total_goals'] = df['goals_home'] + df['goals_away']
df['date'] = pd.to_datetime(df['date'])

# === 3. PROMEDIOS DE GOLES (últimos 3 años) ===
print("⚙️ Calculando promedios de goles (2021–2023)...")
recent_df = df[df['season'] >= 2021]
home_avg = recent_df.groupby('home_team')['goals_home'].mean().rename('avg_goals_home')
away_avg = recent_df.groupby('away_team')['goals_away'].mean().rename('avg_goals_away')
df = df.merge(home_avg, left_on='home_team', right_index=True, how='left')
df = df.merge(away_avg, left_on='away_team', right_index=True, how='left')

# === 4. FORMA RECIENTE (últimos 5 partidos) ===
def calculate_form(team, date, df, n=5):
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    team_matches = team_matches[team_matches['date'] < date].sort_values('date', ascending=False).head(n)
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            if row['goals_home'] > row['goals_away']:
                points += 3
            elif row['goals_home'] == row['goals_away']:
                points += 1
        else:
            if row['goals_away'] > row['goals_home']:
                points += 3
            elif row['goals_away'] == row['goals_home']:
                points += 1
    return points / max(1, len(team_matches))

print("⚙️ Calculando forma reciente (últimos 5 partidos)...")
df['home_form'] = df.apply(lambda x: calculate_form(x['home_team'], x['date'], df), axis=1)
df['away_form'] = df.apply(lambda x: calculate_form(x['away_team'], x['date'], df), axis=1)

# === 5. HEAD-TO-HEAD (últimos 5 enfrentamientos directos) ===
def head_to_head(home, away, date, df, n=5):
    h2h = df[
        (((df['home_team'] == home) & (df['away_team'] == away)) |
         ((df['home_team'] == away) & (df['away_team'] == home))) &
         (df['date'] < date)
    ]
    h2h = h2h.sort_values('date', ascending=False).head(n)
    return h2h['total_goals'].mean() if not h2h.empty else 0

print("⚙️ Calculando head-to-head (últimos 5)...")
df['h2h_avg_goals'] = df.apply(lambda x: head_to_head(x['home_team'], x['away_team'], x['date'], df), axis=1)

# === 6. MARCADO DE LOCAL ===
df['is_home'] = 1

# === 7. GUARDAR RESULTADO ===
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/features_laliga_2015_2023.csv'
df.to_csv(output_path, index=False)
print(f"✅ Características generadas y guardadas en {output_path}")
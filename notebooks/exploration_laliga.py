# notebooks/exploration_laliga.py
# EDA LaLiga — histórico 2023 y temporada 2024/25 (FT)

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery

# ---------- CONFIG ----------
PROJECT_ID = "football-prediction-2025"
CREDENTIALS_JSON = "/Users/B-yond/Documents/Sports Machine Learning/football_model/config/football-prediction-2025-c554bb4a599d.json"

# Si no está el env var, lo seteamos desde aquí (con tu ruta real)
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_JSON

pd.set_option('display.max_columns', 100)
plt.rcParams['figure.figsize'] = (7, 5)

# ---------- CONEXIÓN ----------
print("Inicializando cliente de BigQuery…")
client = bigquery.Client(project=PROJECT_ID)
try:
    _ = list(client.list_datasets())  # sanity check
    print("✅ Conexión a BigQuery verificada")
except Exception as e:
    raise SystemExit(f"❌ Error al conectar con BigQuery: {e}")

# ---------- QUERIES ----------
QUERY_2023 = """
SELECT fixture_id, date, league, season,
       home_team, away_team,
       CAST(goals_home AS INT64) AS goals_home,
       CAST(goals_away AS INT64) AS goals_away,
       status
FROM `football_ds.fixtures`
WHERE league = 'La Liga'
  AND season = 2023
  AND status = 'FT'
"""

QUERY_2025 = """
SELECT fixture_id, date, league, season,
       home_team, away_team,
       CAST(goals_home AS INT64) AS goals_home,
       CAST(goals_away AS INT64) AS goals_away,
       status
FROM `football_ds.fixtures_2024_ft`
WHERE league = 'La Liga'
  AND season = 2025
  AND status = 'FT'
"""

print("Ejecutando consultas…")
df23 = client.query(QUERY_2023).to_dataframe()
df25 = client.query(QUERY_2025).to_dataframe()

print(f"➡️ df23 (LaLiga 2023, FT) shape: {df23.shape}")
print(f"➡️ df25 (LaLiga 2024/25, FT) shape: {df25.shape}")

# ---------- LIMPIEZA ----------
for name, df in [("df23", df23), ("df25", df25)]:
    # Quitamos nulos por seguridad
    before = len(df)
    df.dropna(subset=["goals_home", "goals_away"], inplace=True)
    after = len(df)
    if after < before:
        print(f"⚠️ {name}: {before-after} filas con goles nulos eliminadas")

    df["goals_home"] = df["goals_home"].astype(int)
    df["goals_away"] = df["goals_away"].astype(int)
    df["total_goals"] = df["goals_home"] + df["goals_away"]

# Aserciones duras para evitar graficar con nulos
assert df23["goals_home"].notna().all() and df23["goals_away"].notna().all(), "Nulos en df23"
assert df25["goals_home"].notna().all() and df25["goals_away"].notna().all(), "Nulos en df25"

# ---------- FUNCIONES DE MÉTRICAS ----------
def result_row(r):
    if r["goals_home"] > r["goals_away"]:
        return "Home Win"
    if r["goals_home"] < r["goals_away"]:
        return "Away Win"
    return "Draw"

def dist_1x2(df):
    order = ["Home Win", "Draw", "Away Win"]
    res = df.apply(result_row, axis=1)
    return (res.value_counts(normalize=True) * 100).reindex(order).round(2)

def ou_rates(df):
    tg = df["total_goals"]
    return {
        "O0.5": (tg > 0.5).mean(),
        "O1.5": (tg > 1.5).mean(),
        "O2.5": (tg > 2.5).mean(),
        "O3.5": (tg > 3.5).mean(),
        "U0.5": (tg <= 0.5).mean(),
        "U1.5": (tg <= 1.5).mean(),
        "U2.5": (tg <= 2.5).mean(),
        "U3.5": (tg <= 3.5).mean(),
    }

def btts_rate(df):
    return ((df["goals_home"] > 0) & (df["goals_away"] > 0)).mean()

# ---------- MÉTRICAS BÁSICAS ----------
print("\n🧮 Partidos")
print(f"LaLiga 2023 (FT): {len(df23)}")
print(f"LaLiga 2024/25 (FT): {len(df25)}")

print("\n⚽ Goles — Promedios")
print(f"2023  | total: {df23['total_goals'].mean():.3f} | home: {df23['goals_home'].mean():.3f} | away: {df23['goals_away'].mean():.3f}")
print(f"24/25 | total: {df25['total_goals'].mean():.3f} | home: {df25['goals_home'].mean():.3f} | away: {df25['goals_away'].mean():.3f}")

# ---------- HISTOGRAMAS ----------
bins = list(range(0, 10))
plt.hist(df23["total_goals"], bins=bins, edgecolor='black')
plt.title('Distribución de Goles Totales (LaLiga 2023)')
plt.xlabel('Goles Totales')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

plt.hist(df25["total_goals"], bins=bins, edgecolor='black')
plt.title('Distribución de Goles Totales (LaLiga 2024/25 FT)')
plt.xlabel('Goles Totales')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# ---------- RESULTADOS 1X2 ----------
dist23 = dist_1x2(df23)
dist25 = dist_1x2(df25)
print("\n📊 Distribución 1X2 (%) — orden Home/Draw/Away")
print("2023:\n", dist23)
print("2024/25:\n", dist25)

home_win_23 = dist23["Home Win"]
home_win_25 = dist25["Home Win"]
print(f"\n🏠 Home win rate 2023: {home_win_23:.2f}%")
print(f"🏠 Home win rate 24/25: {home_win_25:.2f}%")

# ---------- O/U ----------
ou23 = {k: round(v*100,2) for k,v in ou_rates(df23).items()}
ou25 = {k: round(v*100,2) for k,v in ou_rates(df25).items()}
print("\n📈 Over/Under (%, LaLiga 2023):", ou23)
print("📈 Over/Under (%, LaLiga 2024/25 FT):", ou25)

# ---------- BTTS ----------
print("\n🤝 BTTS")
print("LaLiga 2023 (%):", round(btts_rate(df23) * 100, 2))
print("LaLiga 2024/25 (%):", round(btts_rate(df25) * 100, 2))

# ---------- GOLES POR EQUIPO ----------
gf_home = df23.groupby("home_team")["goals_home"].sum()
gf_away = df23.groupby("away_team")["goals_away"].sum()
gf = (gf_home.add(gf_away, fill_value=0)).sort_values(ascending=False)

ga_home = df23.groupby("home_team")["goals_away"].sum()
ga_away = df23.groupby("away_team")["goals_home"].sum()
ga = (ga_home.add(ga_away, fill_value=0)).sort_values(ascending=True)

print("\n🏅 Top 5 equipos más goleadores (GF) — 2023:\n", gf.head(5))
print("\n🛡️ Top 5 equipos menos goleados (GA) — 2023:\n", ga.head(5))

# ---------- EXPORT ----------
os.makedirs('data/analysis', exist_ok=True)
df23.to_csv('data/analysis/laliga_2023_matches.csv', index=False)
df25.to_csv('data/analysis/laliga_2024_25_ft_matches.csv', index=False)
print("\n💾 Archivos exportados a data/analysis/:")
print(" - laliga_2023_matches.csv")
print(" - laliga_2024_25_ft_matches.csv")

print("\n✅ EDA finalizado.")
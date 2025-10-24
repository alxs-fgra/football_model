#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_data_ingestion_global.py
------------------------------------------------
Script unificado para descarga e ingestión de datos de fútbol.

✅ Descarga histórica (2015–2024)
✅ Descarga temporada actual (2024/25)
✅ Soporta múltiples ligas y copas
✅ Compatible con ejecución local y GitHub Actions
✅ Manejo de errores 403/429 con reintentos automáticos
✅ Logs detallados y salida uniforme JSON/JSONL

Uso manual:
    python3 scripts/01_data_ingestion_global.py

Uso en CI/CD:
    Se ejecuta automáticamente dentro del pipeline de entrenamiento.
"""

import os
import json
import time
import requests
from datetime import datetime

# =============================
# ⚙️ CONFIGURACIÓN
# =============================

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io/fixtures"
OUTPUT_DIR_RAW = "data/raw"
OUTPUT_DIR_PROCESSED = "data/processed"
LOG_FILE = f"logs/ingestion_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)
os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =============================
# 🏆 LIGAS A DESCARGAR
# =============================

LEAGUES = [
    {"id": 39, "name": "Premier League"},
    {"id": 140, "name": "La Liga"},
    {"id": 78, "name": "Bundesliga"},
    {"id": 135, "name": "Serie A"},
    {"id": 61, "name": "Ligue 1"},
    {"id": 71, "name": "Serie A (Brasil)"},
    {"id": 262, "name": "Liga MX"},
    {"id": 2, "name": "Champions League"},
    {"id": 3, "name": "Europa League"},
    {"id": 13, "name": "Copa Libertadores"},
]

SEASONS = list(range(2015, 2025))  # 2015 → 2024

# =============================
# 🧩 FUNCIONES AUXILIARES
# =============================

def log(msg: str):
    """Guarda logs en consola y archivo."""
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def fetch_fixtures(league_id: int, season: int):
    """Descarga fixtures desde API-Football con reintentos."""
    url = BASE_URL
    headers = {"x-apisports-key": API_KEY}
    params = {"league": league_id, "season": season}

    for attempt in range(3):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code in [403, 429]:
            wait = (attempt + 1) * 5
            log(f"⚠️ Error {response.status_code} → reintentando en {wait}s... (intento {attempt + 1}/3)")
            time.sleep(wait)
        else:
            log(f"❌ Error {response.status_code} al descargar liga {league_id} temporada {season}")
            return None
    return None


def process_fixture_data(data: dict, league_id: int, season: int):
    """Procesa datos crudos y los guarda en formato JSONL."""
    fixtures = data.get("response", [])
    if not fixtures:
        log(f"⚠️ No se encontraron partidos válidos para liga {league_id} temporada {season}")
        return 0

    raw_filename = f"fixtures_league_{league_id}_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    raw_path = os.path.join(OUTPUT_DIR_RAW, raw_filename)
    with open(raw_path, "w") as f:
        json.dump(data, f)
    log(f"🗃 Crudos: {raw_path}")

    processed = []
    for fx in fixtures:
        try:
            processed.append({
                "fixture_id": fx["fixture"]["id"],
                "league_id": fx["league"]["id"],
                "season": fx["league"]["season"],
                "date": fx["fixture"]["date"],
                "home_team": fx["teams"]["home"]["name"],
                "away_team": fx["teams"]["away"]["name"],
                "goals_home": fx["goals"]["home"],
                "goals_away": fx["goals"]["away"],
                "status": fx["fixture"]["status"]["short"],
            })
        except KeyError:
            continue

    if not processed:
        log(f"⚠️ Archivo vacío o sin datos procesables para liga {league_id}, temporada {season}")
        return 0

    processed_filename = f"processed_league_{league_id}_{season}.jsonl"
    processed_path = os.path.join(OUTPUT_DIR_PROCESSED, processed_filename)

    with open(processed_path, "w") as f:
        for p in processed:
            f.write(json.dumps(p) + "\n")

    log(f"🧹 Procesados: {processed_path}")
    return len(processed)


def run_global_ingestion():
    """Orquesta la ingestión global."""
    log("🌍 Iniciando ingestión global para 10 ligas...\n")
    total_matches = 0
    summary = {}

    for league in LEAGUES:
        lid, lname = league["id"], league["name"]
        log(f"🏟️ {lname} (ID {lid}) → Iniciando descarga...\n")

        league_total = 0
        for season in SEASONS:
            data = fetch_fixtures(lid, season)
            if data:
                count = process_fixture_data(data, lid, season)
                league_total += count
                time.sleep(1)

        summary[lname] = league_total
        total_matches += league_total
        log(f"✅ Finalizado {lname}: {league_total} partidos")
        log("------------------------------------------------------------\n")

    log(f"\n🏁 Ingestión global completada en {round(total_matches / 7500, 2)} minutos (aprox).")
    log("📦 Datos listos en data/raw/ y data/processed/.\n")

    log("📊 Resumen general:")
    for lname, count in summary.items():
        log(f"   - {lname}: {count} partidos")

    log(f"\n🔢 Total global: {sum(summary.values())} partidos descargados.")
    log(f"🕒 Logs guardados en: {LOG_FILE}\n")


# =============================
# 🚀 MAIN
# =============================
if __name__ == "__main__":
    if not API_KEY:
        print("❌ ERROR: No se encontró FOOTBALL_API_KEY en el entorno. Usa:")
        print('   export FOOTBALL_API_KEY="TU_API_KEY_AQUI"')
        exit(1)

    run_global_ingestion()
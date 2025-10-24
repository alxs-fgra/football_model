#!/usr/bin/env python3
# ==============================================================
# Script: 02_feature_engineering.py
# Autor:  Alexis Figueroa
# Descripción:
#   Realiza el procesamiento y creación de features básicas
#   a partir de los datos crudos obtenidos por 01_data_ingestion_global.py
# ==============================================================

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# ==============================================================
# CONFIGURACIÓN INICIAL
# ==============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

LEAGUES = [39, 78, 135, 140, 61, 262]  # Premier, Bundesliga, Serie A, LaLiga, Ligue 1, Liga MX
SEASON = 2024

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("🚀 Iniciando feature engineering optimizado...")

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def safe_load_csv(path: str) -> pd.DataFrame | None:
    """Carga segura de CSV con control de errores."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.warning(f"⚠️ Error cargando {path}: {e}")
        return None


def clean_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica y creación de features simples."""
    # Normaliza nombres de columnas
    df.columns = [c.strip().lower() for c in df.columns]

    # Features de goles si existen columnas correspondientes
    if all(col in df.columns for col in ["goals_home", "goals_away"]):
        df["total_goals"] = df["goals_home"] + df["goals_away"]
        df["goal_diff"] = df["goals_home"] - df["goals_away"]
        df["is_home_win"] = np.where(df["goal_diff"] > 0, 1, 0)
        df["is_draw"] = np.where(df["goal_diff"] == 0, 1, 0)
        df["is_away_win"] = np.where(df["goal_diff"] < 0, 1, 0)
    else:
        logging.warning("⚠️ No se encontraron columnas de goles en este dataset.")

    # Manejo de NaN
    df = df.fillna(0)

    return df


# ==============================================================
# PROCESAMIENTO POR LIGA
# ==============================================================

processed_files = []

for league_id in LEAGUES:
    file_pattern = f"league_{league_id}_{SEASON}.csv"
    file_path = os.path.join(RAW_DIR, file_pattern)

    if not os.path.exists(file_path):
        logging.info(f"⚠️ No se encontró archivo para la liga {league_id}")
        continue

    logging.info(f"✅ Procesando datos de la liga {league_id} ({file_pattern})")

    df = safe_load_csv(file_path)
    if df is None or df.empty:
        logging.warning(f"⚠️ Dataset vacío o corrupto para liga {league_id}")
        continue

    df = clean_and_create_features(df)

    out_path = os.path.join(PROCESSED_DIR, f"features_league_{league_id}_{SEASON}.csv")
    df.to_csv(out_path, index=False)
    processed_files.append(out_path)
    logging.info(f"💾 Features guardadas en {out_path}")


# ==============================================================
# CONSOLIDACIÓN FINAL
# ==============================================================

if processed_files:
    logging.info("📊 Consolidando datasets procesados...")
    all_dfs = [pd.read_csv(f) for f in processed_files if os.path.exists(f)]
    merged_df = pd.concat(all_dfs, ignore_index=True)

    final_path = os.path.join(PROCESSED_DIR, f"features_combined_{timestamp}.csv")
    merged_df.to_csv(final_path, index=False)
    logging.info(f"✅ Archivo combinado guardado en: {final_path}")
else:
    logging.warning("⚠️ No se generaron datos procesados válidos.")


logging.info("🎯 Feature engineering completado correctamente.")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_predict.py — Predicciones multi-mercado (1X2, BTTS y Over/Under 2.5)
Autor: Alexis Figueroa
Versión: Paso 9 — Expansión multi-mercado
"""

# ==========================================================
# IMPORTS
# ==========================================================
import os
import joblib
import numpy as np
import pandas as pd
import datetime as dt
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================
os.makedirs("data/predictions", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN DE RUTAS Y MODELOS
# ==========================================================
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = "models"
PREPROC_FILE = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("scaler_imputer")])[-1]

PREPROC_PATH = os.path.join(MODEL_DIR, PREPROC_FILE)
log.info(f"🧩 Cargando preprocesadores desde: {PREPROC_PATH}")
preproc = joblib.load(PREPROC_PATH)
imputer = preproc["imputer"]
scaler = preproc["scaler"]

# Detectar modelos más recientes
models = {}
for market in ["1x2", "btts", "over_2.5"]:
    try:
        model_file = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(f"{market}_model_")])[-1]
        models[market] = joblib.load(os.path.join(MODEL_DIR, model_file))
        log.info(f"✅ Modelo cargado: {model_file}")
    except IndexError:
        log.warning(f"⚠️ No se encontró modelo para {market}")

if not models:
    raise FileNotFoundError("❌ No se encontraron modelos entrenados en 'models/'.")

# ==========================================================
# CARGA DE DATOS
# ==========================================================
DATA_PATH = "data/processed/features_la_liga_2024_25.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
log.info(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

# ==========================================================
# FEATURE ENRICHMENT
# ==========================================================
log.info("🧩 Enriqueciendo features automáticamente...")

if "month" not in df.columns:
    df["month"] = pd.to_datetime(df["date"]).dt.month
if "goal_diff" not in df.columns:
    df["goal_diff"] = df["avg_goals_home"] - df["avg_goals_away"]
if "rolling_avg_goals" not in df.columns:
    df["rolling_avg_goals"] = (df["avg_goals_home"] + df["avg_goals_away"]) / 2

features = [
    "avg_goals_home", "avg_goals_away", "home_form", "away_form",
    "h2h_avg_goals", "is_home", "month", "goal_diff", "rolling_avg_goals"
]
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"❌ Faltan columnas requeridas: {missing}")

# Imputar y escalar
X = imputer.transform(df[features].values)
X = scaler.transform(X)

# ==========================================================
# PREDICCIONES MULTI-MERCADO
# ==========================================================
results = {}
for market, model in models.items():
    log.info(f"⚙️ Realizando predicciones para {market}...")
    preds = model.predict(X)
    probs = model.predict_proba(X)
    df[f"{market}_prediction"] = preds
    df[f"{market}_confidence"] = probs.max(axis=1)
    results[market] = {
        "accuracy": None,
        "mean_confidence": df[f"{market}_confidence"].mean(),
        "distribution": df[f"{market}_prediction"].value_counts().to_dict(),
    }

# ==========================================================
# GUARDADO DE RESULTADOS
# ==========================================================
output_path = f"data/predictions/multi_market_predictions_{timestamp}.csv"
df.to_csv(output_path, index=False)
log.info(f"✅ Predicciones multi-mercado guardadas en: {output_path}")

# ==========================================================
# VISUALIZACIONES
# ==========================================================
for market in models.keys():
    plt.figure(figsize=(5,4))
    sns.countplot(x=df[f"{market}_prediction"])
    plt.title(f"Distribución de predicciones — {market.upper()}")
    plt.tight_layout()
    plot_path = f"data/predictions/distribution_{market}_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    log.info(f"📈 Distribución guardada en: {plot_path}")

# ==========================================================
# RESUMEN FINAL
# ==========================================================
print("\n✅ PREDICCIONES MULTI-MERCADO COMPLETADAS EXITOSAMENTE")
for market, info in results.items():
    print(f"\n🧠 Mercado: {market}")
    print(f"→ Media de confianza: {info['mean_confidence']:.3f}")
    print(f"→ Distribución de predicciones: {info['distribution']}")
print(f"\n📊 Archivo guardado: {output_path}")

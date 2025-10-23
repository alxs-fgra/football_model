#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_predict.py — Predicciones con modelo entrenado (1X2 Fútbol)
Autor: Alexis Figueroa
Versión: Final Pro+++ (Oct 2025)
"""

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------
LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "la_liga"
INPUT_PATH = f"data/processed/features_{LEAGUE.lower().replace(' ', '_')}_2024_25.csv"
PIPELINE_PATH = "models/liga_model_20251022_230930.pkl"  # Ajusta si el timestamp cambia
OUTPUT_DIR = "data/predictions"
LOG_PATH = "logs/prediction_run.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------
def load_model(path: str):
    """Carga el modelo entrenado."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    log.info(f"🧠 Cargando modelo desde {path}")
    return joblib.load(path)

def load_features(path: str) -> pd.DataFrame:
    """Carga los features procesados para predicción."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de features: {path}")
    df = pd.read_csv(path)
    log.info(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas si faltan."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month.fillna(0).astype(int)
    if "goal_diff" not in df.columns and {"avg_goals_home","avg_goals_away"}.issubset(df.columns):
        df["goal_diff"] = df["avg_goals_home"] - df["avg_goals_away"]
    log.info("🧩 Features enriquecidas automáticamente (month, goal_diff)")
    return df

def validate_columns(df: pd.DataFrame, required: list):
    """Valida que existan todas las columnas requeridas."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Faltan columnas requeridas: {missing}")
    df = df[required].astype(float).fillna(df.mean(numeric_only=True))
    return df

def visualize_predictions(df: pd.DataFrame, league: str, ts: str):
    """Genera gráfica de distribución de predicciones."""
    sns.set(style="whitegrid")
    df["prediction"] = df["prediction"].astype(int)
    plt.figure(figsize=(6,4))
    sns.countplot(x="prediction", hue="prediction", data=df, palette="viridis", legend=False)
    plt.title(f"Distribución de predicciones ({league.upper()} 2024/25)")
    plt.xlabel("Resultado (-1=Away, 0=Draw, 1=Home)")
    plt.xticks([-1,0,1], ["Away","Draw","Home"])
    path = f"{OUTPUT_DIR}/distribution_{league}_{ts}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log.info(f"📈 Distribución guardada en: {path}")
    return path

def visualize_confidence_vs_accuracy(df: pd.DataFrame, league: str, ts: str):
    """Grafica relación entre confianza y acierto (si hay resultados reales)."""
    if "actual_result" not in df.columns:
        log.warning("⚠️ No hay resultados reales para comparar.")
        return None
    df["correct"] = (df["prediction"] == df["actual_result"]).astype(int)
    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="confidence", y="correct", hue="prediction",
                    data=df, palette="coolwarm", alpha=0.7)
    plt.title(f"Confianza vs Acierto ({league.upper()} 2024/25)")
    plt.xlabel("Confianza (probabilidad)")
    plt.ylabel("Acierto (1 = Correcto)")
    path = f"{OUTPUT_DIR}/confidence_vs_accuracy_{league}_{ts}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    log.info(f"📊 Relación confianza-acierto guardada en: {path}")
    return path

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"🚀 Iniciando predicciones para {LEAGUE.title()} (2024/25)...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1️⃣ Cargar modelo y preprocesadores
    model = load_model(PIPELINE_PATH)
    preproc_path = os.path.join("models", os.path.basename(PIPELINE_PATH).replace("model", "scaler_imputer"))
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"No se encontró el preprocesador: {preproc_path}")
    preproc = joblib.load(preproc_path)
    imputer, scaler = preproc["imputer"], preproc["scaler"]

    # 2️⃣ Cargar dataset
    df = load_features(INPUT_PATH)
    df = enrich_features(df)

    # Solo las 8 columnas originales del modelo
    required = [
        "avg_goals_home","avg_goals_away","home_form","away_form",
        "h2h_avg_goals","is_home","month","goal_diff"
    ]
    X = validate_columns(df, required)

    # 3️⃣ Aplicar imputación y escalado
    log.info("🔄 Aplicando imputación y escalado antes de predecir...")
    X = imputer.transform(X.values)
    X = scaler.transform(X)

    # 4️⃣ Realizar predicciones
    log.info("Realizando predicciones...")
    preds, probs = model.predict(X), model.predict_proba(X)
    df["prediction"], df["confidence"] = preds, probs.max(axis=1)

    # 5️⃣ Comparar con resultados reales si existen
    if {"goals_home","goals_away"}.issubset(df.columns):
        df["actual_result"] = df.apply(
            lambda x: 1 if x.goals_home > x.goals_away
            else (0 if x.goals_home == x.goals_away else -1),
            axis=1
        )
        acc = accuracy_score(df["actual_result"], df["prediction"])
        log.info(f"📊 Accuracy preliminar vs resultados reales: {acc:.2%}")
    else:
        log.info("⚠️ No se encontraron columnas de goles para comparar.")
        acc = None

    # 6️⃣ Guardar resultados
    out_path = f"{OUTPUT_DIR}/{LEAGUE}_predictions_{ts}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"✅ Predicciones guardadas en: {out_path}")

    # 7️⃣ Visualizaciones
    dist_path = visualize_predictions(df, LEAGUE, ts)
    conf_path = visualize_confidence_vs_accuracy(df, LEAGUE, ts)

    # 8️⃣ Resumen final
    counts = df["prediction"].value_counts().sort_index()
    print("\n✅ PREDICCIONES COMPLETADAS EXITOSAMENTE")
    print(f"Resultados guardados en: {out_path}")
    print(f"Distribución de predicciones:\n{counts}")
    if acc is not None:
        print(f"Accuracy vs resultados reales: {acc:.2%}")
    print(f"📈 Distribución: {dist_path}")
    if conf_path:
        print(f"📊 Confianza vs Acierto: {conf_path}")
    print(f"🧠 Modelo usado: {PIPELINE_PATH}")
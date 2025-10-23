#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_predict.py — Predicciones con modelo entrenado (1X2 Fútbol)
Autor: Alexis Figueroa
Versión: Final Enterprise (Oct 2025)
"""

import os
import sys
import logging
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------
LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "la_liga"
INPUT_PATH = f"data/processed/features_{LEAGUE.lower().replace(' ', '_')}_2024_25.csv"
PIPELINE_PATH = "models/liga_pipeline_20251022_220922.pkl"  # Ajusta si cambia el timestamp
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
def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")
    log.info(f"Cargando pipeline desde {path}")
    return joblib.load(path)

def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de features: {path}")
    df = pd.read_csv(path)
    log.info(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas si no existen."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month.fillna(0).astype(int)
    if "goal_diff" not in df.columns and {"avg_goals_home", "avg_goals_away"}.issubset(df.columns):
        df["goal_diff"] = df["avg_goals_home"] - df["avg_goals_away"]
    if "rolling_avg_goals" not in df.columns:
        df["rolling_avg_goals"] = df[["avg_goals_home", "avg_goals_away"]].mean(axis=1)
    log.info("🧩 Features enriquecidas automáticamente (month, goal_diff, rolling_avg_goals)")
    return df

def validate_columns(df: pd.DataFrame, required_cols: list):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Faltan columnas requeridas: {missing}")
    df = df[required_cols].copy()
    df = df.astype(float).fillna(df.mean(numeric_only=True))
    return df

def visualize_predictions(df: pd.DataFrame, league: str, timestamp: str):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="prediction", hue="prediction", data=df, palette="viridis", legend=False)
    plt.title(f"Distribución de predicciones ({league.upper()} 2024/25)")
    plt.xlabel("Resultado (-1=Away, 0=Draw, 1=Home)")
    plt.xticks(ticks=[-1, 0, 1], labels=["Away", "Draw", "Home"])
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/distribution_{league}_{timestamp}.png"
    plt.savefig(path)
    plt.close()
    log.info(f"📈 Distribución guardada en: {path}")
    return path

def visualize_confidence_vs_accuracy(df: pd.DataFrame, league: str, timestamp: str):
    """Grafica relación entre confianza y acierto (si hay resultados reales)."""
    if "actual_result" not in df.columns:
        log.warning("⚠️ No hay resultados reales para comparar. Se omite la visualización de aciertos.")
        return None
    df["correct"] = (df["prediction"] == df["actual_result"]).astype(int)
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="confidence", y="correct", hue="prediction", data=df, palette="coolwarm", alpha=0.7)
    plt.title(f"Confianza vs Acierto ({league.upper()} 2024/25)")
    plt.xlabel("Confianza (probabilidad)")
    plt.ylabel("Acierto (1=Correcto, 0=Incorrecto)")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/confidence_vs_accuracy_{league}_{timestamp}.png"
    plt.savefig(path)
    plt.close()
    log.info(f"📊 Relación confianza-acierto guardada en: {path}")
    return path

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"🚀 Iniciando predicciones para {LEAGUE.title()} (2024/25)...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1️⃣ Cargar modelo y features
    pipeline = load_pipeline(PIPELINE_PATH)
    df = load_features(INPUT_PATH)
    df = enrich_features(df)

    # 2️⃣ Validar columnas requeridas
    required_cols = [
        "avg_goals_home", "avg_goals_away", "home_form", "away_form",
        "h2h_avg_goals", "is_home", "month", "goal_diff", "rolling_avg_goals"
    ]
    X = validate_columns(df, required_cols).values

    # 3️⃣ Realizar predicciones
    log.info("Realizando predicciones...")
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    df["prediction"] = preds
    df["confidence"] = probs.max(axis=1)

    # 4️⃣ Validación detallada (si hay resultados reales)
    metrics_path = None
    if {"goals_home", "goals_away"}.issubset(df.columns):
        df["actual_result"] = df.apply(
            lambda x: 1 if x["goals_home"] > x["goals_away"]
            else (0 if x["goals_home"] == x["goals_away"] else -1),
            axis=1
        )
        acc = accuracy_score(df["actual_result"], df["prediction"])
        log.info(f"📊 Accuracy preliminar vs resultados reales: {acc:.2%}")

        # Reporte de clasificación
        report = classification_report(
            df["actual_result"],
            df["prediction"],
            target_names=["Away (-1)", "Draw (0)", "Home (1)"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        metrics_path = f"{OUTPUT_DIR}/metrics_{LEAGUE}_{timestamp}.csv"
        report_df.to_csv(metrics_path)
        log.info(f"📋 Métricas detalladas guardadas en: {metrics_path}")
    else:
        log.info("⚠️ No se encontraron columnas de goles para comparar con resultados reales.")

    # 5️⃣ Guardar resultados
    output_path = f"{OUTPUT_DIR}/{LEAGUE}_predictions_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    log.info(f"✅ Predicciones guardadas en: {output_path}")

    # 6️⃣ Visualizaciones
    dist_path = visualize_predictions(df, LEAGUE, timestamp)
    conf_path = visualize_confidence_vs_accuracy(df, LEAGUE, timestamp)

    # 7️⃣ Resumen
    counts = df["prediction"].value_counts().sort_index()
    log.info(f"Distribución de predicciones:\n{counts.to_string()}")

    print("\n✅ PREDICCIONES COMPLETADAS EXITOSAMENTE")
    print(f"Resultados guardados en: {output_path}")
    print(f"Distribución de predicciones:\n{counts}")
    if "actual_result" in df.columns:
        print(f"Accuracy vs resultados reales: {acc:.2%}")
        print(f"📊 Métricas CSV: {metrics_path}")
    print(f"📈 Distribución: {dist_path}")
    if conf_path:
        print(f"📊 Confianza vs Acierto: {conf_path}")
    print(f"🧠 Modelo usado: {PIPELINE_PATH}")
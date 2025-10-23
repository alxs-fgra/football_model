#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_predictions.py — Monitoreo diario de rendimiento del modelo
Autor: Alexis Figueroa
Versión: Paso 8.3 (Oct 2025)
"""

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------
LOG_DIR = "logs"
HISTORY_FILE = os.path.join(LOG_DIR, "metrics_history.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------
def analyze_prediction_logs():
    """Analiza todos los logs diarios y consolida métricas."""
    all_logs = []
    for file in os.listdir(LOG_DIR):
        if file.startswith("predictions_") and file.endswith(".log"):
            path = os.path.join(LOG_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
                if not lines:
                    continue

                # Convertir a DataFrame
                df = pd.DataFrame(lines)
                date_str = file.replace("predictions_", "").replace(".log", "")
                df["date"] = datetime.strptime(date_str, "%Y%m%d").date()

                # Métricas diarias
                total = len(df)
                mean_conf = df["confidence"].mean()
                dist = df["prediction"].value_counts(normalize=True).to_dict()

                away = round(dist.get(-1, 0) * 100, 2)
                draw = round(dist.get(0, 0) * 100, 2)
                home = round(dist.get(1, 0) * 100, 2)

                all_logs.append({
                    "date": date_str,
                    "total_predictions": total,
                    "avg_confidence": round(mean_conf, 3),
                    "away_%": away,
                    "draw_%": draw,
                    "home_%": home
                })

    if not all_logs:
        print("⚠️ No se encontraron logs de predicciones.")
        return

    df_all = pd.DataFrame(all_logs).sort_values("date")

    # Guardar CSV
    df_all.to_csv(HISTORY_FILE, index=False)
    print(f"✅ Métricas consolidadas guardadas en: {HISTORY_FILE}")

    # Graficar evolución del promedio de confianza
    plt.figure(figsize=(8, 4))
    plt.plot(df_all["date"], df_all["avg_confidence"], marker="o", label="Promedio de Confianza")
    plt.title("📈 Evolución de Confianza Promedio")
    plt.xlabel("Fecha")
    plt.ylabel("Confianza Promedio")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, "confidence_trend.png"))
    plt.close()

    print("📊 Gráfico guardado en logs/confidence_trend.png")

    # Mostrar resumen
    print("\nResumen:")
    print(df_all.tail())

# ---------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    analyze_prediction_logs()

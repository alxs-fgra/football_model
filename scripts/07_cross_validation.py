#!/usr/bin/env python3
# ==============================================================
# Script: 07_cross_validation.py
# Autor:  Alexis Figueroa
# Descripción:
#   Ejecuta validación cruzada (K-Fold) sobre el dataset más reciente
#   con targets, evaluando los 3 mercados: 1X2, BTTS y Over/Under 2.5
# ==============================================================

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import logging

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

DATA_DIR = "data/processed"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Buscar automáticamente el dataset más reciente
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "features_with_targets_*.csv")),
    key=os.path.getmtime,
    reverse=True
)
if not files:
    raise FileNotFoundError("❌ No se encontró ningún dataset con targets en data/processed/")
else:
    DATA_PATH = files[0]
    print(f"📂 Using latest dataset for cross-validation: {DATA_PATH}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"logs/cross_validation_{timestamp}.log"

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ==============================================================
# CARGA DE DATOS
# ==============================================================

df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=["number"])target_cols = ["result", "btts", "over_2.5"]
feature_cols = [col for col in df.columns if col not in target_cols]

logging.info(f"✅ Dataset cargado ({len(df)} filas) para validación cruzada.")

# ==============================================================
# CONFIGURACIÓN DE VALIDACIÓN CRUZADA
# ==============================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cross_validate_market(X, y, label):
    """Ejecuta validación cruzada para un mercado específico."""
    acc_scores, f1_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=fold)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        acc_scores.append(acc)
        f1_scores.append(f1)
        logging.info(f"{label.upper()} | Fold {fold}: ACC={acc:.3f}, F1={f1:.3f}")

    return {
        "market": label,
        "acc_mean": np.mean(acc_scores),
        "acc_std": np.std(acc_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores)
    }

# ==============================================================
# EJECUCIÓN DE VALIDACIÓN
# ==============================================================

results = []
for target in target_cols:
    label = target.replace("target_", "")
    y = df[target]
    X = df[feature_cols]

    metrics = cross_validate_market(X, y, label)
    results.append(metrics)
    print(f"✅ {label.upper()} | ACC={metrics['acc_mean']:.3f} ±{metrics['acc_std']:.3f} | F1={metrics['f1_mean']:.3f} ±{metrics['f1_std']:.3f}")

# ==============================================================
# GUARDAR RESULTADOS
# ==============================================================

summary_path = os.path.join(REPORTS_DIR, f"cross_validation_summary_{timestamp}.csv")
avg_path = os.path.join(REPORTS_DIR, f"cross_validation_avg_{timestamp}.csv")

pd.DataFrame(results).to_csv(summary_path, index=False)
pd.DataFrame({
    "metric": ["ACC_mean", "ACC_std", "F1_mean", "F1_std"],
    "result": [np.mean([r["acc_mean"] for r in results]),
               np.mean([r["acc_std"] for r in results]),
               np.mean([r["f1_mean"] for r in results]),
               np.mean([r["f1_std"] for r in results])]
}).to_csv(avg_path, index=False)

logging.info(f"📊 Cross-validation results saved: {summary_path}")
logging.info(f"📈 Average results saved: {avg_path}")

print(f"🏁 Cross-validation completed successfully!\n📊 Summary: {summary_path}\n📈 Averages: {avg_path}")
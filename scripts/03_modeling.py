#!/usr/bin/env python3
# ==============================================================
# Script: 03_modeling.py
# Autor:  Alexis Figueroa
# Descripción:
#   Entrena modelos para los diferentes mercados (1X2, BTTS, Over/Under 2.5)
#   usando el dataset más reciente con targets generados por add_targets.py
# ==============================================================

import os
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

DATA_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
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
    print(f"📂 Loading latest dataset: {DATA_PATH}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=f"logs/model_training_{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("🚀 Starting model training pipeline...")

# ==============================================================
# CARGA DE DATOS
# ==============================================================

df = pd.read_csv(DATA_PATH)
logging.info(f"✅ Dataset loaded: {DATA_PATH} ({len(df)} rows)")

# Features y targets
target_cols = ["target_result", "target_btts", "target_over25"]
feature_cols = [col for col in df.columns if col not in target_cols]

X = df[feature_cols]
y_results = df["target_result"]
y_btts = df["target_btts"]
y_over25 = df["target_over25"]

# ==============================================================
# PREPROCESAMIENTO
# ==============================================================

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Guardar scaler e imputer
joblib.dump(imputer, os.path.join(MODELS_DIR, f"scaler_imputer_{timestamp}.pkl"))
logging.info("💾 Imputer saved.")
joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{timestamp}.pkl"))
logging.info("💾 Scaler saved.")

# ==============================================================
# ENTRENAMIENTO DE MODELOS
# ==============================================================

def train_and_save_model(X, y, label: str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    model_path = os.path.join(MODELS_DIR, f"{label}_randomforest_{timestamp}.pkl")
    joblib.dump(model, model_path)

    logging.info(f"✅ {label.upper()} model trained - ACC={acc:.3f} F1={f1:.3f}")
    print(f"✅ {label.upper()} model trained - ACC={acc:.3f} F1={f1:.3f}")

    return {"model": label, "acc": acc, "f1": f1, "path": model_path}


results = []
results.append(train_and_save_model(X_scaled, y_results, "result"))
results.append(train_and_save_model(X_scaled, y_btts, "btts"))
results.append(train_and_save_model(X_scaled, y_over25, "over_2.5"))

# ==============================================================
# GUARDAR RESULTADOS
# ==============================================================

summary_path = f"reports/model_performance_summary_{timestamp}.csv"
pd.DataFrame(results).to_csv(summary_path, index=False)
logging.info(f"📊 Model training summary saved → {summary_path}")

print(f"🏁 Model training completed successfully! Summary → {summary_path}")
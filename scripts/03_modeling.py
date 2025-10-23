#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_modeling.py — Entrenamiento de modelos 1X2, BTTS y Over/Under 2.5
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import logging

# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ==========================================================
# CARGA DE DATOS
# ==========================================================
DATA_PATH = "data/processed/features_la_liga_2015_2023.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo {DATA_PATH}")

log.info(f"✅ Archivo detectado: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
log.info(f"✅ Datos cargados correctamente: {len(df):,} filas")

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
log.info("🧩 Aplicando feature engineering...")
df["total_goals"] = df["goals_home"] + df["goals_away"]
df["goal_diff"] = df["goals_home"] - df["goals_away"]
df["month"] = pd.to_datetime(df["date"]).dt.month

# Promedio móvil de goles por equipo local (últimos 5 partidos)
df["rolling_avg_goals"] = (
    df.groupby("home_team")["total_goals"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# ==========================================================
# TARGETS PRINCIPALES
# ==========================================================
log.info("🎯 Generando targets adicionales (BTTS, Over/Under 2.5)...")
df["result"] = df.apply(
    lambda x: 1 if x["goals_home"] > x["goals_away"]
    else (0 if x["goals_home"] == x["goals_away"] else -1), axis=1)
df["btts"] = ((df["goals_home"] > 0) & (df["goals_away"] > 0)).astype(int)
df["over_2.5"] = (df["total_goals"] > 2.5).astype(int)

# ==========================================================
# SELECCIÓN DE FEATURES
# ==========================================================
features = [
    "avg_goals_home", "avg_goals_away", "home_form", "away_form",
    "h2h_avg_goals", "is_home", "month", "goal_diff", "rolling_avg_goals"
]
X = df[features].values

# ==========================================================
# SPLIT TRAIN/VALIDATION
# ==========================================================
train_mask = df["season"] <= 2022
val_mask = df["season"] == 2023
log.info(f"📊 Split: Train={train_mask.sum()} | Val={val_mask.sum()}")

# Imputación y escalado
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(X[train_mask]))
X_val = scaler.transform(imputer.transform(X[val_mask]))

# ==========================================================
# CONFIGURACIÓN DE MODELOS
# ==========================================================
from sklearn.model_selection import GridSearchCV
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

models = {
    "1x2": LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced"),
    "btts": RandomForestClassifier(random_state=42),
    "over_2.5": RandomForestClassifier(random_state=42)
}
targets = {"1x2": "result", "btts": "btts", "over_2.5": "over_2.5"}

# ==========================================================
# ENTRENAMIENTO DE MODELOS
# ==========================================================
for name, model in models.items():
    y_train = df[targets[name]][train_mask].values
    y_val = df[targets[name]][val_mask].values

    # Balanceo para draws (solo aplica en 1X2)
    if name == "1x2":
        draw_count = np.sum(y_train == 0)
        safe_target = min(900, max(draw_count, 400))
        smote = SMOTE(random_state=42, sampling_strategy={0: safe_target})
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    log.info(f"🚀 Entrenando modelo: {name}")
    if isinstance(model, LogisticRegression):
        param_grid = {"C": [0.001, 0.005, 0.01], "solver": ["lbfgs", "saga"]}
    else:
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}

    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    best_model = grid.best_estimator_

    # Validación
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)
    acc = (y_pred == y_val).mean()
    logloss = log_loss(y_val, y_pred_proba)
    rocauc = roc_auc_score(y_val, y_pred_proba[:, 1]) if len(np.unique(y_val)) == 2 else None

    log.info(f"✅ {name} | Accuracy={acc:.3f} | LogLoss={logloss:.3f} | ROC-AUC={rocauc}")

    # Guardar modelo
    path = f"models/{name}_model_{timestamp}.pkl"
    joblib.dump(best_model, path)
    log.info(f"💾 Modelo guardado en {path}")

# ==========================================================
# GUARDAR PREPROCESADORES
# ==========================================================
preproc_path = f"models/scaler_imputer_{timestamp}.pkl"
joblib.dump({"imputer": imputer, "scaler": scaler}, preproc_path)
log.info(f"🧩 Preprocesadores guardados en {preproc_path}")

log.info("🏁 ENTRENAMIENTO MULTI-MERCADO COMPLETADO CON ÉXITO")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_modeling.py — Entrenamiento y evaluación de modelo 1X2 (fútbol)
Autor: Alexis Figueroa
Versión: Final Pro (Oct 2025)
"""

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
import os
import sys
import glob
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------
np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/modeling_run.log", mode="a"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# LEAGUE SELECTION
# ---------------------------------------------------------------------
league = sys.argv[1] if len(sys.argv) > 1 else "liga"
pattern = f"data/processed/features_*{league}*_2015_2023.csv"
matches = glob.glob(pattern)
if not matches:
    raise FileNotFoundError(f"No se encontró ningún archivo que coincida con el patrón: {pattern}")
INPUT_PATH = matches[0]
log.info(f"✅ Archivo detectado: {INPUT_PATH}")

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
log.info(f"Datos cargados correctamente. Total de filas: {len(df):,}")

# ---------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------
required_cols = [
    "goals_home", "goals_away", "season", "date",
    "avg_goals_home", "avg_goals_away",
    "home_form", "away_form", "h2h_avg_goals", "is_home"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")

# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
log.info("Aplicando feature engineering...")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["goal_diff"] = df["avg_goals_home"] - df["avg_goals_away"]

# ---------------------------------------------------------------------
# TARGET CREATION (1X2)
# ---------------------------------------------------------------------
log.info("Generando target 1X2...")
df["result"] = df.apply(
    lambda x: 1 if x["goals_home"] > x["goals_away"]
    else (0 if x["goals_home"] == x["goals_away"] else -1),
    axis=1
)

# ---------------------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------------------
features = [
    "avg_goals_home", "avg_goals_away",
    "home_form", "away_form",
    "h2h_avg_goals", "is_home",
    "month", "goal_diff"
]
log.info(f"Usando features: {features}")

X = df[features]
y = df["result"]

# ---------------------------------------------------------------------
# TRAIN/VAL SPLIT
# ---------------------------------------------------------------------
log.info("Dividiendo datos en train (2015–2022) y val (2023)...")
train_mask = df["season"] <= 2022
val_mask = df["season"] == 2023
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

# ---------------------------------------------------------------------
# IMPUTATION AND SCALING (BEFORE SMOTE)
# ---------------------------------------------------------------------
log.info("Imputando valores faltantes antes de aplicar SMOTE...")
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
missing_count = np.isnan(X_train).sum()
log.info(f"🔍 Valores NaN imputados antes de SMOTE: {missing_count}")

# ---------------------------------------------------------------------
# CLASS BALANCING (SMOTE)
# ---------------------------------------------------------------------
log.info("Aplicando SMOTE para balanceo de clases...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------------------------------------------------------------
# SCALING POST-SMOTE
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---------------------------------------------------------------------
# HYPERPARAMETER OPTIMIZATION
# ---------------------------------------------------------------------
log.info("Optimizando hiperparámetros con GridSearchCV...")
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]}
grid = GridSearchCV(
    LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced"),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
log.info(f"✅ Mejor combinación de hiperparámetros: {grid.best_params_}")

# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------
log.info("Evaluando modelo en temporada 2023...")
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)

accuracy = accuracy_score(y_val, y_pred)
logloss = log_loss(y_val, y_pred_proba)
encoder = LabelEncoder()
y_val_encoded = encoder.fit_transform(y_val)
roc_auc = roc_auc_score(y_val_encoded, y_pred_proba, multi_class="ovr", average="weighted")

log.info(f"🎯 Accuracy en 2023 ({league.title()}): {accuracy:.2f}")
log.info(f"📊 Log-loss en 2023: {logloss:.3f}")
log.info(f"ROC-AUC (weighted): {roc_auc:.3f}")

print("\n📊 Reporte de Clasificación:")
print(classification_report(y_val, y_pred, zero_division=0))

# ---------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
log.info(f"Cross-validation mean: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

# ---------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------
log.info("Generando matriz de confusión...")
cm = confusion_matrix(y_val, y_pred, labels=[-1, 0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Away Win", "Draw", "Home Win"],
            yticklabels=["Away Win", "Draw", "Home Win"])
plt.title(f"Matriz de Confusión ({league.title()} 2023)")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cm_path = f"models/confusion_matrix_{league}_{timestamp}.png"
plt.savefig(cm_path)
plt.close()
log.info(f"📊 Matriz de confusión guardada en: {cm_path}")

# ---------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------
log.info("Guardando importancia de features...")
importance = pd.DataFrame({
    "feature": features,
    "importance": best_model.coef_[0]
}).sort_values(by="importance", ascending=False)
importance_path = f"models/feature_importance_{league}_{timestamp}.csv"
importance.to_csv(importance_path, index=False)
log.info(f"📈 Importancia de features guardada en: {importance_path}")

# ---------------------------------------------------------------------
# SAVE MODEL AND PREPROCESSORS
# ---------------------------------------------------------------------
OUTPUT_PATH = f"models/{league}_model_{timestamp}.pkl"
PREPROC_PATH = f"models/{league}_scaler_imputer_{timestamp}.pkl"

joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "imputer": imputer
}, PREPROC_PATH)
joblib.dump(best_model, OUTPUT_PATH)
log.info(f"✅ Modelo guardado en: {OUTPUT_PATH}")
log.info(f"🧩 Preprocesadores guardados en: {PREPROC_PATH}")

# ---------------------------------------------------------------------
# SAVE METRICS
# ---------------------------------------------------------------------
metrics_path = f"models/metrics_{league}_{timestamp}.csv"
pd.DataFrame([{
    "league": league,
    "timestamp": timestamp,
    "accuracy": accuracy,
    "log_loss": logloss,
    "roc_auc": roc_auc,
    "cv_mean": cv_scores.mean(),
    "cv_std": cv_scores.std(),
    "best_C": grid.best_params_["C"],
    "penalty": grid.best_params_["penalty"]
}]).to_csv(metrics_path, index=False)
log.info(f"📊 Métricas guardadas en: {metrics_path}")

# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------
print("\n✅ MODELADO COMPLETADO EXITOSAMENTE")
print(f"Accuracy en 2023: {accuracy:.2%}")
print(f"Log-loss en 2023: {logloss:.3f}")
print(f"ROC-AUC (weighted): {roc_auc:.3f}")
print(f"Cross-validation: {cv_scores.mean():.2%} (± {cv_scores.std()*2:.2%})")
print(f"Modelo guardado en: {OUTPUT_PATH}")
print(f"Preprocesadores guardados en: {PREPROC_PATH}")
print(f"Matriz de confusión: {cm_path}")
print(f"Métricas: {metrics_path}")
print(f"Importancia de features: {importance_path}")
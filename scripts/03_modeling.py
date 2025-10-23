#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_modeling.py — Entrenamiento y evaluación de modelo 1X2 (fútbol)
Autor: Alexis Figueroa
Versión: Final Pro++ (Oct 2025)
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
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    "home_form", "away_form", "h2h_avg_goals", "is_home", "home_team"
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
df["total_goals"] = df["goals_home"] + df["goals_away"]

# Nueva feature: promedio móvil de goles (últimos 5 partidos)
df = df.sort_values(["home_team", "date"])
df["rolling_avg_goals"] = (
    df.groupby("home_team")["total_goals"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

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
    "month", "goal_diff", "rolling_avg_goals"
]
log.info(f"Usando features: {features}")

X = df[features]
y = df["result"]

# ---------------------------------------------------------------------
# TRAIN/VAL SPLIT
# ---------------------------------------------------------------------
train_mask = df["season"] <= 2022
val_mask = df["season"] == 2023
X_train, y_train = X[train_mask].values, y[train_mask].values
X_val, y_val = X[val_mask].values, y[val_mask].values

# ---------------------------------------------------------------------
# IMPUTATION + SMOTE
# ---------------------------------------------------------------------
log.info("Imputando valores faltantes...")
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# SMOTE ajustado para reforzar “Draw” (empates)
draw_count = np.sum(y_train == 0)
safe_target = min(800, draw_count)
smote = SMOTE(random_state=42, sampling_strategy={0: safe_target})
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------------------------------------------------------------
# SCALING
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---------------------------------------------------------------------
# HYPERPARAMETER OPTIMIZATION
# ---------------------------------------------------------------------
param_grid = {"C": [0.001, 0.002, 0.005, 0.01, 0.02], "solver": ["lbfgs", "saga"]}
log.info("Optimizando hiperparámetros con GridSearchCV...")
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
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)
accuracy = accuracy_score(y_val, y_pred)
logloss = log_loss(y_val, y_pred_proba)
encoder = LabelEncoder()
y_val_encoded = encoder.fit_transform(y_val)
roc_auc = roc_auc_score(y_val_encoded, y_pred_proba, multi_class="ovr", average="weighted")

log.info(f"🎯 Accuracy en 2023 ({league.title()}): {accuracy:.2f}")
log.info(f"📊 Log-loss: {logloss:.3f}")
log.info(f"ROC-AUC: {roc_auc:.3f}")

print("\n📊 Reporte de Clasificación:")
print(classification_report(y_val, y_pred, zero_division=0))

# ---------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
log.info(f"Cross-validation mean: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

# ---------------------------------------------------------------------
# CURVAS ROC MULTICLASE
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log.info("Generando curvas ROC multiclase...")

classes = np.unique(y_val)
y_val_bin = label_binarize(y_val, classes=classes)
fpr, tpr, roc_auc_class = {}, {}, {}

plt.figure(figsize=(7, 5))
for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(y_val_bin[:, i], y_pred_proba[:, i])
    roc_auc_class[cls] = auc(fpr[cls], tpr[cls])
    plt.plot(fpr[cls], tpr[cls], lw=2, label=f"Clase {cls} (AUC={roc_auc_class[cls]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Curvas ROC multiclase ({league.title()} 2023)")
plt.legend(loc="lower right")
roc_path = f"models/roc_curve_{league}_{timestamp}.png"
plt.tight_layout()
plt.savefig(roc_path)
plt.close()
log.info(f"📈 Curvas ROC guardadas en: {roc_path}")

# ---------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------
cm = confusion_matrix(y_val, y_pred, labels=[-1, 0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Away Win", "Draw", "Home Win"],
            yticklabels=["Away Win", "Draw", "Home Win"])
plt.title(f"Matriz de Confusión ({league.title()} 2023)")
plt.tight_layout()
cm_path = f"models/confusion_matrix_{league}_{timestamp}.png"
plt.savefig(cm_path)
plt.close()

# ---------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------
importance = pd.DataFrame({
    "feature": features,
    "importance": best_model.coef_[0]
}).sort_values(by="importance", ascending=False)
importance_path = f"models/feature_importance_{league}_{timestamp}.csv"
importance.to_csv(importance_path, index=False)

# ---------------------------------------------------------------------
# SAVE PIPELINE + METRICS
# ---------------------------------------------------------------------
pipeline = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
    ("model", best_model)
])
pipeline_path = f"models/{league}_pipeline_{timestamp}.pkl"
joblib.dump(pipeline, pipeline_path)

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
    "solver": grid.best_params_["solver"]
}]).to_csv(metrics_path, index=False)

# Markdown resumen
summary_md = f"models/summary_{league}_{timestamp}.md"
with open(summary_md, "w") as f:
    f.write(f"# Modelo {league.title()} — {timestamp}\n")
    f.write(f"- Accuracy: {accuracy:.2%}\n")
    f.write(f"- Log-loss: {logloss:.3f}\n")
    f.write(f"- ROC-AUC: {roc_auc:.3f}\n")
    f.write(f"- CV: {cv_scores.mean():.2%} ± {cv_scores.std()*2:.2%}\n")
    f.write(f"- Mejor C: {grid.best_params_['C']} | Solver: {grid.best_params_['solver']}\n")

log.info(f"✅ Pipeline guardado en: {pipeline_path}")
log.info(f"📈 Curvas ROC: {roc_path}")
log.info(f"📊 Matriz de confusión: {cm_path}")
log.info(f"📘 Resumen: {summary_md}")

# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------
print("\n✅ MODELADO COMPLETADO EXITOSAMENTE")
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Modelo guardado en: {pipeline_path}")
print(f"Matriz de confusión: {cm_path}")
print(f"Curvas ROC: {roc_path}")
print(f"Métricas: {metrics_path}")
print(f"Importancia de features: {importance_path}")
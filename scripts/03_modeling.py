#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_modeling.py — Optimización del modelo 1X2 (Paso 7)
Autor: Alexis Figueroa
Versión: 7.2 Final (Oct 2025)
"""
import os, sys, glob, random, warnings, joblib
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# CONFIG GLOBAL
# ------------------------------------------------------------
np.random.seed(42)
random.seed(42)
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

# ------------------------------------------------------------
# LIGA Y ARCHIVO
# ------------------------------------------------------------
league = sys.argv[1] if len(sys.argv) > 1 else "liga"
pattern = f"data/processed/features_*{league}*_2015_2023.csv"
matches = glob.glob(pattern)
if not matches:
    raise FileNotFoundError(f"No se encontró ningún archivo para patrón: {pattern}")
INPUT_PATH = matches[0]
log.info(f"✅ Archivo detectado: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
log.info(f"Datos cargados correctamente: {len(df):,} filas")

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
log.info("Aplicando feature engineering...")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["total_goals"] = df["goals_home"] + df["goals_away"]
df["goal_diff"] = df["avg_goals_home"] - df["avg_goals_away"]

# Media móvil de goles (últimos 5 partidos por equipo local)
df = df.sort_values(["home_team", "date"])
df["rolling_avg_goals"] = (
    df.groupby("home_team")["total_goals"]
      .rolling(5, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

# ------------------------------------------------------------
# TARGET 1X2
# ------------------------------------------------------------
log.info("Generando target 1X2...")
df["result"] = np.where(df["goals_home"] > df["goals_away"], 1,
                 np.where(df["goals_home"] < df["goals_away"], -1, 0))

features = [
    "avg_goals_home", "avg_goals_away",
    "home_form", "away_form",
    "h2h_avg_goals", "is_home",
    "month", "goal_diff", "rolling_avg_goals"
]
log.info(f"Usando features: {features}")

X, y = df[features], df["result"]

# ------------------------------------------------------------
# TRAIN / VAL SPLIT
# ------------------------------------------------------------
log.info("Dividiendo datos: train (2015-2022) vs val (2023)")
train_mask = df["season"] <= 2022
val_mask   = df["season"] == 2023
X_train, y_train = X[train_mask].values, y[train_mask].values
X_val,   y_val   = X[val_mask].values,   y[val_mask].values

# ------------------------------------------------------------
# IMPUTACIÓN + SMOTE + SCALER
# ------------------------------------------------------------
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)

log.info("Aplicando SMOTE (balance Draw = 0)…")
draw_count = int((y_train == 0).sum())
safe_target = min(900, max(draw_count, 400))
log.info(f"SMOTE target para Draw: {safe_target}")
smote = SMOTE(random_state=42, sampling_strategy={0: safe_target})
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ------------------------------------------------------------
# GRID SEARCH LOGISTIC REGRESSION
# ------------------------------------------------------------
log.info("Optimizando hiperparámetros…")
param_grid = {"C": [0.0005, 0.001, 0.002, 0.005, 0.01],
              "solver": ["lbfgs", "saga"]}
grid = GridSearchCV(
    LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced"),
    param_grid, cv=5, n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
log.info(f"✅ Mejor combinación: {grid.best_params_}")

# ------------------------------------------------------------
# EVALUACIÓN
# ------------------------------------------------------------
y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)
accuracy = accuracy_score(y_val, y_pred)
logloss  = log_loss(y_val, y_proba)
enc = LabelEncoder(); y_val_enc = enc.fit_transform(y_val)
roc_auc = roc_auc_score(y_val_enc, y_proba, multi_class="ovr", average="weighted")

print("\n📊 Reporte de Clasificación:")
print(classification_report(y_val, y_pred, zero_division=0))
log.info(f"🎯 Accuracy 2023: {accuracy:.2f}  LogLoss: {logloss:.3f}  ROC-AUC: {roc_auc:.3f}")

cv = cross_val_score(best_model, X_train, y_train, cv=5)
log.info(f"Cross-val mean: {cv.mean():.3f} ± {cv.std()*2:.3f}")

# ------------------------------------------------------------
# GUARDADO Y GRÁFICOS
# ------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cm = confusion_matrix(y_val, y_pred, labels=[-1,0,1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Away","Draw","Home"],
            yticklabels=["Away","Draw","Home"])
plt.title(f"Matriz de Confusión ({league.title()} 2023)")
plt.xlabel("Predicho"); plt.ylabel("Real")
plt.tight_layout()
cm_path = f"models/confusion_matrix_{league}_{timestamp}.png"
plt.savefig(cm_path); plt.close()

imp = pd.DataFrame({"feature": features,
                    "importance": best_model.coef_[0]}).sort_values("importance", ascending=False)
imp_path = f"models/feature_importance_{league}_{timestamp}.csv"
imp.to_csv(imp_path, index=False)

model_path   = f"models/{league}_model_{timestamp}.pkl"
preproc_path = f"models/{league}_scaler_imputer_{timestamp}.pkl"
joblib.dump(best_model, model_path)
joblib.dump({"scaler": scaler, "imputer": imputer, "features": features}, preproc_path)

metrics_path = f"models/metrics_{league}_{timestamp}.csv"
pd.DataFrame([{
    "league": league, "timestamp": timestamp,
    "accuracy": accuracy, "log_loss": logloss, "roc_auc": roc_auc,
    "cv_mean": cv.mean(), "cv_std": cv.std(),
    "best_C": grid.best_params_["C"], "solver": grid.best_params_["solver"]
}]).to_csv(metrics_path, index=False)

print("\n✅ MODELADO COMPLETADO (OPTIMIZADO PASO 7)")
print(f"Accuracy 2023: {accuracy:.2%} | ROC-AUC: {roc_auc:.3f}")
print(f"Modelo: {model_path}")
print(f"Preprocesadores: {preproc_path}")
print(f"Matriz Confusión: {cm_path}")
print(f"Importancia features: {imp_path}")
print(f"Métricas: {metrics_path}")
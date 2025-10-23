"""
03_modeling.py
--------------------------------------
Entrena un modelo predictivo 1X2 (Home Win / Draw / Away Win)
para LaLiga (2015–2023) usando las features generadas en el Paso 4.

Salida:
- Modelo entrenado: models/laliga_model.pkl
- Métricas en consola y logs: logs/model_training.log
"""

import os
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------
LEAGUE = "laliga"  # o "premier_league", "bundesliga"
INPUT_PATH = f"data/processed/features_{LEAGUE}_2015_2023.csv"
MODEL_PATH = f"models/{LEAGUE}_model.pkl"
LOG_PATH = "logs/model_training.log"
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Logging más robusto
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)
log.info(f"🚀 Iniciando entrenamiento para {LEAGUE.upper()} - {datetime.now()}")

# -------------------------------------
# 1. CARGA DE DATOS
# -------------------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"No se encontró el archivo {INPUT_PATH}")

log.info(f"Cargando dataset desde {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

required_cols = [
    "goals_home", "goals_away", "season",
    "avg_goals_home", "avg_goals_away", "home_form", "away_form", "h2h_avg_goals", "is_home"
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas necesarias: {missing}")

# -------------------------------------
# 2. CREAR TARGET (1X2)
# -------------------------------------
df["result"] = df.apply(
    lambda x: 1 if x["goals_home"] > x["goals_away"]
    else (0 if x["goals_home"] == x["goals_away"] else -1),
    axis=1
)

# -------------------------------------
# 3. SELECCIÓN DE FEATURES
# -------------------------------------
FEATURES = ["avg_goals_home", "avg_goals_away", "home_form", "away_form", "h2h_avg_goals", "is_home"]
X = df[FEATURES].fillna(0)
y = df["result"]

log.info(f"Usando features: {FEATURES}")
log.info(f"Dataset final: {df.shape[0]} filas | {X.shape[1]} features")

# -------------------------------------
# 4. DIVISIÓN DE DATOS
# -------------------------------------
X_train = X[df["season"] <= 2022]
y_train = y[df["season"] <= 2022]
X_val = X[df["season"] == 2023]
y_val = y[df["season"] == 2023]

log.info(f"Tamaño entrenamiento: {X_train.shape}, validación: {X_val.shape}")

# -------------------------------------
# 5. ENTRENAMIENTO DEL MODELO
# -------------------------------------
model = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial")
model.fit(X_train, y_train)
log.info("✅ Entrenamiento completado")

# Cross-validation opcional
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
log.info(f"Cross-val (media ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# -------------------------------------
# 6. EVALUACIÓN
# -------------------------------------
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, zero_division=0)

log.info(f"📊 Accuracy 2023 ({LEAGUE}): {accuracy:.3f}")
log.info("Reporte de Clasificación:\n" + report)

print("\n=== RESULTADOS ===")
print(f"Accuracy (2023): {accuracy:.3f}")
print(report)
print("==================\n")

# -------------------------------------
# 7. GUARDAR MODELO
# -------------------------------------
joblib.dump(model, MODEL_PATH)
log.info(f"💾 Modelo guardado en {MODEL_PATH}")

print(f"✅ Modelo guardado en {MODEL_PATH}")
log.info("✅ Finalizado con éxito.\n")
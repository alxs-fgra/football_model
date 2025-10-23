import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import warnings

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# DETECTAR AUTOMÁTICAMENTE EL ARCHIVO DE FEATURES
# ---------------------------------------------------------------------
pattern = "data/processed/features_*liga*_2015_2023.csv"
matches = glob.glob(pattern)

if not matches:
    raise FileNotFoundError(f"No se encontró ningún archivo que coincida con el patrón: {pattern}")

INPUT_PATH = matches[0]
log.info(f"✅ Archivo detectado: {INPUT_PATH}")

# ---------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
log.info(f"Datos cargados correctamente. Total de filas: {len(df):,}")

# ---------------------------------------------------------------------
# VALIDACIÓN DE COLUMNAS NECESARIAS
# ---------------------------------------------------------------------
required_cols = [
    "goals_home", "goals_away", "season",
    "avg_goals_home", "avg_goals_away",
    "home_form", "away_form", "h2h_avg_goals", "is_home"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")

# ---------------------------------------------------------------------
# CREACIÓN DEL TARGET (1X2)
# ---------------------------------------------------------------------
log.info("Generando target 1X2...")
df["result"] = df.apply(
    lambda x: 1 if x["goals_home"] > x["goals_away"]
    else (0 if x["goals_home"] == x["goals_away"] else -1),
    axis=1
)

# ---------------------------------------------------------------------
# DEFINICIÓN DE FEATURES Y TARGET
# ---------------------------------------------------------------------
features = [
    "avg_goals_home", "avg_goals_away",
    "home_form", "away_form",
    "h2h_avg_goals", "is_home"
]

X = df[features].fillna(0)
y = df["result"]

# ---------------------------------------------------------------------
# DIVISIÓN EN TRAIN (2015–2022) Y VALIDACIÓN (2023)
# ---------------------------------------------------------------------
log.info("Dividiendo datos en train (2015–2022) y val (2023)...")
X_train = X[df["season"] <= 2022]
y_train = y[df["season"] <= 2022]
X_val = X[df["season"] == 2023]
y_val = y[df["season"] == 2023]

# ---------------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------------------
log.info("Entrenando modelo de regresión logística (multinomial)...")
model = LogisticRegression(max_iter=1000, multi_class="multinomial")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# EVALUACIÓN EN VALIDACIÓN
# ---------------------------------------------------------------------
log.info("Evaluando modelo en temporada 2023...")
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
log.info(f"🎯 Accuracy en 2023 (LaLiga): {accuracy:.2f}")

print("\n📊 Reporte de Clasificación:")
print(classification_report(y_val, y_pred, zero_division=0))

# ---------------------------------------------------------------------
# VALIDACIÓN CRUZADA OPCIONAL
# ---------------------------------------------------------------------
log.info("Realizando validación cruzada (5-fold) en datos de entrenamiento...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
log.info(f"Cross-validation mean: {cv_scores.mean():.3f}  std: ±{cv_scores.std()*2:.3f}")

# ---------------------------------------------------------------------
# GUARDAR MODELO
# ---------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
OUTPUT_PATH = "models/la_liga_model.pkl"
joblib.dump(model, OUTPUT_PATH)
log.info(f"✅ Modelo guardado en: {OUTPUT_PATH}")

# ---------------------------------------------------------------------
# RESUMEN FINAL
# ---------------------------------------------------------------------
print("\n✅ MODELADO COMPLETADO EXITOSAMENTE")
print(f"Accuracy en 2023: {accuracy:.2%}")
print(f"Cross-validation: {cv_scores.mean():.2%} (± {cv_scores.std()*2:.2%})")
print(f"Modelo guardado en: {OUTPUT_PATH}")
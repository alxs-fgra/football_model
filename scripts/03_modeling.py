import os
import glob
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==========================================================
# 🧠 Configuración de logging
# ==========================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# 📂 Selección dinámica del dataset
# ==========================================================
data_path = os.environ.get("LATEST", "").strip()
if not data_path:
    candidates = sorted(
        glob.glob("data/processed/features_with_targets_*.csv"),
        key=os.path.getmtime,
        reverse=True
    )
    data_path = candidates[0] if candidates else ""

if not data_path or not os.path.exists(data_path):
    logging.error("❌ No se encontró ningún dataset en data/processed/")
    raise FileNotFoundError("No dataset found in data/processed/")

logging.info(f"📦 Dataset seleccionado: {data_path}")
df = pd.read_csv(data_path)

# ==========================================================
# 🔍 Limpieza y selección de columnas
# ==========================================================
df = df.select_dtypes(include=["number"])
logging.info(f"✅ Dataset cargado ({len(df)} filas, {len(df.columns)} columnas)")

# ==========================================================
# 🎯 Variables objetivo y features
# ==========================================================
leakage = ["result", "btts", "over_2.5", "goals_home", "away_goals", "total_goals"]
feature_cols = [col for col in df.columns if col not in leakage]
X = df[feature_cols]

# ==========================================================
# ⚙️ Función auxiliar de entrenamiento
# ==========================================================
def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    logging.info(f"✅ {model_name.upper()} | ACC={acc:.3f} | F1={f1:.3f}")
    return acc, f1

# ==========================================================
# 🚀 Entrenamiento de modelos
# ==========================================================
results = []
for target in ["result", "btts", "over_2.5"]:
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if target == "result":
        model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
    elif target == "btts":
        model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=False, random_seed=42)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

    acc, f1 = train_and_evaluate(target, model, X_train, X_test, y_train, y_test)
    results.append({"target": target, "accuracy": acc, "f1_score": f1})

# ==========================================================
# 💾 Guardado de resultados
# ==========================================================
os.makedirs("reports", exist_ok=True)
results_df = pd.DataFrame(results)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
summary_path = f"reports/model_performance_summary_{timestamp}.csv"
results_df.to_csv(summary_path, index=False)

# Guardar log CSV acumulativo
log_csv = "logs/model_training_log.csv"
if os.path.exists(log_csv):
    prev = pd.read_csv(log_csv)
    results_df = pd.concat([prev, results_df], ignore_index=True)
results_df.to_csv(log_csv, index=False)

logging.info(f"🏁 Entrenamiento completado. Resultados guardados en {summary_path}")
print(f"✅ Model training completed using {data_path}")

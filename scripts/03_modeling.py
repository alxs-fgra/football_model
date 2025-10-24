import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==========================================================
# 🧠 Setup de logging
# ==========================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/model_training.log",  # Archivo separado para logging
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# 📂 Carga del dataset más reciente
# ==========================================================
DATA_PATH = "data/processed/features_with_targets_latest.csv"
if not os.path.exists(DATA_PATH):
    logging.error(f"❌ Dataset not found at {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=["number"])
logging.info(f"✅ Dataset loaded: {DATA_PATH} ({len(df)} rows)")

# ==========================================================
# 🎯 Definición de variables
# ==========================================================
leakage = ["result", "btts", "over_2.5", "home_goals", "away_goals", "total_goals"]
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

    # Selección del modelo base según el target
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

# 1️⃣ Guardar resumen principal
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
summary_path = f"reports/model_performance_summary_{timestamp}.csv"
results_df.to_csv(summary_path, index=False)
logging.info(f"🏁 Model training completed successfully → {summary_path}")

# 2️⃣ Guardar log adicional para evaluación (sin anexado problemático)
log_path = "logs/model_training_log.csv"
if os.path.exists(log_path):
    existing_df = pd.read_csv(log_path)
    results_df = pd.concat([existing_df, results_df], ignore_index=True)
results_df.to_csv(log_path, index=False)
logging.info(f"🧾 Training log saved at {log_path}")

print(f"✅ Model training completed. Summary saved to {summary_path}")
print(f"🧾 Training log saved to {log_path}")

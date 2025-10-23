#!/usr/bin/env python3
# ==========================================
# ⚽ MODEL CROSS-VALIDATION (Data Leakage Safe)
# ==========================================
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# ==========================================
# 📁 CONFIGURACIÓN DE DIRECTORIOS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Buscar dataset más reciente con targets
files = [f for f in os.listdir(DATA_DIR) if "with_targets" in f and f.endswith(".csv")]
if not files:
    raise SystemExit("❌ No se encontró ningún dataset con targets. Ejecuta primero 05_add_targets.py")

latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(DATA_DIR, f)))
dataset_path = os.path.join(DATA_DIR, latest_file)

print(f"📂 Using dataset: {dataset_path}")

# ==========================================
# 📊 CARGA Y VALIDACIÓN DE COLUMNAS
# ==========================================
df = pd.read_csv(dataset_path)

required_targets = ["result", "btts", "over_2.5"]
missing_targets = [t for t in required_targets if t not in df.columns]
if missing_targets:
    raise SystemExit(f"❌ Missing target columns: {missing_targets}")

# ==========================================
# ⚠️ DETECCIÓN DE POSIBLE DATA LEAKAGE
# ==========================================
leakage_cols = [c for c in df.columns if "goal" in c.lower() and c not in ["avg_goals_home", "avg_goals_away", "h2h_avg_goals"]]
if leakage_cols:
    print(f"⚠️ Detected potential leakage columns: {leakage_cols}")

# Excluir columnas que pueden provocar leakage
exclude_cols = [
    "home_team", "away_team", "league", "date", "status",
    "goals_home", "goals_away", "total_goals"
]
features = [c for c in df.columns if c not in exclude_cols + required_targets]
print(f"✅ Using {len(features)} features: {features}")

# ==========================================
# ⚙️ CONFIGURACIÓN DE VALIDACIÓN
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "f1": make_scorer(f1_score, average="weighted", zero_division=0),
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"cross_validation_results_{timestamp}.csv")

# ==========================================
# 🧠 VALIDAR CADA TARGET
# ==========================================
results = []

for target_col in required_targets:
    print(f"\n🏟️ Cross-validating for target: {target_col.upper()}")

    X = df[features]
    y = df[target_col]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    summary = {
        "target": target_col,
        "accuracy": round(scores["test_accuracy"].mean(), 3),
        "precision": round(scores["test_precision"].mean(), 3),
        "recall": round(scores["test_recall"].mean(), 3),
        "f1": round(scores["test_f1"].mean(), 3),
    }

    results.append(summary)

    print(f"📊 [{target_col}] Mean Metrics (5-Fold):")
    for k, v in summary.items():
        if k != "target":
            print(f"   - {k.capitalize():<10}: {v}")

# ==========================================
# 💾 GUARDAR RESULTADOS
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(log_path, index=False)
print(f"\n💾 Cross-validation results saved → {log_path}")

# ==========================================
# ✅ RESUMEN FINAL
# ==========================================
print("\n🏁 Final Cross-validation Summary:")
print(results_df)
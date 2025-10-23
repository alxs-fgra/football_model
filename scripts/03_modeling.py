import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
DATA_PATH = "data/processed/features_with_targets_20251023_161723.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
LOGS_PATH = "logs/model_training_log.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def log_message(msg):
    print(msg)

def plot_feature_importance(importances, feature_names, model_name, target):
    plt.figure(figsize=(10, 6))
    idx = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), np.array(feature_names)[idx], rotation=45, ha="right")
    plt.title(f"Feature Importance: {model_name} ({target})")
    plt.tight_layout()
    filename = f"{REPORTS_DIR}/feature_importance_{target}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return filename


def save_metrics_to_log(target, model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }])

    if os.path.exists(LOGS_PATH):
        log_entry.to_csv(LOGS_PATH, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(LOGS_PATH, index=False)

    return accuracy, precision, recall, f1, cm


# ==========================================
# CARGAR DATASET
# ==========================================
print("🚀 Starting model training pipeline...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"📂 Loaded dataset: {DATA_PATH}")
print(f"✅ Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")

# ==========================================
# DEFINIR FEATURES SEGURAS (SIN LEAKAGE)
# ==========================================
FEATURES = [
    "league_id", "season",
    "home_avg_goals_last5", "home_avg_conceded_last5",
    "away_avg_goals_last5", "away_avg_conceded_last5",
    "season_progress", "league"
]

print(f"✅ Using {len(FEATURES)} clean features: {FEATURES}")

# ==========================================
# MODEL TRAINING FUNCTION
# ==========================================
def train_and_evaluate(df, target):
    print(f"\n🏟️ Training models for: {target}")

    df_target = df.dropna(subset=[target])
    print(f"🔎 Checking data for '{target}': {len(df)} → {len(df_target)} valid rows")

    if len(df_target) < 100:
        print(f"❌ Not enough samples for target '{target}'. Skipping.")
        return None

    X = df_target[FEATURES]
    y = df_target[target]

    # Codificar variables categóricas
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42
        ),
    }

    results = []
    for name, model in models.items():
        print(f"⚙️ Training {name} for {target}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy, precision, recall, f1, cm = save_metrics_to_log(target, name, y_test, y_pred)
        model_filename = f"{MODELS_DIR}/{target.lower()}_{name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        pd.to_pickle(model, model_filename)
        fi_path = plot_feature_importance(model.feature_importances_, X.columns, name.lower(), target)

        print(f"💾 Model saved: {model_filename}")
        print(f"📊 Feature importance saved: {fi_path}")
        print(f"📄 Metrics logged: {LOGS_PATH}")

        results.append({
            "model": name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        })

    df_results = pd.DataFrame(results)
    print(f"\n📊 Results for {target}:\n{df_results}\n")
    return df_results


# ==========================================
# MAIN EXECUTION
# ==========================================
all_results = []
for target in ["result", "btts", "over_2.5"]:
    result_df = train_and_evaluate(df, target)
    if result_df is not None:
        result_df["target"] = target
        all_results.append(result_df)

if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_path = f"reports/model_performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Summary of all models saved: {summary_path}")

print("✅ Training pipeline finished.")
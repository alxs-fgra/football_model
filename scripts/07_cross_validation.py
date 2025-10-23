import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_PATH = "data/processed/features_with_targets_20251023_164110.csv"
REPORTS_DIR = "reports"
LOGS_PATH = "logs/cross_validation_results.csv"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# FEATURES LIMPIAS (sin fuga)
# ==========================================
FEATURES = [
    "league_id", "season",
    "home_avg_goals_last5", "home_avg_conceded_last5",
    "away_avg_goals_last5", "away_avg_conceded_last5",
    "season_progress", "league"
]

# ==========================================
# MÉTRICAS AUXILIARES
# ==========================================
def evaluate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

# ==========================================
# ENTRENAR Y VALIDAR CON K-FOLD
# ==========================================
def cross_validate_model(model_name, model, X, y, target, n_splits=5):
    print(f"\n🔁 Cross-validating {model_name} for target '{target}'...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_metrics(y_test, y_pred)
        metrics["fold"] = fold
        metrics["model"] = model_name
        metrics["target"] = target
        fold_metrics.append(metrics)

        print(f"   Fold {fold}/{n_splits} - Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")

    df = pd.DataFrame(fold_metrics)
    df["accuracy_mean"] = df["accuracy"].mean()
    df["f1_mean"] = df["f1"].mean()
    return df

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
print("🚀 Starting cross-validation pipeline...")
df = pd.read_csv(DATA_PATH)
print(f"📂 Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

results_all = []

for target in ["result", "btts", "over_2.5"]:
    df_target = df.dropna(subset=[target])
    X = pd.get_dummies(df_target[FEATURES], drop_first=True)
    y = df_target[target]

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

    for name, model in models.items():
        df_metrics = cross_validate_model(name, model, X, y, target)
        results_all.append(df_metrics)

# ==========================================
# GUARDAR RESULTADOS
# ==========================================
summary_df = pd.concat(results_all, ignore_index=True)
summary_csv = f"{REPORTS_DIR}/cross_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_df.to_csv(summary_csv, index=False)

# También guardar promedios por modelo
avg_df = (
    summary_df.groupby(["target", "model"])
    [["accuracy", "precision", "recall", "f1"]]
    .mean()
    .reset_index()
)
avg_csv = f"{REPORTS_DIR}/cross_validation_avg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
avg_df.to_csv(avg_csv, index=False)

print("\n✅ Cross-validation finished!")
print(f"📄 Detailed fold metrics: {summary_csv}")
print(f"📊 Averages per model: {avg_csv}")
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# ===============================
# CONFIG
# ===============================
DATA_DIR = "data/processed"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_latest_dataset():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {DATA_DIR}")
    latest = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)))
    return os.path.join(DATA_DIR, latest)

def compute_cv_metrics(model, X, y):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {metric: np.mean(cross_val_score(model, X, y, cv=kfold, scoring=sc))
               for metric, sc in scoring.items()}
    return results

if __name__ == "__main__":
    dataset_path = get_latest_dataset()
    print(f"📂 Using dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    expected_targets = ["result", "btts", "over_2.5"]
    for col in expected_targets:
        if col not in df.columns:
            raise ValueError(f"Missing target column: {col}")

    # Detect and remove leakage columns
    leakage_cols = [c for c in ["goals_home", "goals_away", "total_goals"] if c in df.columns]
    if leakage_cols:
        print(f"⚠️ Detected potential leakage columns: {leakage_cols}")

    # Drop non-numeric text-based columns that are identifiers
    drop_cols = ["date", "home_team", "away_team"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Label encode all remaining object (string) columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        print(f"🔤 Encoded column: {col}")

    # Define features
    base_features = [c for c in df.columns if c not in expected_targets + leakage_cols]
    print(f"✅ Using {len(base_features)} features: {base_features}")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for target in expected_targets:
        print(f"\n🏟️ Cross-validating for target: {target.upper()}")
        X = df[base_features]
        y = df[target]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="most_frequent"), X.columns)
        ])

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        metrics = compute_cv_metrics(pipeline, X, y)
        print(f"📊 [{target}] Mean Metrics (5-Fold):")
        for k, v in metrics.items():
            print(f"   - {k.capitalize():9}: {v:.3f}")

        results.append({"target": target, **metrics})

    results_df = pd.DataFrame(results)
    output_path = os.path.join(LOG_DIR, f"cross_validation_results_{timestamp}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n💾 Cross-validation results saved → {output_path}")
    print("\n🏁 Final Cross-validation Summary:")
    print(results_df.round(3))

#!/usr/bin/env python3
# =====================================
# ⚽ MODEL TRAINING PIPELINE (Safe Feature Selection + Evaluation)
# =====================================
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =====================================
# 📁 CONFIG
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, "model_training_log.csv")

# =====================================
# 📂 LOAD DATASET (latest features file)
# =====================================
files = [f for f in os.listdir(DATA_DIR) if f.startswith("features_la_liga")]
if not files:
    raise SystemExit("❌ No dataset found in data/processed/")
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)
DATA_PATH = os.path.join(DATA_DIR, files[0])

print(f"📂 Using dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# =====================================
# 🧠 AUTO-GENERATE TARGETS IF MISSING
# =====================================
required_targets = ["result", "btts", "over_2.5"]
missing_targets = [col for col in required_targets if col not in df.columns]

if missing_targets:
    print(f"⚠️ Missing targets: {missing_targets} → Generating automatically...")
    
    if {"goals_home", "goals_away", "total_goals"}.issubset(df.columns):
        df["result"] = df.apply(
            lambda row: 1 if row["goals_home"] > row["goals_away"]
            else -1 if row["goals_home"] < row["goals_away"]
            else 0,
            axis=1,
        )
        df["btts"] = df.apply(
            lambda row: 1 if (row["goals_home"] > 0 and row["goals_away"] > 0) else 0,
            axis=1,
        )
        df["over_2.5"] = df["total_goals"].apply(lambda x: 1 if x > 2.5 else 0)

        # 💾 Save auto-generated dataset
        auto_path = os.path.join(DATA_DIR, f"features_la_liga_with_targets_auto_{timestamp}.csv")
        df.to_csv(auto_path, index=False)
        print(f"✅ Targets generated and saved to → {auto_path}")
    else:
        raise SystemExit("❌ Cannot generate targets: missing goal columns (goals_home / goals_away / total_goals).")
else:
    print("✅ All target columns found in dataset.")

# =====================================
# 🧩 DEFINE FEATURES & TARGETS
# =====================================
# Remove columns that would leak real outcomes
leakage_cols = ["result", "btts", "over_2.5", "goals_home", "goals_away", "total_goals"]
numeric_df = df.select_dtypes(include=[np.number])
features = [col for col in numeric_df.columns if col not in leakage_cols]

if not features:
    raise SystemExit("❌ No numeric features found to train models!")

print(f"✅ Selected {len(features)} valid features (no leakage): {features}")

target_map = {
    "1x2": "result",
    "btts": "btts",
    "over_2.5": "over_2.5",
}

# =====================================
# 📊 METRICS FUNCTION
# =====================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 [{model_name}] Metrics:")
    print(f"   - Accuracy : {acc:.3f}")
    print(f"   - Precision: {prec:.3f}")
    print(f"   - Recall   : {rec:.3f}")
    print(f"   - F1 Score : {f1:.3f}")
    print("   - Confusion Matrix:")
    print(cm)

    return acc, prec, rec, f1

# =====================================
# ⚙️ TRAINING LOOP
# =====================================
logs = []

for key, target_col in target_map.items():
    print(f"\n🏟️ Training model for: {key.upper()}")

    X = df[features]
    y = df[target_col]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500) if key == "1x2" else RandomForestClassifier(
        n_estimators=150, random_state=42
    )

    model.fit(X_train, y_train)
    acc, prec, rec, f1 = evaluate_model(model, X_test, y_test, key)

    # Save model
    model_filename = f"{key}_model_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)

    print(f"💾 Model saved: {model_path}")

    logs.append({
        "timestamp": timestamp,
        "model": key,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "dataset": os.path.basename(DATA_PATH),
        "features": len(features)
    })

# =====================================
# 🧾 SAVE LOG
# =====================================
df_logs = pd.DataFrame(logs)
if not os.path.exists(LOG_FILE):
    df_logs.to_csv(LOG_FILE, index=False)
else:
    df_logs.to_csv(LOG_FILE, mode="a", header=False, index=False)

print("\n✅ Training completed successfully!")
print(f"📊 Metrics logged in: {LOG_FILE}")
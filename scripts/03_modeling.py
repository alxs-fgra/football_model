import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

# =====================================================
# 🔧 CONFIGURACIÓN GENERAL
# =====================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "model_training_log.csv")

# =====================================================
# 📦 FUNCIONES AUXILIARES
# =====================================================
def log(msg):
    print(msg)

def get_latest_dataset():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"❌ No CSV found in {DATA_DIR}\nAvailable: {os.listdir(DATA_DIR)}")
    latest = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)))
    latest_path = os.path.join(DATA_DIR, latest)
    log(f"📂 Using dataset: {latest_path}")
    return latest_path

def save_metrics(model_name, accuracy, precision, recall, f1):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"{ts},{model_name},{accuracy:.3f},{precision:.3f},{recall:.3f},{f1:.3f},{os.path.basename(DATA_PATH)}\n"
    header = "timestamp,model,accuracy,precision,recall,f1,dataset\n"
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)

# =====================================================
# ⚙️ ENTRENAMIENTO
# =====================================================
def train_and_evaluate(df, target_col, model_name):
    log(f"\n🏟️ Training model for: {model_name.upper()}")

    # Features (evitamos fuga de datos)
    drop_cols = ["date", "home_team", "away_team", "winner", "league"]
    features = [c for c in df.columns if c not in drop_cols + ["result", "btts", "over_2.5"]]
    log(f"✅ Selected {len(features)} valid features (no leakage): {features}")

    X = df[features]
    y = df[target_col]

    # Imputación + Escalado
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split + Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Métricas
    log(f"\n📊 [{model_name}] Metrics:")
    log(f"   - Accuracy : {acc:.3f}")
    log(f"   - Precision: {prec:.3f}")
    log(f"   - Recall   : {rec:.3f}")
    log(f"   - F1 Score : {f1:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    log(f"   - Confusion Matrix:\n{cm}")

    # Guardar modelo y métricas
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model_{ts}.pkl")
    joblib.dump(model, model_path)
    log(f"💾 Model saved: {model_path}")
    save_metrics(model_name, acc, prec, rec, f1)

# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":
    DATA_PATH = get_latest_dataset()
    df = pd.read_csv(DATA_PATH)

    expected_targets = ["result", "btts", "over_2.5"]
    missing = [col for col in expected_targets if col not in df.columns]
    if missing:
        log(f"❌ Missing target columns: {missing}")
        exit(1)
    else:
        log("✅ All target columns found in dataset.")

    # Entrenamos modelos para los tres mercados
    train_and_evaluate(df, "result", "1x2")
    train_and_evaluate(df, "btts", "btts")
    train_and_evaluate(df, "over_2.5", "over_2.5")

    log("\n✅ Training completed successfully!")
    log(f"📊 Metrics logged in: {LOG_FILE}")

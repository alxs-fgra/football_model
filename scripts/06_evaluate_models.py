import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def log(msg):
    print(f"🧠 {msg}")

# ==================================================
# 📍 DEFINIR RUTA ABSOLUTA CON BASE EN ESTE ARCHIVO
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def get_latest_dataset():
    log(f"🔍 Looking for CSVs inside: {PROCESSED_DIR}")
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    if not csv_files:
        log(f"⚠️ Contents of processed dir: {os.listdir(PROCESSED_DIR)}")
        raise FileNotFoundError(f"❌ No dataset CSV found in {PROCESSED_DIR}")
    latest = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(PROCESSED_DIR, f)))
    path = os.path.join(PROCESSED_DIR, latest)
    log(f"✅ Latest dataset found: {path}")
    return path

def evaluate_dataset(df):
    log(f"📊 Evaluating dataset with {len(df)} rows...")
    stats = {
        "total_matches": len(df),
        "home_wins": (df["result"] == 1).sum(),
        "draws": (df["result"] == 0).sum(),
        "away_wins": (df["result"] == -1).sum(),
        "btts_yes": df["btts"].sum(),
        "over_2.5_yes": df["over_2.5"].sum()
    }
    df_summary = pd.DataFrame([stats])
    log(df_summary.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(REPORTS_DIR, f"dataset_summary_{ts}.csv")
    chart_path = os.path.join(REPORTS_DIR, f"result_distribution_{ts}.png")

    df_summary.to_csv(summary_path, index=False)
    log(f"✅ Summary saved: {summary_path}")

    plt.figure(figsize=(6, 4))
    df["result"].value_counts().sort_index().plot(kind="bar", color=["green", "gray", "red"])
    plt.title("Distribution of Results (1=Home, 0=Draw, -1=Away)")
    plt.xlabel("Result")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(chart_path)
    log(f"📊 Chart saved: {chart_path}")

if __name__ == "__main__":
    dataset_path = get_latest_dataset()
    df = pd.read_csv(dataset_path)
    evaluate_dataset(df)
    log("✅ Evaluation completed successfully.")

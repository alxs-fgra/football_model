import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def log(msg):
    print(f"🧠 {msg}")

# ==================================================
# 🔍 AUTO-DETECTA RUTA BASE
# ==================================================
def find_processed_dir():
    search_depth = 4
    cwd = os.getcwd()
    log(f"🔎 Current working directory: {cwd}")

    for root, dirs, files in os.walk(cwd):
        if "data" in dirs:
            candidate = os.path.join(root, "data", "processed")
            if os.path.exists(candidate):
                log(f"✅ Found processed directory at: {candidate}")
                return candidate
        if root.count(os.sep) - cwd.count(os.sep) >= search_depth:
            break
    raise FileNotFoundError("❌ Could not locate any 'data/processed' directory in the project tree.")

# ==================================================
# 📊 EVALUACIÓN DE MODELOS
# ==================================================
def get_latest_dataset(processed_dir):
    log(f"🔍 Searching for CSV files in: {processed_dir}")
    files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"❌ No dataset found in {processed_dir}")
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
    latest_path = os.path.join(processed_dir, latest)
    log(f"📂 Latest dataset detected: {latest_path}")
    return latest_path

def evaluate_dataset(df, reports_dir):
    log(f"📊 Evaluating dataset with {len(df)} records...")
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

    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 🧾 Save summary CSV
    summary_path = os.path.join(reports_dir, f"dataset_summary_{timestamp}.csv")
    df_summary.to_csv(summary_path, index=False)
    log(f"✅ Saved summary: {summary_path}")

    # 📈 Chart
    plt.figure(figsize=(6, 4))
    df["result"].value_counts().sort_index().plot(kind="bar", color=["green", "gray", "red"])
    plt.title("Distribution of Results (1=Home, 0=Draw, -1=Away)")
    plt.xlabel("Result")
    plt.ylabel("Frequency")
    plt.tight_layout()
    chart_path = os.path.join(reports_dir, f"result_distribution_{timestamp}.png")
    plt.savefig(chart_path)
    log(f"📊 Saved chart: {chart_path}")

# ==================================================
# 🚀 MAIN
# ==================================================
if __name__ == "__main__":
    processed_dir = find_processed_dir()
    reports_dir = os.path.join(os.path.dirname(processed_dir), "reports")

    latest_dataset = get_latest_dataset(processed_dir)
    df = pd.read_csv(latest_dataset)
    evaluate_dataset(df, reports_dir)
    log("✅ Evaluation completed successfully.")

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def log(msg):
    print(f"🧠 {msg}")

# ==================================================
# 🧭 PATH AUTO-DETECTION (FINAL FIX)
# ==================================================
def resolve_base_dir():
    current_dir = os.path.abspath(os.getcwd())
    log(f"🔍 Current working directory: {current_dir}")

    candidates = [
        current_dir,
        os.path.join(current_dir, "football_model"),
        os.path.join(current_dir, "../football_model"),
        os.path.join(current_dir, "../../football_model"),
    ]

    for path in candidates:
        processed_path = os.path.join(path, "data", "processed")
        if os.path.exists(processed_path):
            log(f"✅ Found data directory: {processed_path}")
            return processed_path
    raise FileNotFoundError("❌ Could not locate 'data/processed' folder in any expected location.")

def get_latest_dataset(processed_dir):
    log(f"🔎 Checking for CSVs in {processed_dir}")
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
    if not csv_files:
        log(f"⚠️ No CSVs found in {processed_dir}. Listing contents:")
        for f in os.listdir(processed_dir):
            log(f"   - {f}")
        raise FileNotFoundError(f"❌ No dataset CSV found in {processed_dir}")
    latest = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
    latest_path = os.path.join(processed_dir, latest)
    log(f"📂 Latest dataset: {latest_path}")
    return latest_path

def evaluate_dataset(df, reports_dir):
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

    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(reports_dir, f"dataset_summary_{timestamp}.csv")
    chart_path = os.path.join(reports_dir, f"result_distribution_{timestamp}.png")

    df_summary.to_csv(summary_path, index=False)
    log(f"✅ Saved summary: {summary_path}")

    plt.figure(figsize=(6, 4))
    df["result"].value_counts().sort_index().plot(kind="bar", color=["green", "gray", "red"])
    plt.title("Distribution of Results (1=Home, 0=Draw, -1=Away)")
    plt.xlabel("Result")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(chart_path)
    log(f"📊 Saved chart: {chart_path}")

if __name__ == "__main__":
    processed_dir = resolve_base_dir()
    reports_dir = os.path.join(os.path.dirname(processed_dir), "reports")

    latest_csv = get_latest_dataset(processed_dir)
    df = pd.read_csv(latest_csv)
    evaluate_dataset(df, reports_dir)

    log("✅ Evaluation completed successfully.")

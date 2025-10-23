import os
import pandas as pd
from datetime import datetime

# ======================================
# CONFIGURACIÓN
# ======================================
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

def add_target_columns(df):
    """Genera columnas de targets para predicciones deportivas."""
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    # RESULT: 1 = home win, 0 = draw, 2 = away win
    df["result"] = df.apply(
        lambda x: 1 if x["home_goals"] > x["away_goals"]
        else 0 if x["home_goals"] == x["away_goals"]
        else 2,
        axis=1
    )

    # BTTS: both teams to score
    df["btts"] = df.apply(
        lambda x: 1 if x["home_goals"] > 0 and x["away_goals"] > 0 else 0,
        axis=1
    )

    # OVER 2.5 goals
    df["over_2.5"] = df["total_goals"].apply(lambda x: 1 if x > 2.5 else 0)

    return df

def main():
    # Buscar el CSV más reciente
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "features" in f]
    if not csv_files:
        print("❌ No CSV file found in data/processed/")
        return

    latest_csv = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)))
    input_path = os.path.join(DATA_DIR, latest_csv)

    print(f"📂 Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    print("🧮 Generating target columns...")
    df_targets = add_target_columns(df)

    # Guardar dataset nuevo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(DATA_DIR, f"features_with_targets_{timestamp}.csv")
    df_targets.to_csv(output_path, index=False)

    print(f"\n✅ Targets added successfully!")
    print(f"💾 Saved new dataset with targets → {output_path}")

    print("\n📊 Summary:")
    print("Result distribution:")
    print(df_targets["result"].value_counts())
    print("\nBTTS distribution:")
    print(df_targets["btts"].value_counts())
    print("\nOver 2.5 distribution:")
    print(df_targets["over_2.5"].value_counts())

if __name__ == "__main__":
    main()
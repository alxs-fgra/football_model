#!/usr/bin/env python3
# ============================================
# ⚽ SCRIPT 05 - ADD TARGET COLUMNS
# ============================================
import os
import pandas as pd
from datetime import datetime

# ==========================
# 📁 PATH CONFIG
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "features_la_liga_2015_2023.csv")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", f"features_la_liga_with_targets_{timestamp}.csv"
)

# ==========================
# 🧠 LOAD DATA
# ==========================
print(f"📂 Loading dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

required_cols = ["goals_home", "goals_away", "total_goals"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

# ==========================
# 🧮 GENERATE TARGET COLUMNS
# ==========================
print("🧮 Generating target columns...")

# Resultado (1=local gana, 0=empate, -1=visitante gana)
df["result"] = df.apply(
    lambda row: 1 if row["goals_home"] > row["goals_away"]
    else -1 if row["goals_home"] < row["goals_away"]
    else 0,
    axis=1,
)

# Ambos anotan (1=sí, 0=no)
df["btts"] = df.apply(
    lambda row: 1 if (row["goals_home"] > 0 and row["goals_away"] > 0) else 0,
    axis=1,
)

# Over 2.5 goles (1=sí, 0=no)
df["over_2.5"] = df["total_goals"].apply(lambda x: 1 if x > 2.5 else 0)

# ==========================
# 💾 SAVE OUTPUT
# ==========================
df.to_csv(OUTPUT_FILE, index=False)
print("\n✅ Targets added successfully!")
print(f"💾 Saved new dataset with targets → {OUTPUT_FILE}")

# ==========================
# 🔍 SAMPLE OUTPUT
# ==========================
print("\n📊 Preview:")
print(df[["home_team", "away_team", "goals_home", "goals_away", "result", "btts", "over_2.5"]].head())
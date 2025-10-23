import os
import pandas as pd
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"🧠 {msg}")

# ==========================================
# 🔍 OBTENER ARCHIVO MÁS RECIENTE
# ==========================================
def get_latest_processed_file():
    files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith("processed_") and f.endswith(".jsonl")]
    if not files:
        raise FileNotFoundError("❌ No se encontraron archivos procesados en data/processed/")
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(PROCESSED_DIR, f)))
    return os.path.join(PROCESSED_DIR, latest_file)

# ==========================================
# 🧮 GENERAR TARGETS
# ==========================================
def generate_targets(df):
    df["result"] = df.apply(
        lambda x: 1 if x["home_goals"] > x["away_goals"] else 0 if x["home_goals"] == x["away_goals"] else -1, axis=1
    )
    df["btts"] = df.apply(lambda x: 1 if (x["home_goals"] > 0 and x["away_goals"] > 0) else 0, axis=1)
    df["over_2.5"] = df.apply(lambda x: 1 if (x["home_goals"] + x["away_goals"]) > 2.5 else 0, axis=1)
    return df

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    latest_file = get_latest_processed_file()
    log(f"📂 Cargando dataset más reciente: {latest_file}")

    df = pd.read_json(latest_file, lines=True)
    log(f"📊 Partidos cargados: {len(df)}")

    df = generate_targets(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"features_with_targets_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    log(f"✅ Targets generados correctamente y guardados en: {output_file}")

    print(df.head())

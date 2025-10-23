import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURACIÓN GLOBAL (ABSOLUTE PATH)
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def log(msg):
    print(f"🧠 {msg}")

# ==========================================
# 🔍 OBTENER ARCHIVO MÁS RECIENTE
# ==========================================
def get_latest_dataset():
    log(f"🔎 Buscando datasets en: {PROCESSED_DIR}")
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(f"❌ La carpeta no existe: {PROCESSED_DIR}")

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"❌ No dataset found in {PROCESSED_DIR}")

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(PROCESSED_DIR, f)))
    return os.path.join(PROCESSED_DIR, latest_file)

# ==========================================
# 📊 EVALUACIÓN DE RESULTADOS
# ==========================================
def evaluate_dataset(df):
    log(f"📊 Evaluando dataset con {len(df)} registros...")

    # Métricas básicas
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

    # 📈 Gráfica de resultados
    plt.figure(figsize=(6, 4))
    df["result"].value_counts().sort_index().plot(kind="bar", color=["green", "gray", "red"])
    plt.title("Distribución de Resultados (1=Local, 0=Empate, -1=Visita)")
    plt.xlabel("Resultado")
    plt.ylabel("Frecuencia")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(REPORTS_DIR, f"result_distribution_{timestamp}.png")
    plt.savefig(chart_path)
    log(f"📊 Gráfica guardada en: {chart_path}")

    # 🧾 Guardar resumen CSV
    summary_path = os.path.join(REPORTS_DIR, f"dataset_summary_{timestamp}.csv")
    df_summary.to_csv(summary_path, index=False)
    log(f"✅ Resumen guardado en: {summary_path}")

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    latest_dataset = get_latest_dataset()
    log(f"📂 Cargando dataset más reciente: {latest_dataset}")

    df = pd.read_csv(latest_dataset)
    evaluate_dataset(df)
    log("✅ Evaluación completada exitosamente.")

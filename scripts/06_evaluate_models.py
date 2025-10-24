import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================
# 📂 CONFIGURACIÓN GENERAL
# =====================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "model_training_log.csv")

# =====================================================
# 🧠 FUNCIONES AUXILIARES
# =====================================================
def log(msg):
    print(msg)

def load_logs():
    if not os.path.exists(LOG_FILE):
        log(f"❌ No se encontró archivo de logs: {LOG_FILE}")
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        log(f"✅ Logs cargados correctamente. Registros: {len(df)}")
        return df
    except pd.errors.ParserError as e:
        log(f"❌ Error al parsear {LOG_FILE}: {e}")
        return None

def generate_summary(df):
    if df is None or df.empty:
        log("⚠️ No hay datos para generar resumen.")
        return None, None
    grouped = df.groupby("target")[["accuracy", "f1_score"]].mean().round(3)
    log("\n📊 Métricas promedio por modelo:")
    print(grouped)
    best = df.loc[df.groupby("target")["f1_score"].idxmax()]
    return grouped, best

def plot_performance(df):
    if df is None or df.empty:
        log("⚠️ No hay datos para graficar.")
        return None
    plt.figure(figsize=(8,5))
    grouped = df.groupby("target")[["accuracy", "f1_score"]].mean().reset_index()
    plt.bar(grouped["target"], grouped["accuracy"], label="Accuracy", alpha=0.6)
    plt.bar(grouped["target"], grouped["f1_score"], label="F1 Score", alpha=0.6)
    plt.legend()
    plt.title("📈 Average Model Performance")
    plt.xlabel("Target")
    plt.ylabel("Score")
    plt.grid(alpha=0.4)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(REPORTS_DIR, f"model_performance_{ts}.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    log(f"🖼️ Gráfica guardada en: {chart_path}")
    return chart_path

def generate_html(summary_df, best_df, chart_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORTS_DIR, f"model_evaluation_report_{ts}.html")
    html = f"""
    <html><body style='font-family:Arial'>
    <h1>⚽ Football Model Evaluation Report</h1>
    <h3>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3><hr>
    <h2>📊 Average Metrics</h2>
    {summary_df.to_html(index=True) if summary_df is not None else '<p>No data</p>'}
    <hr><h2>🏆 Best Models</h2>
    {best_df.to_html(index=False) if best_df is not None else '<p>No data</p>'}
    <hr><img src='{os.path.basename(chart_path)}' width='600'>
    </body></html>
    """
    with open(html_path, "w") as f:
        f.write(html)
    log(f"📄 Reporte HTML generado: {html_path}")
    return html_path

# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":
    try:
        df = load_logs()
        summary, best = generate_summary(df)
        chart = plot_performance(df)
        html = generate_html(summary, best, chart)
        log("✅ Evaluación completada.")
    except Exception as e:
        log(f"❌ Error durante la evaluación: {e}")
        exit(1)

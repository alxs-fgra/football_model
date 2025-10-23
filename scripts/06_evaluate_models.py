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
        raise FileNotFoundError(f"❌ No se encontró archivo de logs: {LOG_FILE}")
    df = pd.read_csv(LOG_FILE)
    log(f"✅ Logs cargados correctamente.\n📊 Total de registros: {len(df)}")
    return df

def generate_summary(df):
    grouped = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean().round(3)
    log("\n📊 Métricas promedio por tipo de modelo:")
    print(grouped)

    best_models = df.loc[df.groupby("model")["f1"].idxmax()]
    log("\n🏆 Mejores modelos por tipo:")
    print(best_models[["model", "timestamp", "accuracy", "f1", "dataset"]])
    return grouped, best_models

def plot_performance(df):
    plt.figure(figsize=(8, 5))
    grouped = df.groupby("model")[["accuracy", "f1"]].mean().reset_index()
    plt.bar(grouped["model"], grouped["accuracy"], label="Accuracy", alpha=0.6)
    plt.bar(grouped["model"], grouped["f1"], label="F1 Score", alpha=0.6)
    plt.title("📈 Average Model Performance")
    plt.xlabel("Model Type")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(REPORTS_DIR, f"model_performance_{ts}.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    log(f"\n🖼️ Gráfica guardada: {chart_path}")
    return chart_path

def generate_html_report(summary_df, best_df, chart_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORTS_DIR, f"model_evaluation_report_{ts}.html")

    html = f"""
    <html>
    <head><title>Football ML Model Evaluation</title></head>
    <body style='font-family:Arial; margin:40px;'>
    <h1>⚽ Football Prediction Model Evaluation Report</h1>
    <h3>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
    <hr>
    <h2>📊 Average Metrics</h2>
    {summary_df.to_html(border=0, justify='center')}
    <hr>
    <h2>🏆 Best Models per Category</h2>
    {best_df.to_html(border=0, justify='center')}
    <hr>
    <h2>📈 Performance Chart</h2>
    <img src="{os.path.basename(chart_path)}" width="600">
    </body>
    </html>
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
        report = generate_html_report(summary, best, chart)
        log("\n✅ Evaluación completada con éxito.")
        log("Abre el archivo HTML generado para visualizar resultados de forma interactiva.")
    except Exception as e:
        log(f"❌ Error durante la evaluación: {e}")
        exit(1)

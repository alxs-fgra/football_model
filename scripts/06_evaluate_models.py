#!/usr/bin/env python3
# ==========================================
# ⚽ MODEL EVALUATION DASHBOARD (Offline)
# ==========================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# 📁 CONFIGURACIÓN DE DIRECTORIOS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "model_training_log.csv")

# ==========================================
# 📂 CARGA DE LOGS
# ==========================================
if not os.path.exists(LOG_FILE):
    raise SystemExit("❌ No se encontró el archivo de logs. Asegúrate de haber entrenado modelos con 03_modeling.py.")

df = pd.read_csv(LOG_FILE)

if df.empty:
    raise SystemExit("⚠️ El archivo de logs está vacío. Entrena al menos un modelo antes de evaluar.")

print("✅ Logs cargados correctamente.")
print(f"📊 Total de registros: {len(df)}")

# ==========================================
# 🧹 LIMPIEZA Y PREPARACIÓN
# ==========================================
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S", errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values(by="timestamp")

# ==========================================
# 📈 MÉTRICAS PROMEDIO
# ==========================================
avg_metrics = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean().round(3)
print("\n📊 Métricas promedio por tipo de modelo:")
print(avg_metrics)

# ==========================================
# 🧠 MEJOR MODELO POR MÉTRICA
# ==========================================
best_models = df.loc[df.groupby("model")["f1"].idxmax()][["model", "timestamp", "accuracy", "f1", "dataset"]]
print("\n🏆 Mejores modelos por tipo:")
print(best_models)

# ==========================================
# 🎨 CONFIGURACIÓN DE ESTILO
# ==========================================
sns.set(style="whitegrid", font_scale=1.1)

# ==========================================
# 📈 GRAFICA EVOLUCIÓN DE ACCURACY Y F1
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
sns.lineplot(ax=axes[0], data=df, x="timestamp", y="accuracy", hue="model", marker="o")
axes[0].set_title("Evolución de Accuracy por Modelo")
axes[0].set_ylabel("Accuracy")
axes[0].set_xlabel("Fecha")

sns.lineplot(ax=axes[1], data=df, x="timestamp", y="f1", hue="model", marker="o")
axes[1].set_title("Evolución de F1 Score por Modelo")
axes[1].set_ylabel("F1 Score")
axes[1].set_xlabel("Fecha")

plt.tight_layout()

# ==========================================
# 💾 GUARDAR GRÁFICAS Y REPORTE
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
png_path = os.path.join(REPORT_DIR, f"model_performance_{timestamp}.png")
html_path = os.path.join(REPORT_DIR, f"model_evaluation_report_{timestamp}.html")

plt.savefig(png_path)
print(f"\n🖼️ Gráfica guardada: {png_path}")

# Crear HTML simple con tabla + imagen
with open(html_path, "w") as f:
    f.write("<html><head><title>Model Evaluation Report</title></head><body>")
    f.write("<h1>⚽ Model Evaluation Report</h1>")
    f.write("<h2>📊 Métricas Promedio por Modelo</h2>")
    f.write(avg_metrics.to_html(border=0))
    f.write("<h2>🏆 Mejores Modelos</h2>")
    f.write(best_models.to_html(border=0))
    f.write(f"<h2>📈 Evolución</h2><img src='../reports/{os.path.basename(png_path)}' width='800'>")
    f.write("</body></html>")

print(f"📄 Reporte HTML generado: {html_path}")

# ==========================================
# ✅ RESULTADO FINAL
# ==========================================
print("\n✅ Evaluación completada con éxito.")
print("Abre el archivo HTML generado para visualizar resultados de forma interactiva.")
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn

# ============================
# ⚙️ CONFIGURACIÓN GENERAL
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

app = FastAPI(title="Football Prediction API", version="2.1")

# ============================
# 🧠 MODELOS VÁLIDOS
# ============================
VALID_MODELS = ["1x2", "btts", "over_2.5"]
MODEL_OBJECTS = {}

for file in os.listdir(MODEL_DIR):
    if any(m in file for m in VALID_MODELS) and file.endswith(".pkl"):
        try:
            MODEL_OBJECTS[file] = joblib.load(os.path.join(MODEL_DIR, file))
        except Exception as e:
            print(f"⚠️ No se pudo cargar {file}: {e}")

print(f"✅ Modelos cargados: {list(MODEL_OBJECTS.keys())}")

# ============================
# 📊 LISTA DE LIGAS
# ============================
@app.get("/list_leagues")
def list_leagues():
    leagues = ["la_liga", "premier_league", "serie_a"]
    return {"available_leagues": leagues, "count": len(leagues)}

# ============================
# 🎯 REQUEST BODY
# ============================
class MatchRequest(BaseModel):
    league: str
    home_team: str
    away_team: str

# ============================
# ⚙️ SIMULADOR DE FEATURES
# ============================
def generate_features(home_team: str, away_team: str):
    np.random.seed(abs(hash(home_team + away_team)) % (2**32))
    return np.random.rand(9).reshape(1, -1)  # 9 features esperadas

# ============================
# 🗺️ MAPEO DE ETIQUETAS
# ============================
LABEL_MAP = {
    "1x2": {0: "Away Win", 1: "Draw", 2: "Home Win"},
    "btts": {0: "No", 1: "Yes"},
    "over_2.5": {0: "Under 2.5", 1: "Over 2.5"},
}

# ============================
# 🤖 PREDICCIÓN
# ============================
@app.post("/predict_match")
async def predict_match(request: MatchRequest):
    league = request.league.lower()
    home = request.home_team
    away = request.away_team

    # Filtrar modelos de esa liga (si existen)
    league_models = {k: v for k, v in MODEL_OBJECTS.items() if league in k}
    if not league_models:
        print(f"⚠️ No se encontró liga '{league}', usando modelos genéricos.")
        league_models = MODEL_OBJECTS

    results = {}
    X_input = generate_features(home, away)

    # Evaluar cada modelo
    for name, model in league_models.items():
        try:
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input).max() if hasattr(model, "predict_proba") else 0.5

            # Identificar tipo de modelo
            mtype = next((t for t in VALID_MODELS if t in name), "unknown")
            label = LABEL_MAP.get(mtype, {}).get(int(pred), str(pred))

            results[mtype] = {"label": label, "confidence": round(float(prob), 3)}

        except Exception as e:
            print(f"⚠️ Error en modelo {name}: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="No se pudieron generar predicciones")

    # Generar resumen natural
    readable = []
    if "1x2" in results:
        readable.append(f"Resultado: {results['1x2']['label']}")
    if "btts" in results:
        readable.append(f"Ambos anotan: {results['btts']['label']}")
    if "over_2.5" in results:
        readable.append(f"Goles Totales: {results['over_2.5']['label']}")

    summary_text = ", ".join(readable)
    overall_conf = np.mean([v["confidence"] for v in results.values()])

    friendly_summary = (
        f"🏟️ *{home}* vs *{away}*\n"
        f"🏆 *{league.title()}*\n\n"
        f"🧠 *Predicción:* {summary_text}\n"
        f"🔮 *Confianza global:* {overall_conf*100:.1f}%"
    )

    # Guardar log automático
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    log_path = os.path.join(BASE_DIR, "logs", "predictions_history.csv")

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "league": league,
        "home_team": home,
        "away_team": away,
        "summary": summary_text,
        "confidence": round(overall_conf, 3)
    }])
    log_entry.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))

    return {
        "home_team": home,
        "away_team": away,
        "league": league,
        "summary": summary_text,
        "overall_confidence": round(overall_conf, 2),
        "predictions": results,
        "friendly_summary": friendly_summary,
        "status": "ok",
    }

# ============================
# 🚀 MAIN LOCAL
# ============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
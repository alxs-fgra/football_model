import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

# ==========================
# ⚙️ CONFIGURACIÓN
# ==========================
API_URL = "http://127.0.0.1:8000"
API_FOOTBALL_KEY_PATH = "config/api_football_key.txt"

st.set_page_config(
    page_title="⚽ Football Predictor Dashboard",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Football Predictor Dashboard")
st.markdown("### 🔮 Predicciones automáticas multi-mercado con FastAPI + API-Football")

# ==========================
# 🔐 Cargar API Key
# ==========================
if os.path.exists(API_FOOTBALL_KEY_PATH):
    with open(API_FOOTBALL_KEY_PATH, "r") as f:
        API_FOOTBALL_KEY = f.read().strip()
else:
    API_FOOTBALL_KEY = None
    st.warning("⚠️ No se encontró la API key en config/api_football_key.txt")

# ==========================
# 🏆 Selección de liga
# ==========================
st.sidebar.header("⚙️ Configuración de predicción")

# Mapeo de ligas a sus IDs en API-Football
LEAGUE_IDS = {
    "La Liga 🇪🇸": 140,
    "Premier League 🏴": 39,
    "Serie A 🇮🇹": 135
}

selected_league_name = st.sidebar.selectbox("Selecciona una liga:", list(LEAGUE_IDS.keys()))
selected_league_id = LEAGUE_IDS[selected_league_name]

# ==========================
# 🧠 Obtener equipos desde API-Football
# ==========================
teams = []
if API_FOOTBALL_KEY:
    try:
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
        url = f"https://v3.football.api-sports.io/teams?league={selected_league_id}&season=2024"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            teams = [t["team"]["name"] for t in data["response"]]
        else:
            st.sidebar.error(f"❌ Error al obtener equipos ({res.status_code})")
    except Exception as e:
        st.sidebar.error(f"⚠️ No se pudo conectar con API-Football: {e}")

if not teams:
    teams = ["Real Madrid", "Barcelona", "Sevilla", "Valencia", "Atlético de Madrid"]

# ==========================
# 🏟️ Selección de equipos
# ==========================
st.sidebar.markdown("### 🏟️ Equipos")
home_team = st.sidebar.selectbox("Equipo local 🏠", teams, index=0)
away_team = st.sidebar.selectbox("Equipo visitante 🚗", teams, index=1)

# ==========================
# 🎯 Botón de predicción
# ==========================
if st.sidebar.button("🚀 Predecir resultado"):
    with st.spinner("Calculando predicción... ⏳"):
        payload = {"league": selected_league_name.lower(), "home_team": home_team, "away_team": away_team}

        try:
            res = requests.post(f"{API_URL}/predict_match", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.success("✅ Predicción generada con éxito")

                # ==========================
                # 🧠 RESULTADOS PRINCIPALES
                # ==========================
                st.markdown(f"## 🏟️ {data['home_team']} vs {data['away_team']}")
                st.markdown(f"**🏆 Liga:** `{data['league'].title()}`")

                st.markdown("----")
                st.markdown(f"### 🧩 **Predicción general:**")
                st.markdown(f"🧠 {data['summary']}")
                st.markdown(f"🔮 Confianza global: **{data['overall_confidence']*100:.1f}%**")

                st.markdown("----")
                st.markdown("### 📊 Detalle por mercado")

                for market, info in data["predictions"].items():
                    conf = info["confidence"]
                    label = info["label"]
                    color = "🟢" if conf > 0.75 else "🟡" if conf > 0.6 else "🔴"
                    st.markdown(f"**{market.upper()}** → {label} ({conf*100:.1f}%) {color}")
                    st.progress(conf)

                st.markdown("----")
                st.markdown("### 🗒️ Resumen natural")
                st.info(data["friendly_summary"])

                # ==========================
                # 💾 Historial de predicciones
                # ==========================
                os.makedirs("logs", exist_ok=True)
                log_path = os.path.join("logs", "dashboard_predictions.csv")

                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "league": data["league"],
                    "home": data["home_team"],
                    "away": data["away_team"],
                    "summary": data["summary"],
                    "confidence": data["overall_confidence"],
                }

                df_log = pd.DataFrame([log_entry])
                df_log.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))

                if os.path.exists(log_path):
                    st.markdown("----")
                    st.markdown("### 🕒 Últimas predicciones")
                    df_hist = pd.read_csv(log_path).tail(5)
                    st.dataframe(df_hist)

            else:
                st.error(f"❌ Error en la API ({res.status_code}): {res.text}")

        except Exception as e:
            st.error(f"⚠️ No se pudo conectar con la API: {e}")
else:
    st.info("👈 Selecciona la liga y equipos, luego presiona **Predecir resultado**.")
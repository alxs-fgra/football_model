# 📊 Evaluación y Optimización del Modelo 1X2 — LaLiga 2024/25

**Autor:** Alexis Figueroa  
**Fecha:** 2025-10-23  
**Versión:** Modelo Optimizado (Paso 7.2 Final)  
**Archivo base:** `features_la_liga_2015_2023.csv`

---

## ⚙️ Resumen Técnico

| Parámetro | Valor |
|------------|--------|
| Datos de entrenamiento | 2015 – 2022 (3,420 partidos) |
| Validación | Temporada 2023 (380 partidos) |
| Modelo | Regresión Logística (multiclase balanceada) |
| Accuracy (2023) | **52.11 %** |
| ROC-AUC (weighted) | **0.719** |
| Log-Loss | 0.968 |
| Cross-Validation mean | 0.482 ± 0.128 |
| SMOTE Draw target | 791 |
| GridSearch Best Params | `C = 0.005`, `solver = lbfgs` |

---

## 🎯 Métricas por Clase

| Resultado | Precision | Recall | F1-Score | Support |
|------------|------------|---------|-----------|----------|
| **Away (-1)** | 0.48 | 0.55 | 0.51 | 106 |
| **Draw (0)** | 0.35 | 0.39 | 0.37 | 107 |
| **Home (1)** | 0.70 | 0.59 | 0.64 | 167 |
| **Global Accuracy** |  |  | **52 %** | 380 |

✅ El recall del **Draw (0)** aumentó frente a versiones previas (de 0.27 – 0.38 → **0.39**).

---

## 🔍 Observaciones

- La inclusión de `rolling_avg_goals` mejoró la capacidad del modelo para captar tendencias de ataque recientes.  
- El balanceo con SMOTE (≈ 791 draws) redujo el sesgo hacia victorias o derrotas.  
- ROC-AUC 0.719 confirma buena discriminación global.  
- El modelo mantiene estabilidad y generaliza bien entre temporadas.

---

## 🧩 Archivos Generados

| Tipo | Ruta |
|------|------|
| Modelo | `models/liga_model_20251022_234230.pkl` |
| Preprocesadores | `models/liga_scaler_imputer_20251022_234230.pkl` |
| Matriz de confusión | `models/confusion_matrix_liga_20251022_234230.png` |
| Importancia de features | `models/feature_importance_liga_20251022_234230.csv` |
| Métricas | `models/metrics_liga_20251022_234230.csv` |

---

## 🧠 Recomendaciones

1. **Mantener `rolling_avg_goals`** en futuras versiones.  
2. **Probar modelos no lineales** (RandomForest, XGBoost) para capturar interacciones complejas.  
3. **Evaluar regularización C = 0.003–0.01** en futuros grid search.  
4. **Extender a nuevos mercados:** BTTS y Over/Under (2.5).  
5. **Automatizar evaluación** con pipeline de re-entrenamiento semanal (GCP scheduler + GitHub Actions).

---

## 📈 Próximos Pasos (Paso 8 – Despliegue)

- Implementar API REST (FastAPI/Flask) para servir predicciones 1X2 en tiempo real.  
- Preparar endpoints para múltiples ligas (La Liga, Premier League, Bundesliga).  
- Integrar monitorización (GCP + Grafana Dashboard).

---


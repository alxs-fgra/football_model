#!/bin/bash

# ==============================
# 🧹 FOOTBALL MODEL CLEANER
# ==============================

# Colores
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # Sin color

echo -e "${YELLOW}🧹 Iniciando limpieza del entorno de pruebas...${NC}"

# Carpetas clave
LOGS_DIR="logs"
MODELS_DIR="models"
DATA_DIR="data"
CACHE_DIRS="__pycache__ */__pycache__ .pytest_cache"

# Archivos a eliminar
echo -e "${YELLOW}📂 Limpiando cachés y temporales...${NC}"
rm -rf $CACHE_DIRS *.tmp *.log > /dev/null 2>&1

# Logs
if [ -d "$LOGS_DIR" ]; then
  rm -rf "$LOGS_DIR"/*
  echo -e "${GREEN}✅ Logs limpiados.${NC}"
fi

# Modelos de prueba (mantiene los modelos buenos)
if [ -d "$MODELS_DIR" ]; then
  find "$MODELS_DIR" -type f \
    ! -name "1x2_model_*.pkl" \
    ! -name "btts_model_*.pkl" \
    ! -name "over_2.5_model_*.pkl" \
    ! -name "scaler_*.pkl" \
    ! -name "imputer_*.pkl" \
    -delete
  echo -e "${GREEN}✅ Modelos temporales eliminados (se conservaron los principales).${NC}"
fi

# Datasets de prueba o temporales
if [ -d "$DATA_DIR" ]; then
  find "$DATA_DIR/processed" -type f \( -name "*test*.csv" -o -name "*temp*.csv" \) -delete 2>/dev/null
  find "$DATA_DIR/raw" -type f -name "*tmp*" -delete 2>/dev/null
  echo -e "${GREEN}✅ Datasets de prueba eliminados.${NC}"
fi

echo -e "${GREEN}🎯 Limpieza completada.${NC}"
echo -e "${YELLOW}Tu entorno está listo para la próxima ejecución.${NC}"

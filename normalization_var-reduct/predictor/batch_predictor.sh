#!/bin/bash
#SBATCH --job-name=batch_pred_v6
#SBATCH --output=log_salida.txt
#SBATCH --error=log_error.txt
#SBATCH --partition=organica
#SBATCH --time=72:00:00   # 3 días
#SBATCH --mem=8G
#SBATCH --ntasks=1

# ==== CONFIGURACIÓN ====
subfolder="full-150"  # Cambia esto según el subdirectorio que necesites
training_set="training_set" # Cambia esto según el training set que uses
validation_set="validation_set" # Cambia esto según el validation set que uses

TRAINING_FILE="/home/bcelada.iquir/scripts/Python/$subfolder/$training_set.xlsx"
VALIDATION_FILE="/home/bcelada.iquir/scripts/Python/$subfolder/$validation_set.xlsx"
SCALERS="zscore,minmax,decimal,robust,unit,none"
REDUCTIONS="pca,efa,plsr,lasso,rfe,sparsepca,mihr,none"
MODELS="mlr,svm,dt,rf,xgb,knn,ann,gam,gpr,lgbm,catb"
TRIALS=100
SCRIPT="/home/bcelada.iquir/scripts/Python/$subfolder/batch_predictor.py"
LOG_FOLDER="logs_batch_predictor"

# Crear carpeta de logs y error si no existe
mkdir -p "$LOG_FOLDER"
mkdir -p "$LOG_FOLDER/error"

# === INICIANDO SCRIPT BATCH ===
echo "----------------------------------------"
echo "=== INICIANDO SCRIPT BATCH ==="
echo "Log folder: $LOG_FOLDER"
echo "Training set: $TRAINING_FILE"
echo "Validation set: $VALIDATION_FILE"
echo "Scalers: $SCALERS"
echo "Reductions: $REDUCTIONS"
echo "Models: $MODELS"
echo "Trials: $TRIALS"
echo "Fecha: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "HOME: $HOME"
echo "----------------------------------------"

echo "⚙️ Activando entorno CONDA..."
ls -l $HOME/miniconda3/etc/profile.d/conda.sh || echo "⚠️ No se encontró el conda.sh"
echo "Conda path: $HOME/miniconda3/etc/profile.d/conda.sh"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate calculo_py || echo "⚠️ Falló la activación de Conda"

# Ejecutar script
echo "----------------------------------------"
echo "🚀 Ejecutando script batch_predictor.py"
python "$SCRIPT" \
  --training "$TRAINING_FILE" \
  --validation "$VALIDATION_FILE" \
  --scalers "$SCALERS" \
  --reductions "$REDUCTIONS" \
  --models "$MODELS" \
  --trials "$TRIALS" \
  --logs-dir "$LOG_FOLDER" \
  --error-dir "$LOG_FOLDER/error" \
  --no_intro

echo "✅ Proceso finalizado"

#!/bin/bash
#SBATCH --job-name=procV4
#SBATCH --output=log_salida.txt
#SBATCH --error=log_error.txt
#SBATCH --partition=eth_hi
#SBATCH --time=48:00:00   # 2 días
#SBATCH --mem=8G
#SBATCH --ntasks=4
#SBATCH --nodes=1

# ========================
# Variables configurables
# ========================

# Nombre del subdirectorio donde están los modelos generados
SUBFOLDER_NAME="subfolder"

# Ruta del script de procesamiento
SCRIPT="/home/bcelada.iquir/scripts/Python/predictor_procesor_V4.py"

# Ruta base donde están los modelos generados
BASE_FOLDER="/home/bcelada.iquir/scripts/Python/$SUBFOLDER_NAME"

# Carpeta donde guardar el Master_Table.xlsx
OUTPUT_FOLDER="/home/bcelada.iquir/scripts/Python/$SUBFOLDER_NAME"

# Ruta del archivo Template base
TEMPLATE_PATH="/home/bcelada.iquir/scripts/Python/new_template.xlsx"

# Carpeta con los logs de salida de cada modelo
LOG_FOLDER="/home/bcelada.iquir/scripts/Python/$SUBFOLDER_NAME/logs"

# Líneas de criterio que debe tener el log para considerar terminado correctamente
LINEAS_CRITERIO=("No hubo salida de error.")

# ========================
# Otros parámetros fijos
# ========================

SHEET_NAME="Hoja1"          # Nombre de la hoja del template
FILE_PREFIX="results"        # Prefijo de archivos Excel de resultados
TARGET_COLUMN="C"            # Columna donde insertar en template
SOURCE_COLUMN="C"            # Columna a extraer de Excel fuente
FILA_DATOS=54                 # Fila base de datos
GUARDAR_TEMPLATES="--guardar_templates"   # El flag --guardar_templates se pasa solo si aplica


# ========================
# Entorno
# ========================

echo "⚙️ Activando entorno Conda..."
ls -l $HOME/miniconda3/etc/profile.d/conda.sh || echo "⚠️ No se encontró el conda.sh"
echo "Conda path: $HOME/miniconda3/etc/profile.d/conda.sh"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate calculo_py || echo "⚠️ Falló la activación de Conda"

# ========================
# Comando Python
# ========================

echo "🚀 Ejecutando procesamiento de outliers..."

# === Guardar variables utilizadas ===
echo "----------------------------------------"
echo "=== INICIANDO SCRIPT BATCH_PROCESOR ==="
echo "Ruta del script de procesamiento: $SCRIPT"
echo "Ruta base donde están los modelos generados: $BASE_FOLDER"
echo "Carpeta donde guardar el Master_Table.xlsx: $OUTPUT_FOLDER"
echo "Ruta del archivo Template base: $TEMPLATE_PATH"
echo "Carpeta con los logs de salida de cada modelo: $LOG_FOLDER"
echo "Líneas de criterio que debe tener el log para considerar terminado correctamente: $LINEAS_CRITERIO"
echo "Nombre de la hoja del template: $SHEET_NAME"
echo "Prefijo de archivos Excel de resultados: $FILE_PREFIX"
echo "Columna donde insertar en template: $TARGET_COLUMN"
echo "Columna a extraer de Excel fuente: $SOURCE_COLUMN"
echo "Fila base de datos: $FILA_DATOS"
echo "El flag --guardar_templates se pasa solo si aplica: $GUARDAR_TEMPLATES"
echo "----------------------------------------"

# Ejecutar script
echo "----------------------------------------"
echo "🚀 Ejecutando script batch_predictor.py"
python $SCRIPT \
  --base_folder "$BASE_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --sheet_name "$SHEET_NAME" \
  --file_prefix "$FILE_PREFIX" \
  --template_path "$TEMPLATE_PATH" \
  $GUARDAR_TEMPLATES \
  --target_column "$TARGET_COLUMN" \
  --source_column "$SOURCE_COLUMN" \
  --log_folder "$LOG_FOLDER" \
  --lineas_criterio "${LINEAS_CRITERIO[@]}" \
  --fila_datos $FILA_DATOS

echo "✅ Proceso finalizado"

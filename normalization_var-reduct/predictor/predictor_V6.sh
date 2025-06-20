#!/bin/bash
#SBATCH --job-name=pred_v6
#SBATCH --output=log_salida.txt
#SBATCH --error=log_error.txt
#SBATCH --partition=organica
#SBATCH --time=3-00:00:00   # 3 días
#SBATCH --mem=8G
#SBATCH --ntasks=1

echo "⚙️  Activando entorno CONDA..."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate calculo_py

echo "🚀 Ejecutando script predictor_V6.py"
python /home/bcelada.iquir/test/predictor_V6.py \
  --training /home/bcelada.iquir/test/training_set_sigman_cinchona.xlsx \
  --validation /home/bcelada.iquir/test/validation_set_sigman_cinchona.xlsx \
  --scaler zscore \
  --reduction none \
  --model mlr \
  --trials 500 \
  --no_intro

echo "✅ Proceso finalizado"

#!/bin/bash

#SBATCH --job-name=predict_cmi_demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=../logs/cmi/predict_cmi_demo.out
#SBATCH --error=../logs/cmi/predict_cmi_demo.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

echo "Running external CMI prediction"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

python -m cmi.predict \
  --input_csv "${INPUT_CSV:-demo/input_demo/demo_cmi_external.csv}" \
  --model_dir "${MODEL_DIR:-model/cmi}" \
  --output_dir "${OUTPUT_DIR:-results/cmi/predict_cmi_demo}" \
  --label_col "${LABEL_COL:-relation}" \
  --external_compound_features_file "${EXTERNAL_COMPOUND_FEATURES_FILE:-demo/input_demo/demo_external_compounds_feat.csv}"

echo "End time: $(date)"
echo "Prediction finished."

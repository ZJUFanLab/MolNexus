#!/bin/bash

#SBATCH --job-name=predict_cgi_demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=../logs/cgi/predict_cgi_demo.out
#SBATCH --error=../logs/cgi/predict_cgi_demo.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running external CGI prediction"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

python -m cgi.predict \
  --input_csv "${INPUT_CSV:-demo/input_demo/demo_cgi_external.csv}" \
  --model_dir "${MODEL_DIR:-model/cgi}" \
  --output_dir "${OUTPUT_DIR:-results/cgi/predict_cgi_demo}" \
  --input_format "${INPUT_FORMAT:-triplet}" \
  --compound_id_col "${COMPOUND_ID_COL:-head}" \
  --gene_id_col "${GENE_ID_COL:-tail}" \
  --cpd_features_file "${COMPOUND_FEATURES:-demo/input_demo/demo_external_compounds_feat.csv}"

echo "End time: $(date)"
echo "Prediction finished."

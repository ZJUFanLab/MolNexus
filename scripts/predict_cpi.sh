#!/bin/bash

#SBATCH --job-name=predict_cpi_demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=../logs/cpi/predict_cpi_demo.out
#SBATCH --error=../logs/cpi/predict_cpi_demo.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Running CPI target prediction"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

# SPLIT choices: CCS, DCS, PCS, Warm. It selects model/cpi/<split> and matching hyperparameters.
python -m cpi.predict \
  --task_mode "${TASK_MODE:-compound_screening}" \
  --split "${SPLIT:-DCS}" \
  --compounds "${INPUT_CSV:-demo/input_demo/demo_screening_compounds.csv}" \
  --feat "${COMPOUND_FEATURES:-demo/input_demo/demo_screening_compounds_feat.csv}" \
  --out_dir "${OUTPUT_DIR:-results/cpi/predict_cpi_demo}" \
  --candidate_file "${CANDIDATE_FILE:-demo/input_demo/demo_screening_target.csv}"

echo "End time: $(date)"
echo "Prediction finished."

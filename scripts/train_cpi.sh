#!/bin/bash

#SBATCH --job-name=train_cpi
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_cpi_fold%a.out
#SBATCH --error=../logs/train_cpi_fold%a.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export TRAIN_MODE=PCS
export CGI_MODE=CGI_GLOBAL
export ABLATION_MODE=FULL
export GATE_INIT=0.0

echo "Running fold $SLURM_ARRAY_TASK_ID"
python -m cpi.train
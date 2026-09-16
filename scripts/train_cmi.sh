#!/bin/bash

#SBATCH --job-name=train_cmi
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_cmi_fold%a.out
#SBATCH --error=../logs/train_cmi_fold%a.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export FINETUNE_MODE=warmup_unfreeze
export FINETUNE_LR=1e-3
export FREEZE_EPOCHS=200
export UNFREEZE_LR=1e-4
export RUN_TAG=train_cmi

export RESUME=0
# Use this only to resume an interrupted run.
#export RESUME=1

export ABLATION_MODE=FULL
export GATE_INIT=0.0
export FOLD=${SLURM_ARRAY_TASK_ID}
# Output directory
export RESULTS_DIR=HGT_results/${RUN_TAG}
export CKPT_DIR=${RESULTS_DIR}/checkpoints

mkdir -p ${RESULTS_DIR}
mkdir -p ${CKPT_DIR}
# Base path for the CPI pretraining checkpoint (from CGI_GLOBAL CPI training).
export PRETRAIN_BASE=model/cpi/ccs
# Pretraining checkpoint for the current fold
case ${FOLD} in
  1)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_1.pt
    ;;
  2)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_2.pt
    ;;
  3)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_3.pt
    ;;
  4)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_4.pt
    ;;
  5)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_5.pt
    ;;
esac

echo "Running fold $SLURM_ARRAY_TASK_ID"
python -m cmi.train

#!/bin/bash

#SBATCH --job-name=train_cgi
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_cgi_fold%a.out
#SBATCH --error=../logs/train_cgi_fold%a.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export RESUME=0
# Use this only to resume an interrupted run.
#export RESUME=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export FINETUNE_MODE=warmup_unfreeze
export FINETUNE_LR=1e-3
export FREEZE_EPOCHS=30
export UNFREEZE_LR=1e-4
export RUN_TAG=train_cgi
export MLP_HIDDEN=512

export ABLATION_MODE=FULL
export GATE_INIT=0.0
export CGI_SCORE_BATCH_SIZE=5000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export FOLD=${SLURM_ARRAY_TASK_ID}

export RESULTS_DIR=../results/cgi/${RUN_TAG}
export CKPT_DIR=${RESULTS_DIR}/checkpoints

mkdir -p ${RESULTS_DIR}
mkdir -p ${CKPT_DIR}
#mkdir -p logs

# Base path for the CPI pretraining checkpoint (from CGI_SPLIT CPI training).
export PRETRAIN_BASE=model/cpi/ccs

case ${FOLD} in
  1)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_1*.pt
    ;;
  2)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_2*.pt
    ;;
  3)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_3*.pt
    ;;
  4)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_4*.pt
    ;;
  5)
    export PRETRAIN_CKPT=${PRETRAIN_BASE}/ensemble_5*.pt
    ;;
esac

echo "Running CGI finetune fold ${FOLD}"
echo "RUN_TAG=${RUN_TAG}"
echo "FINETUNE_MODE=${FINETUNE_MODE}"
echo "FINETUNE_LR=${FINETUNE_LR}"
echo "PRETRAIN_CKPT=${PRETRAIN_CKPT}"

python -m cgi.train
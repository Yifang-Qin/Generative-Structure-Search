#!/bin/bash

expname="csp_model_new"
PROJECT_ROOT=$(pwd)

dataset_name="alex_mp_20"

export OUTPUT_DIR="$PROJECT_ROOT/outputs/$expname"
export CACHE_DIR="$PROJECT_ROOT/datasets/cache/$dataset_name"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HYDRA_FULL_ERROR=1


ENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
source $ENV_PATH
python mattergen/scripts/run.py  --config-name csp \
                                 trainer.accumulate_grad_batches=4
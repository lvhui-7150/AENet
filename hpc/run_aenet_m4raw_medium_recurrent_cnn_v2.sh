#!/bin/bash
# M4Raw AENet v2: wider recurrent CNN, 12 equilibrium steps, EMA, NMSE loss.
# Submit with: SEED=42 jsub < run_aenet_m4raw_medium_recurrent_cnn_v2.sh
#JSUB -q ai_share
#JSUB -gpgpu 1
#JSUB -n 4
#JSUB -J m4raw_aenet_v2
#JSUB -o /path/to/aenet/AENet/m4raw_aenet_v2_%J.out
#JSUB -e /path/to/aenet/AENet/m4raw_aenet_v2_%J.err

export PATH="/path/to/aenet/miniconda3/envs/aenet/bin:$PATH"
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/path/to/aenet/miniconda3/envs/aenet/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

M4RAW_ROOT="${M4RAW_ROOT:-/path/to/aenet/data/m4raw}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-48}"
DEPTH="${DEPTH:-8}"
STEPS="${STEPS:-12}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
TAG="medium_recurrent_cnn_v2_s${SEED}"

cd /path/to/aenet/AENet

/path/to/aenet/miniconda3/envs/aenet/bin/python -u experiments/train_aenet_m4raw.py \
  --data "${M4RAW_ROOT}/prepared/m4raw_medium.pt" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --steps "${STEPS}" \
  --hidden "${HIDDEN}" \
  --depth "${DEPTH}" \
  --arch recurrent_cnn \
  --anchor-init 0.05 \
  --augment \
  --lr "${LR}" \
  --nmse-weight 1.0 \
  --ema-decay 0.99 \
  --ema-start-epoch 40 \
  --seed "${SEED}" \
  --only aenet \
  --tag "${TAG}" \
  --output "experiments/m4raw_results_hpc_${TAG}.json"

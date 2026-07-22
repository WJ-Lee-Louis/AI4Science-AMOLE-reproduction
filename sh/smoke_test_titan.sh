#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
MICROMAMBA="${MICROMAMBA:-/home/dsail26s2/micromamba/bin/micromamba}"
GPU_ID="${GPU_ID:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-8}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${MICROMAMBA}" run -p "${ENV_PREFIX}" python pretrain.py \
  --lm SciBERT \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --dynamic_padding \
  --gradient_checkpointing \
  --amp \
  --epochs 1 \
  --max_steps_per_epoch "${MAX_STEPS}" \
  --no_save \
  --text_lr 1e-5 \
  --mol_lr 1e-5 \
  --model AMOLE \
  --dataset TanimotoSTM \
  --data_path ./data/PubChemSTM \
  --target_T 0.1 \
  --T 0.1 \
  --p_aug 0.5 \
  --num_cand 50 \
  --augmentation_strategy baseline \
  --batch_size "${BATCH_SIZE}" \
  --aux_batch_size "${AUX_BATCH_SIZE}" \
  --num_workers 0 \
  --alpha 1.0 \
  --seed 0 \
  --device 0

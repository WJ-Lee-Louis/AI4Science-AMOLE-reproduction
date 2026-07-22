#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
MICROMAMBA="${MICROMAMBA:-/home/dsail26s2/micromamba/bin/micromamba}"
GPU_ID="${GPU_ID:-4}"
BATCH_SIZE="${BATCH_SIZE:-15}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-baseline_b${BATCH_SIZE}_e${EPOCHS}_seq${MAX_SEQ_LEN}_seed${SEED}_gpu${GPU_ID}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./model_checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-./logs}"

cd "${REPO_ROOT}"
mkdir -p "${CHECKPOINT_PATH}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${MICROMAMBA}" run -p "${ENV_PREFIX}" python pretrain.py \
  --lm SciBERT \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --dynamic_padding \
  --gradient_checkpointing \
  --amp \
  --epochs "${EPOCHS}" \
  --text_lr 1e-5 \
  --mol_lr 1e-5 \
  --model AMOLE \
  --dataset TanimotoSTM \
  --data_path ./data/PubChemSTM \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --target_T 0.1 \
  --T 0.1 \
  --p_aug 0.5 \
  --num_cand 50 \
  --augmentation_strategy baseline \
  --batch_size "${BATCH_SIZE}" \
  --aux_batch_size "${AUX_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --alpha 1.0 \
  --seed "${SEED}" \
  --device 0 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

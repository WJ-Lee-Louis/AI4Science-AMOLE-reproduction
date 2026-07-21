#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
TORCHRUN="${TORCHRUN:-${ENV_PREFIX}/bin/torchrun}"
GPU_IDS="${GPU_IDS:-5,6,7}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-15}"
MAX_STEPS="${MAX_STEPS:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-8}"
MASTER_PORT="${MASTER_PORT:-29558}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
SMOKE_CHECKPOINT_PATH="${SMOKE_CHECKPOINT_PATH:-./model_checkpoints/ddp_smoke_test}"

SAVE_ARGS=(--no_save)
if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
  SAVE_ARGS=(--checkpoint_path "${SMOKE_CHECKPOINT_PATH}")
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=False
export PYTHONUNBUFFERED=1

"${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  --master_port="${MASTER_PORT}" \
  pretrain.py \
  --lm SciBERT \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --dynamic_padding \
  --gradient_checkpointing \
  --amp \
  --epochs 1 \
  --max_steps_per_epoch "${MAX_STEPS}" \
  "${SAVE_ARGS[@]}" \
  --text_lr 1e-5 \
  --mol_lr 1e-5 \
  --model AMOLE \
  --dataset TanimotoSTM \
  --data_path ./data/PubChemSTM \
  --target_T 0.1 \
  --T 0.1 \
  --p_aug 0.5 \
  --num_cand 50 \
  --batch_size "${LOCAL_BATCH_SIZE}" \
  --aux_batch_size "${AUX_BATCH_SIZE}" \
  --num_workers 0 \
  --alpha 1.0 \
  --seed 0 \
  --device 0

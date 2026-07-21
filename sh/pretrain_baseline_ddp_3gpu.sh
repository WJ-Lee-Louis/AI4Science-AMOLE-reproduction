#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
TORCHRUN="${TORCHRUN:-${ENV_PREFIX}/bin/torchrun}"
GPU_IDS="${GPU_IDS:-5,6,7}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-15}"
EPOCHS="${EPOCHS:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29557}"
RUN_NAME="${RUN_NAME:-baseline_ddp3_global45_e${EPOCHS}_seq${MAX_SEQ_LEN}_seed${SEED}_gpu567}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./model_checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-./logs}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 3 ]]; then
  echo "GPU_IDS must contain exactly three comma-separated GPU IDs." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${CHECKPOINT_PATH}" "${LOG_DIR}"
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
  --batch_size "${LOCAL_BATCH_SIZE}" \
  --aux_batch_size "${AUX_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --alpha 1.0 \
  --seed "${SEED}" \
  --device 0 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

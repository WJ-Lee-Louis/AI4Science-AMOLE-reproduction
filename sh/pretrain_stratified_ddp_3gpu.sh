#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
TORCHRUN="${TORCHRUN:-${ENV_PREFIX}/bin/torchrun}"
GPU_IDS="${GPU_IDS:-4,5,6}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-10}"
GLOBAL_BATCH_SIZE=$((LOCAL_BATCH_SIZE * 3))
EPOCHS="${EPOCHS:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"
ALPHA="${ALPHA:-1.0}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.25}"
HIGH_PROBABILITY="${HIGH_PROBABILITY:-0.50}"
MID_PROBABILITY="${MID_PROBABILITY:-0.35}"
LOW_PROBABILITY="${LOW_PROBABILITY:-0.15}"
MASTER_PORT="${MASTER_PORT:-29577}"
GPU_TAG="${GPU_IDS//,/}"
ALPHA_TAG="${ALPHA//./p}"
MIN_SIMILARITY_TAG="${MIN_SIMILARITY//./p}"
RUN_NAME="${RUN_NAME:-stratified_h50_m35_l15_min${MIN_SIMILARITY_TAG}_global${GLOBAL_BATCH_SIZE}_e${EPOCHS}_alpha${ALPHA_TAG}_seed${SEED}_gpu${GPU_TAG}}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

"${TORCHRUN}" \
  --nnodes=1 \
  --nproc_per_node=3 \
  --master_addr=127.0.0.1 \
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
  --augmentation_strategy stratified \
  --stratified_min_similarity "${MIN_SIMILARITY}" \
  --stratified_high_probability "${HIGH_PROBABILITY}" \
  --stratified_mid_probability "${MID_PROBABILITY}" \
  --stratified_low_probability "${LOW_PROBABILITY}" \
  --batch_size "${LOCAL_BATCH_SIZE}" \
  --aux_batch_size "${AUX_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --device 0 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

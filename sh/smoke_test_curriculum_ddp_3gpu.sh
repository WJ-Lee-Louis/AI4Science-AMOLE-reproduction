#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/home/dsail26s2/envs/amole-train-titan}"
TORCHRUN="${TORCHRUN:-${ENV_PREFIX}/bin/torchrun}"
GPU_IDS="${GPU_IDS:-1,2,3}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-10}"
MAX_STEPS="${MAX_STEPS:-2}"
MASTER_PORT="${MASTER_PORT:-29568}"

cd "${REPO_ROOT}"
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
  --max_seq_len 512 \
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
  --augmentation_strategy curriculum \
  --curriculum_start_k 10 \
  --curriculum_warmup_epochs 5 \
  --curriculum_rank_increment 4 \
  --batch_size "${LOCAL_BATCH_SIZE}" \
  --aux_batch_size 8 \
  --num_workers 0 \
  --alpha 1.0 \
  --seed 0 \
  --device 0

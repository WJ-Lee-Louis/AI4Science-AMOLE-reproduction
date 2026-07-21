# AMOLE 50k baseline training

## Validated Titan Xp profile

- Dataset: `data/PubChemSTM` (symlink to the audited 50k package)
- GPU: one NVIDIA TITAN Xp selected with `GPU_ID`
- Training batch size: 15
- ER auxiliary microbatch size: 8
- Epochs: 20
- SciBERT maximum sequence length: 512
- Dynamic padding: enabled
- CUDA AMP: enabled
- SciBERT gradient checkpointing: enabled
- AMOLE baseline parameters: `p_aug=0.5`, `num_cand=50`, `T=0.1`,
  `target_T=0.1`, `alpha=1.0`, and learning rates `1e-5`

The ER loss is an independent mean over examples. It is backpropagated in
microbatches and weighted by microbatch size, while the complete batch of 15 is
retained for the S2P `15 x 15` target and prediction matrices.

## Run

From the repository root:

```bash
GPU_ID=4 ./sh/pretrain_baseline_titan.sh
```

The script writes logs under `logs/` and the best epoch weights under
`model_checkpoints/<run-name>/`.

Any default can be overridden without editing the script:

```bash
GPU_ID=5 SEED=1 RUN_NAME=baseline_seed1 ./sh/pretrain_baseline_titan.sh
```

Do not launch two runs with the same `RUN_NAME`, because their checkpoint and
log paths would overlap.

## Re-run the short validation

```bash
GPU_ID=4 BATCH_SIZE=15 MAX_SEQ_LEN=512 AUX_BATCH_SIZE=8 MAX_STEPS=100 \
  ./sh/smoke_test_titan.sh
```

This profile completed 100 optimizer steps on GPU 4 in about 61 seconds. With
67,357 molecule-text rows, one epoch contains 4,491 steps. A 20-epoch run is
therefore expected to take roughly 15--18 hours, with I/O and checkpoint saving
included.

## Three-GPU DDP baseline (GPUs 5, 6, and 7)

The DDP profile uses a local batch of 15 on each GPU and gathers representations
and fingerprints before calculating S2P. The effective global batch is therefore
45, matching the original AMOLE batch size. A distributed sampler gives each
rank a disjoint shard, and only rank 0 writes checkpoints and logs.

Run a short validation:

```bash
GPU_IDS=5,6,7 MAX_STEPS=2 ./sh/smoke_test_ddp_3gpu.sh
```

Start the 20-epoch baseline:

```bash
GPU_IDS=5,6,7 ./sh/pretrain_baseline_ddp_3gpu.sh
```

Do not start another job on GPUs 5, 6, or 7 while this command is running. The
default rendezvous ports are 29557 for training and 29558 for the smoke test;
override `MASTER_PORT` if either port is already occupied.

The validated 100-step run took 80.25 seconds on GPUs 5, 6, and 7. There are
1,496 distributed steps per epoch, giving an estimated 20 minutes per epoch and
approximately 6 hours 40 minutes for 20 epochs. Allow roughly 7--7.5 hours for
checkpoint I/O and normal runtime variation.

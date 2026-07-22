# Rank-expansion curriculum training

## Sampling policy

The curriculum changes only the prefix of the precomputed, self-excluded
Tanimoto neighbor ranking from which an augmented molecule is sampled. Sampling
within the active prefix remains uniform and `p_aug` remains 0.5.

| Epoch | Active neighbor pool |
|---:|---:|
| 1--5 | top-10 |
| 6 | top-14 |
| 7 | top-18 |
| 8 | top-22 |
| 9 | top-26 |
| 10 | top-30 |
| 11 | top-34 |
| 12 | top-38 |
| 13 | top-42 |
| 14 | top-46 |
| 15 | top-50 (expansion reaches its maximum) |
| 16--20 | top-50 (fixed) |

The implementation is selected with `--augmentation_strategy curriculum`.
The default `baseline` strategy always samples uniformly from top-50, so the
existing baseline behavior is unchanged.

## Resource isolation

- Curriculum GPUs: 1, 2, and 3
- Curriculum tmux session: `amole-curriculum`
- Curriculum DDP port: 29567
- Baseline GPUs: 5, 6, and 7
- Baseline tmux session: `amole-baseline`
- Baseline DDP port: 29557

The two runs share read-only dataset and initial checkpoint files but write to
different log and checkpoint directories.

The launch scripts use the static single-node rendezvous backend with an
explicit `127.0.0.1` port. Do not add `--standalone`: PyTorch 1.10.1 overrides
the requested port with `localhost:29400` in standalone mode, which makes two
concurrent jobs share one rendezvous store.

## Commands

Run the smoke test before starting a full run:

```bash
GPU_IDS=1,2,3 ./sh/smoke_test_curriculum_ddp_3gpu.sh
```

Start the full run inside its own tmux session:

```bash
tmux new -s amole-curriculum -c /home/dsail26s2/AMOLE
GPU_IDS=1,2,3 ./sh/pretrain_curriculum_ddp_3gpu.sh
```

Outputs:

- Log: `logs/curriculum_rank_expand_top10_to50_global30_e20_alpha1p0_seed0_gpu123.log`
- Checkpoints: `model_checkpoints/curriculum_rank_expand_top10_to50_global30_e20_alpha1p0_seed0_gpu123/`

For the alpha-2 diagnostic, use `ALPHA=2.0`. The alpha value is included in the
automatic run name, so it cannot overwrite the alpha-1 checkpoint:

```bash
GPU_IDS=1,2,3 ALPHA=2.0 ./sh/pretrain_curriculum_ddp_3gpu.sh
```

The full-run scripts default to local batch 10 (global batch 30). Local batch
15 completed several epochs but eventually hit a rare peak-memory batch on a
TITAN Xp. A local batch of 10 leaves substantial VRAM headroom while retaining
the 512-token cap. Baseline and curriculum use the same batch size for a fair
comparison.

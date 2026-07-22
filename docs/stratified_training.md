# Similarity-aware stratified AMOLE training

## Policy

The `stratified` augmentation strategy keeps the same top-50 neighbor artifact
as the baseline and partitions each molecule's ranked candidates into three
groups split at the 20th and 80th rank percentiles:

| Group | Neighbor ranks | Group-selection probability |
| --- | ---: | ---: |
| High | 1--10 (top 20%) | 0.50 |
| Mid | 11--40 (middle 60%) | 0.35 |
| Low | 41--50 (bottom 20%) | 0.15 |

After selecting a group, the loader samples uniformly from candidates whose
stored Tanimoto similarity is at least 0.25. If the selected group has no
eligible candidate, augmentation is cancelled for that pair and the original
molecule is retained. Probability mass is not reassigned to another group.
The independent AMOLE replacement probability remains `p_aug=0.5`.

On the current 50K dataset, the similarity gate cancels approximately 5% of
attempted replacements, giving an expected effective replacement rate close to
0.475. This avoids materially changing augmentation frequency relative to the
baseline while rejecting the weakest candidates.

## Validation

Run a two-step, three-GPU smoke test before a full experiment:

```bash
GPU_IDS=4,5,6 MASTER_PORT=29578 bash sh/smoke_test_stratified_ddp_3gpu.sh
```

The startup log must report ranks `1-10`, `11-40`, and `41-50`, probabilities
`0.50/0.35/0.15`, and minimum similarity `0.25`.

## Full training

The defaults reproduce the existing global-batch-30, 20-epoch setting with
`alpha=1.0`:

```bash
bash sh/pretrain_stratified_ddp_3gpu.sh
```

All important settings can be overridden without editing the script:

```bash
GPU_IDS=4,5,6 ALPHA=2.0 MASTER_PORT=29577 \
  bash sh/pretrain_stratified_ddp_3gpu.sh
```

The strategy, alpha value, similarity threshold, global batch size, seed, and
physical GPU IDs are encoded in the default run name. Checkpoints and logs are
therefore isolated from baseline and curriculum runs.

For an alpha-2 curriculum diagnostic running concurrently on GPUs 1--3:

```bash
GPU_IDS=1,2,3 ALPHA=2.0 MASTER_PORT=29567 \
  bash sh/pretrain_curriculum_ddp_3gpu.sh
```

Do not assign overlapping physical GPUs or the same rendezvous port to
concurrent jobs.

## Loss logging

New runs report the following epoch-level values:

- `CL Loss`: summed S2P loss, retained for compatibility with earlier logs;
- `S2P Mean`: mean S2P loss per optimizer step;
- `ER Mean`: unweighted mean expertise reconstruction loss;
- `Weighted ER Mean`: `alpha * ER Mean`;
- `Total Mean`: `S2P Mean + alpha * ER Mean`.

Checkpoint selection remains based on summed S2P loss, matching the behavior of
the completed baseline and curriculum runs.

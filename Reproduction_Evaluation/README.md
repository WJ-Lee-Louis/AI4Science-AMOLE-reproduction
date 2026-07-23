# AMOLE reproduction: zero-shot cross-modal retrieval

This directory compares six completed global-batch-30 checkpoints: baseline,
rank-expansion curriculum, and similarity-aware stratified augmentation, each
trained with alpha 1.0 and alpha 2.0.

## Evaluation protocol

- Official DrugBank Description, Pharmacodynamics, and ATC-5 files and their
  official negative-index files are used.
- Both Given Molecule (retrieve text) and Given Text (retrieve molecule) are
  evaluated.
- Candidate counts are **4, 10, and 20**, matching the official AMOLE
  repository and Appendix evaluation protocol.
- Results are mean and population standard deviation over seeds 0--4.
- The same sampled negative indices are used for every model, enabling paired
  comparisons.
- Encoders run in evaluation mode. Each deterministic molecule/text embedding
  is cached once; only negative sampling and ranking are repeated. This is
  equivalent to the official metric while avoiding redundant model forwards.
- The maximum text length is 512 tokens.

Dataset sizes:

| Dataset | Pairs |
| --- | ---: |
| Description | 1,154 |
| Pharmacodynamics | 1,005 |
| ATC | 3,007 |

## Outputs

- `table1_at4.md`, `table1_at10.md`, `table1_at20.md`: compact human-readable
  comparisons at each candidate count.
- Matching `.csv` files provide numeric wide tables for all four methods.
- `retrieval_all_metrics.csv`: all @4/@10/@20 means, standard deviations, and
  per-trial values.
- `retrieval_deltas_vs_baseline.csv`: paired method-minus-baseline differences.
- `raw/baseline.json`, `raw/baseline_alpha2.json`, `raw/curriculum.json`,
  `raw/curriculum_alpha2.json`, `raw/stratified.json`, and
  `raw/stratified_alpha2.json`: checkpoint hashes, settings, and complete
  structured results.
- `raw/parts/`: four GPU-partitioned intermediate result files.
- `logs/`: evaluation logs for the four GPU jobs.

The old pretraining data was replaced with the curated PubChem324kV2 50k
subset, so these numbers compare the reproduction variants directly and should
not be presented as exact reproduction of the paper's absolute scores.

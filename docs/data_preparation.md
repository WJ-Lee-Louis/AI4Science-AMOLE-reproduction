# PubChem324kV2 data preparation

This pipeline keeps AMOLE's original Python 3.7 environment untouched. It uses a
separate CPU environment to download, audit, subset, and export the dataset.

```bash
MICROMAMBA=/home/dsail26s2/micromamba/bin/micromamba

$MICROMAMBA create -y -f environment-data.yml -p /home/dsail26s2/envs/amole-data
bash scripts/download_pubchem324kv2.sh
$MICROMAMBA run -p /home/dsail26s2/envs/amole-data \
  python scripts/prepare_pubchem324kv2.py
$MICROMAMBA run -p /home/dsail26s2/envs/amole-data \
  python scripts/validate_pubchem324kv2.py --validate-sdf
```

The fixed source revision is `e449660d39ec83c4ccf0bff2dcfb9bbf6943ab89`.
The default subset:

- uses only `pretrain.pt` as the training pool;
- excludes canonical-SMILES overlap with the official train/valid/test files;
- keeps every remaining molecule with multiple newline-separated descriptions;
- fills the remaining slots to 50,000 using seeded scaffold-diverse round-robin
  sampling of single-description molecules;
- writes portable molecule, text, and molecular-property Parquet tables;
- writes AMOLE-compatible `CID2SMILES.csv`, `CID2text.json`, and `molecules.sdf`;
- records exact source checksums, parameters, quality counts, and summary
  distributions.

Generated data is ignored by Git. The main output is
`data/PubChem324kV2/processed_50k/`; `manifest.json` makes the selection
reproducible, while `statistics/` contains machine-readable distributions ready
for later visualization.

## Current audited build (seed 42)

- 50,000 unique canonical SMILES and 67,357 molecule-text rows
- 12,381 multi-description molecules and 37,619 single-description molecules
- 39,833 scaffold values (including the empty scaffold for acyclic molecules)
- 31 pretrain records excluded for overlap with official evaluation splits
- 73 duplicate canonical-SMILES records merged in the eligible pool

`statistics/quality_flags.json` deliberately reports, but does not silently
remove, large structures, disconnected SMILES, long text, and globally repeated
descriptions. The current build contains 1,333 molecules with more than 128
heavy atoms and only 32,094 unique exact texts among 67,357 rows. These choices
must be reviewed before neighbor generation because filtering later would change
all molecule indices.

Top-k Morgan/Tanimoto neighbor calculation is intentionally a separate stage.
It should run only after this molecule table has been reviewed, because changing
the subset invalidates all neighbor indices and scores.

`environment-train-titan.yml` separately specifies the legacy-compatible
PyTorch 1.10.1/CUDA 11.3/PyG 2.0.3 training stack for the TITAN Xp hosts. It is
not used to prepare the data, so updating training dependencies cannot silently
change the selected subset or its statistics.

## Freeze the final AMOLE 50k package

After reviewing the audit flags, freeze the selected molecule and text order and
build AMOLE-compatible graphs and fingerprints in the training environment:

```bash
$MICROMAMBA run -p /home/dsail26s2/envs/amole-train-titan \
  python scripts/finalize_amole50k.py
$MICROMAMBA run -p /home/dsail26s2/envs/amole-train-titan \
  python scripts/validate_amole50k.py
```

The current finalized package is `data/PubChemSTM_50k`, with
`data/PubChemSTM` linked to it for compatibility with the original AMOLE paths.
It contains:

- all three original raw inputs under `raw/`;
- 50,000 unique GraphMVP/AMOLE-style graphs and 67,357 text-aligned graphs;
- unpacked and bit-packed Morgan radius-2, 1024-bit fingerprints;
- fixed text-to-molecule indices and the original `same_CID.pt` relation;
- explicit policy, order hashes, artifact checksums, statistics, and quality flags.

All 50,000 structures converted successfully. Large molecules, disconnected
SMILES, long descriptions, and repeated exact descriptions remain present by
policy. Raw long descriptions are retained; the 512-token truncation is an
explicit runtime tokenizer rule. Tanimoto neighbors are derived only after this
freeze so that their indices remain tied to the fixed molecule order; the
derived artifacts are described below.

## Exact Tanimoto top-100 neighbors

The frozen build now includes exact self-excluded top-100 Tanimoto neighbors,
computed across all 50,000 Morgan fingerprints with GPUs 4–7:

```bash
$MICROMAMBA run -p /home/dsail26s2/envs/amole-train-titan \
  python scripts/compute_tanimoto_topk.py --gpus 4,5,6,7 --k 100
```

`processed/neighbors_top100.pt` is the canonical artifact. It stores an
`int32 [50000, 100]` molecule-index tensor and a matching `float32` Tanimoto
score tensor, tied to the frozen molecule-order hash. `similarities_CID.pt` and
`similarity_scores_CID.pt` provide the corresponding CID-keyed compatibility
format. Self neighbors are excluded before ranking. GPU results were checked by
recomputing selected scores for 256 rows and the global 50,000-candidate top-100
score sets for 32 rows on CPU.

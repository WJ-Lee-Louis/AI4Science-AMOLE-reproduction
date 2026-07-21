#!/usr/bin/env python3
"""Validate the frozen AMOLE 50k raw and processed package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def graph_count(path: Path) -> int:
    _, slices = torch.load(path, map_location="cpu")
    return int(slices[next(iter(slices))].numel() - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/PubChemSTM_50k"))
    args = parser.parse_args()
    root = args.root
    molecules = pd.read_parquet(root / "processed" / "molecules.parquet")
    texts = pd.read_parquet(root / "processed" / "texts.parquet")
    fingerprints = np.load(root / "processed" / "morgan_fingerprints_1024.npy", mmap_mode="r")
    packed = np.load(root / "processed" / "morgan_fingerprints_1024_packed.npy", mmap_mode="r")
    text_to_molecule = np.load(root / "processed" / "text_to_molecule.npy")
    same_cid = torch.load(root / "processed" / "same_CID.pt")
    failures = json.loads((root / "processed" / "graph_conversion_failures.json").read_text())
    policy = json.loads((root / "DATA_POLICY.json").read_text())
    manifest = json.loads((root / "MANIFEST.json").read_text())

    assert len(molecules) == 50_000 and molecules.canonical_smiles.is_unique
    assert len(texts) == 67_357
    assert fingerprints.shape == (50_000, 1024) and fingerprints.dtype == np.uint8
    assert packed.shape == (50_000, 128) and packed.dtype == np.uint8
    assert np.array_equal(np.unpackbits(packed, axis=1), fingerprints)
    assert np.array_equal(text_to_molecule, texts.molecule_id.to_numpy(dtype=np.int32))
    assert graph_count(root / "processed" / "molecule_graphs.pt") == 50_000
    assert graph_count(root / "processed" / "geometric_data_processed.pt") == len(texts)
    assert len(same_cid) == len(texts) and not failures
    assert policy["dataset_status"] == "frozen_final_raw_50k"
    assert (root / "FINALIZED").is_file()
    result = {
        "status": "ok",
        "molecules": len(molecules),
        "texts": len(texts),
        "unique_graphs": 50_000,
        "text_aligned_graphs": len(texts),
        "fingerprints": list(fingerprints.shape),
        "graph_conversion_failures": len(failures),
    }

    neighbor_path = root / "processed" / "neighbors_top100.pt"
    if neighbor_path.exists():
        neighbors = torch.load(neighbor_path, map_location="cpu")
        indices = neighbors["indices"].numpy()
        scores = neighbors["scores"].numpy()
        rows = np.arange(len(molecules), dtype=np.int64)[:, None]
        assert indices.shape == (50_000, 100) and scores.shape == (50_000, 100)
        assert not np.any(indices == rows)
        assert np.all(scores[:, :-1] >= scores[:, 1:] - 1e-7)
        assert neighbors["molecule_order_sha256"] == manifest["molecule_order_sha256"]
        cid_neighbors = torch.load(root / "processed" / "similarities_CID.pt")
        cid_scores = torch.load(root / "processed" / "similarity_scores_CID.pt")
        cids = molecules.cid.astype(np.int64).to_numpy()
        assert len(cid_neighbors) == len(cid_scores) == 50_000
        for row in range(50_000):
            cid = int(cids[row])
            assert np.array_equal(cid_neighbors[cid], cids[indices[row]])
            assert np.array_equal(cid_scores[cid], scores[row])
        result["neighbors"] = {
            "shape": list(indices.shape),
            "self_excluded": True,
            "scores_saved": True,
            "amole_cid_mapping": True,
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

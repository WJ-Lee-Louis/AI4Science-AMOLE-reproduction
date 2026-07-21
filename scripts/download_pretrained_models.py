#!/usr/bin/env python3
"""Download and validate the SciBERT and GraphMVP-G initial checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download


GRAPH_MVP_URL = (
    "https://huggingface.co/chao1224/MoleculeSTM/resolve/main/"
    "pretrained_GraphMVP/GraphMVP_G/model.pth"
)
GRAPH_MVP_SHA256 = "ef7e38bb5b239c2fd223ab3e5f48ad06d03c224ed2141be282f41c476c2d4ff7"
SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
SCIBERT_REQUIRED = ["config.json", "pytorch_model.bin", "vocab.txt"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/PubChemSTM"))
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    scibert_path = data_path / "pretrained_SciBERT"
    graphmvp_path = data_path / "pretrained_GraphMVP" / "GraphMVP_G" / "model.pth"

    if any(not (scibert_path / name).is_file() for name in SCIBERT_REQUIRED):
        snapshot_download(
            repo_id="allenai/scibert_scivocab_uncased",
            revision=SCIBERT_REVISION,
            local_dir=str(scibert_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5"],
        )

    graphmvp_path.parent.mkdir(parents=True, exist_ok=True)
    if not graphmvp_path.exists() or sha256(graphmvp_path) != GRAPH_MVP_SHA256:
        temporary_path = graphmvp_path.with_suffix(".pth.part")
        urlretrieve(GRAPH_MVP_URL, temporary_path)
        temporary_path.replace(graphmvp_path)

    actual_hash = sha256(graphmvp_path)
    if actual_hash != GRAPH_MVP_SHA256:
        raise RuntimeError(
            f"GraphMVP-G checksum mismatch: expected {GRAPH_MVP_SHA256}, got {actual_hash}"
        )

    missing = [name for name in SCIBERT_REQUIRED if not (scibert_path / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete SciBERT snapshot; missing: {missing}")

    print(f"SciBERT: {scibert_path}")
    print(f"GraphMVP-G: {graphmvp_path}")
    print(f"GraphMVP-G SHA256: {actual_hash}")


if __name__ == "__main__":
    main()

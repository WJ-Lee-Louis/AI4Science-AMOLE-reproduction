#!/usr/bin/env python3
"""Freeze the audited 50k subset into AMOLE-compatible raw/processed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.chem import mol_to_graph_data_obj_simple

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/PubChem324kV2/processed_50k"))
    parser.add_argument("--output-root", type=Path, default=Path("data/PubChemSTM_50k"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def collate(graphs: list[Any]) -> tuple[Any, dict[str, torch.Tensor]]:
    return InMemoryDataset.collate(graphs)


def build_graphs_and_fingerprints(molecules: pd.DataFrame) -> tuple[list[Any], np.ndarray, list[dict[str, Any]]]:
    graphs: list[Any] = []
    fingerprints = np.empty((len(molecules), 1024), dtype=np.uint8)
    failures: list[dict[str, Any]] = []
    for row in tqdm(molecules.itertuples(index=False), total=len(molecules), desc="graphs + fingerprints"):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            failures.append({"molecule_id": int(row.molecule_id), "cid": str(row.cid), "reason": "SMILES parse failure"})
            continue
        try:
            graph = mol_to_graph_data_obj_simple(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            array = np.zeros(1024, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, array)
            graph.fp = torch.from_numpy(array.copy()).reshape(1, -1)
        except Exception as exc:
            failures.append({"molecule_id": int(row.molecule_id), "cid": str(row.cid), "reason": repr(exc)})
            continue
        graphs.append(graph)
        fingerprints[int(row.molecule_id)] = array
    return graphs, fingerprints, failures


def build_same_cid(texts: pd.DataFrame) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for text_index, molecule_id in enumerate(texts.molecule_id.astype(int)):
        groups[molecule_id].append(text_index)
    same_cid: dict[int, np.ndarray] = {}
    for indices in groups.values():
        values = np.asarray(indices, dtype=np.int64)
        for index in indices:
            same_cid[index] = values[values != index]
    return same_cid


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"

    molecules = pd.read_parquet(input_root / "molecules.parquet")
    texts = pd.read_parquet(input_root / "texts.parquet").sort_values(
        ["molecule_id", "description_id"], kind="stable"
    ).reset_index(drop=True)
    properties = pd.read_parquet(input_root / "molecular_properties.parquet")
    if len(molecules) != 50_000 or not molecules.molecule_id.tolist() == list(range(50_000)):
        raise ValueError("Expected a frozen contiguous 50,000-molecule table")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(output_root)
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    graphs, fingerprints, failures = build_graphs_and_fingerprints(molecules)
    atomic_json(processed_dir / "graph_conversion_failures.json", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} selected molecules cannot be converted. The 50k selection must be refilled before finalization."
        )

    unique_graphs, unique_slices = collate(graphs)
    torch.save((unique_graphs, unique_slices), processed_dir / "molecule_graphs.pt")
    np.save(processed_dir / "morgan_fingerprints_1024.npy", fingerprints)
    np.save(processed_dir / "morgan_fingerprints_1024_packed.npy", np.packbits(fingerprints, axis=1))

    text_to_molecule = texts.molecule_id.to_numpy(dtype=np.int32)
    np.save(processed_dir / "text_to_molecule.npy", text_to_molecule)
    repeated_graphs = [graphs[molecule_id] for molecule_id in text_to_molecule]
    geometric_data, geometric_slices = collate(repeated_graphs)
    torch.save((geometric_data, geometric_slices), processed_dir / "geometric_data_processed.pt")

    cid_text = texts.merge(molecules[["molecule_id", "cid"]], on=["molecule_id", "cid"], how="inner", validate="many_to_one")
    cid_text[["cid", "text"]].rename(columns={"cid": "CID"}).to_csv(
        processed_dir / "CID_text_list.csv", index=False
    )
    same_cid = build_same_cid(texts)
    torch.save(same_cid, processed_dir / "same_CID.pt")

    source_raw = input_root / "amole" / "raw"
    for name in ("CID2SMILES.csv", "CID2text.json", "molecules.sdf"):
        shutil.copy2(source_raw / name, raw_dir / name)
    molecules.to_parquet(processed_dir / "molecules.parquet", index=False)
    texts.to_parquet(processed_dir / "texts.parquet", index=False)
    properties.to_parquet(processed_dir / "molecular_properties.parquet", index=False)
    shutil.copy2(input_root / "statistics" / "statistics.json", processed_dir / "statistics.json")
    shutil.copy2(input_root / "statistics" / "quality_flags.json", processed_dir / "quality_flags.json")

    policy = {
        "dataset_status": "frozen_final_raw_50k",
        "molecule_count": len(molecules),
        "text_count": len(texts),
        "selection_seed": 42,
        "structure_policy": {
            "keep_large_molecules": True,
            "keep_disconnected_smiles": True,
            "exclude_only_unparseable_or_graph_conversion_failures": True,
            "graph_conversion_failures": len(failures),
        },
        "text_policy": {
            "keep_exact_repeated_descriptions": True,
            "keep_raw_long_descriptions": True,
            "runtime_tokenizer_truncation": {"enabled": True, "max_tokens": 512},
        },
        "fingerprint": {"type": "Morgan bit vector", "radius": 2, "n_bits": 1024},
        "neighbor_policy": "Not generated yet; compute top-k only from this frozen molecule order and save scores with indices.",
    }
    atomic_json(output_root / "DATA_POLICY.json", policy)

    artifacts = sorted(path for path in output_root.rglob("*") if path.is_file())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256(input_root / "manifest.json"),
        "molecule_order_sha256": sequence_hash(molecules.canonical_smiles.astype(str).tolist()),
        "text_order_sha256": sequence_hash(
            (texts.molecule_id.astype(str) + "\t" + texts.description_id.astype(str) + "\t" + texts.text).tolist()
        ),
        "artifacts": {
            str(path.relative_to(output_root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        },
    }
    atomic_json(output_root / "MANIFEST.json", manifest)
    (output_root / "FINALIZED").write_text("AMOLE PubChemSTM 50k dataset finalized successfully.\n")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a reproducible 50k AMOLE-ready subset from PubChem324kV2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


SOURCE_SPLITS = ("pretrain", "train", "valid", "test")


class PubChemDataset(InMemoryDataset):
    def __init__(self, path: Path):
        super().__init__()
        self.data, self.slices = torch.load(path, map_location="cpu")

    def __len__(self) -> int:
        return int(self.slices[next(iter(self.slices))].numel() - 1)

    def drop_graph_tensors(self) -> None:
        """Release graph arrays; this stage only needs metadata and rebuilds AMOLE graphs from SMILES."""
        for key in ("x", "edge_index", "edge_attr"):
            if key in self._data:
                del self._data[key]
        gc.collect()


@dataclass
class MoleculeRecord:
    source_index: int
    cid: str
    smiles: str
    canonical_smiles: str
    descriptions: list[str]
    mol: Chem.Mol
    scaffold: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/PubChem324kV2/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/PubChem324kV2/processed_50k"))
    parser.add_argument("--target-molecules", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-eval-overlap",
        action="store_true",
        help="Keep pretrain molecules also present in the official train/valid/test files.",
    )
    parser.add_argument("--skip-sdf", action="store_true", help="Skip AMOLE-compatible molecules.sdf export.")
    return parser.parse_args()


def resolve_split_files(source_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for split in SOURCE_SPLITS:
        matches = sorted(source_dir.rglob(f"{split}.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {split}.pt under {source_dir}, found {matches}")
        files[split] = matches[0]
    return files


def clean_descriptions(text: Any) -> list[str]:
    candidates: Iterable[Any]
    if isinstance(text, str):
        candidates = text.splitlines()
    elif isinstance(text, (list, tuple)):
        candidates = text
    else:
        candidates = [text]
    seen: set[str] = set()
    output: list[str] = []
    for item in candidates:
        value = " ".join(str(item).split())
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def dataset_field(dataset: PubChemDataset, index: int, key: str) -> Any:
    if key not in dataset._data:
        raise KeyError(f"Dataset item is missing required field {key!r}")
    value = dataset._data[key]
    if isinstance(value, (list, tuple)):
        return scalar(value[index])
    start = int(dataset.slices[key][index])
    end = int(dataset.slices[key][index + 1])
    return scalar(value[start:end])


def canonicalize(smiles: str) -> tuple[str, Chem.Mol] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def read_canonical_smiles(path: Path) -> tuple[set[str], dict[str, int]]:
    dataset = PubChemDataset(path)
    dataset.drop_graph_tensors()
    values: set[str] = set()
    invalid = 0
    for idx in tqdm(range(len(dataset)), desc=f"audit {path.stem}"):
        result = canonicalize(str(dataset_field(dataset, idx, "smiles")))
        if result is None:
            invalid += 1
        else:
            values.add(result[0])
    return values, {"records": len(dataset), "canonical_smiles": len(values), "invalid_smiles": invalid}


def load_pretrain(path: Path, excluded_smiles: set[str]) -> tuple[list[MoleculeRecord], dict[str, int]]:
    dataset = PubChemDataset(path)
    dataset.drop_graph_tensors()
    merged: dict[str, MoleculeRecord] = {}
    counters: Counter[str] = Counter(records=len(dataset))
    for idx in tqdm(range(len(dataset)), desc="load pretrain"):
        smiles = str(dataset_field(dataset, idx, "smiles"))
        result = canonicalize(smiles)
        if result is None:
            counters["invalid_smiles"] += 1
            continue
        canonical, mol = result
        descriptions = clean_descriptions(dataset_field(dataset, idx, "text"))
        if not descriptions:
            counters["empty_text"] += 1
            continue
        if canonical in excluded_smiles:
            counters["eval_overlap_excluded"] += 1
            if len(descriptions) > 1:
                counters["eval_overlap_multi_excluded"] += 1
            continue
        cid = str(dataset_field(dataset, idx, "cid"))
        if canonical in merged:
            counters["duplicate_canonical_smiles"] += 1
            existing = merged[canonical]
            existing.descriptions = list(dict.fromkeys(existing.descriptions + descriptions))
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        merged[canonical] = MoleculeRecord(idx, cid, smiles, canonical, descriptions, mol, scaffold)
    records = list(merged.values())
    counters["eligible_unique_molecules"] = len(records)
    counters["eligible_multi_description"] = sum(len(r.descriptions) > 1 for r in records)
    counters["eligible_texts"] = sum(len(r.descriptions) for r in records)
    return records, dict(counters)


def diverse_round_robin(records: list[MoleculeRecord], count: int, seed: int) -> list[MoleculeRecord]:
    if count < 0 or count > len(records):
        raise ValueError(f"Cannot select {count} molecules from {len(records)} candidates")
    rng = random.Random(seed)
    groups: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for record in records:
        groups[record.scaffold].append(record)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    for _, members in group_items:
        rng.shuffle(members)
    selected: list[MoleculeRecord] = []
    active = [members for _, members in group_items]
    depth = 0
    while active and len(selected) < count:
        next_active: list[list[MoleculeRecord]] = []
        for members in active:
            if depth < len(members):
                selected.append(members[depth])
                if len(selected) == count:
                    break
            if depth + 1 < len(members):
                next_active.append(members)
        active = next_active
        depth += 1
    return selected


def select_records(records: list[MoleculeRecord], target: int, seed: int) -> tuple[list[MoleculeRecord], dict[str, int]]:
    multi = [record for record in records if len(record.descriptions) > 1]
    single = [record for record in records if len(record.descriptions) == 1]
    if len(multi) > target:
        raise ValueError(f"{len(multi)} multi-description molecules exceed target {target}")
    chosen_single = diverse_round_robin(single, target - len(multi), seed)
    selected = multi + chosen_single
    random.Random(seed).shuffle(selected)
    return selected, {
        "target_molecules": target,
        "selected_multi_description": len(multi),
        "selected_single_description": len(chosen_single),
        "selected_total_texts": sum(len(record.descriptions) for record in selected),
    }


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    points = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    result = {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in points}
    result.update(mean=float(array.mean()), std=float(array.std()), count=int(array.size))
    return result


def quality_summary(molecules: pd.DataFrame, texts: pd.DataFrame, properties: pd.DataFrame) -> dict[str, Any]:
    text_frequency = texts["text"].value_counts()
    return {
        "structure_outliers": {
            "disconnected_smiles": int(molecules.canonical_smiles.str.contains(".", regex=False).sum()),
            "heavy_atoms_gt_64": int((properties.heavy_atoms > 64).sum()),
            "heavy_atoms_gt_128": int((properties.heavy_atoms > 128).sum()),
            "heavy_atoms_gt_256": int((properties.heavy_atoms > 256).sum()),
            "heavy_atoms_gt_512": int((properties.heavy_atoms > 512).sum()),
            "molecular_weight_gt_1000": int((properties.molecular_weight > 1000).sum()),
            "molecular_weight_gt_2000": int((properties.molecular_weight > 2000).sum()),
            "molecular_weight_gt_5000": int((properties.molecular_weight > 5000).sum()),
        },
        "text_outliers": {
            "word_count_gt_128": int((texts.word_count > 128).sum()),
            "word_count_gt_256": int((texts.word_count > 256).sum()),
            "word_count_gt_512": int((texts.word_count > 512).sum()),
            "unique_exact_texts": int(texts.text.nunique()),
            "rows_in_repeated_exact_text_groups": int(texts.duplicated("text", keep=False).sum()),
            "repeated_exact_text_groups": int((text_frequency > 1).sum()),
            "maximum_exact_text_reuse": int(text_frequency.max()),
        },
        "note": "These are audit flags, not automatic exclusion rules. Review them before fixing a training subset.",
    }


def molecule_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "h_bond_donors": float(Lipinski.NumHDonors(mol)),
        "h_bond_acceptors": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "atoms": float(mol.GetNumAtoms()),
        "bonds": float(mol.GetNumBonds()),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    selected: list[MoleculeRecord],
    output_dir: Path,
    source_files: dict[str, Path],
    config: dict[str, Any],
    source_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    skip_sdf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    amole_raw = output_dir / "amole" / "raw"
    stats_dir = output_dir / "statistics"
    amole_raw.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    molecule_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    element_counts: Counter[str] = Counter()
    cid2text: dict[str, list[str]] = {}

    for molecule_id, record in enumerate(tqdm(selected, desc="compute statistics")):
        molecule_rows.append({
            "molecule_id": molecule_id,
            "cid": record.cid,
            "smiles": record.smiles,
            "canonical_smiles": record.canonical_smiles,
            "scaffold_smiles": record.scaffold,
            "description_count": len(record.descriptions),
            "source_split": "pretrain",
            "source_index": record.source_index,
        })
        cid2text[record.cid] = record.descriptions
        properties = molecule_properties(record.mol)
        property_rows.append({"molecule_id": molecule_id, **properties})
        element_counts.update(atom.GetSymbol() for atom in record.mol.GetAtoms())
        for description_id, description in enumerate(record.descriptions):
            text_rows.append({
                "molecule_id": molecule_id,
                "cid": record.cid,
                "description_id": description_id,
                "text": description,
                "character_count": len(description),
                "word_count": len(description.split()),
            })

    molecules = pd.DataFrame(molecule_rows)
    texts = pd.DataFrame(text_rows)
    properties = pd.DataFrame(property_rows)
    molecules.to_parquet(output_dir / "molecules.parquet", index=False)
    texts.to_parquet(output_dir / "texts.parquet", index=False)
    properties.to_parquet(output_dir / "molecular_properties.parquet", index=False)
    molecules.to_csv(output_dir / "molecules.csv", index=False)

    description_distribution = (
        molecules["description_count"].value_counts().sort_index().rename_axis("description_count").reset_index(name="molecule_count")
    )
    scaffold_frequency = (
        molecules["scaffold_smiles"].fillna("").value_counts().rename_axis("scaffold_smiles").reset_index(name="molecule_count")
    )
    description_distribution.to_csv(stats_dir / "description_count_distribution.csv", index=False)
    scaffold_frequency.to_csv(stats_dir / "scaffold_frequency.csv", index=False)
    pd.DataFrame(sorted(element_counts.items()), columns=["element", "atom_count"]).to_csv(
        stats_dir / "element_frequency.csv", index=False
    )

    property_summary = pd.DataFrame({name: quantiles(properties[name]) for name in properties.columns if name != "molecule_id"}).T
    property_summary.index.name = "property"
    property_summary.to_csv(stats_dir / "molecular_property_summary.csv")
    text_summary = {
        "character_count": quantiles(texts["character_count"]),
        "word_count": quantiles(texts["word_count"]),
    }
    atomic_write_json(stats_dir / "text_length_summary.json", text_summary)
    quality = quality_summary(molecules, texts, properties)
    atomic_write_json(stats_dir / "quality_flags.json", quality)

    pd.DataFrame({"CID": molecules["cid"], "SMILES": molecules["canonical_smiles"]}).to_csv(
        amole_raw / "CID2SMILES.csv", index=False
    )
    atomic_write_json(amole_raw / "CID2text.json", cid2text)
    if not skip_sdf:
        writer = Chem.SDWriter(str(amole_raw / "molecules.sdf"))
        for record in tqdm(selected, desc="write SDF"):
            mol = Chem.Mol(record.mol)
            mol.SetProp("PUBCHEM_COMPOUND_CID", record.cid)
            writer.write(mol)
        writer.close()

    overall_stats = {
        "source": source_stats,
        "selection": selection_stats,
        "selected": {
            "molecules": len(molecules),
            "texts": len(texts),
            "multi_description_molecules": int((molecules.description_count > 1).sum()),
            "unique_canonical_smiles": int(molecules.canonical_smiles.nunique()),
            "unique_scaffolds_including_empty": int(molecules.scaffold_smiles.nunique()),
            "empty_scaffold_molecules": int((molecules.scaffold_smiles == "").sum()),
        },
        "text_lengths": text_summary,
        "molecular_properties": property_summary.to_dict(orient="index"),
        "quality_flags": quality,
    }
    atomic_write_json(stats_dir / "statistics.json", overall_stats)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "source_files": {
            split: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for split, path in source_files.items()
        },
        "artifacts": [
            "molecules.parquet",
            "texts.parquet",
            "molecular_properties.parquet",
            "molecules.csv",
            "amole/raw/CID2SMILES.csv",
            "amole/raw/CID2text.json",
            *([] if skip_sdf else ["amole/raw/molecules.sdf"]),
            "statistics/statistics.json",
            "statistics/quality_flags.json",
        ],
    }
    atomic_write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    source_files = resolve_split_files(args.source_dir)
    eval_smiles: set[str] = set()
    source_stats: dict[str, Any] = {}
    if not args.allow_eval_overlap:
        for split in ("train", "valid", "test"):
            split_smiles, split_stats = read_canonical_smiles(source_files[split])
            eval_smiles.update(split_smiles)
            source_stats[split] = split_stats
        source_stats["combined_eval_unique_canonical_smiles"] = len(eval_smiles)

    records, pretrain_stats = load_pretrain(source_files["pretrain"], eval_smiles)
    source_stats["pretrain"] = pretrain_stats
    selected, selection_stats = select_records(records, args.target_molecules, args.seed)
    config = {
        "target_molecules": args.target_molecules,
        "seed": args.seed,
        "include_all_eligible_multi_description": True,
        "single_description_selection": "scaffold-diverse seeded round-robin",
        "eval_overlap_policy": "allow" if args.allow_eval_overlap else "exclude official train/valid/test canonical SMILES",
        "description_parsing": "split newline, normalize whitespace, exact deduplication per molecule",
    }
    write_outputs(
        selected,
        args.output_dir,
        source_files,
        config,
        source_stats,
        selection_stats,
        args.skip_sdf,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **selection_stats}, indent=2))


if __name__ == "__main__":
    main()

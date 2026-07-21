#!/usr/bin/env python3
"""Fast integrity checks for the prepared PubChem324kV2 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/PubChem324kV2/processed_50k"))
    parser.add_argument("--validate-sdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    molecules = pd.read_parquet(root / "molecules.parquet")
    texts = pd.read_parquet(root / "texts.parquet")
    properties = pd.read_parquet(root / "molecular_properties.parquet")
    stats = json.loads((root / "statistics" / "statistics.json").read_text())
    cid2text = json.loads((root / "amole" / "raw" / "CID2text.json").read_text())
    cid2smiles = pd.read_csv(root / "amole" / "raw" / "CID2SMILES.csv", dtype={"CID": str})

    assert len(molecules) == stats["selected"]["molecules"]
    assert len(texts) == stats["selected"]["texts"]
    assert len(properties) == len(molecules)
    assert molecules.molecule_id.tolist() == list(range(len(molecules)))
    assert molecules.canonical_smiles.is_unique
    assert molecules.cid.astype(str).is_unique
    assert set(texts.molecule_id) == set(molecules.molecule_id)
    assert int(molecules.description_count.sum()) == len(texts)
    assert texts.groupby("molecule_id").size().eq(molecules.set_index("molecule_id").description_count).all()
    assert len(cid2text) == len(molecules)
    assert len(cid2smiles) == len(molecules)
    assert cid2smiles.CID.is_unique
    assert not molecules.isna().any().any()
    assert not texts.isna().any().any()
    assert not properties.isna().any().any()

    sdf_records = None
    if args.validate_sdf:
        supplier = Chem.SDMolSupplier(str(root / "amole" / "raw" / "molecules.sdf"))
        sdf_records = sum(mol is not None for mol in supplier)
        assert sdf_records == len(molecules)

    print(json.dumps({
        "status": "ok",
        "molecules": len(molecules),
        "texts": len(texts),
        "multi_description_molecules": int((molecules.description_count > 1).sum()),
        "unique_scaffolds_including_empty": int(molecules.scaffold_smiles.nunique()),
        "sdf_records": sdf_records,
    }, indent=2))


if __name__ == "__main__":
    main()


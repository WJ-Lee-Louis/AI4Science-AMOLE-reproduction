#!/usr/bin/env python3
"""Efficient zero-shot DrugBank retrieval evaluation for AMOLE checkpoints.

The official evaluator recomputes deterministic eval-mode embeddings for every
negative draw. This implementation caches each molecule/text embedding once,
then runs five seeded negative-sampling trials for T in {4, 10, 20}.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers import GNN, GNN_graphpred
from utils.chem import mol_to_graph_data_obj_simple


DATASETS = {
    "description": {
        "file": "SMILES_description_removed_from_PubChem_full.txt",
        "processed_text": lambda columns: columns[1],
        "split_count": 1,
    },
    "pharmacodynamics": {
        "file": "SMILES_pharmacodynamics_removed_from_PubChem_full.txt",
        "processed_text": lambda columns: columns[1],
        "split_count": 1,
    },
    "ATC": {
        "file": "SMILES_ATC_5_full.txt",
        "processed_text": lambda columns: f"This molecule is for {columns[2]}.",
        "split_count": 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/Drugbank"))
    parser.add_argument(
        "--scibert_dir", type=Path, default=Path("data/PubChemSTM/pretrained_SciBERT")
    )
    parser.add_argument("--output_dir", type=Path, default=Path("Reproduction_Evaluation/raw"))
    parser.add_argument("--output_name", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--graph_batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])
    return parser.parse_args()


def discover_checkpoint(checkpoint_dir, component):
    files = sorted((checkpoint_dir / component).glob("*.pth"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {checkpoint_dir / component}, found {len(files)}"
        )
    return files[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(args, device):
    tokenizer = AutoTokenizer.from_pretrained(str(args.scibert_dir), local_files_only=True)
    text_model = AutoModel.from_pretrained(str(args.scibert_dir), local_files_only=True)

    molecule_node_model = GNN(
        num_layer=5,
        emb_dim=300,
        JK="last",
        drop_ratio=0.5,
        gnn_type="gin",
    )
    molecule_model = GNN_graphpred(
        num_layer=5,
        emb_dim=300,
        JK="last",
        graph_pooling="mean",
        num_tasks=1,
        molecule_node_model=molecule_node_model,
    )
    text2latent = nn.Linear(768, 256)
    mol2latent = nn.Linear(300, 256)

    components = {
        "text": text_model,
        "molecule": molecule_model,
        "text2latent": text2latent,
        "mol2latent": mol2latent,
    }
    checkpoint_manifest = {}
    for name, model in components.items():
        path = discover_checkpoint(args.checkpoint_dir, name)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        checkpoint_manifest[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
        model.to(device).eval()

    return tokenizer, text_model, molecule_model, text2latent, mol2latent, checkpoint_manifest


def load_dataset(data_dir, dataset_name):
    spec = DATASETS[dataset_name]
    raw_path = data_dir / "raw" / spec["file"]
    index_path = data_dir / "index" / spec["file"]
    raw_lines = raw_path.read_text().splitlines()
    index_lines = index_path.read_text().splitlines()
    if len(raw_lines) != len(index_lines):
        raise RuntimeError(f"Raw/index row mismatch for {dataset_name}")

    smiles, texts, negative_candidates = [], [], []
    for row_number, (raw_line, index_line) in enumerate(zip(raw_lines, index_lines)):
        columns = raw_line.split("\t", spec["split_count"])
        molecule = AllChem.MolFromSmiles(columns[0])
        if molecule is None:
            raise RuntimeError(f"Invalid SMILES at {raw_path}:{row_number + 1}")
        candidates = np.asarray([int(value) for value in index_line.split(",")], dtype=np.int64)
        if len(candidates) < 19 or np.any(candidates == row_number):
            raise RuntimeError(f"Invalid negative candidates at {index_path}:{row_number + 1}")
        smiles.append(columns[0])
        texts.append(spec["processed_text"](columns))
        negative_candidates.append(candidates)
    return smiles, texts, negative_candidates


@torch.no_grad()
def encode_texts(texts, tokenizer, text_model, text2latent, device, batch_size, max_seq_len):
    encoded = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        representation = text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )["pooler_output"]
        encoded.append(text2latent(representation).cpu())
    return F.normalize(torch.cat(encoded, dim=0), dim=-1).numpy()


@torch.no_grad()
def encode_molecules(smiles, molecule_model, mol2latent, device, batch_size):
    graphs = []
    for row_number, value in enumerate(smiles):
        graph = mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(value))
        graph.id = torch.tensor([row_number])
        graphs.append(graph)

    encoded = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0):
        representation, _ = molecule_model(batch.to(device))
        encoded.append(mol2latent(representation).cpu())
    return F.normalize(torch.cat(encoded, dim=0), dim=-1).numpy()


def sample_negative_indices(candidates, seed, count):
    rng = np.random.RandomState(seed)
    return np.stack([rng.choice(row, count) for row in candidates])


def evaluate_embeddings(text_embeddings, molecule_embeddings, candidates, seeds, T_list):
    positive_scores = np.sum(text_embeddings * molecule_embeddings, axis=1)
    trial_results = {
        "given_molecule": {str(T): [] for T in T_list},
        "given_text": {str(T): [] for T in T_list},
    }
    max_negatives = max(T_list) - 1

    for seed in seeds:
        negative_indices = sample_negative_indices(candidates, seed, max_negatives)
        molecule_to_text_scores = np.einsum(
            "nd,nkd->nk", molecule_embeddings, text_embeddings[negative_indices]
        )
        text_to_molecule_scores = np.einsum(
            "nd,nkd->nk", text_embeddings, molecule_embeddings[negative_indices]
        )
        for T in T_list:
            negative_count = T - 1
            # Positive is placed first in the official evaluator, so a tie is
            # counted as correct by argmax.
            given_molecule = np.mean(
                positive_scores >= molecule_to_text_scores[:, :negative_count].max(axis=1)
            )
            given_text = np.mean(
                positive_scores >= text_to_molecule_scores[:, :negative_count].max(axis=1)
            )
            trial_results["given_molecule"][str(T)].append(float(given_molecule))
            trial_results["given_text"][str(T)].append(float(given_text))

    summary = {}
    for direction, by_T in trial_results.items():
        summary[direction] = {}
        for T, values in by_T.items():
            summary[direction][T] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "trials": values,
            }
    return summary


def main():
    args = parse_args()
    if max(args.T_list) > 20:
        raise ValueError("Official AMOLE retrieval evaluation supports T up to 20")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        tokenizer,
        text_model,
        molecule_model,
        text2latent,
        mol2latent,
        checkpoint_manifest,
    ) = load_models(args, device)

    result = {
        "strategy": args.strategy,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_manifest": checkpoint_manifest,
        "seeds": args.seeds,
        "T_list": args.T_list,
        "max_seq_len": args.max_seq_len,
        "datasets": {},
    }
    for dataset_name in args.datasets:
        print(f"[{args.strategy}] encoding {dataset_name}", flush=True)
        smiles, texts, candidates = load_dataset(args.data_dir, dataset_name)
        text_embeddings = encode_texts(
            texts,
            tokenizer,
            text_model,
            text2latent,
            device,
            args.text_batch_size,
            args.max_seq_len,
        )
        molecule_embeddings = encode_molecules(
            smiles, molecule_model, mol2latent, device, args.graph_batch_size
        )
        metrics = evaluate_embeddings(
            text_embeddings, molecule_embeddings, candidates, args.seeds, args.T_list
        )
        result["datasets"][dataset_name] = {
            "size": len(smiles),
            "metrics": metrics,
        }
        for direction in ("given_molecule", "given_text"):
            metric = metrics[direction]["20"]
            print(
                f"[{args.strategy}] {dataset_name} {direction} @20: "
                f"{100 * metric['mean']:.2f} +/- {100 * metric['std']:.2f}",
                flush=True,
            )

    output_name = args.output_name or args.strategy
    output_path = args.output_dir / f"{output_name}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute exact self-excluded Morgan/Tanimoto top-k neighbors on multiple GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/PubChemSTM_50k"))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_ranges(size: int, parts: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, size, parts + 1, dtype=np.int64)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(parts)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker(
    rank: int,
    gpu: int,
    start: int,
    end: int,
    fingerprint_path: str,
    shard_path: str,
    k: int,
    chunk_size: int,
) -> None:
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    fingerprints = np.load(fingerprint_path, mmap_mode="r")
    reference = torch.from_numpy(np.asarray(fingerprints, dtype=np.float32)).to(device)
    reference_t = reference.t().contiguous()
    reference_count = reference.sum(dim=1).reshape(1, -1)
    shard_indices = np.empty((end - start, k), dtype=np.int32)
    shard_scores = np.empty((end - start, k), dtype=np.float32)
    started = time.time()

    with torch.no_grad():
        for query_start in range(start, end, chunk_size):
            query_end = min(query_start + chunk_size, end)
            query = reference[query_start:query_end]
            intersection = torch.mm(query, reference_t)
            union = query.sum(dim=1, keepdim=True) + reference_count - intersection
            intersection.div_(union.clamp_min_(1.0))
            local_rows = torch.arange(query_end - query_start, device=device)
            global_rows = torch.arange(query_start, query_end, device=device)
            intersection[local_rows, global_rows] = -1.0
            scores, indices = torch.topk(intersection, k=k, dim=1, largest=True, sorted=True)
            offset_start = query_start - start
            offset_end = query_end - start
            shard_indices[offset_start:offset_end] = indices.cpu().numpy().astype(np.int32, copy=False)
            shard_scores[offset_start:offset_end] = scores.cpu().numpy().astype(np.float32, copy=False)
            print(
                f"GPU {gpu} rank {rank}: {query_end - start}/{end - start} rows "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    np.savez_compressed(
        shard_path,
        start=np.asarray(start, dtype=np.int64),
        end=np.asarray(end, dtype=np.int64),
        indices=shard_indices,
        scores=shard_scores,
    )


def validate_exact(fingerprints: np.ndarray, indices: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    n, k = indices.shape
    rows = np.arange(n, dtype=np.int64)[:, None]
    if np.any(indices == rows):
        raise AssertionError("Self neighbor found")
    if np.any(scores[:, :-1] < scores[:, 1:] - 1e-7):
        raise AssertionError("Neighbor scores are not sorted in descending order")
    if np.any(scores < 0) or np.any(scores > 1):
        raise AssertionError("Tanimoto scores outside [0, 1]")

    rng = np.random.default_rng(42)
    sampled_rows = rng.choice(n, size=min(256, n), replace=False)
    maximum_error = 0.0
    for row in sampled_rows:
        neighbors = indices[row]
        query = fingerprints[row].astype(bool)
        candidates = fingerprints[neighbors].astype(bool)
        intersection = np.logical_and(query, candidates).sum(axis=1)
        union = np.logical_or(query, candidates).sum(axis=1)
        exact = intersection / np.maximum(union, 1)
        maximum_error = max(maximum_error, float(np.max(np.abs(exact - scores[row]))))
    if maximum_error > 1e-6:
        raise AssertionError(f"CPU exact Tanimoto validation failed: max error {maximum_error}")

    global_sampled_rows = sampled_rows[:32]
    maximum_topk_score_error = 0.0
    reference_bool = fingerprints.astype(bool)
    for row in global_sampled_rows:
        query = reference_bool[row]
        intersection = np.logical_and(query, reference_bool).sum(axis=1)
        union = np.logical_or(query, reference_bool).sum(axis=1)
        all_scores = intersection / np.maximum(union, 1)
        all_scores[row] = -1.0
        expected_scores = np.sort(all_scores)[-k:][::-1]
        maximum_topk_score_error = max(
            maximum_topk_score_error,
            float(np.max(np.abs(expected_scores - scores[row]))),
        )
    if maximum_topk_score_error > 1e-6:
        raise AssertionError(f"Global top-k validation failed: max error {maximum_topk_score_error}")
    return {
        "sampled_rows": int(len(sampled_rows)),
        "global_topk_sampled_rows": int(len(global_sampled_rows)),
        "maximum_absolute_error": maximum_error,
        "maximum_global_topk_score_error": maximum_topk_score_error,
        "minimum_top1_score": float(scores[:, 0].min()),
        "median_top1_score": float(np.median(scores[:, 0])),
        "median_top100_score": float(np.median(scores[:, -1])),
    }


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    root = args.root
    processed = root / "processed"
    fingerprint_path = processed / "morgan_fingerprints_1024.npy"
    output_path = processed / f"neighbors_top{args.k}.pt"
    cid_output_path = processed / "similarities_CID.pt"
    score_cid_output_path = processed / "similarity_scores_CID.pt"
    report_path = processed / f"neighbors_top{args.k}_validation.json"
    temporary = processed / f".tanimoto_top{args.k}_shards"
    existing = [path for path in (output_path, cid_output_path, score_cid_output_path, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}; pass --overwrite to rebuild")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    fingerprints = np.load(fingerprint_path, mmap_mode="r")
    if fingerprints.ndim != 2 or fingerprints.shape[1] != 1024:
        raise ValueError(f"Unexpected fingerprint shape: {fingerprints.shape}")
    n = len(fingerprints)
    if args.k >= n:
        raise ValueError("k must be smaller than the number of molecules for self exclusion")
    ranges = split_ranges(n, len(gpus))
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    started = time.time()
    for rank, (gpu, (start, end)) in enumerate(zip(gpus, ranges)):
        shard_path = temporary / f"shard_{rank:02d}.npz"
        process = context.Process(
            target=worker,
            args=(rank, gpu, start, end, str(fingerprint_path), str(shard_path), args.k, args.query_chunk_size),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [(index, process.exitcode) for index, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"GPU workers failed: {failed}")

    indices = np.empty((n, args.k), dtype=np.int32)
    scores = np.empty((n, args.k), dtype=np.float32)
    for rank, (start, end) in enumerate(ranges):
        shard = np.load(temporary / f"shard_{rank:02d}.npz")
        if int(shard["start"]) != start or int(shard["end"]) != end:
            raise AssertionError(f"Shard {rank} range mismatch")
        indices[start:end] = shard["indices"]
        scores[start:end] = shard["scores"]

    validation = validate_exact(fingerprints, indices, scores)
    source_manifest = json.loads((root / "MANIFEST.json").read_text())
    artifact = {
        "version": 1,
        "metric": "Tanimoto",
        "fingerprint": {"type": "Morgan bit vector", "radius": 2, "n_bits": 1024},
        "molecule_order_sha256": source_manifest["molecule_order_sha256"],
        "self_excluded": True,
        "k": args.k,
        "indices": torch.from_numpy(indices),
        "scores": torch.from_numpy(scores),
    }
    torch.save(artifact, output_path)

    molecules = pd.read_parquet(processed / "molecules.parquet")
    cids = molecules.cid.astype(np.int64).to_numpy()
    similarities_cid = {int(cids[row]): cids[indices[row]].copy() for row in range(n)}
    scores_cid = {int(cids[row]): scores[row].copy() for row in range(n)}
    torch.save(similarities_cid, cid_output_path)
    torch.save(scores_cid, score_cid_output_path)

    validation.update({
        "status": "ok",
        "molecules": n,
        "k": args.k,
        "self_excluded": True,
        "gpus": gpus,
        "query_chunk_size": args.query_chunk_size,
        "elapsed_seconds": time.time() - started,
        "indices_dtype": str(indices.dtype),
        "scores_dtype": str(scores.dtype),
    })
    report_path.write_text(json.dumps(validation, indent=2) + "\n")

    policy_path = root / "DATA_POLICY.json"
    policy = json.loads(policy_path.read_text())
    policy["neighbor_policy"] = {
        "status": "complete",
        "metric": "exact Tanimoto",
        "self_excluded": True,
        "k": args.k,
        "indices_and_scores": str(output_path.relative_to(root)),
        "amole_cid_neighbors": str(cid_output_path.relative_to(root)),
        "amole_cid_scores": str(score_cid_output_path.relative_to(root)),
    }
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")

    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (policy_path, output_path, cid_output_path, score_cid_output_path, report_path):
        manifest["artifacts"][str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(temporary)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

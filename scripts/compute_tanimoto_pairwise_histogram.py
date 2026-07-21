#!/usr/bin/env python3
"""Count an exact histogram over every unique molecule pair using multiple GPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/PubChemSTM_50k"))
    parser.add_argument("--output-dir", type=Path, default=Path("Tanimoto_Analysis"))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--bin-width", type=float, default=0.05)
    return parser.parse_args()


def worker(
    gpu: int,
    start: int,
    end: int,
    fingerprint_path: str,
    output_path: str,
    chunk_size: int,
    boundaries: list[float],
) -> None:
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    fingerprints = np.load(fingerprint_path, mmap_mode="r")
    reference = torch.from_numpy(np.asarray(fingerprints, dtype=np.float32)).to(device)
    reference_t = reference.t().contiguous()
    reference_count = reference.sum(dim=1).reshape(1, -1)
    bin_boundaries = torch.tensor(boundaries[1:-1], dtype=torch.float32, device=device)
    counts = torch.zeros(len(boundaries) - 1, dtype=torch.int64, device=device)
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
            intersection[local_rows, global_rows] = -1.0  # temporarily counted in bin 0, removed after merge
            bin_ids = torch.bucketize(intersection, bin_boundaries, right=True)
            chunk_counts = torch.bincount(bin_ids.reshape(-1), minlength=len(boundaries))
            counts += chunk_counts[: len(boundaries) - 1]
            print(f"GPU {gpu}: {query_end - start}/{end - start} rows ({time.time() - started:.1f}s)", flush=True)
    np.save(output_path, counts.cpu().numpy())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    fingerprint_path = args.root / "processed" / "morgan_fingerprints_1024.npy"
    fingerprints = np.load(fingerprint_path, mmap_mode="r")
    n = len(fingerprints)
    boundaries = np.arange(0.0, 1.0 + args.bin_width, args.bin_width, dtype=np.float64)
    boundaries[-1] = 1.0
    split_points = np.linspace(0, n, len(gpus) + 1, dtype=np.int64)
    temporary = args.output_dir / ".pairwise_histogram_shards"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    started = time.time()
    for rank, gpu in enumerate(gpus):
        start, end = int(split_points[rank]), int(split_points[rank + 1])
        process = context.Process(
            target=worker,
            args=(gpu, start, end, str(fingerprint_path), str(temporary / f"{rank}.npy"), args.query_chunk_size, boundaries.tolist()),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"GPU workers failed: {failed}")
    directed_counts = sum((np.load(temporary / f"{rank}.npy") for rank in range(len(gpus))), start=np.zeros(len(boundaries) - 1, dtype=np.int64))
    directed_counts[0] -= n
    expected_directed = n * (n - 1)
    if int(directed_counts.sum()) != expected_directed or np.any(directed_counts % 2):
        raise AssertionError("Directed pair counts are incomplete or not symmetric")
    counts = directed_counts // 2
    expected_unique = n * (n - 1) // 2
    frame = pd.DataFrame({
        "bin_left": boundaries[:-1],
        "bin_right": boundaries[1:],
        "count": counts,
        "percent": counts / expected_unique * 100.0,
    })
    frame.to_csv(args.output_dir / "all_unique_pairwise_histogram.csv", index=False)
    summary = {
        "status": "ok",
        "molecules": n,
        "self_pairs_excluded": True,
        "symmetric_duplicates_removed": True,
        "unique_pair_count": expected_unique,
        "bin_width": args.bin_width,
        "gpus": gpus,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "all_unique_pairwise_histogram_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.rmtree(temporary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

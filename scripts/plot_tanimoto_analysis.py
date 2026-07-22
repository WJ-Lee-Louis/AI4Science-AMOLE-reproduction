#!/usr/bin/env python3
"""Create distribution plots and data-driven case studies for top-100 Tanimoto neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


COLORS = {
    "overall": "#2A788E",
    "isolated_extreme": "#6C757D",
    "spiky_high": "#E67E22",
    "evenly_high": "#2E8B57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/PubChemSTM_50k"))
    parser.add_argument("--output-dir", type=Path, default=Path("Tanimoto_Analysis"))
    parser.add_argument("--bin-width", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_cases(scores: np.ndarray) -> dict[str, int]:
    means = scores.mean(axis=1)
    high_counts = (scores >= 0.8).sum(axis=1)

    # Lowest best-neighbor score; mean breaks ties toward the most isolated neighborhood.
    isolated = int(np.lexsort((means, scores[:, 0]))[0])

    # Require a genuinely high first neighbor, but at most 20 high-similarity neighbors.
    # Maximize the drop from ranks 1-5 to ranks 21-100.
    spike_strength = scores[:, :5].mean(axis=1) - scores[:, 20:].mean(axis=1)
    eligible = np.flatnonzero((scores[:, 0] >= 0.8) & (high_counts <= 20))
    spiky = int(eligible[np.argmax(spike_strength[eligible])])

    # Maximize the weakest (rank-100) neighbor; mean breaks ties.
    evenly_high = int(np.lexsort((-means, -scores[:, -1]))[0])
    return {
        "isolated_extreme": isolated,
        "spiky_high": spiky,
        "evenly_high": evenly_high,
    }


def histogram_frame(values: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
        "percent": counts / counts.sum() * 100.0,
    })


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def bin_labels(frame: pd.DataFrame) -> list[str]:
    labels = [f"{left:.2f}–{right:.2f}" for left, right in zip(frame.bin_left, frame.bin_right)]
    labels[-1] = labels[-1].replace("–1.00", "–1.00]")
    return labels


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_overall(frame: pd.DataFrame, values: np.ndarray, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(frame))
    axis.bar(positions, frame.percent, color=COLORS["overall"], width=0.86)
    axis.set_xticks(positions, bin_labels(frame), rotation=50, ha="right")
    axis.set_ylabel("Share of saved top-100 similarities (%)")
    axis.set_xlabel("Tanimoto similarity interval")
    axis.set_title("Distribution of 5,000,000 Saved Top-100 Tanimoto Similarities", pad=14, weight="bold")
    axis.text(
        0.99,
        0.96,
        f"50,000 molecules × 100 neighbors\nMedian={np.median(values):.3f}  Mean={np.mean(values):.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "01_overall_top100_similarity_distribution")


def plot_top50(
    frame: pd.DataFrame,
    values: np.ndarray,
    output_dir: Path,
    minimum_similarity: float = 0.25,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(frame))
    axis.bar(positions, frame.percent, color="#3B6FB6", width=0.86)
    axis.set_xticks(positions, bin_labels(frame), rotation=50, ha="right")
    axis.set_ylabel("Share of saved top-50 similarities (%)")
    axis.set_xlabel("Tanimoto similarity interval")
    axis.set_title("Distribution of 2,500,000 Saved Top-50 Tanimoto Similarities", pad=14, weight="bold")
    axis.text(
        0.99,
        0.96,
        f"50,000 molecules × 50 neighbors\n"
        f"Median={np.median(values):.3f}  Mean={np.mean(values):.3f}\n"
        f"≥{minimum_similarity:.2f}: {100 * np.mean(values >= minimum_similarity):.2f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "04_overall_top50_similarity_distribution")


def plot_all_pairs(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(frame))
    axis.bar(positions, frame.percent, color="#414487", width=0.86)
    axis.set_xticks(positions, bin_labels(frame), rotation=50, ha="right")
    axis.set_ylabel("Share of all unique molecule pairs (%)")
    axis.set_xlabel("Tanimoto similarity interval")
    axis.set_title("Distribution of All 1,249,975,000 Unique Molecule-Pair Similarities", pad=14, weight="bold")
    axis.text(
        0.99,
        0.96,
        "50,000 choose 2 pairs\nSelf-pairs and symmetric duplicates excluded",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "00_all_unique_pairwise_similarity_distribution")


def case_metrics(case: str, molecule_id: int, scores: np.ndarray, molecules: pd.DataFrame) -> dict[str, Any]:
    values = scores[molecule_id]
    row = molecules.iloc[molecule_id]
    return {
        "case": case,
        "molecule_id": molecule_id,
        "cid": str(row.cid),
        "canonical_smiles": row.canonical_smiles,
        "top1": float(values[0]),
        "top5_mean": float(values[:5].mean()),
        "top10_mean": float(values[:10].mean()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "top100": float(values[-1]),
        "count_ge_0_8": int((values >= 0.8).sum()),
        "count_ge_0_6": int((values >= 0.6).sum()),
    }


def plot_case_histogram(
    case: str,
    molecule_id: int,
    scores: np.ndarray,
    molecules: pd.DataFrame,
    bins: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    values = scores[molecule_id]
    frame = histogram_frame(values, bins)
    cid = molecules.iloc[molecule_id].cid
    figure, axis = plt.subplots(figsize=(12, 5.8))
    positions = np.arange(len(frame))
    axis.bar(positions, frame["count"], color=COLORS[case], width=0.86)
    axis.set_xticks(positions, bin_labels(frame), rotation=50, ha="right")
    axis.set_ylabel("Neighbor count (out of 100)")
    axis.set_xlabel("Tanimoto similarity interval")
    title = case.replace("_", " ").title()
    axis.set_title(f"{title}: Molecule ID {molecule_id}, CID {cid}", pad=14, weight="bold")
    axis.text(
        0.99,
        0.96,
        f"Top-1={values[0]:.3f}  Median={np.median(values):.3f}\nRank-100={values[-1]:.3f}  ≥0.8: {(values >= 0.8).sum()}/100",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    axis.set_ylim(0, max(10, frame["count"].max() * 1.18))
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / f"case_{case}_top100_histogram")
    return frame


def plot_case_panel(
    cases: dict[str, int], scores: np.ndarray, molecules: pd.DataFrame, bins: np.ndarray, output_dir: Path
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
    positions = np.arange(len(bins) - 1)
    for axis, (case, molecule_id) in zip(axes, cases.items()):
        frame = histogram_frame(scores[molecule_id], bins)
        cid = molecules.iloc[molecule_id].cid
        axis.bar(positions, frame["count"], color=COLORS[case], width=0.86)
        axis.set_ylabel("Count")
        axis.set_title(
            f"{case.replace('_', ' ').title()} — Molecule ID {molecule_id}, CID {cid}", loc="left", weight="bold"
        )
        axis.text(
            0.99,
            0.9,
            f"Top-1 {scores[molecule_id, 0]:.3f} | Median {np.median(scores[molecule_id]):.3f} | "
            f"Rank-100 {scores[molecule_id, -1]:.3f}",
            transform=axis.transAxes,
            ha="right",
        )
        style_axis(axis)
    axes[-1].set_xticks(positions, bin_labels(histogram_frame(scores[next(iter(cases.values()))], bins)), rotation=50, ha="right")
    axes[-1].set_xlabel("Tanimoto similarity interval")
    figure.suptitle("Top-100 Similarity Distributions for Three Data-Selected Cases", y=1.005, weight="bold", fontsize=16)
    figure.tight_layout()
    save_figure(figure, output_dir / "02_selected_cases_histogram_panel")


def plot_rank_profiles(cases: dict[str, int], scores: np.ndarray, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.2))
    ranks = np.arange(1, scores.shape[1] + 1)
    for case, molecule_id in cases.items():
        axis.plot(ranks, scores[molecule_id], color=COLORS[case], linewidth=2.2, label=case.replace("_", " ").title())
    axis.set_xlim(1, 100)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Neighbor rank")
    axis.set_ylabel("Tanimoto similarity")
    axis.set_title("Rank Profiles of the Three Selected Molecules", pad=14, weight="bold")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "03_selected_cases_rank_profiles")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.root / "processed" / "neighbors_top100.pt"
    artifact = torch.load(artifact_path, map_location="cpu")
    scores = artifact["scores"].numpy()
    indices = artifact["indices"].numpy()
    molecules = pd.read_parquet(args.root / "processed" / "molecules.parquet")
    if scores.shape != (50_000, 100):
        raise ValueError(f"Expected [50000, 100] scores, got {scores.shape}")

    all_pairs_path = args.output_dir / "all_unique_pairwise_histogram.csv"
    if all_pairs_path.exists():
        plot_all_pairs(pd.read_csv(all_pairs_path), args.output_dir)

    bins = np.arange(0.0, 1.0 + args.bin_width, args.bin_width)
    bins[-1] = 1.0 + np.finfo(np.float32).eps
    overall = histogram_frame(scores.ravel(), bins)
    overall.to_csv(args.output_dir / "overall_top100_histogram.csv", index=False)
    plot_overall(overall, scores.ravel(), args.output_dir)

    top50_values = scores[:, :50].ravel()
    top50 = histogram_frame(top50_values, bins)
    top50.to_csv(args.output_dir / "overall_top50_histogram.csv", index=False)
    plot_top50(top50, top50_values, args.output_dir)

    cases = choose_cases(scores)
    summaries: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    for case, molecule_id in cases.items():
        summaries.append(case_metrics(case, molecule_id, scores, molecules))
        frame = plot_case_histogram(case, molecule_id, scores, molecules, bins, args.output_dir)
        frame.to_csv(args.output_dir / f"case_{case}_histogram.csv", index=False)
        for rank, (neighbor_id, similarity) in enumerate(zip(indices[molecule_id], scores[molecule_id]), start=1):
            neighbor = molecules.iloc[int(neighbor_id)]
            neighbor_rows.append({
                "case": case,
                "query_molecule_id": molecule_id,
                "query_cid": str(molecules.iloc[molecule_id].cid),
                "rank": rank,
                "neighbor_molecule_id": int(neighbor_id),
                "neighbor_cid": str(neighbor.cid),
                "similarity": float(similarity),
                "neighbor_canonical_smiles": neighbor.canonical_smiles,
            })

    pd.DataFrame(summaries).to_csv(args.output_dir / "selected_case_summary.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(args.output_dir / "selected_case_top100_neighbors.csv", index=False)
    plot_case_panel(cases, scores, molecules, bins, args.output_dir)
    plot_rank_profiles(cases, scores, args.output_dir)

    selection = {
        "scope": "Both all 1,249,975,000 unique molecule pairs and the 5,000,000 saved self-excluded top-100 similarities.",
        "bin_width": args.bin_width,
        "top50": {
            "count": int(top50_values.size),
            "mean": float(top50_values.mean()),
            "median": float(np.median(top50_values)),
            "proposed_minimum_similarity": 0.25,
            "fraction_at_or_above_minimum": float(np.mean(top50_values >= 0.25)),
        },
        "criteria": {
            "isolated_extreme": "Minimum top-1 similarity; lower top-100 mean breaks ties.",
            "spiky_high": "Among top-1 >= 0.8 and <=20 neighbors >=0.8, maximum mean(rank 1-5) minus mean(rank 21-100).",
            "evenly_high": "Maximum rank-100 similarity; higher top-100 mean breaks ties.",
        },
        "cases": summaries,
        "source": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256(artifact_path),
            "molecule_order_sha256": artifact["molecule_order_sha256"],
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Tanimoto top-100 analysis\n\n"
        "This directory separately analyzes all 1,249,975,000 unique molecule pairs and the 5,000,000 "
        "saved self-excluded top-100 similarities. Histogram bins have width 0.05. Case-selection criteria and exact "
        "molecule metadata are recorded in `analysis_summary.json`; all plot values are also exported as CSV. "
        "The top-50 candidate distribution is provided separately with the proposed 0.25 minimum-similarity "
        "threshold marked on the plot.\n"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

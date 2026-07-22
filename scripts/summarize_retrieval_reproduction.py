#!/usr/bin/env python3
"""Create long-form and Table-1-style summaries from retrieval JSON files."""

import argparse
import csv
import json
from pathlib import Path


DATASET_LABELS = {
    "description": "Description",
    "pharmacodynamics": "Pharmacodynamics",
    "ATC": "ATC",
}
DIRECTION_LABELS = {
    "given_molecule": "Given Molecule",
    "given_text": "Given Text",
}
METHOD_ORDER = ["baseline", "curriculum", "stratified", "curriculum_alpha2"]
METHOD_LABELS = {
    "baseline": "baseline (alpha=1.0)",
    "curriculum": "curriculum (alpha=1.0)",
    "stratified": "stratified (alpha=1.0)",
    "curriculum_alpha2": "curriculum (alpha=2.0)",
}


def strategy_sort_key(strategy):
    try:
        return (METHOD_ORDER.index(strategy), strategy)
    except ValueError:
        return (len(METHOD_ORDER), strategy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("Reproduction_Evaluation/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("Reproduction_Evaluation"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_results = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("*.json"))]
    if not partial_results:
        raise RuntimeError(f"No JSON results found in {args.input_dir}")

    merged = {}
    for result in partial_results:
        strategy = result["strategy"]
        if strategy not in merged:
            merged[strategy] = {**result, "datasets": {}}
        target = merged[strategy]
        for dataset, payload in result["datasets"].items():
            if dataset in target["datasets"]:
                raise RuntimeError(f"Duplicate {strategy}/{dataset} result")
            target["datasets"][dataset] = payload
    results = [merged[strategy] for strategy in sorted(merged, key=strategy_sort_key)]
    required_datasets = set(DATASET_LABELS)
    for result in results:
        missing = required_datasets - set(result["datasets"])
        if missing:
            raise RuntimeError(f"Missing datasets for {result['strategy']}: {sorted(missing)}")

    canonical_raw_dir = args.output_dir / "raw"
    canonical_raw_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (canonical_raw_dir / f"{result['strategy']}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    long_rows = []
    for result in results:
        for dataset, payload in result["datasets"].items():
            for direction, by_T in payload["metrics"].items():
                for T, metric in by_T.items():
                    long_rows.append(
                        {
                            "method": result["strategy"],
                            "dataset": dataset,
                            "direction": direction,
                            "T": int(T),
                            "mean_accuracy": metric["mean"],
                            "std_accuracy": metric["std"],
                            "mean_percent": 100 * metric["mean"],
                            "std_percent": 100 * metric["std"],
                            "trials": json.dumps(metric["trials"]),
                        }
                    )
    long_path = args.output_dir / "retrieval_all_metrics.csv"
    with long_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=long_rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(long_rows)

    by_strategy = {result["strategy"]: result for result in results}
    delta_path = args.output_dir / "retrieval_deltas_vs_baseline.csv"
    if "baseline" in by_strategy:
        delta_rows = []
        baseline = by_strategy["baseline"]
        for strategy, result in by_strategy.items():
            if strategy == "baseline":
                continue
            for dataset in DATASET_LABELS:
                for direction in DIRECTION_LABELS:
                    for T in baseline["T_list"]:
                        key = str(T)
                        baseline_metric = baseline["datasets"][dataset]["metrics"][direction][key]
                        metric = result["datasets"][dataset]["metrics"][direction][key]
                        paired_deltas = [
                            100 * (value - base_value)
                            for value, base_value in zip(
                                metric["trials"], baseline_metric["trials"]
                            )
                        ]
                        delta_rows.append(
                            {
                                "method": strategy,
                                "dataset": dataset,
                                "direction": direction,
                                "T": T,
                                "baseline_percent": 100 * baseline_metric["mean"],
                                "method_percent": 100 * metric["mean"],
                                "delta_percentage_points": 100
                                * (metric["mean"] - baseline_metric["mean"]),
                                "paired_trial_deltas": json.dumps(paired_deltas),
                            }
                        )
        if delta_rows:
            with delta_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=delta_rows[0].keys(), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(delta_rows)

    columns = []
    for dataset in DATASET_LABELS:
        for direction in DIRECTION_LABELS:
            columns.append((dataset, direction))
    T_values = results[0]["T_list"]
    for result in results[1:]:
        if result["T_list"] != T_values:
            raise RuntimeError("All strategies must use the same T_list")

    table_paths = []
    for T in T_values:
        key = str(T)
        wide_rows = []
        for result in results:
            row = {"method": result["strategy"]}
            for dataset, direction in columns:
                metric = result["datasets"][dataset]["metrics"][direction][key]
                label = f"{DATASET_LABELS[dataset]} {DIRECTION_LABELS[direction]} @{T}"
                row[label] = 100 * metric["mean"]
                row[f"{label} std"] = 100 * metric["std"]
            wide_rows.append(row)

        wide_path = args.output_dir / f"table1_at{T}.csv"
        with wide_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=wide_rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(wide_rows)

        markdown_path = args.output_dir / f"table1_at{T}.md"
        headers = ["Method"] + [
            f"{DATASET_LABELS[dataset]}<br>{DIRECTION_LABELS[direction]} @{T}"
            for dataset, direction in columns
        ]
        lines = [
            f"# Zero-shot cross-modal retrieval (@{T})",
            "",
            "Values are five-trial mean accuracy (%) with population standard deviation.",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for result in results:
            cells = [METHOD_LABELS.get(result["strategy"], result["strategy"])]
            for dataset, direction in columns:
                metric = result["datasets"][dataset]["metrics"][direction][key]
                cells.append(f"{100 * metric['mean']:.2f} +/- {100 * metric['std']:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        markdown_path.write_text("\n".join(lines) + "\n")
        table_paths.extend([wide_path, markdown_path])

    print(f"Saved {long_path}")
    for path in table_paths:
        print(f"Saved {path}")
    if delta_path.exists():
        print(f"Saved {delta_path}")


if __name__ == "__main__":
    main()

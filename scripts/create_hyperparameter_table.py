#!/usr/bin/env python3
"""Render the completed AMOLE reproduction hyperparameter table."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("Reproduction_Evaluation/hyperparameter_setting")

ROWS = [
    ("Training epochs", "20"),
    (r"Learning rate for text encoder $f_{text}$", r"$1 \times 10^{-5}$"),
    (r"Learning rate for molecule encoder $f_{mol}$", r"$1 \times 10^{-5}$"),
    (r"Temperature for pseudo-label $\tau_1$", "0.1"),
    (r"Temperature for model prediction $\tau_2$", "0.1"),
    (r"Maximum number of similar molecules $k$", "50"),
    (r"Replacement probability $p$", "0.5"),
    (r"Weight of expertise reconstruction loss $\alpha$", "{1.0, 2.0}"),
    ("Global batch size", "30 (10 per GPU × 3 GPUs)"),
    ("Maximum text sequence length", "512 tokens"),
]


def write_csv():
    path = OUTPUT_DIR / "hyperparameter_setting.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Hyperparameter", "Value"])
        writer.writerows(ROWS)
    return path


def write_markdown():
    path = OUTPUT_DIR / "hyperparameter_setting.md"
    lines = [
        "# Hyperparameter specifications for AMOLE reproduction pretraining",
        "",
        "| Hyperparameter | Value |",
        "| --- | --- |",
    ]
    for parameter, value in ROWS:
        parameter = parameter.replace("$", "")
        value = value.replace("$", "")
        lines.append(f"| {parameter} | {value} |")
    lines.extend(
        [
            "",
            "The table reports the common pretraining settings and the evaluated ER-loss weights.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def render_table():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )
    figure = plt.figure(figsize=(15, 6.5), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()

    left, right = 0.075, 0.925
    value_x = 0.655
    top = 0.955
    header_y = 0.923
    header_rule_y = 0.893
    bottom_rule_y = 0.205

    axis.plot([left, right], [top, top], color="black", linewidth=2.4)
    axis.text(left + 0.018, header_y, "Hyperparameter", fontsize=22, va="center")
    axis.text(value_x, header_y, "Value", fontsize=22, va="center")
    axis.plot([left, right], [header_rule_y, header_rule_y], color="black", linewidth=1.35)

    row_top = 0.845
    row_bottom = 0.245
    spacing = (row_top - row_bottom) / (len(ROWS) - 1)
    for index, (parameter, value) in enumerate(ROWS):
        y = row_top - index * spacing
        axis.text(left + 0.018, y, parameter, fontsize=15.5, va="center")
        value_size = 13.0 if index == len(ROWS) - 1 else 15.0
        axis.text(value_x, y, value, fontsize=value_size, va="center")

    axis.plot([left, right], [bottom_rule_y, bottom_rule_y], color="black", linewidth=2.4)
    axis.text(
        left,
        0.105,
        "Table: Hyperparameter specifications for AMOLE reproduction pretraining.",
        fontsize=18,
        fontweight="bold",
        va="center",
    )

    png_path = OUTPUT_DIR / "hyperparameter_setting.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return (png_path,)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [write_csv(), write_markdown(), *render_table()]
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

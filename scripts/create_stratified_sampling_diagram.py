#!/usr/bin/env python3
"""Render an intuitive one-row diagram of stratified top-50 sampling."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path("Reproduction_Evaluation/hyperparameter_setting")
OUTPUT_PATH = OUTPUT_DIR / "stratified_sampling_diagram.png"

GROUPS = [
    {
        "name": "HIGH",
        "start": 1,
        "end": 10,
        "probability": "0.50",
        "color": "#C95A5A",
        "light": "#F5DADA",
    },
    {
        "name": "MID",
        "start": 11,
        "end": 40,
        "probability": "0.35",
        "color": "#D69E2E",
        "light": "#FAEBC8",
    },
    {
        "name": "LOW",
        "start": 41,
        "end": 50,
        "probability": "0.15",
        "color": "#4C78A8",
        "light": "#DCE9F5",
    },
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    figure = plt.figure(figsize=(16, 5), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    left, right = 0.045, 0.955
    axis.plot([left, right], [0.955, 0.955], color="black", linewidth=2.2)
    axis.text(
        0.5,
        0.895,
        "Similarity-aware Stratified Top-50 Sampling",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    axis.plot([left, right], [0.835, 0.835], color="black", linewidth=1.2)

    axis.annotate(
        "",
        xy=(right, 0.770),
        xytext=(left, 0.770),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#444444"},
    )
    axis.text(
        left,
        0.790,
        "Higher Tanimoto similarity",
        ha="left",
        va="bottom",
        fontsize=12.5,
        color="#333333",
    )
    axis.text(
        right,
        0.790,
        "Lower Tanimoto similarity",
        ha="right",
        va="bottom",
        fontsize=12.5,
        color="#333333",
    )

    gap = 0.0015
    total_width = right - left
    box_width = (total_width - gap * 49) / 50
    box_y, box_height = 0.505, 0.135

    for rank in range(1, 51):
        group = next(item for item in GROUPS if item["start"] <= rank <= item["end"])
        x = left + (rank - 1) * (box_width + gap)
        axis.add_patch(
            Rectangle(
                (x, box_y),
                box_width,
                box_height,
                facecolor=group["light"],
                edgecolor=group["color"],
                linewidth=1.0,
            )
        )
        axis.text(
            x + box_width / 2,
            box_y + box_height / 2,
            str(rank),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#222222",
        )

    for group in GROUPS:
        start_x = left + (group["start"] - 1) * (box_width + gap)
        end_x = left + group["end"] * box_width + (group["end"] - 1) * gap
        center_x = (start_x + end_x) / 2

        axis.plot(
            [start_x, end_x],
            [0.675, 0.675],
            color=group["color"],
            linewidth=5.0,
            solid_capstyle="butt",
        )
        axis.text(
            center_x,
            0.705,
            f'{group["name"]}  (ranks {group["start"]}–{group["end"]})',
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=group["color"],
        )

        bracket_y = 0.455
        axis.plot([start_x, end_x], [bracket_y, bracket_y], color=group["color"], linewidth=1.6)
        axis.plot(
            [start_x, start_x],
            [bracket_y, bracket_y + 0.018],
            color=group["color"],
            linewidth=1.6,
        )
        axis.plot(
            [end_x, end_x],
            [bracket_y, bracket_y + 0.018],
            color=group["color"],
            linewidth=1.6,
        )
        axis.text(
            center_x,
            0.385,
            f'P(select {group["name"].lower()}) = {group["probability"]}',
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=group["color"],
        )

    axis.text(
        0.5,
        0.275,
        "1) Select one group with the probability shown above."
        "    2) Sample one eligible molecule uniformly within that group.",
        ha="center",
        va="center",
        fontsize=13.5,
    )
    axis.text(
        0.5,
        0.205,
        r"Conditional on augmentation ($A_i=1$); candidates with Tanimoto similarity $<0.25$ are excluded.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#444444",
    )
    axis.text(
        0.5,
        0.145,
        "If the selected group has no eligible candidate, retain the original molecule.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#444444",
    )
    axis.plot([left, right], [0.080, 0.080], color="black", linewidth=2.2)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render curriculum sampling with fixed start/end rows and one expansion row."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path("Reproduction_Evaluation/hyperparameter_setting")
OUTPUT_PATH = OUTPUT_DIR / "curriculum_sampling_diagram.png"


def blend(color_a, color_b, amount):
    """Linearly blend two RGB colors; amount=0 returns color_a."""
    a = to_rgb(color_a)
    b = to_rgb(color_b)
    return tuple((1 - amount) * x + amount * y for x, y in zip(a, b))


def draw_boxes(
    axis,
    grid_left,
    grid_right,
    center_y,
    mode,
    active_color="#C95A5A",
    active_fill="#F5DADA",
    inactive_edge="#B7BDC5",
    inactive_fill="#F1F3F5",
):
    gap = 0.0012
    box_width = (grid_right - grid_left - gap * 49) / 50
    box_height = 0.105
    box_y = center_y - box_height / 2

    for rank in range(1, 51):
        if mode == "start":
            is_active = rank <= 10
            edge = active_color if is_active else inactive_edge
            fill = active_fill if is_active else inactive_fill
        elif mode == "final":
            edge, fill = active_color, active_fill
        else:
            if rank <= 10:
                edge, fill = active_color, active_fill
            else:
                progress = (rank - 11) / 39
                # Earlier-added ranks look more active; later-added ranks fade
                # toward the inactive color to depict a moving boundary.
                fade = 0.20 + 0.72 * progress
                edge = blend(active_color, inactive_edge, fade)
                fill = blend(active_fill, inactive_fill, fade)

        x = grid_left + (rank - 1) * (box_width + gap)
        axis.add_patch(
            Rectangle(
                (x, box_y),
                box_width,
                box_height,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.95,
            )
        )
        if rank in {1, 10, 14, 30, 50}:
            axis.text(
                x + box_width / 2,
                center_y,
                str(rank),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#222222" if mode == "final" or rank <= 10 else "#777777",
            )

    return box_width, gap, box_y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    figure = plt.figure(figsize=(16, 7), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    left, right = 0.055, 0.955
    grid_left, grid_right = 0.170, 0.925
    active_color = "#C95A5A"

    axis.plot([left, right], [0.960, 0.960], color="black", linewidth=2.2)
    axis.text(
        0.5,
        0.915,
        "Rank-expansion Curriculum: Top-$K(e)$ Sampling",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    axis.plot([left, right], [0.865, 0.865], color="black", linewidth=1.2)

    axis.annotate(
        "",
        xy=(grid_right, 0.815),
        xytext=(grid_left, 0.815),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#444444"},
    )
    axis.text(
        grid_left,
        0.832,
        "Higher Tanimoto similarity",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#333333",
    )
    axis.text(
        grid_right,
        0.832,
        "Lower Tanimoto similarity",
        ha="right",
        va="bottom",
        fontsize=12,
        color="#333333",
    )

    rows = [
        ("Epochs 1–5", 0.690, "start", r"$K(e)=10$"),
        ("Epochs 6–15", 0.485, "expansion", r"$K(e):\ 14\longrightarrow50$"),
        ("Epochs 16–20", 0.280, "final", r"$K(e)=50$"),
    ]

    axis.annotate(
        "",
        xy=(0.083, 0.205),
        xytext=(0.083, 0.755),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "#555555"},
    )
    axis.text(0.083, 0.772, "Training", ha="center", va="bottom", fontsize=12, color="#444444")
    axis.text(0.083, 0.185, "progress", ha="center", va="top", fontsize=12, color="#444444")

    for label, center_y, mode, k_label in rows:
        axis.text(
            0.145,
            center_y,
            label,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        box_width, gap, box_y = draw_boxes(
            axis, grid_left, grid_right, center_y, mode
        )
        axis.text(
            grid_right + 0.012,
            center_y,
            k_label,
            ha="left",
            va="center",
            fontsize=13.5,
            color=active_color,
            fontweight="bold",
        )

        if mode == "start":
            active_end = grid_left + 10 * box_width + 9 * gap
            axis.plot(
                [grid_left, active_end],
                [box_y - 0.014, box_y - 0.014],
                color=active_color,
                linewidth=3.0,
                solid_capstyle="butt",
            )
        elif mode == "final":
            axis.plot(
                [grid_left, grid_right],
                [box_y - 0.014, box_y - 0.014],
                color=active_color,
                linewidth=3.0,
                solid_capstyle="butt",
            )
        else:
            rank10_end = grid_left + 10 * box_width + 9 * gap
            axis.annotate(
                "",
                xy=(grid_right, box_y - 0.020),
                xytext=(rank10_end + gap, box_y - 0.020),
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": 3.2,
                    "color": "#D47A7A",
                    "mutation_scale": 14,
                },
            )
            axis.text(
                (rank10_end + grid_right) / 2,
                box_y - 0.055,
                "+4 ranks per epoch",
                ha="center",
                va="top",
                fontsize=11.5,
                color="#A84B4B",
                fontweight="bold",
            )

    axis.text(
        0.5,
        0.112,
        r"At each epoch, sample uniformly from the active prefix $\{1,\ldots,K(e)\}$.",
        ha="center",
        va="center",
        fontsize=13.5,
    )
    axis.text(
        0.5,
        0.068,
        r"The independent augmentation probability remains $p_{\mathrm{aug}}=0.5$.",
        ha="center",
        va="center",
        fontsize=12,
        color="#444444",
    )
    axis.plot([left, right], [0.025, 0.025], color="black", linewidth=2.2)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

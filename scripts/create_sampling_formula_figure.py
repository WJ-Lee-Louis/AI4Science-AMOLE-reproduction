#!/usr/bin/env python3
"""Render intuitive curriculum and stratified top-k sampling equations."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("Reproduction_Evaluation/hyperparameter_setting")
OUTPUT_PATH = OUTPUT_DIR / "topk_sampling_formulas.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    figure = plt.figure(figsize=(14, 9.2), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()

    left, right = 0.065, 0.935
    axis.plot([left, right], [0.955, 0.955], color="black", linewidth=2.2)
    axis.text(
        0.5,
        0.916,
        "Top-$k$ Molecule Subset Sampling",
        fontsize=25,
        ha="center",
        va="center",
        fontweight="bold",
    )
    axis.plot([left, right], [0.875, 0.875], color="black", linewidth=1.2)

    axis.text(
        left,
        0.825,
        r"$G_i^{(r)}$: the $r$-th most similar, self-excluded neighbor of molecule $G_i$",
        fontsize=17,
        va="center",
    )
    axis.text(
        left,
        0.770,
        r"$A_i\sim\mathrm{Bernoulli}(0.5)$"
        r"$\quad\Rightarrow\quad A_i=0:\ \widetilde{G}_i=G_i$",
        fontsize=18,
        va="center",
    )

    axis.plot([left, right], [0.720, 0.720], color="#555555", linewidth=0.9)
    axis.text(
        left,
        0.678,
        "1. Rank-expansion curriculum",
        fontsize=20,
        fontweight="bold",
        va="center",
    )

    axis.text(0.135, 0.550, r"$K(e)=$", fontsize=25, va="center")
    axis.text(0.255, 0.550, "{", fontsize=104, va="center", ha="center")
    axis.text(0.300, 0.610, r"$10,$", fontsize=22, va="center")
    axis.text(0.505, 0.610, r"$1\leq e\leq5$", fontsize=20, va="center")
    axis.text(0.300, 0.550, r"$10+4(e-5),$", fontsize=22, va="center")
    axis.text(0.505, 0.550, r"$6\leq e\leq15$", fontsize=20, va="center")
    axis.text(0.300, 0.490, r"$50,$", fontsize=22, va="center")
    axis.text(0.505, 0.490, r"$16\leq e\leq20$", fontsize=20, va="center")

    axis.text(
        left + 0.025,
        0.405,
        r"$A_i=1:\quad r\sim\mathrm{Uniform}\{1,\ldots,K(e)\},"
        r"\qquad \widetilde{G}_i=G_i^{(r)}$",
        fontsize=20,
        va="center",
    )
    axis.text(
        left + 0.025,
        0.365,
        "The uniformly sampled rank range expands from top-10 to top-50.",
        fontsize=12.5,
        color="#444444",
        va="center",
    )

    axis.plot([left, right], [0.325, 0.325], color="#555555", linewidth=0.9)
    axis.text(
        left,
        0.284,
        "2. Similarity-aware stratified sampling",
        fontsize=20,
        fontweight="bold",
        va="center",
    )
    axis.text(
        left + 0.025,
        0.230,
        r"$H_i=\{G_i^{(1)},\ldots,G_i^{(10)}\},\quad"
        r"M_i=\{G_i^{(11)},\ldots,G_i^{(40)}\},\quad"
        r"L_i=\{G_i^{(41)},\ldots,G_i^{(50)}\}$",
        fontsize=18,
        va="center",
    )
    axis.text(
        left + 0.025,
        0.187,
        r"Keep only candidates with Tanimoto similarity $\geq 0.25$ in each group.",
        fontsize=12.5,
        color="#444444",
        va="center",
    )
    axis.text(
        left + 0.025,
        0.137,
        r"$A_i=1:\quad P(Z_i=H_i)=0.50,\quad"
        r"P(Z_i=M_i)=0.35,\quad P(Z_i=L_i)=0.15$",
        fontsize=17,
        va="center",
    )
    axis.text(
        left + 0.025,
        0.082,
        r"$Z_i\neq\varnothing:\ \widetilde{G}_i\sim\mathrm{Uniform}(Z_i)"
        r"\qquad;\qquad Z_i=\varnothing:\ \widetilde{G}_i=G_i$",
        fontsize=18,
        va="center",
    )
    axis.plot([left, right], [0.035, 0.035], color="black", linewidth=2.2)

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

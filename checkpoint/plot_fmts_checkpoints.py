#!/usr/bin/env python3
"""Render the corrected FMTS checkpoint audit as a compact two-panel figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("artifacts/fmts-checkpoints/full_results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fmts-checkpoints/fmts_checkpoint_audit"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    rows = payload["rows"]
    colors = {256: "#8C564B", 4096: "#1F77B4", 8192: "#2CA02C", 16384: "#9467BD"}
    markers = {3: "o", 7: "s", 11: "^", 15: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for row in rows:
        m = int(row["M"])
        k = int(row["prefix_k"])
        competent = float(row["noisy_risk_ratio"]) < 1.2
        x, y = float(row["G_mean"]), float(row["S"])
        xci, yci = row["G_mean_ci95"], row["S_ci95"]
        axes[0].errorbar(
            x,
            y,
            xerr=[[x - xci[0]], [xci[1] - x]],
            yerr=[[y - yci[0]], [yci[1] - y]],
            color=colors[m],
            marker=markers[k],
            markerfacecolor=colors[m] if competent else "white",
            markeredgewidth=1.2,
            capsize=2,
            linestyle="none",
            alpha=0.9,
        )
        axes[1].plot(
            k,
            row["noisy_risk_ratio"],
            color=colors[m],
            marker=markers[k],
            markerfacecolor=colors[m] if competent else "white",
            markeredgewidth=1.2,
            linestyle="none",
        )

    axes[0].axhline(0, color="#555555", linewidth=0.9)
    axes[0].axvline(3, color="#555555", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel(r"coverage gap $G=\log_2 N_2-\log_2 M$")
    axes[0].set_ylabel(r"morphology $S=(\Delta_D-\Delta_C)/(\Delta_D+\Delta_C)$")
    axes[0].set_title("Prediction morphology on shared unseen tasks")
    axes[0].grid(alpha=0.18)

    for m, color in colors.items():
        points = sorted((row for row in rows if int(row["M"]) == m), key=lambda row: row["prefix_k"])
        axes[1].plot(
            [row["prefix_k"] for row in points],
            [row["noisy_risk_ratio"] for row in points],
            color=color,
            linewidth=1.4,
            label=f"M={m:,}",
        )
    axes[1].axhline(1.2, color="#555555", linewidth=0.9, linestyle="--", label="competence gate")
    axes[1].set_xlabel("observed-label prefix k")
    axes[1].set_ylabel("Transformer / ridge noisy risk")
    axes[1].set_title("Continuous-Bayes competence")
    axes[1].set_xticks(sorted(markers))
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Eight-layer interrupted checkpoints; 4,096 identical iid Gaussian tasks", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()

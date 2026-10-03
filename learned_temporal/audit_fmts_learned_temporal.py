#!/usr/bin/env python3
"""Frozen endpoint analysis for the learned temporal coverage experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import fmts_learned_temporal as experiment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def ratio_interval(numerator: np.ndarray, denominator: np.ndarray,
                   indices: np.ndarray) -> dict[str, float]:
    numerator_means = numerator[indices].mean(axis=1)
    denominator_means = denominator[indices].mean(axis=1)
    draws = numerator_means / np.maximum(denominator_means, 1e-30)
    return {
        "mean": float(numerator.mean() / max(denominator.mean(), 1e-30)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def morphology_interval(discrete: np.ndarray, continuous: np.ndarray,
                        indices: np.ndarray) -> dict[str, float]:
    discrete_means = discrete[indices].mean(axis=1)
    continuous_means = continuous[indices].mean(axis=1)
    draws = (discrete_means - continuous_means) / np.maximum(
        discrete_means + continuous_means, 1e-30
    )
    return {
        "mean": float((discrete.mean() - continuous.mean())
                      / max(discrete.mean() + continuous.mean(), 1e-30)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def load_prediction(root: Path, M: int) -> tuple[np.ndarray, dict]:
    arm = root / f"M{M}"
    summary = json.loads((arm / "summary.json").read_text())
    if summary["phase"] != "complete" or int(summary["step"]) != experiment.ENDPOINT:
        raise RuntimeError(f"M={M} is not complete")
    identity = summary["identity"]
    if identity["assets_sha256"] != sha256(root / "assets.pt"):
        raise RuntimeError(f"M={M} assets hash mismatch")
    if identity["source_sha256"] != sha256(Path(experiment.__file__)):
        raise RuntimeError(f"M={M} source hash mismatch")
    if identity["preregistration_sha256"] != sha256(experiment.PREREG):
        raise RuntimeError(f"M={M} preregistration hash mismatch")
    prediction_path = arm / f"predictions_step_{experiment.ENDPOINT:06d}.npz"
    if summary["endpoint_predictions_sha256"] != sha256(prediction_path):
        raise RuntimeError(f"M={M} endpoint prediction hash mismatch")
    prediction = np.load(prediction_path)["predictions"]
    if prediction.shape != (experiment.EVAL_COUNT, experiment.PREDICTIONS, experiment.DIM):
        raise RuntimeError(f"M={M} endpoint prediction shape mismatch")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"M={M} non-finite predictions")
    return prediction, summary


def audit(root256: Path, root4096: Path, output: Path) -> dict:
    assets_path = root4096 / "assets.pt"
    assets = torch.load(assets_path, map_location="cpu", weights_only=True)
    if sha256(root256 / "assets.pt") != sha256(assets_path):
        raise RuntimeError("the two arms do not use identical frozen assets")
    states = assets["eval_states"]
    basis = assets["basis"]
    continuous, log2_n2 = experiment.continuous_bayes(states, basis, 96)
    fine, fine_n2 = experiment.continuous_bayes(states[:128], basis, 128)
    quadrature = {
        "prediction_max_abs_96_vs_128": float(torch.max(torch.abs(continuous[:128] - fine))),
        "log2_N2_max_abs_96_vs_128": float(torch.max(torch.abs(log2_n2[:128] - fine_n2))),
    }
    target = torch.stack([states[:, prefix + 1] for prefix in experiment.PREFIXES], dim=1).numpy()
    continuous_np = continuous.numpy()
    log2_n2_np = log2_n2.numpy()
    rng = np.random.default_rng(experiment.seed_for("bootstrap"))
    bootstrap_indices = rng.integers(0, len(states), size=(2000, len(states)))
    results: dict[str, dict] = {}

    for M in (256, 4096):
        root = root256 if M == 256 else root4096
        prediction, summary = load_prediction(root, M)
        selected_prediction = prediction[:, list(experiment.PREFIXES)]
        discrete = experiment.discrete_bayes(states, basis, assets["library"][:M]).numpy()
        per_prompt_transformer_risk = np.mean((selected_prediction - target) ** 2, axis=2)
        per_prompt_continuous_risk = np.mean((continuous_np - target) ** 2, axis=2)
        distance_continuous = np.mean((selected_prediction - continuous_np) ** 2, axis=2)
        distance_discrete = np.mean((selected_prediction - discrete) ** 2, axis=2)
        gap = log2_n2_np - np.log2(M)
        prefix_rows = []
        for index, prefix in enumerate(experiment.PREFIXES):
            risk_ratio = ratio_interval(per_prompt_transformer_risk[:, index],
                                        per_prompt_continuous_risk[:, index],
                                        bootstrap_indices)
            prefix_rows.append({
                "prefix": prefix,
                "transformer_risk": interval(per_prompt_transformer_risk[:, index], bootstrap_indices),
                "continuous_bayes_risk": interval(per_prompt_continuous_risk[:, index], bootstrap_indices),
                "risk_ratio": risk_ratio,
                "distance_to_continuous": interval(distance_continuous[:, index], bootstrap_indices),
                "distance_to_discrete": interval(distance_discrete[:, index], bootstrap_indices),
                "S": morphology_interval(distance_discrete[:, index],
                                           distance_continuous[:, index],
                                           bootstrap_indices),
                "G_bits": interval(gap[:, index], bootstrap_indices),
            })
        competence = all(row["risk_ratio"]["mean"] < 1.2 for row in prefix_rows)
        certificate = (
            prefix_rows[-1]["S"]["lower"] > 0
            and prefix_rows[-1]["G_bits"]["lower"] > 3
        )
        results[str(M)] = {
            "summary": summary,
            "prefixes": prefix_rows,
            "competence_all_prefixes": competence,
            "coverage_certificate_k15": certificate,
        }

    high = results["4096"]
    if not high["competence_all_prefixes"]:
        outcome = "temporal_training_failure"
    elif high["coverage_certificate_k15"]:
        outcome = "temporal_generalization_beyond_retrieval"
    else:
        outcome = "library_resolved_or_no_certificate"
    transition = (
        not results["256"]["competence_all_prefixes"]
        and results["4096"]["competence_all_prefixes"]
    )
    report = {
        "stage": "fmts_learned_temporal_coverage_v1",
        "quadrature": quadrature,
        "arms": results,
        "frozen_outcome": outcome,
        "transition_evidence": transition,
        "assets_sha256": sha256(assets_path),
        "source_sha256": sha256(Path(experiment.__file__)),
        "preregistration_sha256": sha256(experiment.PREREG),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "learned_temporal_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    colors = {256: "#D55E00", 4096: "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for M in (256, 4096):
        rows = results[str(M)]["prefixes"]
        prefixes = [row["prefix"] for row in rows]
        ratios = np.array([row["risk_ratio"]["mean"] for row in rows])
        ratio_low = np.array([row["risk_ratio"]["lower"] for row in rows])
        ratio_high = np.array([row["risk_ratio"]["upper"] for row in rows])
        axes[0].errorbar(prefixes, ratios, yerr=[ratios - ratio_low, ratio_high - ratios],
                         marker="o", color=colors[M], capsize=2, label=f"M={M:,}")
        s_mean = np.array([row["S"]["mean"] for row in rows])
        s_low = np.array([row["S"]["lower"] for row in rows])
        s_high = np.array([row["S"]["upper"] for row in rows])
        g_mean = np.array([row["G_bits"]["mean"] for row in rows])
        g_low = np.array([row["G_bits"]["lower"] for row in rows])
        g_high = np.array([row["G_bits"]["upper"] for row in rows])
        axes[1].errorbar(g_mean, s_mean, xerr=[g_mean - g_low, g_high - g_mean],
                         yerr=[s_mean - s_low, s_high - s_mean], marker="o",
                         color=colors[M], capsize=2, label=f"M={M:,}")
        for x, y, prefix in zip(g_mean, s_mean, prefixes):
            axes[1].annotate(str(prefix), (x, y), xytext=(3, 2),
                             textcoords="offset points", fontsize=7)
    axes[0].axhline(1.2, color="0.45", ls="--", lw=1)
    axes[0].set(xlabel="context transitions $k$", ylabel="risk / continuous Bayes risk")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="0.55", lw=1)
    axes[1].axvline(3, color="0.55", ls="--", lw=1)
    axes[1].set(xlabel="$G=\\log_2 N_2-\\log_2 M$ (bits)", ylabel="continuous-vs-library score $S$")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "learned_temporal_result.pdf", bbox_inches="tight")
    fig.savefig(output / "learned_temporal_result.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m256-root", type=Path, required=True)
    parser.add_argument("--m4096-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.m256_root.resolve(), args.m4096_root.resolve(), args.output.resolve())
    print(json.dumps({"frozen_outcome": report["frozen_outcome"],
                      "transition_evidence": report["transition_evidence"],
                      "quadrature": report["quadrature"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

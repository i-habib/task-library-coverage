#!/usr/bin/env python3
"""Verify GRU snapshots and apply the frozen architecture-comparison decision."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import fmts_checkpoint_audit as checkpoint


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/fmts-architecture-baseline"
GRU_ARCHIVES = {replicate: ARTIFACT / f"snapshots/r{replicate}-final.zip" for replicate in (1, 2)}
TRANSFORMER_ARCHIVES = {
    replicate: ROOT / f"artifacts/fmts-m4096-replication/snapshots/r{replicate}-final.zip"
    for replicate in (1, 2)
}
PROMPTS = ROOT / "artifacts/fmts-checkpoints/local-reference/fresh_unseen_prompts.npz"
PREFIXES = (3, 7, 11, 15)
BOOTSTRAPS = 2000


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def verified_payloads(path: Path) -> tuple[dict[str, bytes], str]:
    blob = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failure: {path}")
        manifest = json.loads(archive.read("snapshot_manifest.json"))
        payloads = {}
        for name, expected in manifest.items():
            payload = archive.read(name)
            if sha256(payload) != expected:
                raise RuntimeError(f"manifest mismatch: {path}:{name}")
            payloads[name] = payload
    return payloads, sha256(blob)


def bootstrap(values: dict[str, np.ndarray], seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    count = len(values["lt"])
    ratios, morphologies, gaps = [], [], []
    for start in range(0, BOOTSTRAPS, 128):
        batch = min(128, BOOTSTRAPS - start)
        indices = rng.integers(0, count, size=(batch, count), dtype=np.int32)
        means = {key: array[indices].mean(axis=1) for key, array in values.items()}
        ratios.append(means["lt"] / means["lc"])
        morphologies.append((means["dd"] - means["dc"]) /
                            np.maximum(means["dd"] + means["dc"], 1e-30))
        gaps.append(means["g"])
    return {
        "noisy_risk_ratio_ci95": np.quantile(np.concatenate(ratios), [0.025, 0.975]).tolist(),
        "S_ci95": np.quantile(np.concatenate(morphologies), [0.025, 0.975]).tolist(),
        "G_mean_ci95": np.quantile(np.concatenate(gaps), [0.025, 0.975]).tolist(),
    }


def point_metrics(prediction: np.ndarray, target: np.ndarray, ridge: np.ndarray,
                  discrete: np.ndarray, gap: np.ndarray, seed: int) -> dict:
    values = {
        "lt": (prediction - target) ** 2,
        "lc": (ridge - target) ** 2,
        "dc": (prediction - ridge) ** 2,
        "dd": (prediction - discrete) ** 2,
        "g": gap,
    }
    means = {key: float(value.mean()) for key, value in values.items()}
    result = {
        "noisy_risk_ratio": means["lt"] / means["lc"],
        "Delta_C": means["dc"],
        "Delta_D": means["dd"],
        "S": (means["dd"] - means["dc"]) / max(means["dd"] + means["dc"], 1e-30),
        "G_mean": means["g"],
    }
    result.update(bootstrap(values, seed))
    return result


def load_prompt_bank() -> dict[str, np.ndarray]:
    with np.load(PROMPTS, allow_pickle=False) as saved:
        prompts = {key: saved[key] for key in saved.files}
    if set(prompts) != {"data", "tasks", "targets", "signal"}:
        raise RuntimeError("unexpected prompt schema")
    return prompts


def evaluate_replicate(replicate: int, prompts: dict[str, np.ndarray]) -> dict:
    gru_payloads, gru_archive_hash = verified_payloads(GRU_ARCHIVES[replicate])
    transformer_payloads, transformer_archive_hash = verified_payloads(TRANSFORMER_ARCHIVES[replicate])
    gru_root = f"results/replicate_{replicate}/GRU_M4096"
    transformer_root = f"results/replicate_{replicate}/M4096"
    summary = json.loads(gru_payloads[f"{gru_root}/summary.json"])
    if summary["phase"] != "complete" or int(summary["step"]) != 290000:
        raise RuntimeError(f"GRU replicate {replicate} did not finish at the frozen endpoint")
    gru_saved = np.load(io.BytesIO(gru_payloads[f"{gru_root}/predictions_step_290000.npz"]),
                        allow_pickle=False)
    transformer_saved = np.load(
        io.BytesIO(transformer_payloads[f"{transformer_root}/predictions_step_290000.npz"]),
        allow_pickle=False,
    )
    gru_prediction = gru_saved["predictions"].astype(np.float64)
    transformer_prediction = transformer_saved["predictions"].astype(np.float64)
    gru_library_blob = gru_payloads["task_library_M4096.pt"]
    transformer_library_blob = transformer_payloads[f"results/replicate_{replicate}/task_library_M4096.pt"]
    gru_library = torch.load(io.BytesIO(gru_library_blob), map_location="cpu", weights_only=True)
    transformer_library = torch.load(io.BytesIO(transformer_library_blob), map_location="cpu", weights_only=True)
    if not torch.equal(gru_library, transformer_library):
        raise RuntimeError(f"replicate {replicate} did not reuse the exact Transformer library")

    data = torch.from_numpy(prompts["data"])
    targets = torch.from_numpy(prompts["targets"])
    rows = []
    for prefix in PREFIXES:
        ridge = checkpoint.exact_ridge_predictions(data, targets, prefix)
        discrete = checkpoint.exact_atomic_predictions(
            data[:, :prefix], targets[:, :prefix], data[:, prefix], gru_library,
            task_chunk=4096, prompt_chunk=64,
        )
        log2_n2 = checkpoint.exact_log2_n2_batch(
            prompts["data"][:, :prefix], prompts["targets"][:, :prefix]
        )
        gap = log2_n2 - math.log2(len(gru_library))
        target = prompts["targets"][:, prefix].astype(np.float64)
        for architecture, prediction in (
            ("Transformer", transformer_prediction[:, prefix]),
            ("GRU", gru_prediction[:, prefix]),
        ):
            metrics = point_metrics(
                prediction, target, ridge, discrete, gap,
                seed=2026091500 + 100 * replicate + 10 * prefix + (architecture == "GRU"),
            )
            rows.append({"replicate": replicate, "architecture": architecture,
                         "prefix_k": prefix, **metrics})
    gru_rows = [row for row in rows if row["architecture"] == "GRU"]
    competent = all(row["noisy_risk_ratio"] < 1.2 for row in gru_rows)
    final = next(row for row in gru_rows if row["prefix_k"] == 15)
    if competent and final["S_ci95"][0] > 0:
        category = "architecture_general"
    elif not competent and final["S_ci95"][1] <= 0:
        category = "recurrent_retrieval_like"
    else:
        category = "optimization_or_morphology_mismatch"
    return {
        "replicate": replicate,
        "gru_archive_sha256": gru_archive_hash,
        "transformer_archive_sha256": transformer_archive_hash,
        "exact_library_reuse": True,
        "parameter_count": summary["identity"]["parameter_count"],
        "causal_query_label_max_abs": summary["causal_query_label_max_abs"],
        "competent_all_prefixes": competent,
        "frozen_category": category,
        "rows": rows,
    }


def make_plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    colors = {"Transformer": "#1764A3", "GRU": "#C96113"}
    styles = {1: "-", 2: "--"}
    for architecture in ("Transformer", "GRU"):
        for replicate in (1, 2):
            selected = sorted(
                (row for row in rows if row["architecture"] == architecture
                 and row["replicate"] == replicate),
                key=lambda row: row["prefix_k"],
            )
            label = f"{architecture}, seed {replicate}"
            axes[0].plot([row["prefix_k"] for row in selected],
                         [row["noisy_risk_ratio"] for row in selected], marker="o",
                         color=colors[architecture], linestyle=styles[replicate], label=label)
            axes[1].plot([row["prefix_k"] for row in selected],
                         [row["S"] for row in selected], marker="o",
                         color=colors[architecture], linestyle=styles[replicate], label=label)
    axes[0].axhline(1.2, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel("model / ridge noisy risk")
    axes[0].set_title("Continuous-Bayes competence")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel(r"morphology $S$ (continuous $\leftarrow$ $\rightarrow$ retrieval)")
    axes[1].set_title("Prediction morphology")
    for axis in axes:
        axis.set_xlabel("observed-label prefix k")
        axis.set_xticks(PREFIXES)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(ARTIFACT / "architecture_comparison.pdf")
    fig.savefig(ARTIFACT / "architecture_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    prompts = load_prompt_bank()
    results = [evaluate_replicate(replicate, prompts) for replicate in (1, 2)]
    categories = [result["frozen_category"] for result in results]
    if categories == ["architecture_general", "architecture_general"]:
        outcome = "architecture_general"
        interpretation = "Both GRUs learn the continuous rule beyond literal library support; the effect is not Transformer-specific here."
    elif categories == ["recurrent_retrieval_like", "recurrent_retrieval_like"]:
        outcome = "recurrent_retrieval_like"
        interpretation = "Both GRUs remain closer to finite-library Bayes and fail competence under the matched schedule; this is architecture-comparative evidence, not a universal theorem."
    elif categories[0] == categories[1]:
        outcome = "optimization_or_morphology_mismatch"
        interpretation = "The recurrent models differ in competence or morphology without becoming retrieval-like; do not attribute the result specifically to retrieval."
    else:
        outcome = "mixed"
        interpretation = "The recurrent comparison is seed-sensitive and inconclusive."
    rows = [row for result in results for row in result["rows"]]
    payload = {"status": "complete", "frozen_outcome": outcome,
               "interpretation": interpretation, "replicates": results, "rows": rows,
               "limits": [
                   "Matched parameters and schedules do not equalize optimization difficulty.",
                   "Two seeds do not establish a universal architecture effect.",
                   "A competence failure is called retrieval-like only when S also favors finite-library Bayes.",
               ]}
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "architecture_comparison_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (ARTIFACT / "architecture_comparison_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "replicate", "architecture", "prefix_k", "noisy_risk_ratio", "S", "G_mean",
            "Delta_C", "Delta_D",
        ])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    make_plot(rows)
    lines = ["# Parameter-matched recurrent comparison", "",
             f"Frozen outcome: **{outcome}**.", "", interpretation, "",
             "| seed | architecture | k=3 risk | k=7 risk | k=11 risk | k=15 risk | k=15 S |",
             "|---:|:---|---:|---:|---:|---:|---:|"]
    for replicate in (1, 2):
        for architecture in ("Transformer", "GRU"):
            selected = sorted((row for row in rows if row["replicate"] == replicate
                               and row["architecture"] == architecture), key=lambda row: row["prefix_k"])
            lines.append(f"| {replicate} | {architecture} | "
                         + " | ".join(f"{row['noisy_risk_ratio']:.4f}" for row in selected)
                         + f" | {selected[-1]['S']:.4f} |")
    lines.extend(["", "The task libraries and held-out prompts are byte-identical within each paired seed.",
                  "The comparison uses a four-layer, 256-hidden-unit GRU with roughly 1.48M parameters.",
                  ""])
    (ARTIFACT / "FMTS_ARCHITECTURE_BASELINE_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"outcome": outcome, "categories": categories,
                      "parameter_counts": [result["parameter_count"] for result in results]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

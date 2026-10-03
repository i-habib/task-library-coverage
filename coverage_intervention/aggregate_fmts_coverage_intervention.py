#!/usr/bin/env python3
"""Aggregate frozen shards from the FMTS coverage intervention."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fmts-coverage-intervention-mpl")

import numpy as np

import fmts_temporal as temporal


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / "fmts_coverage_intervention_preregistration.json"
WORKER_SOURCE = ROOT / "fmts_coverage_intervention.py"
DEFAULT_OUTPUT = ROOT / "artifacts/fmts-coverage-intervention"
VERSION = "FMTS-coverage-intervention-v1"
BASE_SEED = 2026091501
N_VALUES = (16, 64, 256, 1024)
TARGET_R = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
HORIZONS = (1, 4, 16)
CONTEXTS_PER_N = 256
LIBRARY_REPLICATES = 32
BOOTSTRAP_REPLICATES = 2000
QUADRATURE_ORDER = 96


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def config(smoke: bool = False) -> dict:
    return {
        "version": VERSION,
        "base_seed": BASE_SEED,
        "n_values": list((16,) if smoke else N_VALUES),
        "target_R": list((0.5, 2.0, 8.0) if smoke else TARGET_R),
        "horizons": list((1, 16) if smoke else HORIZONS),
        "contexts_per_n": 8 if smoke else CONTEXTS_PER_N,
        "library_replicates": 3 if smoke else LIBRARY_REPLICATES,
        "bootstrap_replicates": 100 if smoke else BOOTSTRAP_REPLICATES,
        "quadrature_order": 48 if smoke else QUADRATURE_ORDER,
        "a_max": temporal.A_MAX,
    }


def make_context(n: int, context: int, quadrature_order: int) -> dict:
    rng = np.random.default_rng(stable_seed(BASE_SEED, "target", n, context))
    a_true = float(rng.uniform(-temporal.A_MAX, temporal.A_MAX))
    x = np.zeros(n + 1, dtype=float)
    for index, innovation in enumerate(rng.standard_normal(n)):
        x[index + 1] = a_true * x[index] + innovation
    sxx = float(np.dot(x[:-1], x[:-1]))
    sxy = float(np.dot(x[:-1], x[1:]))
    posterior = temporal.posterior_from_sufficient(sxx, sxy, quadrature_order)
    return {
        "n": n,
        "context": context,
        "a_true": a_true,
        "x_n": float(x[-1]),
        "sxx": sxx,
        "sxy": sxy,
        "posterior": posterior,
        "n2": math.exp(float(posterior["log_n2"])),
    }


def evaluate_library(context: dict, atoms: np.ndarray, horizons: tuple[int, ...]) -> dict:
    posterior = context["posterior"]
    log_weights = context["sxy"] * atoms - 0.5 * context["sxx"] * atoms * atoms
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)
    weights /= weights.sum()
    discrete_mean_a = float(np.dot(weights, atoms))
    parameter_error = (discrete_mean_a - float(posterior["mean_a"])) ** 2
    posterior_variance = max(float(posterior["var_a"]), np.finfo(float).tiny)
    forecasts = {}
    for horizon in horizons:
        continuous_mean, continuous_variance = temporal.predictive_moments(
            posterior["posterior_weights"], posterior["a"], context["x_n"], horizon
        )
        discrete_mean, _ = temporal.predictive_moments(weights, atoms, context["x_n"], horizon)
        error = (discrete_mean - continuous_mean) ** 2
        forecasts[horizon] = {
            "forecast_mean_error_sq": error,
            "forecast_mean_error_sq_scaled": error / max(continuous_variance, np.finfo(float).tiny),
        }
    return {
        "posterior_mean_error_sq": parameter_error,
        "posterior_mean_error_sq_scaled": parameter_error / posterior_variance,
        "ess": 1.0 / float(np.sum(weights * weights)),
        "maxweight": float(np.max(weights)),
        "forecasts": forecasts,
    }


def run_n(n: int, cfg: dict, output: Path) -> dict:
    started = time.time()
    contexts = [make_context(n, index, cfg["quadrature_order"]) for index in range(cfg["contexts_per_n"])]
    n2 = np.asarray([item["n2"] for item in contexts])
    permutation_rng = np.random.default_rng(stable_seed(BASE_SEED, "shuffle", n))
    identity = np.arange(len(contexts))
    permutation = permutation_rng.permutation(len(contexts))
    while np.any(permutation == identity):
        permutation = permutation_rng.permutation(len(contexts))
    shuffled_n2 = n2[permutation]
    rows: list[dict[str, int | float | str]] = []
    targets = tuple(float(value) for value in cfg["target_R"])
    horizons = tuple(int(value) for value in cfg["horizons"])
    for context_index, current in enumerate(contexts):
        true_sizes = [max(1, int(math.ceil(target * current["n2"]))) for target in targets]
        shuffled_sizes = [max(1, int(math.ceil(target * shuffled_n2[context_index]))) for target in targets]
        maximum_size = max(true_sizes + shuffled_sizes)
        for replicate in range(cfg["library_replicates"]):
            rng = np.random.default_rng(stable_seed(BASE_SEED, "library", n, context_index, replicate))
            library = rng.uniform(-temporal.A_MAX, temporal.A_MAX, maximum_size)
            for assignment, assigned_n2, sizes in (
                ("true", current["n2"], true_sizes),
                ("shuffled", float(shuffled_n2[context_index]), shuffled_sizes),
            ):
                for target, size in zip(targets, sizes):
                    result = evaluate_library(current, library[:size], horizons)
                    common = {
                        "n": n,
                        "context": context_index,
                        "replicate": replicate,
                        "assignment": assignment,
                        "target_R": target,
                        "M": size,
                        "N2_context": current["n2"],
                        "N2_assigned": assigned_n2,
                        "achieved_R": size / current["n2"],
                        "achieved_log2_R": math.log2(size / current["n2"]),
                        "posterior_mean_error_sq": result["posterior_mean_error_sq"],
                        "posterior_mean_error_sq_scaled": result["posterior_mean_error_sq_scaled"],
                        "ESS": result["ess"],
                        "maxweight": result["maxweight"],
                    }
                    rows.append({
                        **common,
                        **{
                            f"{metric}_h{horizon}": value
                            for horizon in horizons
                            for metric, value in result["forecasts"][horizon].items()
                        },
                    })
        if (context_index + 1) % 32 == 0:
            print(f"n={n}: {context_index + 1}/{len(contexts)} contexts", flush=True)
    fieldnames = list(rows[0])
    csv_path = output / f"rows_n{n}.csv.gz"
    temporary = output / f"rows_n{n}.csv.gz.tmp"
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    metadata = {
        "status": "complete",
        "version": VERSION,
        "n": n,
        "configuration": cfg,
        "rows": len(rows),
        "contexts": len(contexts),
        "elapsed_seconds": time.time() - started,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "temporal_source_sha256": sha256_file(Path(temporal.__file__).resolve()),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "rows_sha256": sha256_file(csv_path),
        "n2_summary": {
            "minimum": float(n2.min()),
            "median": float(np.median(n2)),
            "maximum": float(n2.max()),
        },
        "shuffle_permutation_sha256": hashlib.sha256(np.ascontiguousarray(permutation).tobytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(output / f"metadata_n{n}.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> list[float]:
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def read_all_rows(cfg: dict, output: Path) -> list[dict]:
    rows = []
    source_hash = sha256_file(WORKER_SOURCE)
    prereg_hash = sha256_file(PREREGISTRATION)
    for n in cfg["n_values"]:
        metadata_path = output / f"metadata_n{n}.json"
        csv_path = output / f"rows_n{n}.csv.gz"
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_sha256"] != source_hash or metadata["preregistration_sha256"] != prereg_hash:
            raise RuntimeError(f"hash mismatch for n={n}")
        if metadata["rows_sha256"] != sha256_file(csv_path):
            raise RuntimeError(f"row checksum mismatch for n={n}")
        with gzip.open(csv_path, "rt", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    numeric = {
        "n": int,
        "context": int,
        "replicate": int,
        "target_R": float,
        "M": int,
        "N2_context": float,
        "N2_assigned": float,
        "achieved_R": float,
        "achieved_log2_R": float,
        "posterior_mean_error_sq": float,
        "posterior_mean_error_sq_scaled": float,
        "ESS": float,
        "maxweight": float,
    }
    for horizon in cfg["horizons"]:
        numeric[f"forecast_mean_error_sq_h{horizon}"] = float
        numeric[f"forecast_mean_error_sq_scaled_h{horizon}"] = float
    for row in rows:
        for key, converter in numeric.items():
            row[key] = converter(row[key])
    return rows


def aggregate_context_rows(rows: list[dict], cfg: dict) -> dict[tuple[int, str, float, int], dict[str, float]]:
    metrics = [
        "M",
        "achieved_R",
        "posterior_mean_error_sq",
        "posterior_mean_error_sq_scaled",
        "ESS",
        "maxweight",
    ]
    for horizon in cfg["horizons"]:
        metrics.extend((f"forecast_mean_error_sq_h{horizon}", f"forecast_mean_error_sq_scaled_h{horizon}"))
    accumulated: dict[tuple[int, str, float, int], dict[str, float]] = {}
    for row in rows:
        key = (row["n"], row["assignment"], row["target_R"], row["context"])
        current = accumulated.setdefault(key, {"count": 0.0, **{metric: 0.0 for metric in metrics}})
        current["count"] += 1.0
        for metric in metrics:
            current[metric] += float(row[metric])
    for current in accumulated.values():
        count = current.pop("count")
        for metric in metrics:
            current[metric] /= count
    return accumulated


def summarize(rows: list[dict], cfg: dict, output: Path) -> dict:
    context_rows = aggregate_context_rows(rows, cfg)
    summary_rows = []
    for n in cfg["n_values"]:
        for assignment in ("true", "shuffled"):
            for target in cfg["target_R"]:
                condition = [
                    values
                    for (row_n, row_assignment, row_target, _), values in context_rows.items()
                    if row_n == n and row_assignment == assignment and row_target == target
                ]
                for horizon in cfg["horizons"]:
                    values = np.asarray([row["posterior_mean_error_sq"] for row in condition], dtype=float)
                    scaled_values = np.asarray([row["posterior_mean_error_sq_scaled"] for row in condition], dtype=float)
                    forecast = np.asarray([row[f"forecast_mean_error_sq_scaled_h{horizon}"] for row in condition], dtype=float)
                    rng = np.random.default_rng(stable_seed(BASE_SEED, "bootstrap", n, assignment, target, horizon))
                    summary_rows.append({
                        "n": n,
                        "assignment": assignment,
                        "target_R": target,
                        "horizon": horizon,
                        "contexts": len(values),
                        "mean_M": float(np.mean([row["M"] for row in condition])),
                        "mean_achieved_R": float(np.mean([row["achieved_R"] for row in condition])),
                        "mean_posterior_mean_error_sq": float(values.mean()),
                        "posterior_mean_error_sq_ci95": bootstrap_mean_ci(values, rng, cfg["bootstrap_replicates"]),
                        "mean_posterior_mean_error_sq_scaled": float(scaled_values.mean()),
                        "posterior_mean_error_sq_scaled_ci95": bootstrap_mean_ci(scaled_values, rng, cfg["bootstrap_replicates"]),
                        "mean_forecast_mean_error_sq_scaled": float(forecast.mean()),
                        "forecast_mean_error_sq_scaled_ci95": bootstrap_mean_ci(forecast, rng, cfg["bootstrap_replicates"]),
                    })
    slopes = []
    high_targets = np.asarray([value for value in cfg["target_R"] if value >= 2.0], dtype=float)
    for n in cfg["n_values"]:
        context_curves = []
        for context in range(cfg["contexts_per_n"]):
            means = []
            for target in high_targets:
                means.append(max(context_rows[(n, "true", float(target), context)]["posterior_mean_error_sq"], np.finfo(float).tiny))
            context_curves.append(means)
        context_curves = np.asarray(context_curves)
        mean_curve = context_curves.mean(axis=0)
        slope = float(np.polyfit(np.log2(high_targets), np.log2(mean_curve), 1)[0])
        rng = np.random.default_rng(stable_seed(BASE_SEED, "slope-bootstrap", n))
        boot = []
        for _ in range(cfg["bootstrap_replicates"]):
            indices = rng.integers(0, len(context_curves), len(context_curves))
            curve = context_curves[indices].mean(axis=0)
            boot.append(float(np.polyfit(np.log2(high_targets), np.log2(np.maximum(curve, np.finfo(float).tiny)), 1)[0]))
        slopes.append({"n": n, "slope": slope, "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])]})
    shuffled_contrasts = []
    for n in cfg["n_values"]:
        for target in cfg["target_R"]:
            true_values = np.asarray([context_rows[(n, "true", target, context)]["posterior_mean_error_sq"] for context in range(cfg["contexts_per_n"])])
            shuffled_values = np.asarray([context_rows[(n, "shuffled", target, context)]["posterior_mean_error_sq"] for context in range(cfg["contexts_per_n"])])
            difference = shuffled_values - true_values
            rng = np.random.default_rng(stable_seed(BASE_SEED, "shuffle-contrast", n, target))
            shuffled_contrasts.append({
                "n": n,
                "target_R": target,
                "mean_shuffled_minus_true": float(difference.mean()),
                "ci95": bootstrap_mean_ci(difference, rng, cfg["bootstrap_replicates"]),
                "fraction_contexts_favoring_true": float(np.mean(difference > 0)),
            })
    primary = {
        "endpoint_decrease_every_n": all(
            next(row for row in summary_rows if row["n"] == n and row["assignment"] == "true" and row["target_R"] == min(cfg["target_R"]) and row["horizon"] == cfg["horizons"][0])["mean_posterior_mean_error_sq"]
            > next(row for row in summary_rows if row["n"] == n and row["assignment"] == "true" and row["target_R"] == max(cfg["target_R"]) and row["horizon"] == cfg["horizons"][0])["mean_posterior_mean_error_sq"]
            for n in cfg["n_values"]
        ),
        "all_high_R_point_slopes_negative": all(row["slope"] < 0 for row in slopes),
        "all_high_R_ci_upper_bounds_negative": all(row["ci95"][1] < 0 for row in slopes),
    }
    result = {
        "status": "complete",
        "version": VERSION,
        "configuration": cfg,
        "rows": len(rows),
        "summary_rows": summary_rows,
        "high_R_slopes": slopes,
        "shuffled_N2_contrasts": shuffled_contrasts,
        "primary_decision": {**primary, "positive_pattern": all(primary.values())},
        "worker_source_sha256": sha256_file(WORKER_SOURCE),
        "aggregation_source_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
    }
    atomic_json(output / "summary.json", result)
    return result


def make_figure(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {16: "#1764A3", 64: "#C96113", 256: "#008C77", 1024: "#8E44AD"}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    for n in summary["configuration"]["n_values"]:
        primary = [row for row in summary["summary_rows"] if row["n"] == n and row["assignment"] == "true" and row["horizon"] == 1]
        primary.sort(key=lambda row: row["target_R"])
        x = np.asarray([row["target_R"] for row in primary])
        y = np.asarray([row["mean_posterior_mean_error_sq_scaled"] for row in primary])
        ci = np.asarray([row["posterior_mean_error_sq_scaled_ci95"] for row in primary])
        axes[0].plot(x, y, marker="o", color=colors[n], label=f"n={n}")
        axes[0].fill_between(x, ci[:, 0], ci[:, 1], color=colors[n], alpha=0.12)
        forecast = [row for row in summary["summary_rows"] if row["n"] == n and row["assignment"] == "true" and row["horizon"] == 16]
        forecast.sort(key=lambda row: row["target_R"])
        fy = np.asarray([row["mean_forecast_mean_error_sq_scaled"] for row in forecast])
        fci = np.asarray([row["forecast_mean_error_sq_scaled_ci95"] for row in forecast])
        axes[1].plot(x, fy, marker="o", color=colors[n], label=f"n={n}")
        axes[1].fill_between(x, fci[:, 0], fci[:, 1], color=colors[n], alpha=0.12)
        shuffled = [row for row in summary["summary_rows"]
                    if row["n"] == n and row["assignment"] == "shuffled" and row["horizon"] == 1]
        shuffled.sort(key=lambda row: row["target_R"])
        ratio = np.asarray([
            shuffled_row["mean_posterior_mean_error_sq"]
            / max(true_row["mean_posterior_mean_error_sq"], np.finfo(float).tiny)
            for true_row, shuffled_row in zip(primary, shuffled)
        ])
        axes[2].plot(x, ratio, marker="o", color=colors[n], label=f"n={n}")
    for axis in axes[:2]:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.set_xlabel(r"assigned coverage $\mathcal{R}$")
    axes[0].set_ylabel(r"$\mathbb{E}(\hat a_M-\hat a)^2 / \mathrm{Var}(a\mid C)$")
    axes[0].set_title("Posterior mean approximation")
    axes[1].set_ylabel("forecast-mean error / predictive variance")
    axes[1].set_title("Long-horizon forecast (h=16)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[2].grid(alpha=0.2)
    axes[2].set_xlabel(r"assigned coverage $\mathcal{R}$")
    axes[2].set_ylabel("shuffled / matched mean error")
    axes[2].set_title("Budget-matched control")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(output / "coverage_intervention.pdf")
    fig.savefig(output / "coverage_intervention.png", dpi=220)
    plt.close(fig)


def make_control_figure(summary: dict, output: Path) -> None:
    """Show raw-budget organization and the shuffled-allocation control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {16: "#1764A3", 64: "#C96113", 256: "#008C77", 1024: "#8E44AD"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    rows = [row for row in summary["summary_rows"] if row["horizon"] == 1]
    for n in summary["configuration"]["n_values"]:
        true = sorted(
            (row for row in rows if row["n"] == n and row["assignment"] == "true"),
            key=lambda row: row["target_R"],
        )
        shuffled = sorted(
            (row for row in rows if row["n"] == n and row["assignment"] == "shuffled"),
            key=lambda row: row["target_R"],
        )
        axes[0].plot(
            [row["mean_M"] for row in true],
            [row["mean_posterior_mean_error_sq_scaled"] for row in true],
            marker="o",
            color=colors[n],
            label=f"n={n}",
        )
        ratios = [
            shuffled_row["mean_posterior_mean_error_sq"]
            / max(true_row["mean_posterior_mean_error_sq"], np.finfo(float).tiny)
            for true_row, shuffled_row in zip(true, shuffled)
        ]
        axes[1].plot(
            [row["target_R"] for row in true],
            ratios,
            marker="o",
            color=colors[n],
            label=f"n={n}",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("mean library size $M$")
    axes[0].set_ylabel(r"$\mathbb{E}(\hat a_M-\hat a)^2 / \mathrm{Var}(a\mid C)$")
    axes[0].set_title("Raw library size does not align curves")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel(r"assigned coverage $\mathcal{R}$")
    axes[1].set_ylabel("shuffled / matched mean error")
    axes[1].set_title("Context-matched allocation helps")
    for axis in axes:
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(output / "coverage_intervention_controls.pdf")
    fig.savefig(output / "coverage_intervention_controls.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, choices=N_VALUES)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config(args.smoke)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        rows = read_all_rows(cfg, output)
        result = summarize(rows, cfg, output)
        make_figure(result, output)
        make_control_figure(result, output)
        print(json.dumps({"status": "complete", "rows": result["rows"], "primary_decision": result["primary_decision"]}, indent=2))
        return
    n = args.n if args.n is not None else cfg["n_values"][0]
    if n not in cfg["n_values"]:
        raise SystemExit(f"n={n} is not in this run configuration")
    run_n(n, cfg, output)


if __name__ == "__main__":
    main()

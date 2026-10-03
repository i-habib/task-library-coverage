#!/usr/bin/env python3
"""Resumable exact-posterior temporal FMTS simulation.

The simulator is deliberately narrow: an AR(1) task family with a bounded
uniform prior.  It measures whether the finite library approximation is
organized by R=M/N2 for posterior calibration and forecast quality.  It does
not make claims about neural temporal foundation models.
"""
from __future__ import annotations

import argparse
import csv
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
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fmts-temporal-mpl")

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import logsumexp, roots_legendre

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts/fmts-temporal"
PREREGISTRATION = ROOT / "fmts_temporal_preregistration.json"
VERSION = "FMTS-temporal-v1"
BASE_SEED = 2026091401
A_MAX = 0.95
N2_LOWER_TOLERANCE = 1e-10
MATCHED_R_REPORTING_FLOOR = 30
N_VALUES = (16, 64, 256, 1024)
HORIZONS = (1, 4, 16)
M_GRID = (4, 16, 64, 256, 1024)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def experiment_config(smoke: bool = False) -> dict:
    return {
        "version": VERSION,
        "base_seed": BASE_SEED,
        "a_max": A_MAX,
        "n_values": list((16, 64) if smoke else N_VALUES),
        "horizons": list(HORIZONS),
        "M_grid": list(M_GRID),
        "contexts_per_n": 3 if smoke else 512,
        "future_paths_per_context": 2 if smoke else 8,
        "quadrature_order": 48 if smoke else 96,
        "quadrature_audit_order": 96 if smoke else 192,
        "scipy_quad_tolerance": 2e-10,
        "bootstrap_replicates": 50 if smoke else 1000,
        "direct_simulation_draws": 10000 if smoke else 60000,
        "credible_mass": 0.90,
        "description": "Independent target trajectories and independent nested library task prefixes; all M/h comparisons remain paired within context cluster.",
    }


def composite_nodes(sxx: float, sxy: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted nodes and prior-measure weights on the full bounded prior."""
    mode = float(np.clip(sxy / sxx if sxx > 0 else 0.0, -A_MAX, A_MAX))
    scale = 1.0 / math.sqrt(max(float(sxx), 1.0))
    knots = [-A_MAX, 0.0, A_MAX, mode]
    for multiplier in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        knots.extend((mode - multiplier * scale, mode + multiplier * scale))
    breaks = np.unique(np.clip(np.asarray(knots, dtype=float), -A_MAX, A_MAX))
    base_x, base_w = roots_legendre(order)
    nodes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right <= left:
            continue
        half = (right - left) / 2.0
        nodes.append((left + right) / 2.0 + half * base_x)
        # Include the Uniform[-A_MAX,A_MAX] prior density in every weight.
        weights.append(base_w * half / (2.0 * A_MAX))
    return np.concatenate(nodes), np.concatenate(weights)


def posterior_from_sufficient(sxx: float, sxy: float, order: int) -> dict:
    """Stable normalized posterior and N2 from one-dimensional quadrature."""
    a, prior_weights = composite_nodes(sxx, sxy, order)
    log_kernel = sxy * a - 0.5 * sxx * a * a
    shift = float(np.max(log_kernel))
    scaled1 = np.exp(log_kernel - shift)
    z1_scaled = float(np.sum(prior_weights * scaled1))
    z2_scaled = float(np.sum(prior_weights * scaled1 * scaled1))
    log_z1 = shift + math.log(z1_scaled)
    log_z2 = 2.0 * shift + math.log(z2_scaled)
    log_n2 = log_z2 - 2.0 * log_z1
    if log_n2 < -N2_LOWER_TOLERANCE:
        raise ArithmeticError(f"N2 < 1 beyond tolerance: log_N2={log_n2}")
    posterior_weights = prior_weights * np.exp(log_kernel - log_z1)
    posterior_weights /= np.sum(posterior_weights)
    if not np.all(np.isfinite(posterior_weights)) or not np.isfinite(log_n2):
        raise FloatingPointError("non-finite posterior quadrature")
    return {
        "a": a,
        "prior_weights": prior_weights,
        "log_kernel": log_kernel,
        "posterior_weights": posterior_weights,
        "log_z1": log_z1,
        "log_z2": log_z2,
        "log_n2": log_n2,
        "mean_a": float(np.dot(posterior_weights, a)),
        "var_a": float(np.dot(posterior_weights, a * a) - np.dot(posterior_weights, a) ** 2),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cdf = np.cumsum(sorted_weights)
    cdf /= cdf[-1]
    return float(np.interp(probability, np.r_[0.0, cdf], np.r_[sorted_values[0], sorted_values]))


def predictive_moments(weights: np.ndarray, atoms: np.ndarray, x_n: float, horizon: int) -> tuple[float, float]:
    powers = atoms ** horizon
    conditional_variance = sum(atoms ** (2 * j) for j in range(horizon))
    mean = float(x_n * np.dot(weights, powers))
    variance = float(np.dot(weights, conditional_variance) + x_n * x_n * (np.dot(weights, powers * powers) - np.dot(weights, powers) ** 2))
    return mean, variance


def normal_logpdf(y: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi * variance) + (y - mean) ** 2 / variance)


def predictive_log_density(y: np.ndarray, weights: np.ndarray, atoms: np.ndarray, x_n: float, horizon: int) -> np.ndarray:
    means = x_n * atoms ** horizon
    variances = sum(atoms ** (2 * j) for j in range(horizon))
    log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
    component = log_weights[None, :] + (-0.5 * (math.log(2.0 * math.pi) + np.log(variances)[None, :] + (y[:, None] - means[None, :]) ** 2 / variances[None, :]))
    return logsumexp(component, axis=1)


def scipy_quad_log_n2(sxx: float, sxy: float) -> float:
    mode = float(np.clip(sxy / sxx if sxx else 0.0, -A_MAX, A_MAX))
    shift = max(0.0, sxy * mode - 0.5 * sxx * mode * mode)

    def scaled(power: int) -> float:
        value, error = quad(lambda a: math.exp(power * (sxy * a - 0.5 * sxx * a * a - shift)) / (2.0 * A_MAX), -A_MAX, A_MAX, points=[mode], epsabs=1e-13, epsrel=1e-13, limit=400)
        if error > 2e-10:
            raise ArithmeticError(f"scipy quad reported large error {error}")
        return value

    z1 = scaled(1)
    z2 = scaled(2)
    return (2.0 * shift + math.log(z2)) - 2.0 * (shift + math.log(z1))


def quad_audits(config: dict) -> dict:
    """Independent quadrature checks, including concentrated boundary cases."""
    cases = [
        {"name": "low_information_interior", "sxx": 1.0, "sxy": 0.1},
        {"name": "concentrated_interior", "sxx": 1024.0, "sxy": 0.1 * 1024.0},
        {"name": "concentrated_positive_boundary", "sxx": 1024.0, "sxy": 0.95 * 1024.0},
        {"name": "concentrated_negative_boundary", "sxx": 1024.0, "sxy": -0.95 * 1024.0},
    ]
    rows = []
    for case in cases:
        low = posterior_from_sufficient(case["sxx"], case["sxy"], config["quadrature_order"])
        high = posterior_from_sufficient(case["sxx"], case["sxy"], config["quadrature_audit_order"])
        independent = scipy_quad_log_n2(case["sxx"], case["sxy"])
        rows.append({**case, "log_n2_primary": low["log_n2"], "log_n2_doubled": high["log_n2"], "log_n2_quad": independent, "order_abs_error": abs(low["log_n2"] - high["log_n2"]), "quad_abs_error": abs(high["log_n2"] - independent)})
    maximum = max(max(row["order_abs_error"], row["quad_abs_error"]) for row in rows)
    if maximum > config["scipy_quad_tolerance"]:
        raise ArithmeticError(f"synthetic quadrature audit failed: {maximum}")
    return {"cases": rows, "maximum_abs_error": maximum, "tolerance": config["scipy_quad_tolerance"]}


def direct_simulation_audit(config: dict) -> dict:
    """Check analytic mixture moments with direct multi-step posterior draws."""
    posterior = posterior_from_sufficient(1024.0, 0.95 * 1024.0, config["quadrature_audit_order"])
    rng = np.random.default_rng(stable_seed(BASE_SEED, "direct-mixture-audit"))
    draws = config["direct_simulation_draws"]
    atom_index = rng.choice(len(posterior["a"]), size=draws, p=posterior["posterior_weights"])
    sampled_a = posterior["a"][atom_index]
    x_n = 0.7
    rows = []
    for horizon in HORIZONS:
        innovations = rng.standard_normal((draws, horizon))
        state = np.full(draws, x_n)
        for step in range(horizon):
            state = sampled_a * state + innovations[:, step]
        mean, variance = predictive_moments(posterior["posterior_weights"], posterior["a"], x_n, horizon)
        empirical_mean = float(state.mean())
        empirical_variance = float(state.var())
        mean_se = math.sqrt(variance / draws)
        variance_se = variance * math.sqrt(2.0 / max(draws - 1, 1))
        rows.append({"horizon": horizon, "analytic_mean": mean, "empirical_mean": empirical_mean, "mean_z": abs(empirical_mean - mean) / mean_se, "analytic_variance": variance, "empirical_variance": empirical_variance, "variance_z": abs(empirical_variance - variance) / variance_se})
    maximum_z = max(max(row["mean_z"], row["variance_z"]) for row in rows)
    if maximum_z > 8.0:
        raise ArithmeticError(f"direct predictive simulation audit failed: z={maximum_z}")
    return {"draws": draws, "rows": rows, "maximum_z": maximum_z, "acceptance_threshold_z": 8.0}


def context_snapshot(config: dict, n: int, context: int) -> dict[str, object]:
    target_rng = np.random.default_rng(stable_seed(BASE_SEED, "target", n, context))
    library_rng = np.random.default_rng(stable_seed(BASE_SEED, "library", n, context))
    future_rng = np.random.default_rng(stable_seed(BASE_SEED, "future", n, context))
    a_true = float(target_rng.uniform(-A_MAX, A_MAX))
    x = np.zeros(n + 1, dtype=float)
    innovations = target_rng.standard_normal(n)
    for index in range(n):
        x[index + 1] = a_true * x[index] + innovations[index]
    x_n = float(x[-1])
    sxx = float(np.dot(x[:-1], x[:-1]))
    sxy = float(np.dot(x[:-1], x[1:]))
    primary = posterior_from_sufficient(sxx, sxy, config["quadrature_order"])
    doubled = posterior_from_sufficient(sxx, sxy, config["quadrature_audit_order"])
    log_n2 = primary["log_n2"]
    order_abs_error = abs(log_n2 - doubled["log_n2"])
    audit_contexts = {0, max(0, config["contexts_per_n"] // 2), config["contexts_per_n"] - 1}
    quad_context_error = float("nan")
    if context in audit_contexts:
        quad_context_error = abs(doubled["log_n2"] - scipy_quad_log_n2(sxx, sxy))
    library_a = library_rng.uniform(-A_MAX, A_MAX, max(M_GRID))
    future_paths = config["future_paths_per_context"]
    future = np.zeros((future_paths, max(HORIZONS)), dtype=float)
    for replicate in range(future_paths):
        state = x_n
        for step in range(max(HORIZONS)):
            state = a_true * state + future_rng.standard_normal()
            future[replicate, step] = state
    credible_mass = config["credible_mass"]
    tail = (1.0 - credible_mass) / 2.0
    continuous_interval = (weighted_quantile(primary["a"], primary["posterior_weights"], tail), weighted_quantile(primary["a"], primary["posterior_weights"], 1.0 - tail))
    continuous_cov = float(continuous_interval[0] <= a_true <= continuous_interval[1])
    h1_identity_error = 0.0
    rows: dict[str, list[list[float]]] = {name: [] for name in ("coverage", "support", "ess", "maxweight", "achieved_log2_R", "forecast_discrepancy", "forecast_discrepancy_norm", "mse_ratio", "score_difference", "finite_variance", "continuous_variance")}
    continuous_mse = []
    oracle_mse = []
    continuous_nll = []
    oracle_nll = []
    for m in M_GRID:
        atoms = library_a[:m]
        discrete_log_weights = primary["log_kernel"] if False else (sxy * atoms - 0.5 * sxx * atoms * atoms)
        discrete_weights = np.exp(discrete_log_weights - logsumexp(discrete_log_weights))
        ess = 1.0 / float(np.sum(discrete_weights * discrete_weights))
        maximum_weight = float(np.max(discrete_weights))
        discrete_interval = (weighted_quantile(atoms, discrete_weights, tail), weighted_quantile(atoms, discrete_weights, 1.0 - tail))
        discrete_coverage = float(discrete_interval[0] <= a_true <= discrete_interval[1])
        support = float(np.any((atoms >= continuous_interval[0]) & (atoms <= continuous_interval[1])))
        achieved_log2_r = math.log2(m) - log_n2 / math.log(2.0)
        coverage_h, support_h, discrepancy_h, discrepancy_norm_h, mse_ratio_h, score_diff_h, variance_h, continuous_variance_h = [], [], [], [], [], [], [], []
        for horizon_index, horizon in enumerate(HORIZONS):
            continuous_mean, continuous_var = predictive_moments(primary["posterior_weights"], primary["a"], x_n, horizon)
            discrete_mean, discrete_var = predictive_moments(discrete_weights, atoms, x_n, horizon)
            oracle_mean = a_true ** horizon * x_n
            oracle_var = sum(a_true ** (2 * j) for j in range(horizon))
            future_y = future[:, horizon - 1]
            continuous_density = predictive_log_density(future_y, primary["posterior_weights"], primary["a"], x_n, horizon)
            discrete_density = predictive_log_density(future_y, discrete_weights, atoms, x_n, horizon)
            oracle_density = normal_logpdf(future_y, oracle_mean, oracle_var)
            current_continuous_mse = float(np.mean((continuous_mean - future_y) ** 2))
            current_oracle_mse = float(np.mean((oracle_mean - future_y) ** 2))
            current_continuous_nll = float(np.mean(-continuous_density))
            current_oracle_nll = float(np.mean(-oracle_density))
            current_discrepancy = float((discrete_mean - continuous_mean) ** 2)
            denom = max(current_oracle_mse, np.finfo(float).tiny)
            coverage_h.append(discrete_coverage)
            support_h.append(support)
            discrepancy_h.append(current_discrepancy)
            discrepancy_norm_h.append(current_discrepancy / denom)
            mse_ratio_h.append(float(np.mean((discrete_mean - future_y) ** 2)) / denom)
            score_diff_h.append(float(np.mean(-discrete_density + continuous_density)))
            variance_h.append(discrete_var)
            continuous_variance_h.append(continuous_var)
            continuous_mse.append(current_continuous_mse)
            oracle_mse.append(current_oracle_mse)
            continuous_nll.append(current_continuous_nll)
            oracle_nll.append(current_oracle_nll)
            if horizon == 1:
                h1_identity_error = max(h1_identity_error, abs(continuous_var - (1.0 + x_n * x_n * primary["var_a"])))
        for name, value in (("coverage", coverage_h), ("support", support_h), ("forecast_discrepancy", discrepancy_h), ("forecast_discrepancy_norm", discrepancy_norm_h), ("mse_ratio", mse_ratio_h), ("score_difference", score_diff_h), ("finite_variance", variance_h), ("continuous_variance", continuous_variance_h)):
            rows[name].append(value)
        rows["ess"].append([ess] * len(HORIZONS))
        rows["maxweight"].append([maximum_weight] * len(HORIZONS))
        rows["achieved_log2_R"].append([achieved_log2_r] * len(HORIZONS))
    arrays = {key: np.asarray(value, dtype=float) for key, value in rows.items()}
    arrays.update({"n": np.array(n), "context": np.array(context), "a_true": np.array(a_true), "x_n": np.array(x_n), "sxx": np.array(sxx), "sxy": np.array(sxy), "log_n2": np.array(log_n2), "quadrature_order_abs_error": np.array(order_abs_error), "quadrature_context_abs_error": np.array(quad_context_error), "continuous_coverage": np.array(continuous_cov), "h1_identity_max_abs_error": np.array(h1_identity_error), "continuous_mse": np.asarray(continuous_mse, dtype=float).reshape(len(M_GRID), len(HORIZONS)), "oracle_mse": np.asarray(oracle_mse, dtype=float).reshape(len(M_GRID), len(HORIZONS)), "continuous_nll": np.asarray(continuous_nll, dtype=float).reshape(len(M_GRID), len(HORIZONS)), "oracle_nll": np.asarray(oracle_nll, dtype=float).reshape(len(M_GRID), len(HORIZONS)), "library_a": library_a})
    finite_values = np.concatenate([value.ravel() for value in arrays.values() if np.issubdtype(value.dtype, np.number) and value.dtype.kind != "u"])
    if not np.all(np.isfinite(finite_values)):
        # The two audit fields are intentionally NaN for non-audited contexts.
        finite_values = np.concatenate([value.ravel() for key, value in arrays.items() if key not in {"quadrature_context_abs_error"} and np.issubdtype(value.dtype, np.number)])
    if not np.all(np.isfinite(finite_values)):
        raise FloatingPointError(f"non-finite context arrays at n={n}, context={context}")
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    arrays["checksum_hex"] = np.array(digest.hexdigest())
    return arrays


def snapshot_path(output: Path, n: int, context: int) -> Path:
    return output / "raw" / f"n{n}" / f"context_{context:04d}.npz"


def run_contexts(config: dict, output: Path) -> None:
    for n in config["n_values"]:
        started = time.time()
        missing = 0
        for context in range(config["contexts_per_n"]):
            path = snapshot_path(output, n, context)
            if path.exists():
                continue
            atomic_npz(path, **context_snapshot(config, n, context))
            missing += 1
        print(f"n={n}: verified {config['contexts_per_n']} context snapshots ({missing} newly computed, {time.time()-started:.1f}s)", flush=True)


def load_contexts(config: dict, output: Path, n: int) -> list[dict[str, np.ndarray]]:
    contexts = []
    for context in range(config["contexts_per_n"]):
        path = snapshot_path(output, n, context)
        if not path.exists():
            raise FileNotFoundError(f"missing resumable snapshot {path}")
        with np.load(path, allow_pickle=False) as stored:
            data = {name: stored[name] for name in stored.files}
        if int(data["n"]) != n or int(data["context"]) != context:
            raise RuntimeError(f"snapshot identity mismatch: {path}")
        contexts.append(data)
    return contexts


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bootstrap_ci expects one-dimensional context values")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite bootstrap values")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def aggregate(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | int]] = []
    condition_summary = []
    context_checksums = []
    max_order_error = 0.0
    max_quad_error = 0.0
    max_h1_error = 0.0
    for n in config["n_values"]:
        contexts = load_contexts(config, output, n)
        for data in contexts:
            context_checksums.append(str(data["checksum_hex"]))
            max_order_error = max(max_order_error, float(data["quadrature_order_abs_error"]))
            if np.isfinite(data["quadrature_context_abs_error"]):
                max_quad_error = max(max_quad_error, float(data["quadrature_context_abs_error"]))
            max_h1_error = max(max_h1_error, float(data["h1_identity_max_abs_error"]))
        for m_index, m in enumerate(M_GRID):
            for h_index, horizon in enumerate(HORIZONS):
                metrics = {"coverage": np.array([data["coverage"][m_index, h_index] for data in contexts]), "support": np.array([data["support"][m_index, h_index] for data in contexts]), "ess": np.array([data["ess"][m_index, h_index] for data in contexts]), "maxweight": np.array([data["maxweight"][m_index, h_index] for data in contexts]), "forecast_discrepancy": np.array([data["forecast_discrepancy"][m_index, h_index] for data in contexts]), "forecast_discrepancy_norm": np.array([data["forecast_discrepancy_norm"][m_index, h_index] for data in contexts]), "mse_ratio": np.array([data["mse_ratio"][m_index, h_index] for data in contexts]), "score_difference": np.array([data["score_difference"][m_index, h_index] for data in contexts]), "achieved_log2_R": np.array([data["achieved_log2_R"][m_index, h_index] for data in contexts])}
                rng = np.random.default_rng(stable_seed(BASE_SEED, "condition-bootstrap", n, m, horizon))
                condition_summary.append({"n": n, "M": m, "horizon": horizon, "contexts": len(contexts), "mean_achieved_log2_R": float(metrics["achieved_log2_R"].mean()), "mean_coverage": float(metrics["coverage"].mean()), "coverage_ci95": bootstrap_ci(metrics["coverage"], rng, config["bootstrap_replicates"]), "mean_continuous_coverage": float(np.mean([data["continuous_coverage"] for data in contexts])), "mean_support": float(metrics["support"].mean()), "mean_ess": float(metrics["ess"].mean()), "mean_maxweight": float(metrics["maxweight"].mean()), "mean_forecast_discrepancy": float(metrics["forecast_discrepancy"].mean()), "forecast_discrepancy_raw_ci95": bootstrap_ci(metrics["forecast_discrepancy"], rng, config["bootstrap_replicates"]), "mean_forecast_discrepancy_norm": float(metrics["forecast_discrepancy_norm"].mean()), "forecast_discrepancy_ci95": bootstrap_ci(metrics["forecast_discrepancy_norm"], rng, config["bootstrap_replicates"]), "mean_mse_ratio": float(metrics["mse_ratio"].mean()), "mse_ratio_ci95": bootstrap_ci(metrics["mse_ratio"], rng, config["bootstrap_replicates"]), "mean_score_difference_nats": float(metrics["score_difference"].mean()), "score_difference_ci95": bootstrap_ci(metrics["score_difference"], rng, config["bootstrap_replicates"])})
                for context, data in enumerate(contexts):
                    all_rows.append({"n": n, "context": context, "M": m, "horizon": horizon, "achieved_log2_R": float(data["achieved_log2_R"][m_index, h_index]), "coverage": float(data["coverage"][m_index, h_index]), "support": float(data["support"][m_index, h_index]), "ESS": float(data["ess"][m_index, h_index]), "maxweight": float(data["maxweight"][m_index, h_index]), "forecast_discrepancy": float(data["forecast_discrepancy"][m_index, h_index]), "forecast_discrepancy_norm": float(data["forecast_discrepancy_norm"][m_index, h_index]), "mse_ratio": float(data["mse_ratio"][m_index, h_index]), "score_difference_nats": float(data["score_difference"][m_index, h_index])})
    fieldnames = list(all_rows[0])
    with (output / "temporal_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    bins = np.arange(-6.0, 12.01, 0.5)
    binned = []
    for n in config["n_values"]:
        n_rows = [row for row in all_rows if row["n"] == n]
        for horizon in HORIZONS:
            h_rows = [row for row in n_rows if row["horizon"] == horizon]
            for left in bins:
                right = left + 0.5
                selected = [row for row in h_rows if left <= row["achieved_log2_R"] < right]
                if not selected:
                    continue
                # A context cluster contributes at most once to each bin.
                by_context: dict[int, list[dict]] = {}
                for row in selected:
                    by_context.setdefault(int(row["context"]), []).append(row)
                cluster_rows = []
                for context, rows in by_context.items():
                    cluster_rows.append({"context": context, "x": float(np.mean([row["achieved_log2_R"] for row in rows])), **{key: float(np.mean([row[key] for row in rows])) for key in ("coverage", "support", "forecast_discrepancy", "forecast_discrepancy_norm", "mse_ratio", "score_difference_nats")}})
                x = np.array([row["x"] for row in cluster_rows])
                row_out = {"n": n, "horizon": horizon, "left": left, "right": right, "contexts": len(cluster_rows), "mean_achieved_log2_R": float(x.mean())}
                for key in ("coverage", "support", "forecast_discrepancy", "forecast_discrepancy_norm", "mse_ratio", "score_difference_nats"):
                    values = np.array([row[key] for row in cluster_rows])
                    rng = np.random.default_rng(stable_seed(BASE_SEED, "bin-bootstrap", n, horizon, left, key))
                    row_out[f"mean_{key}"] = float(values.mean())
                    row_out[f"{key}_ci95"] = bootstrap_ci(values, rng, config["bootstrap_replicates"])
                binned.append(row_out)
    matched_r = []
    for horizon in HORIZONS:
        for left in bins:
            rows = [row for row in binned if row["horizon"] == horizon and row["left"] == left]
            if len(rows) < 2:
                continue
            minimum_contexts = min(int(row["contexts"]) for row in rows)
            matched_r.append({"horizon": horizon, "left": left, "right": left + 0.5, "n_conditions": len(rows), "minimum_contexts": minimum_contexts, "reporting_floor": MATCHED_R_REPORTING_FLOOR, "eligible_for_primary_reporting": minimum_contexts >= MATCHED_R_REPORTING_FLOOR, "discrepancy_range": float(max(row["mean_forecast_discrepancy_norm"] for row in rows) - min(row["mean_forecast_discrepancy_norm"] for row in rows)), "mse_ratio_range": float(max(row["mean_mse_ratio"] for row in rows) - min(row["mean_mse_ratio"] for row in rows)), "score_difference_range_nats": float(max(row["mean_score_difference_nats"] for row in rows) - min(row["mean_score_difference_nats"] for row in rows)), "conditions": [{"n": row["n"], "contexts": row["contexts"], "mean_log2_R": row["mean_achieved_log2_R"], "discrepancy": row["mean_forecast_discrepancy_norm"], "mse_ratio": row["mean_mse_ratio"], "score_difference_nats": row["mean_score_difference_nats"]} for row in rows]})
    checksums = hashlib.sha256("\n".join(sorted(context_checksums)).encode()).hexdigest()
    summary = {"status": "complete", "version": VERSION, "configuration": config, "contexts_total": sum(config["contexts_per_n"] for _ in config["n_values"]), "rows": len(all_rows), "maximum_quadrature_order_abs_error": max_order_error, "maximum_scipy_quad_abs_error": max_quad_error, "maximum_h1_identity_abs_error": max_h1_error, "context_checksum_manifest_sha256": checksums, "condition_summary": condition_summary, "binned_summary": binned, "matched_R_heterogeneity": matched_r, "reporting_revision": {"reason": "Post-review interpretation correction using retained data only; no contexts regenerated.", "matched_R_minimum_contexts_per_n": MATCHED_R_REPORTING_FLOOR, "source_sha256": sha256_file(Path(__file__).resolve())}, "interpretation": {"achieved_R": "All curves use achieved log2(M/N2).", "bootstrap": "Intervals resample independent context clusters and preserve paired M/horizon rows.", "forecast": "Finite-library MSE and posterior-mean discrepancy are oracle-MSE normalized; score differences are finite-library minus continuous in nats.", "universality": "Matched-R heterogeneity is descriptive and no universal collapse is assumed; sparse-bin maxima are diagnostics, not falsification tests."}}
    atomic_json(output / "summary.json", summary)
    return summary


def make_figures(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {16: "#1764A3", 64: "#C96113", 256: "#008C77", 1024: "#8E44AD"}
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 10.5), sharex="col", constrained_layout=True)
    for col, horizon in enumerate(HORIZONS):
        for n in summary["configuration"]["n_values"]:
            points = [row for row in summary["binned_summary"] if row["n"] == n and row["horizon"] == horizon]
            points.sort(key=lambda row: row["mean_achieved_log2_R"])
            if not points:
                continue
            x = np.array([row["mean_achieved_log2_R"] for row in points])
            for row_index, (metric, ylabel) in enumerate((("coverage", "90% posterior interval coverage"), ("forecast_discrepancy_norm", "mean discrepancy / oracle MSE"), ("score_difference_nats", "NLL difference (nats)"))):
                y = np.array([row[f"mean_{metric}"] for row in points])
                ci = np.array([row[f"{metric}_ci95"] for row in points])
                axes[row_index, col].plot(x, y, color=colors.get(n, "#333333"), marker="o", markersize=3, linewidth=1.3, label=f"n={n}")
                axes[row_index, col].fill_between(x, ci[:, 0], ci[:, 1], color=colors.get(n, "#333333"), alpha=0.10)
            continuous_baseline = [row["mean_continuous_coverage"] for row in summary["condition_summary"] if row["n"] == n and row["horizon"] == horizon and row["M"] == min(M_GRID)]
            if continuous_baseline:
                axes[0, col].axhline(float(continuous_baseline[0]), color=colors.get(n, "#333333"), linestyle="--", linewidth=0.8, alpha=0.55)
        axes[0, col].set_title(f"h={horizon}")
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[0, col].axhline(0.90, color="#555555", linestyle=":", linewidth=0.8)
        axes[1, col].axhline(0.0, color="#555555", linewidth=0.8)
        axes[2, col].axhline(0.0, color="#555555", linewidth=0.8)
        for row_index in range(3):
            axes[row_index, col].grid(alpha=0.18)
            axes[row_index, col].set_xlabel(r"achieved $\log_2 R$")
    axes[0, 0].set_ylabel("posterior coverage")
    axes[1, 0].set_ylabel("forecast discrepancy")
    axes[2, 0].set_ylabel("score gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("FMTS temporal AR(1): posterior calibration and paired forecast quality\nLines are n; bands are context-cluster bootstrap 95% intervals; all x-axes use achieved R=M/N2", fontsize=13)
    fig.savefig(output / "fmts_temporal_primary.png", dpi=220)
    fig.savefig(output / "fmts_temporal_primary.pdf")
    plt.close(fig)


def write_report(output: Path, summary: dict, audit: dict, simulation: dict) -> None:
    conditions = summary["condition_summary"]
    endpoint = [row for row in conditions if row["M"] == max(M_GRID)]
    start = [row for row in conditions if row["M"] == min(M_GRID)]
    endpoint_discrepancy = float(np.mean([row["mean_forecast_discrepancy_norm"] for row in endpoint]))
    start_discrepancy = float(np.mean([row["mean_forecast_discrepancy_norm"] for row in start]))
    endpoint_score = float(np.mean([row["mean_score_difference_nats"] for row in endpoint]))
    start_score = float(np.mean([row["mean_score_difference_nats"] for row in start]))
    matched = summary["matched_R_heterogeneity"]
    raw_max = max(matched, key=lambda row: row["discrepancy_range"], default=None)
    reportable = [row for row in matched if row["eligible_for_primary_reporting"]]
    reportable_max = max(reportable, key=lambda row: row["discrepancy_range"], default=None)
    direction_disc = "decreased" if endpoint_discrepancy < start_discrepancy else "did not decrease uniformly"
    direction_score = "improved" if endpoint_score < start_score else "did not improve uniformly"
    lines = [
        "# FMTS temporal results",
        "",
        "## Outcome",
        "",
        f"The frozen AR(1) study completed {summary['contexts_total']:,} independent context clusters and {summary['rows']:,} paired context/library/horizon rows. Increasing the fixed library from M={min(M_GRID)} to M={max(M_GRID)} {direction_disc} the oracle-normalized continuous-versus-finite forecast-mean discrepancy overall ({start_discrepancy:.4g} to {endpoint_discrepancy:.4g}) and {direction_score} the finite-minus-continuous predictive NLL gap overall ({start_score:.4g} to {endpoint_score:.4g} nats). These are simulator-specific empirical statements, not a universality claim.",
        "",
        ("The largest raw matched-R discrepancy is retained as a sparse-bin diagnostic, not a falsification result: "
         + (f"{raw_max['discrepancy_range']:.4g} discrepancy units and {raw_max['score_difference_range_nats']:.4g} nats at h={raw_max['horizon']} in [{raw_max['left']:.1f},{raw_max['right']:.1f}), with only {raw_max['minimum_contexts']} contexts in its sparsest represented n."
            if raw_max is not None else "no achieved-R bin contained two n values.")),
        (f"With the post-review reporting floor of {MATCHED_R_REPORTING_FLOOR} contexts per represented n, the largest descriptive cross-n range is {reportable_max['discrepancy_range']:.4g} discrepancy units and {reportable_max['score_difference_range_nats']:.4g} nats."
         if reportable_max is not None else f"No matched-R bin met the post-review reporting floor of {MATCHED_R_REPORTING_FLOOR} contexts per represented n."),
        "",
        "## Integrity checks",
        "",
        f"- Maximum primary-versus-doubled quadrature log-N2 difference: {summary['maximum_quadrature_order_abs_error']:.3e}.",
        f"- Maximum independent scipy.integrate.quad log-N2 difference among audited contexts: {summary['maximum_scipy_quad_abs_error']:.3e}; synthetic interior and concentrated positive/negative boundary audits had maximum error {audit['maximum_abs_error']:.3e}.",
        f"- Maximum h=1 variance-identity error, |Var_mix-(1+x_n^2 Var(a))|: {summary['maximum_h1_identity_abs_error']:.3e}.",
        f"- Direct posterior-mixture multi-step simulation maximum standardized moment error: {simulation['maximum_z']:.3f} SD (acceptance threshold {simulation['acceptance_threshold_z']:.1f}).",
        f"- Context checksum manifest: `{summary['context_checksum_manifest_sha256']}`.",
        "",
        "## Selected condition means",
        "",
        "| n | h | M | achieved log2 R | coverage | ESS | max weight | normalized discrepancy | NLL gap (nats) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        if row["M"] in (min(M_GRID), max(M_GRID)):
            lines.append(f"| {row['n']} | {row['horizon']} | {row['M']} | {row['mean_achieved_log2_R']:.3f} | {row['mean_coverage']:.3f} | {row['mean_ess']:.2f} | {row['mean_maxweight']:.3f} | {row['mean_forecast_discrepancy_norm']:.4g} | {row['mean_score_difference_nats']:.4g} |")
    lines += [
        "",
        "## Files and interpretation limits",
        "",
        "- `fmts_temporal_primary.png` and `.pdf` are the publication-ready primary figure: posterior coverage, normalized forecast-mean discrepancy, and predictive score gap versus achieved R by horizon.",
        "- `temporal_rows.csv` contains paired context-level rows; `summary.json` contains bootstrap intervals, matched-R summaries, audits, and checksums.",
        f"- Matched-R maxima from bins with fewer than {MATCHED_R_REPORTING_FLOOR} contexts for any represented n remain in `summary.json` as diagnostics only; the original extreme bin was driven by one n=64 outlier among two contexts.",
        "- The continuous forecast integrates the atom-specific multi-step variance; the finite posterior uses every library atom. Future draws are independent and paired across predictors.",
        "- This controlled one-dimensional AR(1) simulator does not justify claims about temporal foundation models, neural mechanisms, or universal R-collapse.",
    ]
    (output / "FMTS_TEMPORAL_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reporting-only", action="store_true", help="Rebuild summaries and figures from retained context snapshots without regenerating data.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args.smoke)
    if args.reporting_only:
        audits = json.loads((output / "quadrature_audit.json").read_text())
        simulation = json.loads((output / "direct_simulation_audit.json").read_text())
        summary = aggregate(config, output)
        make_figures(output, summary)
        write_report(output, summary, audits, simulation)
        print(json.dumps({"status": summary["status"], "reporting_only": True, "contexts_total": summary["contexts_total"], "rows": summary["rows"]}, indent=2))
        return
    identity = {"version": VERSION, "configuration": config, "source_sha256": sha256_file(Path(__file__).resolve()), "preregistration_sha256": sha256_file(PREREGISTRATION)}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Cannot resume with changed source, preregistration, or configuration")
    atomic_json(identity_path, identity)
    atomic_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    audits = quad_audits(config)
    simulation = direct_simulation_audit(config)
    atomic_json(output / "quadrature_audit.json", audits)
    atomic_json(output / "direct_simulation_audit.json", simulation)
    if args.audit_only:
        print(json.dumps({"quadrature": audits, "direct_simulation": simulation}, indent=2))
        return
    if not args.figures_only:
        run_contexts(config, output)
    summary = aggregate(config, output)
    make_figures(output, summary)
    write_report(output, summary, audits, simulation)
    print(json.dumps({"status": summary["status"], "contexts_total": summary["contexts_total"], "rows": summary["rows"], "maximum_quadrature_order_abs_error": summary["maximum_quadrature_order_abs_error"], "maximum_scipy_quad_abs_error": summary["maximum_scipy_quad_abs_error"], "maximum_h1_identity_abs_error": summary["maximum_h1_identity_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()

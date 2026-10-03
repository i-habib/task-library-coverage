#!/usr/bin/env python3
"""Resumable learned temporal coverage experiment on coupled stable dynamics."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "fmts_temporal_model_preregistration.json"
DIM = 8
STATES = 17
PREDICTIONS = STATES - 1
NOISE = 0.1
THETA_MAX = 0.9
ENDPOINT = 200_000
WARMUP = 100_000
PEAK_LR = 1e-3
PREFIXES = (3, 7, 11, 15)
BASE_SEED = 2026091517
EVAL_COUNT = 1024


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(*parts: object) -> int:
    payload = "|".join(map(str, (BASE_SEED, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_torch(path: Path, payload: Any) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


class TemporalTransformer(nn.Module):
    def __init__(self, smoke: bool = False) -> None:
        super().__init__()
        width = 32 if smoke else 128
        layers = 2 if smoke else 8
        hidden = 128 if smoke else 512
        self.input_projection = nn.Linear(DIM, width, bias=False)
        self.position = nn.Embedding(PREDICTIONS, width)
        self.blocks = nn.ModuleList([GPT2Block(width, 2, layers, hidden) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width, eps=1e-5)
        self.output = nn.Linear(width, DIM, bias=False)
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(states.shape[1], device=states.device)
        hidden = self.input_projection(states) + self.position(positions)[None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


class GPT2Block(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.ln_1 = nn.LayerNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_projection = nn.Linear(width, width)
        self.ln_2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp_in = nn.Linear(width, hidden)
        self.mlp_out = nn.Linear(hidden, width)
        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.attention_projection.weight, mean=0.0,
                        std=0.02 / math.sqrt(2 * layers))
        nn.init.zeros_(self.attention_projection.bias)
        nn.init.normal_(self.mlp_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp_in.bias)
        nn.init.normal_(self.mlp_out.weight, mean=0.0,
                        std=0.02 / math.sqrt(2 * layers))
        nn.init.zeros_(self.mlp_out.bias)
        nn.init.ones_(self.ln_1.weight)
        nn.init.zeros_(self.ln_1.bias)
        nn.init.ones_(self.ln_2.weight)
        nn.init.zeros_(self.ln_2.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(hidden)
        batch, length, channels = normalized.shape
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        hidden = hidden + self.attention_projection(attended)
        normalized = self.ln_2(hidden)
        return hidden + self.mlp_out(F.gelu(self.mlp_in(normalized), approximate="tanh"))


def create_assets(output: Path) -> dict[str, torch.Tensor]:
    path = output / "assets.pt"
    if path.exists():
        assets = torch.load(path, map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed_for("assets"))
        basis = torch.linalg.qr(torch.randn(DIM, DIM, generator=generator), mode="reduced").Q
        library = (2 * torch.rand(4096, DIM, generator=generator) - 1) * THETA_MAX
        eval_theta = (2 * torch.rand(EVAL_COUNT, DIM, generator=generator) - 1) * THETA_MAX
        eval_states = simulate(eval_theta, basis, generator)
        assets = {"basis": basis, "library": library, "eval_theta": eval_theta,
                  "eval_states": eval_states}
        atomic_torch(path, assets)
    required = {"basis", "library", "eval_theta", "eval_states"}
    if set(assets) != required or tuple(assets["library"].shape) != (4096, DIM):
        raise RuntimeError("invalid assets")
    if float(torch.max(torch.abs(assets["basis"].T @ assets["basis"] - torch.eye(DIM)))) > 1e-5:
        raise RuntimeError("basis is not orthogonal")
    return assets


def simulate(theta: torch.Tensor, basis: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    count = len(theta)
    latent = torch.empty(count, STATES, DIM)
    latent[:, 0] = torch.randn(count, DIM, generator=generator)
    for step in range(PREDICTIONS):
        latent[:, step + 1] = theta * latent[:, step] + NOISE * torch.randn(
            count, DIM, generator=generator
        )
    return latent @ basis.T


def training_batch(library: torch.Tensor, basis: torch.Tensor, batch: int,
                   generator: torch.Generator, device: str) -> torch.Tensor:
    indices = torch.randint(len(library), (batch,), generator=generator, device=device)
    theta = library[indices]
    latent = torch.empty(batch, STATES, DIM, device=device)
    latent[:, 0] = torch.randn(batch, DIM, generator=generator, device=device)
    for step in range(PREDICTIONS):
        latent[:, step + 1] = theta * latent[:, step] + NOISE * torch.randn(
            batch, DIM, generator=generator, device=device
        )
    return latent @ basis.T


def learning_rate(step: int) -> float:
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    return PEAK_LR * max(ENDPOINT - step, 0) / (ENDPOINT - WARMUP)


def evaluation_steps(endpoint: int, smoke: bool) -> set[int]:
    if smoke:
        return {0, 1, 3, 6, 12}
    return {0, 1000, 5000, 10_000, *range(25_000, endpoint + 1, 25_000), endpoint}


def continuous_bayes(states: torch.Tensor, basis: torch.Tensor, order: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior predictive means and log2 N2 for every frozen prefix."""
    latent = states.to(torch.float64) @ basis.to(torch.float64)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(order)
    nodes = torch.from_numpy(nodes_np).to(latent) * THETA_MAX
    prior_weights = torch.from_numpy(weights_np / 2).to(latent)
    predictions = torch.empty(len(states), len(PREFIXES), DIM, dtype=torch.float64)
    log2_n2 = torch.empty(len(states), len(PREFIXES), dtype=torch.float64)
    for prefix_index, prefix in enumerate(PREFIXES):
        x = latent[:, :prefix]
        y = latent[:, 1:prefix + 1]
        residual = y[:, :, :, None] - x[:, :, :, None] * nodes[None, None, None, :]
        log_likelihood = -torch.sum(residual * residual, dim=1) / (2 * NOISE**2)
        maximum = log_likelihood.max(dim=-1, keepdim=True).values
        likelihood = torch.exp(log_likelihood - maximum)
        z1 = torch.sum(likelihood * prior_weights, dim=-1)
        z2 = torch.sum(likelihood * likelihood * prior_weights, dim=-1)
        posterior_mean = torch.sum(likelihood * prior_weights * nodes, dim=-1) / z1
        predictions[:, prefix_index] = (posterior_mean * latent[:, prefix]) @ basis.to(torch.float64).T
        log2_n2[:, prefix_index] = torch.sum(torch.log2(z2) - 2 * torch.log2(z1), dim=1)
    return predictions, log2_n2


def discrete_bayes(states: torch.Tensor, basis: torch.Tensor, library: torch.Tensor) -> torch.Tensor:
    latent = states.to(torch.float64) @ basis.to(torch.float64)
    tasks = library.to(torch.float64)
    output = torch.empty(len(states), len(PREFIXES), DIM, dtype=torch.float64)
    for prefix_index, prefix in enumerate(PREFIXES):
        x = latent[:, :prefix]
        y = latent[:, 1:prefix + 1]
        # Sufficient statistics avoid materializing [prompt,time,task,dimension].
        # Terms independent of the task cancel in the softmax.
        sum_xy = torch.sum(x * y, dim=1)
        sum_x2 = torch.sum(x * x, dim=1)
        log_likelihood = (
            sum_xy @ tasks.T - 0.5 * sum_x2 @ (tasks * tasks).T
        ) / NOISE**2
        weights = torch.softmax(log_likelihood, dim=1)
        posterior_theta = weights @ tasks
        output[:, prefix_index] = (posterior_theta * latent[:, prefix]) @ basis.to(torch.float64).T
    return output


def predict(model: nn.Module, states: torch.Tensor, batch: int = 256) -> np.ndarray:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch):
            rows.append(model(states[start:start + batch, :-1]).cpu())
    model.train()
    result = torch.cat(rows).numpy().astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite predictions")
    return result


def identity(M: int, endpoint: int, assets_path: Path, smoke: bool) -> dict[str, Any]:
    return {"stage": "fmts_learned_temporal_coverage_v1", "M": M,
            "endpoint": endpoint, "smoke": smoke,
            "source_sha256": sha256_path(Path(__file__)),
            "preregistration_sha256": sha256_path(PREREG),
            "assets_sha256": sha256_path(assets_path),
            "model_seed": seed_for("model", M), "training_seed": seed_for("training", M),
            "torch": torch.__version__}


def latest_resume(arm: Path, ident: dict, device: str) -> dict | None:
    candidates = []
    for metadata_path in arm.glob("resume_*.json"):
        metadata = json.loads(metadata_path.read_text())
        state_path = arm / metadata["file"]
        if state_path.is_file() and sha256_path(state_path) == metadata["sha256"]:
            candidates.append((int(metadata["step"]), state_path))
    for _, path in sorted(candidates, reverse=True):
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("identity") == ident:
            return state
    if candidates:
        raise RuntimeError("resume identity mismatch")
    return None


def save_resume(arm: Path, step: int, ident: dict, model: nn.Module,
                optimizer: torch.optim.Optimizer, generator: torch.Generator, smoke: bool) -> None:
    interval = 2 if smoke else 5000
    slot = (step // interval) % 2
    path = arm / f"resume_{slot}.pt"
    atomic_torch(path, {"identity": ident, "step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "generator_state": generator.get_state().cpu(),
                        "cpu_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": [x.cpu() for x in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available() else []})
    atomic_json(arm / f"resume_{slot}.json", {"step": step, "file": path.name,
                                              "sha256": sha256_path(path), "identity": ident})


def train(M: int, output: Path, device: str, smoke: bool) -> dict:
    endpoint = 12 if smoke else ENDPOINT
    batch_size = 8 if smoke else 256
    output.mkdir(parents=True, exist_ok=True)
    assets = create_assets(output)
    assets_path = output / "assets.pt"
    basis = assets["basis"].to(device)
    library = assets["library"][:M].contiguous().to(device)
    eval_states = assets["eval_states"][:32 if smoke else EVAL_COUNT].to(device)
    arm = output / f"M{M}"
    arm.mkdir(exist_ok=True)
    torch.manual_seed(seed_for("model", M))
    if device == "cuda":
        torch.cuda.manual_seed_all(seed_for("model", M))
        torch.set_float32_matmul_precision("high")
    model = TemporalTransformer(smoke).to(device)
    ident = identity(M, endpoint, assets_path, smoke)
    identity_path = arm / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != ident:
        raise RuntimeError("identity changed")
    atomic_json(identity_path, ident)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    generator = torch.Generator(device=device).manual_seed(ident["training_seed"])
    state = latest_resume(arm, ident, device)
    if state is None:
        step = 0
    else:
        step = int(state["step"])
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator_state"].cpu())
        torch.set_rng_state(state["cpu_rng_state"].cpu())
        if device == "cuda" and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_states"]])
    metrics_path = arm / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    metrics = [row for row in metrics if int(row["step"]) <= step]
    evals = evaluation_steps(endpoint, smoke)
    latest_loss = None
    started = time.time()

    def evaluate(current: int) -> None:
        nonlocal metrics
        prediction = predict(model, eval_states)
        row = {"step": current, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "latest_train_loss": latest_loss}
        metrics = [old for old in metrics if int(old["step"]) != current] + [row]
        metrics.sort(key=lambda item: int(item["step"]))
        atomic_json(metrics_path, metrics)
        atomic_npz(arm / f"predictions_step_{current:06d}.npz", predictions=prediction)
        status = {"phase": "training" if current < endpoint else "finalizing", "M": M,
                  "step": current, "endpoint": endpoint, "latest_metrics": row,
                  "elapsed_seconds_this_invocation": time.time() - started,
                  "updated_unix": time.time()}
        atomic_json(arm / "status.json", status)
        print(json.dumps(status, sort_keys=True), flush=True)

    if step in evals and not any(int(row["step"]) == step for row in metrics):
        evaluate(step)
    for current in range(step + 1, endpoint + 1):
        lr = learning_rate(current) if not smoke else 1e-3 * current / endpoint
        for group in optimizer.param_groups:
            group["lr"] = lr
        states = training_batch(library, basis, batch_size, generator, device)
        prediction = model(states[:, :-1])
        loss = torch.mean((prediction - states[:, 1:]) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at {current}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        latest_loss = float(loss.item())
        if current % (2 if smoke else 5000) == 0 or current == endpoint:
            save_resume(arm, current, ident, model, optimizer, generator, smoke)
        if current in evals:
            evaluate(current)
    final_model = arm / "final_model.pt"
    atomic_torch(final_model, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    summary = {"phase": "complete", "M": M, "step": endpoint, "identity": ident,
               "latest_metrics": metrics[-1], "parameter_count": sum(p.numel() for p in model.parameters()),
               "final_model_sha256": sha256_path(final_model),
               "endpoint_predictions_sha256": sha256_path(arm / f"predictions_step_{endpoint:06d}.npz")}
    atomic_json(arm / "summary.json", summary)
    atomic_json(arm / "status.json", {"phase": "complete", "M": M, "step": endpoint,
                                      "endpoint": endpoint, "summary": summary,
                                      "updated_unix": time.time()})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def audit_assets(output: Path) -> dict:
    assets = create_assets(output)
    states = assets["eval_states"][:128]
    coarse_pred, coarse_n2 = continuous_bayes(states, assets["basis"], 96)
    fine_pred, fine_n2 = continuous_bayes(states, assets["basis"], 128)
    _, all_n2 = continuous_bayes(assets["eval_states"], assets["basis"], 96)
    result = {"prediction_max_abs_96_vs_128": float(torch.max(torch.abs(coarse_pred - fine_pred))),
              "log2_N2_max_abs_96_vs_128": float(torch.max(torch.abs(coarse_n2 - fine_n2))),
              "maximum_library_abs_theta": float(torch.max(torch.abs(assets["library"]))),
              "basis_orthogonality_max_abs": float(torch.max(torch.abs(
                  assets["basis"].T @ assets["basis"] - torch.eye(DIM)))),
              "coverage_by_prefix": {
                  str(prefix): {
                      "mean_log2_N2": float(all_n2[:, index].mean()),
                      "median_log2_N2": float(torch.quantile(all_n2[:, index], 0.5)),
                      "q025_log2_N2": float(torch.quantile(all_n2[:, index], 0.025)),
                      "q975_log2_N2": float(torch.quantile(all_n2[:, index], 0.975)),
                      "mean_G_M256": float(all_n2[:, index].mean() - 8),
                      "mean_G_M4096": float(all_n2[:, index].mean() - 12),
                  }
                  for index, prefix in enumerate(PREFIXES)
              }}
    atomic_json(output / "quadrature_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, choices=(256, 4096))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-assets", action="store_true")
    args = parser.parse_args()
    if args.audit_assets:
        audit_assets(args.output.resolve())
    elif args.M is None:
        raise SystemExit("--M is required for training")
    else:
        train(args.M, args.output.resolve(), args.device, args.smoke)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resumable parameter-matched GRU baseline for the FMTS coverage paper."""
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


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "fmts_architecture_baseline_preregistration.json"
PROMPTS = ROOT / "fresh_unseen_prompts.npz"
LIBRARY = ROOT / "task_library_M4096.pt"
DIMENSION = 8
POINTS = 16
NOISE = 0.5
M = 4096
ENDPOINT = 290_000
NOMINAL_TOTAL = 500_000
WARMUP = 250_000
PEAK_LR = 1e-3
BASE_SEED = 2026091402
PREFIXES = (3, 7, 11, 15)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(replicate: int, stream: str) -> int:
    payload = f"fmts_boundary_seed_replication_v1|{BASE_SEED}|{replicate}|{stream}".encode()
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


def learning_rate(step: int) -> float:
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    return PEAK_LR * (NOMINAL_TOTAL - step) / (NOMINAL_TOTAL - WARMUP)


def evaluation_steps(endpoint: int, smoke: bool) -> set[int]:
    if smoke:
        return {0, 1, 3, 6, 12}
    return {0, 1000, 5000, 10_000, *range(25_000, endpoint + 1, 25_000), endpoint}


class RecurrentRegressor(nn.Module):
    """Causal GRU using the exact x/y tokenization of PublishedRegressor."""

    def __init__(self, smoke: bool = False) -> None:
        super().__init__()
        projection = 32 if smoke else 128
        hidden = 64 if smoke else 256
        layers = 2 if smoke else 4
        self.input_projection = nn.Linear(DIMENSION + 1, projection, bias=False)
        self.recurrent = nn.GRU(projection, hidden, num_layers=layers, batch_first=True)
        self.final_norm = nn.LayerNorm(hidden, eps=1e-5)
        self.output = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch, points, dimension = data.shape
        data_tokens = torch.cat(
            [torch.zeros(batch, points, 1, device=data.device, dtype=data.dtype), data], dim=2
        )
        target_tokens = torch.cat(
            [targets.unsqueeze(-1), torch.zeros(batch, points, dimension,
                                                device=data.device, dtype=data.dtype)], dim=2
        )
        sequence = torch.stack((data_tokens, target_tokens), dim=2).reshape(
            batch, 2 * points, dimension + 1
        )
        hidden, _ = self.recurrent(self.input_projection(sequence))
        predictions = self.output(self.final_norm(hidden)).squeeze(-1)
        return predictions[:, ::2]


def load_assets(device: str, smoke: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    library = torch.load(LIBRARY, map_location="cpu", weights_only=True)
    if not isinstance(library, torch.Tensor) or tuple(library.shape) != (M, DIMENSION):
        raise RuntimeError("uploaded task library has the wrong shape")
    with np.load(PROMPTS, allow_pickle=False) as saved:
        if set(saved.files) != {"data", "tasks", "targets", "signal"}:
            raise RuntimeError("unexpected prompt schema")
        count = 32 if smoke else 4096
        prompts = {key: torch.from_numpy(saved[key][:count]).to(device) for key in saved.files}
    if tuple(prompts["data"].shape) != (count, POINTS, DIMENSION):
        raise RuntimeError("unexpected prompt shape")
    return library.to(device), prompts


def ridge_predictions(data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(targets)
    identity = torch.eye(DIMENSION, dtype=data.dtype, device=data.device)[None]
    for k in range(1, POINTS):
        x = data[:, :k]
        y = targets[:, :k]
        gram = x.transpose(1, 2) @ x
        rhs = x.transpose(1, 2) @ y.unsqueeze(-1)
        mean = torch.linalg.solve(gram + NOISE**2 * identity, rhs).squeeze(-1)
        result[:, k] = torch.sum(data[:, k] * mean, dim=1)
    return result


def predict(model: nn.Module, prompts: dict[str, torch.Tensor], batch: int = 256) -> np.ndarray:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts["data"]), batch):
            rows.append(model(prompts["data"][start:start + batch],
                              prompts["targets"][start:start + batch]).cpu())
    model.train()
    output = torch.cat(rows).numpy().astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite predictions")
    return output


def identity(replicate: int, endpoint: int, model: nn.Module, smoke: bool) -> dict[str, Any]:
    return {
        "stage": "fmts_parameter_matched_gru_v1",
        "replicate": replicate,
        "M": M,
        "endpoint": endpoint,
        "smoke": smoke,
        "preregistration_sha256": sha256_path(PREREG),
        "source_sha256": sha256_path(Path(__file__)),
        "prompt_sha256": sha256_path(PROMPTS),
        "library_sha256": sha256_path(LIBRARY),
        "model_seed": derived_seed(replicate, "model"),
        "training_seed": derived_seed(replicate, "training"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch": torch.__version__,
    }


def latest_resume(arm: Path, ident: dict[str, Any], device: str) -> dict[str, Any] | None:
    candidates = []
    for metadata_path in arm.glob("resume_*.json"):
        metadata = json.loads(metadata_path.read_text())
        state_path = arm / metadata["file"]
        if state_path.is_file() and sha256_path(state_path) == metadata["sha256"]:
            candidates.append((int(metadata["step"]), state_path))
    for _, state_path in sorted(candidates, reverse=True):
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("identity") == ident:
            return state
    if candidates:
        raise RuntimeError("resume state exists but does not match the frozen identity")
    return None


def save_resume(arm: Path, step: int, ident: dict[str, Any], model: nn.Module,
                optimizer: torch.optim.Optimizer, generator: torch.Generator, smoke: bool) -> None:
    interval = 2 if smoke else 5000
    slot = (step // interval) % 2
    path = arm / f"resume_{slot}.pt"
    payload = {
        "identity": ident,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": generator.get_state().cpu(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }
    atomic_torch(path, payload)
    atomic_json(arm / f"resume_{slot}.json", {
        "step": step, "file": path.name, "sha256": sha256_path(path), "identity": ident,
    })


def train(replicate: int, output: Path, device: str, smoke: bool) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    endpoint = 12 if smoke else ENDPOINT
    batch_size = 8 if smoke else 256
    library, prompts = load_assets(device, smoke)
    arm = output / f"replicate_{replicate}" / "GRU_M4096"
    arm.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(derived_seed(replicate, "model"))
    if device == "cuda":
        torch.cuda.manual_seed_all(derived_seed(replicate, "model"))
        torch.set_float32_matmul_precision("high")
    model = RecurrentRegressor(smoke).to(device)
    ident = identity(replicate, endpoint, model, smoke)
    identity_path = arm / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != ident:
        raise RuntimeError("arm identity changed; refusing mixed resume")
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
        generator.set_state(state["training_generator_state"].cpu())
        torch.set_rng_state(state["cpu_rng_state"].cpu())
        if device == "cuda" and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])

    ridge = ridge_predictions(prompts["data"], prompts["targets"]).cpu().numpy()
    metrics_path = arm / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    metrics = [row for row in metrics if int(row["step"]) <= step]
    evals = evaluation_steps(endpoint, smoke)
    latest_loss = None
    started = time.time()

    def status(phase: str, current: int, **extra: Any) -> None:
        payload = {"phase": phase, "replicate": replicate, "step": current,
                   "endpoint": endpoint, "updated_unix": time.time(),
                   "elapsed_seconds_this_invocation": time.time() - started, **extra}
        atomic_json(arm / "status.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    def evaluate(current: int) -> None:
        nonlocal metrics
        predictions = predict(model, prompts)
        target = prompts["targets"].cpu().numpy().astype(np.float64)
        ratios = {}
        for k in PREFIXES:
            ratios[str(k)] = float(
                np.mean((predictions[:, k].astype(np.float64) - target[:, k]) ** 2)
                / np.mean((ridge[:, k].astype(np.float64) - target[:, k]) ** 2)
            )
        row = {"step": current, "learning_rate": learning_rate(current),
               "latest_train_loss": latest_loss, "noisy_risk_ratio_by_prefix": ratios}
        metrics = [old for old in metrics if int(old["step"]) != current] + [row]
        metrics.sort(key=lambda item: int(item["step"]))
        atomic_json(metrics_path, metrics)
        atomic_npz(arm / f"predictions_step_{current:06d}.npz", predictions=predictions)
        status("training" if current < endpoint else "finalizing", current, latest_metrics=row)

    if step in evals and not any(int(row["step"]) == step for row in metrics):
        evaluate(step)
    for current in range(step + 1, endpoint + 1):
        lr = learning_rate(current) if not smoke else 1e-3 * current / endpoint
        for group in optimizer.param_groups:
            group["lr"] = lr
        data = torch.randn(batch_size, POINTS, DIMENSION, generator=generator, device=device)
        indices = torch.randint(M, (batch_size,), generator=generator, device=device)
        tasks = library[indices]
        targets = torch.einsum("bpd,bd->bp", data, tasks)
        targets += NOISE * torch.randn(targets.shape, generator=generator, device=device)
        predictions = model(data, targets)
        loss = torch.mean((predictions - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {current}")
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
    changed = prompts["targets"][:128].clone()
    changed[:, -1] += 1_000_000
    model.eval()
    with torch.inference_mode():
        causal_error = float(torch.max(torch.abs(
            model(prompts["data"][:128], prompts["targets"][:128])[:, -1]
            - model(prompts["data"][:128], changed)[:, -1]
        )).cpu())
    if causal_error > 1e-7:
        raise RuntimeError(f"causal query-label intervention failed: {causal_error}")
    summary = {"phase": "complete", "replicate": replicate, "step": endpoint,
               "identity": ident, "latest_metrics": metrics[-1],
               "causal_query_label_max_abs": causal_error,
               "final_model_sha256": sha256_path(final_model),
               "endpoint_predictions_sha256": sha256_path(arm / f"predictions_step_{endpoint:06d}.npz")}
    atomic_json(arm / "summary.json", summary)
    status("complete", endpoint, summary=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.replicate, args.output.resolve(), args.device, args.smoke),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired M=4096 seed replication with dense fixed-time prediction checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import fmts_seed_replication as base


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "fmts_m4096_replication_preregistration.json"
M = 4096
ENDPOINT = 290_000
PREFIXES = (3, 7, 11, 15)


def evaluation_steps(endpoint: int, smoke: bool) -> set[int]:
    if smoke:
        return {0, 1, 3, 6, 12}
    return {0, 1000, 5000, 10_000, *range(25_000, endpoint + 1, 25_000), endpoint}


def prepare_library(output: Path, replicate: int, smoke: bool) -> tuple[Path, Path, torch.Tensor]:
    prefix_path = ROOT / "task_library_M2048.pt"
    prefix = torch.load(prefix_path, map_location="cpu", weights_only=True)
    if not isinstance(prefix, torch.Tensor) or tuple(prefix.shape) != (2048, base.DIMENSION):
        raise RuntimeError("uploaded M2048 prefix has the wrong shape")
    extension_generator = torch.Generator(device="cpu").manual_seed(
        base.derived_seed(replicate, "library_M4096_extension")
    )
    if smoke:
        check_rows = 32
        extension = torch.randn((32, base.DIMENSION), generator=extension_generator, dtype=torch.float32)
        generated = torch.cat((prefix[:check_rows], extension), dim=0)
    else:
        check_rows = len(prefix)
        extension = torch.randn((M - check_rows, base.DIMENSION),
                                generator=extension_generator, dtype=torch.float32)
        generated = torch.cat((prefix, extension), dim=0)
    if not torch.equal(generated[:check_rows], prefix[:check_rows]):
        raise RuntimeError("M4096 library is not a literal extension of the retained M2048 library")
    rep_root = output / f"replicate_{replicate}"
    library_path = rep_root / "task_library_M4096.pt"
    base.atomic_torch(library_path, generated)
    base.atomic_json(rep_root / "library_audit.json", {
        "replicate": replicate,
        "shape": list(generated.shape),
        "sha256": base.sha256_path(library_path),
        "prefix_source_sha256": base.sha256_path(prefix_path),
        "prefix_equal_rows": check_rows,
        "coordinate_mean": float(generated.double().mean()),
        "coordinate_std": float(generated.double().std(unbiased=False)),
    })
    return prefix_path, library_path, generated


def identity(replicate: int, prefix_path: Path, library_path: Path, smoke: bool) -> dict:
    return {
        "stage": "fmts_M4096_seed_replication_v1",
        "replicate": replicate,
        "M": M,
        "endpoint": 12 if smoke else ENDPOINT,
        "smoke": smoke,
        "preregistration_sha256": base.sha256_path(PREREG),
        "source_sha256": base.sha256_path(Path(__file__)),
        "base_source_sha256": base.sha256_path(ROOT / "fmts_seed_replication.py"),
        "model_source_sha256": base.sha256_path(ROOT / "positive_control_training.py"),
        "prompt_sha256": base.sha256_path(ROOT / "fresh_unseen_prompts.npz"),
        "prefix_source_sha256": base.sha256_path(prefix_path),
        "library_sha256": base.sha256_path(library_path),
        "model_seed": base.derived_seed(replicate, "model"),
        "training_seed": base.derived_seed(replicate, "training"),
        "torch": torch.__version__,
    }


def save_resume(arm: Path, step: int, ident: dict, model: torch.nn.Module,
                optimizer: torch.optim.Optimizer, generator: torch.Generator, smoke: bool) -> None:
    interval = 2 if smoke else 5000
    slot = (step // interval) % 2
    state_path = arm / f"resume_{slot}.pt"
    state = {
        "identity": ident,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": generator.get_state().cpu(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }
    base.atomic_torch(state_path, state)
    base.atomic_json(arm / f"resume_{slot}.json", {
        "step": step,
        "file": state_path.name,
        "sha256": base.sha256_path(state_path),
        "identity": ident,
    })


def latest_resume(arm: Path, ident: dict, device: str) -> dict | None:
    candidates = []
    for metadata_path in arm.glob("resume_*.json"):
        metadata = json.loads(metadata_path.read_text())
        state_path = arm / metadata["file"]
        if state_path.is_file() and base.sha256_path(state_path) == metadata["sha256"]:
            candidates.append((int(metadata["step"]), state_path))
    for _, state_path in sorted(candidates, reverse=True):
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("identity") == ident:
            return state
    if candidates:
        raise RuntimeError("resume state exists but does not match the frozen identity")
    return None


def train(replicate: int, output: Path, device: str, smoke: bool) -> dict:
    if replicate not in (1, 2):
        raise ValueError("replicate must be 1 or 2")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    endpoint = 12 if smoke else ENDPOINT
    batch_size = 8 if smoke else 256
    prefix_path, library_path, library_cpu = prepare_library(output, replicate, smoke)
    library = library_cpu.to(device)
    prompts = base.load_prompts(device, smoke)
    ridge = base.published_ridge_predictions(prompts["data"], prompts["targets"], base.NOISE).cpu().numpy()
    arm = output / f"replicate_{replicate}" / "M4096"
    arm.mkdir(parents=True, exist_ok=True)
    ident = identity(replicate, prefix_path, library_path, smoke)
    identity_path = arm / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != ident:
        raise RuntimeError("arm identity changed; refusing a mixed resume")
    base.atomic_json(identity_path, ident)

    torch.manual_seed(ident["model_seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(ident["model_seed"])
        torch.set_float32_matmul_precision("high")
    model = base.make_model(device, smoke)
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
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_states"]])

    metrics_path = arm / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    metrics = [row for row in metrics if int(row["step"]) <= step]
    evals = evaluation_steps(endpoint, smoke)
    latest_loss = None
    started = time.time()

    def evaluate(current: int) -> None:
        nonlocal metrics
        prediction = base.predict(model, prompts)
        targets = prompts["targets"].cpu().numpy().astype(np.float64)
        ratios = {}
        for k in PREFIXES:
            numerator = np.mean((prediction[:, k].astype(np.float64) - targets[:, k]) ** 2)
            denominator = np.mean((ridge[:, k].astype(np.float64) - targets[:, k]) ** 2)
            ratios[str(k)] = float(numerator / denominator)
        row = {
            "step": current,
            "learning_rate": base.learning_rate(current),
            "latest_train_loss": latest_loss,
            "noisy_risk_ratio_by_prefix": ratios,
            "all_prefix_pass": all(value < 1.2 for value in ratios.values()),
        }
        metrics = [old for old in metrics if int(old["step"]) != current] + [row]
        metrics.sort(key=lambda item: int(item["step"]))
        base.atomic_json(metrics_path, metrics)
        base.atomic_npz(arm / f"predictions_step_{current:06d}.npz", predictions=prediction)
        base.save_status(arm / "status.json", "training" if current < endpoint else "finalizing",
                         replicate=replicate, M=M, step=current, endpoint=endpoint,
                         latest_metrics=row, elapsed_seconds_this_invocation=time.time() - started)

    if step in evals and not any(int(row["step"]) == step for row in metrics):
        evaluate(step)
    for current in range(step + 1, endpoint + 1):
        lr = base.learning_rate(current) if not smoke else 1e-3 * current / endpoint
        for group in optimizer.param_groups:
            group["lr"] = lr
        data = torch.randn(batch_size, base.POINTS, base.DIMENSION, generator=generator, device=device)
        indices = torch.randint(len(library), (batch_size,), generator=generator, device=device)
        tasks = library[indices]
        targets = torch.einsum("bpd,bd->bp", data, tasks)
        targets += base.NOISE * torch.randn(targets.shape, generator=generator, device=device)
        output_values = model(data, targets)
        loss = torch.mean((output_values - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {current}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        latest_loss = float(loss.item())
        interval = 2 if smoke else 5000
        if current % interval == 0 or current == endpoint:
            save_resume(arm, current, ident, model, optimizer, generator, smoke)
        if current in evals:
            evaluate(current)

    base.atomic_torch(arm / "final_model.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
    changed = prompts["targets"][:128].clone()
    changed[:, -1] += 1_000_000
    model.eval()
    with torch.inference_mode():
        ordinary = model(prompts["data"][:128], prompts["targets"][:128])[:, -1]
        intervened = model(prompts["data"][:128], changed)[:, -1]
    causal_error = float(torch.max(torch.abs(ordinary - intervened)).cpu())
    if causal_error > 1e-7:
        raise RuntimeError(f"causal query-label intervention failed: {causal_error}")
    summary = {
        "phase": "complete",
        "replicate": replicate,
        "M": M,
        "step": endpoint,
        "identity": ident,
        "latest_metrics": metrics[-1],
        "causal_query_label_max_abs": causal_error,
        "final_model_sha256": base.sha256_path(arm / "final_model.pt"),
    }
    base.atomic_json(arm / "summary.json", summary)
    base.save_status(arm / "status.json", "complete", replicate=replicate, M=M, step=endpoint,
                     endpoint=endpoint, summary=summary,
                     elapsed_seconds_this_invocation=time.time() - started)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fmts-m4096-replication"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = train(args.replicate, args.output.resolve(), args.device, args.smoke)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

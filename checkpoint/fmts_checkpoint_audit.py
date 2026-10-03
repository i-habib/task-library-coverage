#!/usr/bin/env python3
"""FMTS checkpoint extraction and fresh-context inference audit.

This file intentionally has no network, cloud, subprocess, or training path.
It verifies the four source snapshots, extracts model-only state dictionaries,
and evaluates fresh prompts against exact ridge and finite-library dMMSE
baselines. An explicitly exported model-only bundle can be replayed with
``--use-extracted``; no training or optimizer state is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT_ROOT = Path(
    "ANONYMIZED_SOURCE_SNAPSHOTS"
)
OUTPUT_ROOT = ROOT / "artifacts" / "fmts-checkpoints"
INFERENCE_SEED = 2026091414
DIMENSION = 8
POINTS = 16
NOISE = 0.5
NOISE_SQUARED = NOISE**2
PREFIXES = (3, 7, 11, 15)
PROMPTS = 4096
BOOTSTRAP_REPLICATES = 2000

CONDITIONS = (
    ("M256", 8, 256, 300000),
    ("M4096", 12, 4096, 290000),
    ("M8192", 13, 8192, 280000),
    ("M16384", 14, 16384, 280000),
)

EXPECTED_EXTRACTED_SHA256 = {
    "model_M256_step300000.pt": "b4b0c753f34a1a0c279439918cf0b915cd54abe8368c1d4f445584bd343572bc",
    "model_M4096_step290000.pt": "6b25b20579a6868b393739d1b190ca5383e447fd0fe0de32a0be169b82e91a6c",
    "model_M8192_step280000.pt": "40c6e78d02394b20529825b72682ff4d908512235143f1263416b4d1bfe222a5",
    "model_M16384_step280000.pt": "a38e22e592d0c9643e4e1a5320a6503f02e9adbc315fa2761dd66d97d4230345",
    "master_library_prefix_M16384.pt": "61719fda90c09f53e15ffb9ab6707dcb78abbe233920234b9159b4e449dd5688",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_fileobj(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_extracted_bundle(output_root: Path) -> dict[str, Any]:
    """Verify the exact model-only bundle used for a cloud replay."""
    rows = []
    for filename, expected in EXPECTED_EXTRACTED_SHA256.items():
        path = output_root / filename
        if not path.is_file():
            raise RuntimeError(f"missing extracted replay file: {path}")
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"replay file hash mismatch for {filename}: {actual} != {expected}")
        rows.append({"file": filename, "sha256": actual, "bytes": path.stat().st_size})
    return {
        "mode": "verified_extracted_model_only_bundle",
        "files": rows,
        "optimizer_or_rng_state_present": False,
    }


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"missing archive member: {name}")
    with extracted:
        return extracted.read()


def member_sha256(archive: tarfile.TarFile, name: str) -> str:
    member = archive.getmember(name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"missing archive member: {name}")
    with extracted:
        return sha256_fileobj(extracted)


def condition_archive(snapshot_root: Path, name: str) -> Path:
    matches = sorted(snapshot_root.glob(f"screen-{name}-step*.tar.gz"))
    expected = {condition[0]: condition[3] for condition in CONDITIONS}[name]
    matches = [path for path in matches if f"step{expected}.tar.gz" in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one archive for {name}, found {matches}")
    return matches[0]


def verify_and_extract(snapshot_root: Path, output_root: Path) -> dict[str, Any]:
    """Verify all manifest entries and save only model parameter tensors."""
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    library_tensor: torch.Tensor | None = None
    expected_model_source: str | None = None
    expected_spec: str | None = None
    expected_master: str | None = None

    for name, log2_m, m, step in CONDITIONS:
        archive_path = condition_archive(snapshot_root, name)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            manifest = json.loads(member_bytes(archive, "snapshot_manifest.json"))
            if not isinstance(manifest, dict) or not manifest:
                raise RuntimeError(f"invalid snapshot manifest in {archive_path}")
            actual_members = set(archive.getnames())
            missing = sorted(set(manifest) - actual_members)
            if missing:
                raise RuntimeError(f"manifest members missing from {archive_path}: {missing}")
            mismatches = []
            for member_name, expected_hash in sorted(manifest.items()):
                actual_hash = member_sha256(archive, member_name)
                if actual_hash != expected_hash:
                    mismatches.append((member_name, expected_hash, actual_hash))
            if mismatches:
                raise RuntimeError(f"manifest hash mismatch in {archive_path}: {mismatches}")

            identity = json.loads(member_bytes(archive, "identity.json"))
            resume_metadata = json.loads(member_bytes(archive, "resume_0.json"))
            status = json.loads(member_bytes(archive, "status.json"))
            if int(identity["log2_M"]) != log2_m:
                raise RuntimeError(f"identity M mismatch in {archive_path}: {identity}")
            if int(resume_metadata["step"]) != step:
                raise RuntimeError(f"resume step mismatch in {archive_path}: {resume_metadata}")
            if int(status["step"]) != step or int(status["M"]) != m:
                raise RuntimeError(f"status mismatch in {archive_path}: {status}")
            if manifest["resume_0.pt"] != resume_metadata["sha256"]:
                raise RuntimeError(f"resume hash metadata mismatch in {archive_path}")

            model_source = identity["model_source_sha256"]
            spec_hash = identity["spec_sha256"]
            master_hash = identity["master_library_sha256"]
            if expected_model_source is None:
                expected_model_source = model_source
                expected_spec = spec_hash
                expected_master = master_hash
            elif (model_source, spec_hash, master_hash) != (
                expected_model_source,
                expected_spec,
                expected_master,
            ):
                raise RuntimeError(f"source identity differs across snapshots: {archive_path}")

            resume_payload = torch.load(
                io.BytesIO(member_bytes(archive, "resume_0.pt")),
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(resume_payload, dict) or "model" not in resume_payload:
                raise RuntimeError(f"unexpected checkpoint payload in {archive_path}")
            model_state = resume_payload["model"]
            if not isinstance(model_state, dict) or not model_state:
                raise RuntimeError(f"empty model state in {archive_path}")
            if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                       for key, value in model_state.items()):
                raise RuntimeError(f"non-tensor model parameter in {archive_path}")
            if any(key.startswith(("optimizer", "generator", "rng")) for key in model_state):
                raise RuntimeError(f"nested optimizer/RNG key in model state: {archive_path}")
            if "optimizer" not in resume_payload or "generator_state" not in resume_payload:
                raise RuntimeError(f"full-state checkpoint unexpectedly lacks audit fields: {archive_path}")

            cpu_state = {key: value.detach().cpu().clone() for key, value in model_state.items()}
            model_path = output_root / f"model_{name}_step{step}.pt"
            atomic_torch_save(model_path, cpu_state)

            if library_tensor is None:
                library_tensor = torch.load(
                    io.BytesIO(member_bytes(archive, "source/transition_master_library.pt")),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(library_tensor, torch.Tensor):
                    raise RuntimeError("master library is not a tensor")
                if tuple(library_tensor.shape) != (2**20, DIMENSION):
                    raise RuntimeError(f"unexpected master library shape: {library_tensor.shape}")
                if library_tensor.dtype != torch.float32:
                    raise RuntimeError(f"unexpected master library dtype: {library_tensor.dtype}")

            prefix = library_tensor[:m].contiguous()
            prefix_raw_hash = sha256_bytes(prefix.numpy().tobytes(order="C"))
            rows.append(
                {
                    "condition": name,
                    "M": m,
                    "log2_M": log2_m,
                    "step": step,
                    "archive": str(archive_path),
                    "archive_sha256": sha256_path(archive_path),
                    "manifest_entries": len(manifest),
                    "manifest_verified": True,
                    "model_parameter_file": str(model_path),
                    "model_parameter_sha256": sha256_path(model_path),
                    "model_parameter_count": len(cpu_state),
                    "model_parameter_bytes": int(sum(value.numel() * value.element_size() for value in cpu_state.values())),
                    "resume_top_level_keys": sorted(resume_payload),
                    "optimizer_excluded": True,
                    "rng_excluded": True,
                    "model_source_sha256": model_source,
                    "spec_sha256": spec_hash,
                    "master_library_sha256": master_hash,
                    "library_prefix_raw_sha256": prefix_raw_hash,
                    "library_prefix_shape": list(prefix.shape),
                }
            )

    if library_tensor is None or expected_master is None:
        raise RuntimeError("no library extracted")
    local_model_source = ROOT / "positive_control_training.py"
    local_model_source_sha256 = sha256_path(local_model_source) if local_model_source.is_file() else None
    if local_model_source_sha256 != expected_model_source:
        raise RuntimeError(
            "local positive_control_training.py does not match the checkpoint source identity: "
            f"{local_model_source_sha256} != {expected_model_source}"
        )
    prefix_path = output_root / "master_library_prefix_M16384.pt"
    atomic_torch_save(prefix_path, library_tensor[: max(item[2] for item in CONDITIONS)].contiguous())
    extraction = {
        "snapshot_root": str(snapshot_root),
        "conditions": rows,
        "common_model_source_sha256": expected_model_source,
        "local_model_source_sha256": local_model_source_sha256,
        "common_spec_sha256": expected_spec,
        "common_master_library_sha256": expected_master,
        "library_prefix_file": str(prefix_path),
        "library_prefix_file_sha256": sha256_path(prefix_path),
        "library_prefix_file_shape": list(library_tensor[:16384].shape),
        "library_prefix_file_dtype": str(library_tensor.dtype),
        "local_only": True,
        "export_performed": False,
    }
    write_json(output_root / "extraction_audit.json", extraction)
    return extraction


class GPT2Block(nn.Module):
    """Byte-for-byte architecture equivalent of the published source class."""

    def __init__(self, d_model: int, heads: int, layers: int, mlp_hidden: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attention_projection = nn.Linear(d_model, d_model, bias=True)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp_in = nn.Linear(d_model, mlp_hidden, bias=True)
        self.mlp_out = nn.Linear(mlp_hidden, d_model, bias=True)
        self.reset_parameters(layers)

    def reset_parameters(self, layers: int) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.attention_projection.weight, mean=0.0, std=0.02 / math.sqrt(2 * layers))
        nn.init.zeros_(self.attention_projection.bias)
        nn.init.normal_(self.mlp_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp_in.bias)
        nn.init.normal_(self.mlp_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * layers))
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
        hidden = hidden + self.mlp_out(F.gelu(self.mlp_in(normalized), approximate="tanh"))
        return hidden


class PublishedRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dimension = DIMENSION
        self.points = POINTS
        self.input_projection = nn.Linear(DIMENSION + 1, 128, bias=False)
        self.position = nn.Embedding(2 * POINTS, 128)
        self.blocks = nn.ModuleList([GPT2Block(128, 2, 8, 512) for _ in range(8)])
        self.final_norm = nn.LayerNorm(128, eps=1e-5)
        self.output = nn.Linear(128, 1, bias=False)

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch, points, dimension = data.shape
        data_tokens = torch.cat(
            [torch.zeros(batch, points, 1, device=data.device, dtype=data.dtype), data], dim=2
        )
        target_tokens = torch.cat(
            [targets.unsqueeze(-1), torch.zeros(batch, points, dimension, device=data.device, dtype=data.dtype)],
            dim=2,
        )
        sequence = torch.stack((data_tokens, target_tokens), dim=2).reshape(
            batch, 2 * points, dimension + 1
        )
        positions = torch.arange(2 * points, device=data.device)
        hidden = self.input_projection(sequence) + self.position(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        outputs = self.output(self.final_norm(hidden)).squeeze(-1)
        return outputs[:, ::2]


def exact_ridge_predictions(data: torch.Tensor, targets: torch.Tensor, k: int) -> np.ndarray:
    data64 = data[:, :k].to(torch.float64)
    targets64 = targets[:, :k].to(torch.float64)
    query64 = data[:, k].to(torch.float64)
    gram = data64.transpose(1, 2) @ data64
    rhs = data64.transpose(1, 2) @ targets64.unsqueeze(-1)
    identity = torch.eye(DIMENSION, dtype=torch.float64, device=data.device)[None]
    posterior_mean = torch.linalg.solve(gram + NOISE_SQUARED * identity, rhs).squeeze(-1)
    return torch.sum(query64 * posterior_mean, dim=1).cpu().numpy()


def exact_log2_n2_batch(data: np.ndarray, targets: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    hessian = np.einsum("bni,bnj->bij", data, data) / NOISE_SQUARED
    linear = np.einsum("bni,bn->bi", data, targets) / NOISE_SQUARED
    identity = np.eye(DIMENSION, dtype=np.float64)[None]
    j1, j2 = identity + hessian, identity + 2 * hessian
    sign1, logdet1 = np.linalg.slogdet(j1)
    sign2, logdet2 = np.linalg.slogdet(j2)
    if not np.all(sign1 > 0) or not np.all(sign2 > 0):
        raise RuntimeError("non-positive Gaussian precision")
    q1 = np.einsum("bi,bi->b", linear, np.linalg.solve(j1, linear[..., None])[..., 0])
    q2 = np.einsum("bi,bi->b", linear, np.linalg.solve(j2, linear[..., None])[..., 0])
    return (logdet1 - 0.5 * logdet2 + 2 * q2 - q1) / math.log(2.0)


def exact_log2_n2_reference(data: np.ndarray, targets: np.ndarray) -> np.ndarray:
    rows = []
    for x, y in zip(np.asarray(data, dtype=np.float64), np.asarray(targets, dtype=np.float64)):
        hessian = x.T @ x / NOISE_SQUARED
        linear = x.T @ y / NOISE_SQUARED
        j1 = np.eye(DIMENSION) + hessian
        j2 = np.eye(DIMENSION) + 2 * hessian
        _, ld1 = np.linalg.slogdet(j1)
        _, ld2 = np.linalg.slogdet(j2)
        rows.append((ld1 - 0.5 * ld2 + 2 * linear @ np.linalg.solve(j2, linear) - linear @ np.linalg.solve(j1, linear)) / math.log(2.0))
    return np.asarray(rows)


def exact_atomic_predictions(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    library: torch.Tensor,
    *,
    task_chunk: int = 32768,
    prompt_chunk: int = 64,
) -> np.ndarray:
    """Stream every atom in the actual prefix, accumulating posterior moments in float64."""
    predictions = torch.empty(len(context_x), dtype=torch.float64, device="cpu")
    for prompt_start in range(0, len(context_x), prompt_chunk):
        prompt_stop = min(len(context_x), prompt_start + prompt_chunk)
        x = context_x[prompt_start:prompt_stop]
        y = context_y[prompt_start:prompt_stop]
        query = query_x[prompt_start:prompt_stop]
        gram = x.transpose(1, 2) @ x
        aty = (x.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)
        count = prompt_stop - prompt_start
        maximum = torch.full((count,), -torch.inf, dtype=torch.float64, device=x.device)
        total = torch.zeros(count, dtype=torch.float64, device=x.device)
        weighted = torch.zeros((count, DIMENSION), dtype=torch.float64, device=x.device)
        for task_start in range(0, len(library), task_chunk):
            tasks = library[task_start:task_start + task_chunk]
            linear = aty @ tasks.T
            quadratic = torch.einsum("ki,pij,kj->pk", tasks, gram, tasks)
            log_weight = (linear - 0.5 * quadratic) / NOISE_SQUARED
            block_maximum = log_weight.max(dim=1).values.to(torch.float64)
            new_maximum = torch.maximum(maximum, block_maximum)
            old_scale = torch.where(
                torch.isfinite(maximum), torch.exp(maximum - new_maximum), 0.0
            )
            shifted = torch.exp(log_weight.to(torch.float64) - new_maximum[:, None])
            total = total * old_scale + shifted.sum(dim=1)
            weighted = weighted * old_scale[:, None] + shifted @ tasks.to(torch.float64)
            maximum = new_maximum
        posterior_mean = weighted / total[:, None]
        predictions[prompt_start:prompt_stop] = torch.sum(
            posterior_mean * query.to(torch.float64), dim=1
        ).cpu()
    return predictions.numpy()


def materialized_atomic_float64(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    library: np.ndarray,
) -> np.ndarray:
    rows = []
    for x, y, query in zip(context_x, context_y, query_x):
        residual = y[None] - library @ x.T
        log_weight = -np.sum(residual * residual, axis=1) / (2 * NOISE_SQUARED)
        log_weight -= log_weight.max()
        weights = np.exp(log_weight)
        weights /= weights.sum()
        rows.append(weights @ (library @ query))
    return np.asarray(rows, dtype=np.float64)


def derive_seed(*values: int) -> int:
    sequence = np.random.SeedSequence([INFERENCE_SEED, *values])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def fresh_prompts(count: int, seed: int) -> dict[str, np.ndarray]:
    """Generate one shared unseen-task bank without consulting any library."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((count, POINTS, DIMENSION), dtype=np.float32)
    tasks = rng.standard_normal((count, DIMENSION), dtype=np.float32)
    signal = np.einsum("bpd,bd->bp", data, tasks, dtype=np.float32)
    targets = signal + np.float32(NOISE) * rng.standard_normal(signal.shape, dtype=np.float32)
    return {
        "data": data,
        "tasks": tasks,
        "targets": targets.astype(np.float32),
        "signal": signal.astype(np.float32),
    }


def validate_fresh_prompt_generator(count: int, seed: int) -> dict[str, Any]:
    """Test determinism, seed isolation, Gaussian task moments, and no index path."""
    first = fresh_prompts(count, seed)
    replay = fresh_prompts(count, seed)
    if set(first) != {"data", "tasks", "targets", "signal"}:
        raise RuntimeError(f"fresh prompt schema exposes a non-Gaussian/index field: {sorted(first)}")
    if not all(np.array_equal(first[key], replay[key]) for key in first):
        raise RuntimeError("fresh prompt generator is not deterministic under the frozen seed")
    alternate = fresh_prompts(count, seed + 1)
    if np.array_equal(first["tasks"], alternate["tasks"]):
        raise RuntimeError("fresh task seed isolation failed")
    task_mean = float(np.mean(first["tasks"], dtype=np.float64))
    task_std = float(np.std(first["tasks"], dtype=np.float64))
    if abs(task_mean) >= 0.1 or not (0.9 < task_std < 1.1):
        raise RuntimeError(f"fresh task Gaussian moment check failed: mean={task_mean}, std={task_std}")
    if not np.all(np.isfinite(first["tasks"])):
        raise RuntimeError("fresh Gaussian task draw is non-finite")
    return {
        "task_generator": "numpy Generator.standard_normal directly into task vectors; no library indexing",
        "schema": sorted(first),
        "deterministic_replay": True,
        "alternate_seed_changes_tasks": True,
        "task_mean": task_mean,
        "task_std": task_std,
    }


def infer_model(
    model_path: Path,
    prompts: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model = PublishedRegressor().to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"model-only parameter file is malformed: {model_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    data = torch.from_numpy(prompts["data"]).to(device)
    targets = torch.from_numpy(prompts["targets"]).to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            outputs.append(model(data[start:start + batch_size], targets[start:start + batch_size]).cpu())
    result = torch.cat(outputs).numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"non-finite model prediction from {model_path}")
    return result


def causal_mask_checks(
    model_path: Path,
    prompts: dict[str, np.ndarray],
    prediction: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    model = PublishedRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    data = torch.from_numpy(prompts["data"])
    targets = torch.from_numpy(prompts["targets"])
    checks = {}
    with torch.inference_mode():
        for k in PREFIXES:
            changed = targets.clone()
            generator = torch.Generator(device="cpu").manual_seed(derive_seed(9000, k))
            changed[:, k:] = torch.randn(changed[:, k:].shape, generator=generator)
            altered_batches = []
            for start in range(0, len(data), batch_size):
                altered_batches.append(
                    model(
                        data[start:start + batch_size].to(device),
                        changed[start:start + batch_size].to(device),
                    ).cpu()
                )
            altered = torch.cat(altered_batches).numpy()
            checks[f"k{k}"] = float(np.max(np.abs(altered[:, k] - prediction[:, k])))
    return checks


def bootstrap_ci(
    prompt_values: dict[str, np.ndarray],
    seed: int,
    replicates: int,
    batch_size: int = 128,
) -> dict[str, list[float]]:
    n = len(next(iter(prompt_values.values())))
    rng = np.random.default_rng(seed)
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "noisy_risk_ratio", "signal_risk_ratio", "Delta_C", "Delta_D", "S", "G_mean", "C"
    )}
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        indices = rng.integers(0, n, size=(count, n), dtype=np.int64)
        means = {key: np.mean(values[indices], axis=1) for key, values in prompt_values.items()}
        denominator = means["ld"] - means["lc"]
        c = (means["lt"] - means["lc"]) / denominator
        s = (means["dd"] - means["dc"]) / np.maximum(means["dd"] + means["dc"], 1e-30)
        derived = {
            "noisy_risk_ratio": means["lt"] / means["lc"],
            "signal_risk_ratio": means["signal_t"] / np.maximum(means["signal_c"], 1e-30),
            "Delta_C": means["dc"],
            "Delta_D": means["dd"],
            "S": s,
            "G_mean": means["g"],
            "C": c,
        }
        for key, values in derived.items():
            collected[key].append(values)
    return {
        f"{key}_ci95": np.quantile(np.concatenate(values), [0.025, 0.975]).tolist()
        for key, values in collected.items()
    }


def evaluate_condition(
    condition: tuple[str, int, int, int],
    output_root: Path,
    library: torch.Tensor,
    prompts: dict[str, np.ndarray],
    prompt_path: Path,
    prompt_seed: int,
    count: int,
    bootstrap_replicates: int,
    prompt_chunk: int,
    save_arrays: bool,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name, log2_m, m, step = condition
    if len(prompts["data"]) != count:
        raise RuntimeError(f"shared prompt count changed for {name}: {len(prompts['data'])} != {count}")
    model_path = output_root / f"model_{name}_step{step}.pt"
    model_prediction = infer_model(model_path, prompts, device)
    causal_checks = causal_mask_checks(model_path, prompts, model_prediction, device)
    # Fused CUDA attention shows small input-dependent roundoff even for masked
    # future positions; the CPU reference remains exact to the original 1e-7
    # threshold. Record rather than hide the accelerator-specific tolerance.
    causal_tolerance = 5e-5 if device.type == "cuda" else 1e-7
    if max(causal_checks.values()) > causal_tolerance:
        raise RuntimeError(f"causal masking check failed for {name}: {causal_checks}")

    # Keep exact baselines on CPU so their float64 reference checks are
    # hardware-stable; only neural inference uses the requested accelerator.
    data = torch.from_numpy(prompts["data"])
    targets = torch.from_numpy(prompts["targets"])
    data_np = prompts["data"].astype(np.float64)
    targets_np = prompts["targets"].astype(np.float64)
    tasks_np = prompts["tasks"].astype(np.float64)
    signal_np = prompts["signal"].astype(np.float64)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for k in PREFIXES:
        started = time.time()
        ridge = exact_ridge_predictions(data, targets, k)
        library_condition = library[:m]
        atomic = exact_atomic_predictions(
            data[:, :k],
            targets[:, :k],
            data[:, k],
            library_condition,
            task_chunk=32768,
            prompt_chunk=prompt_chunk,
        )
        n2 = exact_log2_n2_batch(data_np[:, :k], targets_np[:, :k])
        subset_count = min(8, count)
        subset_tasks = min(4096, m)
        reference = materialized_atomic_float64(
            data_np[:subset_count, :k],
            targets_np[:subset_count, :k],
            data_np[:subset_count, k],
            library[:subset_tasks].numpy().astype(np.float64),
        )
        streamed_subset = exact_atomic_predictions(
            data[:subset_count, :k],
            targets[:subset_count, :k],
            data[:subset_count, k],
            library_condition[:subset_tasks],
            task_chunk=subset_tasks,
            prompt_chunk=subset_count,
        )
        atomic_spot_error = float(np.max(np.abs(reference - streamed_subset)))
        n2_reference = exact_log2_n2_reference(data_np[:subset_count, :k], targets_np[:subset_count, :k])
        n2_spot_error = float(np.max(np.abs(n2[:subset_count] - n2_reference)))
        if atomic_spot_error > 2e-4:
            raise RuntimeError(f"atomic float64 subset check failed for {name}/k{k}: {atomic_spot_error}")
        if n2_spot_error > 1e-10:
            raise RuntimeError(f"N2 float64 subset check failed for {name}/k{k}: {n2_spot_error}")

        transformer = model_prediction[:, k]
        target = targets_np[:, k]
        signal = signal_np[:, k]
        continuous = ridge.astype(np.float64)
        discrete = atomic.astype(np.float64)
        prompt_values = {
            "lt": (transformer - target) ** 2,
            "lc": (continuous - target) ** 2,
            "ld": (discrete - target) ** 2,
            "dc": (transformer - continuous) ** 2,
            "dd": (transformer - discrete) ** 2,
            "signal_t": (transformer - signal) ** 2,
            "signal_c": (continuous - signal) ** 2,
            "g": n2 - math.log2(m),
        }
        means = {key: float(np.mean(value)) for key, value in prompt_values.items()}
        denominator = means["ld"] - means["lc"]
        result = {
            "condition": name,
            "M": m,
            "log2_M": log2_m,
            "checkpoint_step": step,
            "prefix_k": k,
            "prompt_count": count,
            "prompt_seed": prompt_seed,
            "bootstrap_seed": derive_seed(2000, log2_m, k),
            "noisy_risk_ratio": means["lt"] / means["lc"],
            "signal_risk_ratio": means["signal_t"] / max(means["signal_c"], 1e-30),
            "Delta_C": means["dc"],
            "Delta_D": means["dd"],
            "S": (means["dd"] - means["dc"]) / max(means["dd"] + means["dc"], 1e-30),
            "G_mean": means["g"],
            "G_median": float(np.median(prompt_values["g"])),
            "L_T": means["lt"],
            "L_C": means["lc"],
            "L_D": means["ld"],
            "C": None if abs(denominator) <= max(1e-8, 0.01 * means["lc"]) else (means["lt"] - means["lc"]) / denominator,
            "C_denominator": denominator,
            "atomic_float64_subset_max_abs": atomic_spot_error,
            "n2_float64_subset_max_abs": n2_spot_error,
            "causal_future_label_max_abs": causal_checks[f"k{k}"],
            "elapsed_seconds": time.time() - started,
        }
        result.update(bootstrap_ci(prompt_values, result["bootstrap_seed"], bootstrap_replicates))
        rows.append(result)
        arrays[f"k{k}_transformer"] = transformer
        arrays[f"k{k}_ridge"] = continuous
        arrays[f"k{k}_atomic"] = discrete
        arrays[f"k{k}_target"] = target
        arrays[f"k{k}_signal"] = signal
        arrays[f"k{k}_log2_n2"] = n2

    if save_arrays:
        np.savez_compressed(output_root / f"audit_arrays_{name}.npz", **arrays)
    metadata = {
        "condition": name,
        "prompt_seed": prompt_seed,
        "shared_prompt_bank": True,
        "shared_prompt_bank_sha256": sha256_path(prompt_path),
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": sha256_path(prompt_path),
        "causal_future_label_max_abs": max(causal_checks.values()),
        "causal_future_label_tolerance": causal_tolerance,
        "all_predictions_finite": True,
    }
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--count", type=int, default=PROMPTS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--prompt-chunk", type=int, default=64)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--use-extracted", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--execution-location", choices=("local", "colab"), default="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot and args.full:
        raise SystemExit("choose at most one of --pilot and --full")
    args.output_root.mkdir(parents=True, exist_ok=True)
    extraction = (
        verify_extracted_bundle(args.output_root)
        if args.use_extracted
        else verify_and_extract(args.snapshot_root, args.output_root)
    )
    if args.inspect_only:
        print(json.dumps(clean(extraction), indent=2, sort_keys=True))
        return
    count = args.count
    bootstrap_replicates = args.bootstrap_replicates
    if args.pilot:
        count = min(count, 128)
        bootstrap_replicates = min(bootstrap_replicates, 100)
    if not args.full and not args.pilot:
        raise SystemExit("inference requires --pilot or --full")
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested_device == "auto":
        requested_device = "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(requested_device)
    library = torch.load(
        args.output_root / "master_library_prefix_M16384.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(library, torch.Tensor) or tuple(library.shape) != (16384, DIMENSION):
        raise RuntimeError(f"unexpected extracted prefix: {getattr(library, 'shape', None)}")
    started = time.time()
    prompt_seed = derive_seed(1000)
    prompts = fresh_prompts(count, prompt_seed)
    prompt_path = args.output_root / "fresh_unseen_prompts.npz"
    np.savez_compressed(prompt_path, **prompts)
    generator_checks = validate_fresh_prompt_generator(count, prompt_seed)
    all_rows: list[dict[str, Any]] = []
    metadata = []
    for condition in CONDITIONS:
        rows, condition_metadata = evaluate_condition(
            condition,
            args.output_root,
            library,
            prompts,
            prompt_path,
            prompt_seed,
            count,
            bootstrap_replicates,
            args.prompt_chunk,
            args.save_arrays,
            device,
        )
        all_rows.extend(rows)
        metadata.append(condition_metadata)
        print(json.dumps(clean(condition_metadata), sort_keys=True), flush=True)
    run = {
        "mode": "pilot" if args.pilot else "full",
        "count": count,
        "bootstrap_replicates": bootstrap_replicates,
        "prefixes": list(PREFIXES),
        "conditions": [name for name, _, _, _ in CONDITIONS],
        "inference_seed": INFERENCE_SEED,
        "prompt_seed": prompt_seed,
        "shared_prompt_bank": True,
        "shared_prompt_bank_sha256": sha256_path(prompt_path),
        "fresh_prompt_generator_checks": generator_checks,
        "noise": NOISE,
        "dimension": DIMENSION,
        "points": POINTS,
        "elapsed_seconds": time.time() - started,
        "extraction_audit": extraction,
        "condition_metadata": metadata,
        "rows": all_rows,
        "execution_location": args.execution_location,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "local_only": args.execution_location == "local",
        "export_performed": args.execution_location == "colab",
    }
    write_json(args.output_root / ("pilot_results.json" if args.pilot else "full_results.json"), run)
    print(json.dumps({"mode": run["mode"], "elapsed_seconds": run["elapsed_seconds"], "rows": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

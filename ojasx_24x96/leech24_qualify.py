"""Reproducible real-checkpoint Leech24 qualification on a tiny trained GPT.

The default source is HeathLoganCampbell/one-million-params v0.1.0,
``tiny-gpt-1m-mixed.pt`` (1,006,976 unique parameters).  Its published release
SHA-256 is pinned below.  The architecture is mirrored from the MIT-licensed
source at commit 170de2c5760cfdea3f375f6e5aaf2a8c45933850.

This is a real trained checkpoint, not a pretrained-quality or broad benchmark
claim.  Validation uses pinned Tiny Shakespeare byte windows, not the source
model's original held-out mixed corpus.  Dense means the same checkpoint after
an explicit BF16 cast.  Every quality/memory result is scoped to that setup.

Run ``qualify`` for conversion/planning/gates/save and fresh-process workers;
run ``worker`` only through that parent command.  Quantized workers receive no
source checkpoint path and reconstruct modules on meta tensors before loading
encoded shards.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .gates import QualityGateThresholds
from .generalized import quantize_model_generalized
from .leech_codec import LeechQuantizationConfig
from .module import quantization_manifest, restore_vq_modules
from .planner import PlanningConstraints, default_leech_candidate_configs
from .runtime import (
    runtime_storage_report,
    set_decode_row_block_size,
    set_leech_decode_backend,
    set_runtime_profile,
)


MODEL_SHA256 = "05b9bc603f9a826867dadc67302021d3dd77248af6ef42b3682844759d974228"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
MODEL_PROVENANCE = {
    "repository": "HeathLoganCampbell/one-million-params",
    "release": "v0.1.0",
    "asset": "tiny-gpt-1m-mixed.pt",
    "asset_sha256": MODEL_SHA256,
    "architecture_commit": "170de2c5760cfdea3f375f6e5aaf2a8c45933850",
    "training": "mixed corpus, checkpoint step 19999; real trained weights",
    "license": "MIT",
}
DATA_PROVENANCE = {
    "repository": "karpathy/char-rnn",
    "commit": "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e",
    "path": "data/tinyshakespeare/input.txt",
    "sha256": DATA_SHA256,
    "scope": "Pinned substitute byte corpus, not the checkpoint's original validation split",
}


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 128
    mlp_hidden: Optional[int] = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.mlp_hidden is None:
            self.mlp_hidden = int(8 * self.n_embd / 3 / 8) * 8


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm = inputs.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * inputs * norm


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    frequencies = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    return torch.cos(frequencies), torch.sin(frequencies)


def apply_rope(inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos[: inputs.size(-2)].to(inputs.dtype), sin[: inputs.size(-2)].to(inputs.dtype)
    first, second = inputs[..., 0::2], inputs[..., 1::2]
    return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("Embedding width must be divisible by attention heads")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, positions, channels = inputs.shape
        query, key, value = self.qkv(inputs).split(channels, dim=2)
        query = query.view(batch, positions, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, positions, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, positions, self.n_head, self.head_dim).transpose(1, 2)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, positions, channels)
        return self.proj(output)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.w2 = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(inputs)) * self.w3(inputs))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attn(self.norm1(inputs), cos, sin)
        return inputs + self.mlp(self.norm2(inputs))


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.cfg = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError("Input exceeds checkpoint context length")
        hidden = self.drop(self.tok_emb(input_ids))
        for block in self.blocks:
            hidden = block(hidden, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm_f(hidden))
        loss = (
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if targets is not None
            else None
        )
        return SimpleNamespace(logits=logits, loss=loss)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _json_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _restore_runtime_buffers(model: TinyGPT) -> None:
    cos, sin = precompute_rope(
        model.config.n_embd // model.config.n_head,
        model.config.block_size,
    )
    model.rope_cos = cos
    model.rope_sin = sin
    model.lm_head.weight = model.tok_emb.weight


def _load_shard_into_meta(model: TinyGPT, shard_path: Path) -> TinyGPT:
    from safetensors.torch import load_file

    shard = load_file(str(shard_path), device="cpu")
    expected = set(model.state_dict())
    unexpected = sorted(set(shard) - expected)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint tensors: {unexpected[:8]}")
    model.load_state_dict(shard, strict=False, assign=True)
    missing = expected - set(shard)
    if not missing.issubset({"lm_head.weight", "tok_emb.weight"}):
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)[:8]}")
    # safetensors.save_model deliberately stores one member of a tied alias.
    # Restore the missing member from whichever alias was actually persisted.
    if "tok_emb.weight" in missing:
        model.tok_emb.weight = model.lm_head.weight
    _restore_runtime_buffers(model)
    meta = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta:
        raise ValueError(f"Unrestored meta tensors: {meta[:8]}")
    model.eval()
    return model


def _load_dense(directory: Path) -> TinyGPT:
    from accelerate import init_empty_weights

    config = GPTConfig(**json.loads((directory / "config.json").read_text()))
    with init_empty_weights():
        model = TinyGPT(config)
    return _load_shard_into_meta(model, directory / "model.safetensors")


def _load_quantized(directory: Path) -> TinyGPT:
    from accelerate import init_empty_weights

    config = GPTConfig(**json.loads((directory / "config.json").read_text()))
    manifest = json.loads((directory / "vq_manifest.json").read_text())
    with init_empty_weights():
        model = TinyGPT(config).to(dtype=torch.bfloat16)
    restore_vq_modules(model, manifest)
    model = _load_shard_into_meta(model, directory / "model.safetensors")
    plan = json.loads((directory / "vq_plan.json").read_text())
    set_runtime_profile(model, "memory")
    set_decode_row_block_size(model, int(plan["constraints"]["decode_row_block_size"]))
    return model


def _byte_batches(data: bytes, offsets: Iterable[int], length: int):
    return [
        {
            "input_ids": torch.tensor(
                [[byte for byte in data[offset : offset + length]]],
                dtype=torch.int64,
            )
        }
        for offset in offsets
    ]


def _prepare_source(source: Path, workspace: Path) -> Path:
    from safetensors.torch import save_model

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != 19999 or checkpoint.get("corpus") != "mixed":
        raise ValueError("Checkpoint identity/metadata differs from the pinned qualification source")
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(dtype=torch.bfloat16).eval()
    directory = workspace / "source-bf16"
    directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(directory / "model.safetensors"), metadata={"format": "pt"})
    _json_write(directory / "config.json", asdict(config))
    provenance = {
        **MODEL_PROVENANCE,
        "source_sha256_actual": _sha256(source),
        "bf16_conversion": "Load FP32 state dict, strict architecture match, cast all floating state to BF16",
        "bf16_shard_sha256": _sha256(directory / "model.safetensors"),
        "unique_parameters": 1_006_976,
        "state_dict_elements_including_tied_alias": sum(
            tensor.numel() for tensor in model.state_dict().values()
        ),
    }
    _json_write(directory / "provenance.json", provenance)
    return directory


class _RSSMonitor:
    """Sample actual process RSS during one phase; also retain ru_maxrss."""

    def __init__(self) -> None:
        import psutil

        self.process = psutil.Process(os.getpid())
        self.maximum = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.maximum = max(self.maximum, self.process.memory_info().rss)
            self._stop.wait(0.001)

    def __enter__(self):
        self.maximum = max(self.maximum, self.process.memory_info().rss)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.maximum = max(self.maximum, self.process.memory_info().rss)

    @property
    def result(self) -> Dict[str, int]:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process_peak = peak * 1024 if sys.platform.startswith("linux") else peak
        return {
            "sampled_phase_peak_rss_bytes": self.maximum,
            "process_peak_rss_bytes": process_peak,
        }


def _runtime_worker(args) -> Dict[str, object]:
    directory = Path(args.artifact)
    started = time.perf_counter()
    with _RSSMonitor() as monitor:
        model = _load_dense(directory) if args.mode == "dense" else _load_quantized(directory)
    load_result = monitor.result
    load_seconds = time.perf_counter() - started
    set_runtime_profile(model, args.profile)
    leech_backend = args.leech_backend if args.mode == "quantized" else "python"
    leech_layers = set_leech_decode_backend(model, leech_backend)
    payload = json.loads((directory / "runtime-input.json").read_text())
    input_ids = torch.tensor(payload["input_ids"], dtype=torch.int64)
    prefill_times = []
    logits = None
    with _RSSMonitor() as monitor, torch.inference_mode():
        for _ in range(3):
            start = time.perf_counter()
            logits = model(input_ids=input_ids).logits
            prefill_times.append(time.perf_counter() - start)
    prefill_result = monitor.result
    if logits is None or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Nonfinite prefill logits")
    sequence = input_ids.clone()
    generated = []
    decode_times = []
    with _RSSMonitor() as monitor, torch.inference_mode():
        for _ in range(4):
            start = time.perf_counter()
            logits = model(input_ids=sequence).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            decode_times.append(time.perf_counter() - start)
            generated.append(int(next_token[0, 0].item()))
            sequence = torch.cat((sequence, next_token), dim=1)
    decode_result = monitor.result
    storage = runtime_storage_report(model)
    from .leech24 import initialized_shared_table_bytes

    native_status = None
    if args.mode == "quantized" and leech_backend != "python":
        from .leech_native import native_backend_status

        native_status = native_backend_status()
    return {
        "mode": args.mode,
        "profile": args.profile,
        "leech_decode_backend": leech_backend,
        "leech_layers": leech_layers,
        "native_backend": native_status,
        "initialized_python_shared_table_bytes": initialized_shared_table_bytes(),
        "artifact": str(directory),
        "load": {"seconds": load_seconds, **load_result},
        "prefill": {
            "cold_seconds": prefill_times[0],
            "warm_median_seconds": statistics.median(prefill_times[1:]),
            "logits_sha256": _tensor_sha256(logits),
            **prefill_result,
        },
        "decode_without_kv_cache": {
            "tokens": 4,
            "median_seconds_per_recomputed_token": statistics.median(decode_times),
            "generated_token_ids": generated,
            **decode_result,
        },
        "persistent_storage_after_run": storage,
        "source_dense_checkpoint_passed_to_worker": False if args.mode == "quantized" else True,
        "scope": (
            "Fresh CPython process; sampled phase RSS plus ru_maxrss. Four greedy "
            "tokens recompute the context without KV caching. Not a throughput benchmark."
        ),
    }


def _run_worker(
    artifact: Path,
    mode: str,
    profile: str,
    leech_backend: str = "python",
) -> Dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "ojasx_vector_quantization_runner.leech24_qualify",
        "worker",
        "--artifact",
        str(artifact),
        "--mode",
        mode,
        "--profile",
        profile,
        "--leech-backend",
        leech_backend,
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Runtime worker failed ({mode}/{profile}):\n{completed.stdout}\n{completed.stderr}"
        )
    prefix = "LEECH24_WORKER_JSON="
    line = next((line for line in completed.stdout.splitlines() if line.startswith(prefix)), None)
    if line is None:
        raise RuntimeError(f"Worker returned no JSON: {completed.stdout}\n{completed.stderr}")
    return json.loads(line[len(prefix) :])


def _runtime_comparison(
    workers: Iterable[Dict[str, object]],
    dense_persistent_bytes: int,
    quantized_persistent_bytes: int,
) -> Dict[str, object]:
    """Explicitly compare measured workers without turning RSS into a guarantee."""

    by_mode = {(item["mode"], item["profile"]): item for item in workers}
    dense = by_mode[("dense", "memory")]
    memory = by_mode[("quantized", "memory")]
    speed = by_mode[("quantized", "speed")]

    def phase_delta(candidate: Dict[str, object], phase: str) -> int:
        return int(candidate[phase]["sampled_phase_peak_rss_bytes"]) - int(
            dense[phase]["sampled_phase_peak_rss_bytes"]
        )

    dense_decode = float(
        dense["decode_without_kv_cache"]["median_seconds_per_recomputed_token"]
    )
    memory_decode = float(
        memory["decode_without_kv_cache"]["median_seconds_per_recomputed_token"]
    )
    speed_decode = float(
        speed["decode_without_kv_cache"]["median_seconds_per_recomputed_token"]
    )
    speed_cache = int(speed["persistent_storage_after_run"]["decoded_cache_bytes"])
    speed_state_cache = int(
        speed["persistent_storage_after_run"]["model_state_plus_decoded_cache_bytes"]
    )
    token_sequences = [
        item["decode_without_kv_cache"]["generated_token_ids"] for item in workers
    ]
    return {
        "persistent_savings_bytes": dense_persistent_bytes - quantized_persistent_bytes,
        "persistent_savings_fraction": (
            dense_persistent_bytes - quantized_persistent_bytes
        )
        / dense_persistent_bytes,
        "quantized_memory_sampled_rss_delta_bytes": {
            phase: phase_delta(memory, phase) for phase in ("load", "prefill", "decode_without_kv_cache")
        },
        "quantized_speed_sampled_rss_delta_bytes": {
            phase: phase_delta(speed, phase) for phase in ("load", "prefill", "decode_without_kv_cache")
        },
        "decode_time_ratio_to_dense": {
            "quantized_memory": memory_decode / dense_decode,
            "quantized_speed": speed_decode / dense_decode,
        },
        "speed_decoded_cache_bytes": speed_cache,
        "speed_state_plus_cache_delta_to_dense_bytes": (
            speed_state_cache - dense_persistent_bytes
        ),
        "all_worker_generated_token_ids_match": all(
            sequence == token_sequences[0] for sequence in token_sequences[1:]
        ),
        "conclusion": (
            "Persistent tensor storage is smaller. On this tiny model, sampled process RSS "
            "does not establish a RAM saving because runtime/library overhead dominates. "
            "Streaming trades latency for zero retained decoded cache; speed mode restores "
            "latency by retaining full decoded weights. These are not throughput claims."
        ),
    }


def qualify(args) -> int:
    torch.set_num_threads(1)
    source = Path(args.source).resolve()
    data_path = Path(args.data).resolve()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if _sha256(source) != MODEL_SHA256:
        raise ValueError("Source checkpoint SHA-256 does not match the pinned release")
    if _sha256(data_path) != DATA_SHA256:
        raise ValueError("Validation data SHA-256 does not match the pinned corpus")
    source_directory = _prepare_source(source, workspace)
    data = data_path.read_bytes()
    sequence_length = 128
    calibration = _byte_batches(data, (4096, 8192), sequence_length)
    validation = _byte_batches(
        data,
        (len(data) - 32768, len(data) - 16384, len(data) - 8192),
        sequence_length,
    )
    runtime_input = validation[0]["input_ids"].tolist()
    _json_write(source_directory / "runtime-input.json", {"input_ids": runtime_input})

    dense = _load_dense(source_directory)
    reference = copy.deepcopy(dense)
    candidate = copy.deepcopy(dense)
    dense_storage = runtime_storage_report(dense)
    configs = default_leech_candidate_configs(
        LeechQuantizationConfig(
            max_gain_samples=512,
            gain_grid_size=16,
            chunk_size=64,
            min_weight_elements=256,
            require_compression=True,
            seed=20261104,
        )
    )
    thresholds = QualityGateThresholds(
        max_relative_logits_l2_error=args.max_relative_logits,
        max_mean_kl=args.max_kl,
        min_top1_agreement=args.min_top1,
        max_mean_nll_delta=args.max_nll_delta,
    )
    result = quantize_model_generalized(
        candidate,
        constraints=PlanningConstraints(
            target_state_ratio=args.target_state_ratio,
            max_tensor_estimated_error=2.0,
            runtime_profile="memory",
            decode_row_block_size=16,
        ),
        calibration_inputs=calibration,
        candidate_configs=configs,
        include=(args.include,),
        reference_model=reference,
        quality_thresholds=thresholds,
        validation_inputs=validation,
        max_positions_per_batch=512,
    )
    leech_selected = sum(
        item.codec == "leech24_shell2_residual"
        for item in result.plan.selected_candidates.values()
    )
    evidence = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": result.status,
        "model_provenance": MODEL_PROVENANCE,
        "data_provenance": DATA_PROVENANCE,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(),
        },
        "dense_bf16_storage": dense_storage,
        "leech_selected_tensors": leech_selected,
        "generalized": result.to_dict(include_alternatives=True),
        "scope": (
            "One small real trained checkpoint cast to BF16 and a pinned substitute "
            "byte corpus. Passing gates are not broad model-quality certification."
        ),
    }
    if result.quality is None or not result.quality.passed or not leech_selected:
        evidence["qualification_error"] = (
            "End-to-end gates failed or no Leech tensor was selected; no artifact was saved"
        )
        _json_write(Path(args.evidence), evidence)
        print(json.dumps({"status": evidence["status"], "error": evidence["qualification_error"]}))
        return 2

    from safetensors.torch import save_model

    output = workspace / "encoded"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Encoded qualification output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    save_model(result.model, str(output / "model.safetensors"), metadata={"format": "pt"})
    _json_write(output / "config.json", asdict(result.model.config))
    _json_write(output / "vq_manifest.json", quantization_manifest(result.model))
    _json_write(output / "vq_plan.json", result.plan.to_dict())
    _json_write(output / "quality.json", result.quality.to_dict())
    _json_write(output / "runtime-input.json", {"input_ids": runtime_input})

    reloaded = _load_quantized(output)
    reload_storage = runtime_storage_report(reloaded)
    predicted = int(result.plan.memory["predicted_model_state_bytes"])
    if reload_storage["model_unique_state_tensor_bytes"] != predicted:
        raise RuntimeError("Reloaded persistent state differs from the planner prediction")
    del reloaded, candidate, dense, reference
    evidence["artifact"] = {
        "path": str(output),
        "file_bytes": {
            path.name: path.stat().st_size
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "total_file_bytes": sum(
            path.stat().st_size for path in output.iterdir() if path.is_file()
        ),
        "model_shard_sha256": _sha256(output / "model.safetensors"),
        "predicted_persistent_bytes": predicted,
        "actual_reloaded_persistent_bytes": reload_storage[
            "model_unique_state_tensor_bytes"
        ],
        "dense_source_materialized_on_quantized_reload": False,
    }
    workers = [
        _run_worker(source_directory, "dense", "memory"),
        _run_worker(output, "quantized", "memory", args.leech_backend),
        _run_worker(output, "quantized", "speed", args.leech_backend),
    ]
    evidence["runtime_workers"] = workers
    evidence["runtime_comparison"] = _runtime_comparison(
        workers,
        int(dense_storage["model_unique_state_tensor_bytes"]),
        predicted,
    )
    evidence["status"] = "passed"
    _json_write(Path(args.evidence), evidence)
    quality = result.quality.metrics
    print(
        "Leech24 real-checkpoint qualification: "
        f"status=passed, selected={leech_selected}, "
        f"persistent={predicted}/{dense_storage['model_unique_state_tensor_bytes']} bytes, "
        f"relative-logits={quality['relative_logits_l2_error']:.6f}, "
        f"NLL-delta={quality['mean_nll_delta']:.6f}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("qualify")
    convert.add_argument("--source", required=True)
    convert.add_argument("--data", required=True)
    convert.add_argument("--workspace", required=True)
    convert.add_argument("--evidence", required=True)
    convert.add_argument("--include", default="*")
    convert.add_argument("--target-state-ratio", type=float, default=0.90)
    convert.add_argument("--max-relative-logits", type=float, default=0.25)
    convert.add_argument("--max-kl", type=float, default=0.10)
    convert.add_argument("--min-top1", type=float, default=0.70)
    convert.add_argument("--max-nll-delta", type=float, default=0.25)
    convert.add_argument(
        "--leech-backend",
        choices=("python", "native", "auto"),
        default="python",
    )
    worker = subparsers.add_parser("worker")
    worker.add_argument("--artifact", required=True)
    worker.add_argument("--mode", choices=("dense", "quantized"), required=True)
    worker.add_argument("--profile", choices=("memory", "speed"), required=True)
    worker.add_argument(
        "--leech-backend",
        choices=("python", "native", "auto"),
        default="python",
    )
    args = parser.parse_args()
    if args.command == "worker":
        payload = _runtime_worker(args)
        print("LEECH24_WORKER_JSON=" + json.dumps(payload, sort_keys=True))
        return
    raise SystemExit(qualify(args))


if __name__ == "__main__":
    main()

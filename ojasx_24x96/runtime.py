"""Explicit memory/speed runtime profiles and reproducible forward benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .module import ResidualVQFusedExperts, ResidualVQLinear


RUNTIME_PROFILES = ("memory", "speed")


def set_runtime_profile(model: torch.nn.Module, profile: str) -> Dict[str, object]:
    """Configure decoded-weight reuse without changing the checkpoint format.

    ``memory`` never retains a dense decoded matrix and streams row/expert chunks.
    ``speed`` reuses one full decoded matrix per VQ linear layer after its first
    forward. The latter deliberately trades additional RAM for lower repeated
    decoding work.
    """

    if profile not in RUNTIME_PROFILES:
        raise ValueError(f"Runtime profile must be one of {RUNTIME_PROFILES}")
    changed = 0
    fused = 0
    for module in model.modules():
        if isinstance(module, ResidualVQFusedExperts):
            module.streaming_decode = profile == "memory"
            fused += 1
        if isinstance(module, ResidualVQLinear):
            module.set_weight_cache(profile == "speed")
            module.set_streaming_decode(
                profile == "memory",
                row_block_size=module.decode_row_block_size,
            )
            changed += 1
    return {
        "profile": profile,
        "vq_linear_layers": changed,
        "fused_expert_banks": fused,
        "dense_weight_cache": profile == "speed",
        "streaming_decode": profile == "memory",
        "scope": "Runtime memory/speed tradeoff only; no latency claim",
    }


def set_decode_row_block_size(model: torch.nn.Module, row_block_size: int) -> int:
    """Set the ordinary-projection streaming row bound and return layer count."""

    if row_block_size <= 0:
        raise ValueError("Decode row block size must be positive")
    changed = 0
    for module in model.modules():
        if isinstance(module, ResidualVQLinear):
            module.set_streaming_decode(
                module.streaming_decode,
                row_block_size=row_block_size,
            )
            changed += 1
    return changed


def set_leech_decode_backend(model: torch.nn.Module, backend: str) -> int:
    """Select ``python``, ``native``, or ``auto`` on every Leech linear layer."""

    if backend not in {"python", "native", "auto"}:
        raise ValueError("Leech decode backend must be python, native, or auto")
    changed = 0
    for module in model.modules():
        if isinstance(module, ResidualVQLinear) and module.codec == "leech24_shell2_residual":
            module.set_leech_decode_backend(backend)
            changed += 1
    return changed


def _unique_state_tensors(model: torch.nn.Module) -> List[torch.Tensor]:
    """Return persistent state tensors without counting exact tied aliases twice."""

    unique: Dict[Tuple[object, ...], torch.Tensor] = {}
    for tensor in model.state_dict().values():
        if tensor.device.type == "meta":
            key = ("meta-id", id(tensor))
        else:
            key = (
                str(tensor.device),
                tensor.dtype,
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
        unique.setdefault(key, tensor)
    return list(unique.values())


def runtime_storage_report(model: torch.nn.Module) -> Dict[str, int]:
    """Measure persistent tensor bytes and optional decoded-cache bytes.

    This is a tensor inventory, not an allocator/RSS measurement. Exact tied
    aliases are counted once; temporary activations and KV cache are excluded.
    """

    encoded = 0
    cached = 0
    layers = 0
    for module in model.modules():
        if isinstance(module, ResidualVQLinear):
            encoded += module.storage_bytes
            cached += module.cached_weight_bytes
            layers += 1
    persistent = sum(
        tensor.numel() * tensor.element_size() for tensor in _unique_state_tensors(model)
    )
    return {
        "vq_layers": layers,
        "vq_storage_bytes_including_bias": encoded,
        "decoded_cache_bytes": cached,
        "combined_vq_bytes": encoded + cached,
        "model_unique_state_tensor_bytes": persistent,
        "non_vq_state_tensor_bytes": max(0, persistent - encoded),
        "model_state_plus_decoded_cache_bytes": persistent + cached,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_with_inputs(model: torch.nn.Module, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    inputs,
    *,
    profiles: Iterable[str] = RUNTIME_PROFILES,
    warmup: int = 2,
    iterations: int = 5,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> Dict[str, object]:
    """Benchmark the same forward under explicit runtime profiles.

    The returned numbers describe only the supplied model, input, and environment.
    Numerical output agreement is checked across profiles. None tolerance uses
    PyTorch's dtype-specific defaults, accommodating chunked-GEMM rounding.
    Generation, allocator, and service throughput require separate benchmarks.
    """

    if warmup < 0 or iterations <= 0:
        raise ValueError("Benchmark warmup must be nonnegative and iterations positive")
    if (rtol is not None and rtol < 0) or (atol is not None and atol < 0):
        raise ValueError("Benchmark tolerances must be nonnegative")
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("Benchmark requires at least one runtime profile")
    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None)
    device = (
        parameter.device
        if parameter is not None
        else buffer.device
        if buffer is not None
        else torch.device("cpu")
    )
    reference: Optional[torch.Tensor] = None
    results = []
    try:
        for profile in profiles:
            set_runtime_profile(model, profile)
            for _ in range(warmup):
                output = _forward_with_inputs(model, inputs)
            _synchronize(device)
            measurements: List[float] = []
            for _ in range(iterations):
                start = time.perf_counter()
                output = _forward_with_inputs(model, inputs)
                _synchronize(device)
                measurements.append(time.perf_counter() - start)
            tensor = output.logits if hasattr(output, "logits") else output
            if reference is None:
                reference = tensor.detach().cpu().clone()
            else:
                torch.testing.assert_close(
                    tensor.detach().cpu(),
                    reference,
                    rtol=rtol,
                    atol=atol,
                )
            results.append({
                "profile": profile,
                "warmup": warmup,
                "iterations": iterations,
                "median_seconds": statistics.median(measurements),
                "mean_seconds": statistics.fmean(measurements),
                "min_seconds": min(measurements),
                "max_seconds": max(measurements),
                **runtime_storage_report(model),
            })
    finally:
        set_runtime_profile(model, "memory")
    return {
        "schema_version": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "profile_output_rtol": rtol,
        "profile_output_atol": atol,
        "results": results,
        "scope": (
            "Measured forward latency on one input and host; not a universal "
            "speedup, generation-throughput, or memory allocator guarantee"
        ),
    }

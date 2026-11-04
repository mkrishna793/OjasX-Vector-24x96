"""Qualify a pinned, trained BF16 causal-LM checkpoint at larger weight scales.

The parent command verifies source provenance, calibrates mixed codec candidates,
plans against persistent-byte budgets, executes independent quality gates, saves
an encoded artifact, and repeats those gates after a meta-only reload. Fresh
workers then measure dense and quantized load/prefill/decode RSS without giving
quantized workers access to the dense source path.

This module is a harness, not evidence that any particular larger checkpoint has
been run. A full pass requires the declared runtime gates. Explicitly disabling
those gates produces ``passed_without_full_ram_qualification`` and is intended
only for small integration tests.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .codec import QuantizationConfig
from .gates import QualityGateThresholds, evaluate_quality_gates
from .generalized import quantize_model_generalized
from .hf import (
    _weight_shards,
    load_quantized_pretrained,
    save_quantized_pretrained,
)
from .leech24_qualify import _RSSMonitor, _json_write, _sha256, _tensor_sha256
from .leech_codec import LeechQuantizationConfig
from .planner import PlanningConstraints
from .runtime import (
    runtime_storage_report,
    set_leech_decode_backend,
    set_runtime_profile,
)


_WORKER_PREFIX = "LARGER_BF16_WORKER_JSON="
_PHASES = ("load", "prefill", "decode_without_kv_cache")
_UNACCEPTABLE_LICENSES = {
    "",
    "other",
    "to-be-decided",
    "tbd",
    "unknown",
    "unlicensed",
    "none",
    "not-specified",
}


def _process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if sys.platform.startswith("linux") else peak


def _current_rss_bytes() -> int:
    import psutil

    return int(psutil.Process(os.getpid()).memory_info().rss)


def _relative_file(base: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe provenance-relative path: {relative!r}")
    path = base.joinpath(*pure.parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Provenance path escapes the source directory: {relative!r}")
    return resolved


def _audit_source_dtypes(source: Path) -> Dict[str, object]:
    shards, weight_map = _weight_shards(source)
    dtype_bytes: Dict[str, int] = {}
    tensor_counts: Dict[str, int] = {}
    seen = set()
    from safetensors import safe_open

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"Duplicate source tensor across shards: {key!r}")
                seen.add(key)
                if weight_map.get(key) != shard.relative_to(source).as_posix():
                    raise ValueError(f"Source shard index disagrees for tensor {key!r}")
                description = handle.get_slice(key)
                dtype = str(description.get_dtype())
                shape = description.get_shape()
                elements = math.prod(shape)
                dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + elements * {
                    "BF16": 2,
                    "F16": 2,
                    "F32": 4,
                    "F64": 8,
                    "I8": 1,
                    "U8": 1,
                    "I16": 2,
                    "U16": 2,
                    "I32": 4,
                    "U32": 4,
                    "I64": 8,
                    "U64": 8,
                    "BOOL": 1,
                }.get(dtype, 0)
                tensor_counts[dtype] = tensor_counts.get(dtype, 0) + 1
    floating = {"BF16", "F16", "F32", "F64"}
    floating_bytes = sum(value for key, value in dtype_bytes.items() if key in floating)
    bf16_bytes = dtype_bytes.get("BF16", 0)
    if floating_bytes <= 0 or bf16_bytes <= 0:
        raise ValueError("The source does not contain auditable BF16 floating-point weights")
    return {
        "tensor_counts_by_dtype": tensor_counts,
        "tensor_bytes_by_dtype": dtype_bytes,
        "floating_tensor_bytes": floating_bytes,
        "bf16_tensor_bytes": bf16_bytes,
        "bf16_fraction_of_floating_tensor_bytes": bf16_bytes / floating_bytes,
        "weight_files": [path.relative_to(source).as_posix() for path in shards],
    }


def validate_source_provenance(
    source: Path,
    provenance_path: Path,
    *,
    minimum_bf16_fraction: float = 0.95,
) -> Dict[str, object]:
    """Verify file identities, trained-source declarations, license, and BF16 bytes."""

    if not 0 < minimum_bf16_fraction <= 1:
        raise ValueError("minimum_bf16_fraction must be in (0, 1]")
    source = source.resolve()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Source provenance must use schema_version=1")
    for field in ("repository", "revision", "license", "training"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"Source provenance requires a nonempty {field!r}")
    if payload.get("trained_checkpoint") is not True:
        raise ValueError("Source provenance must explicitly identify a trained checkpoint")
    if payload.get("bf16_checkpoint") is not True:
        raise ValueError("Source provenance must explicitly identify a BF16 checkpoint")
    unique_parameters = payload.get("unique_parameters")
    if (
        not isinstance(unique_parameters, int)
        or isinstance(unique_parameters, bool)
        or unique_parameters <= 0
    ):
        raise ValueError("Source provenance requires a positive unique_parameters count")
    if payload["license"].strip().lower() in _UNACCEPTABLE_LICENSES:
        raise ValueError("The checkpoint license is missing or too unclear for qualification")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source provenance requires a pinned file mapping")
    verified = {}
    for relative, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Provenance file entry must be an object: {relative!r}")
        expected_sha = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError(f"Invalid pinned size/SHA-256 for {relative!r}")
        if not isinstance(relative, str):
            raise ValueError("Provenance file names must be strings")
        path = _relative_file(source, relative)
        if path.relative_to(source).as_posix() != relative:
            raise ValueError(f"Provenance file name is not canonical: {relative!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source file is missing: {relative}")
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise ValueError(f"Source file identity mismatch: {relative!r}")
        verified[relative] = {"bytes": actual_bytes, "sha256": actual_sha}
    actual_files = set()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Source file resolves outside the source directory: {path}")
        actual_files.add(path.relative_to(source).as_posix())
    undeclared = sorted(actual_files - verified.keys())
    if undeclared:
        raise ValueError(f"Every regular source file must be hash-pinned; missing: {undeclared}")
    for required in ("config.json",):
        if required not in verified:
            raise ValueError(f"Source provenance must pin {required}")
    shards, _ = _weight_shards(source)
    for shard in shards:
        relative = shard.relative_to(source).as_posix()
        if relative not in verified:
            raise ValueError(f"Every weight shard must be hash-pinned: {relative}")
    dtype = _audit_source_dtypes(source)
    if dtype["bf16_fraction_of_floating_tensor_bytes"] < minimum_bf16_fraction:
        raise ValueError(
            "Source BF16 byte fraction is below the declared minimum: "
            f"{dtype['bf16_fraction_of_floating_tensor_bytes']:.6f}"
        )
    return {
        "declaration": payload,
        "provenance_file_sha256": _sha256(provenance_path),
        "verified_files": verified,
        "dtype_audit": dtype,
    }


def _load_source_tokenizer(directory: Path, declaration: Mapping[str, object]):
    """Load stock tokenizers, with an explicit local generic-class override."""

    import transformers

    class_name = declaration.get("tokenizer_class")
    if class_name is None:
        return transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if not isinstance(class_name, str) or not class_name.isidentifier():
        raise ValueError("Provenance tokenizer_class must be a Transformers class name")
    tokenizer_class = getattr(transformers, class_name, None)
    if tokenizer_class is None or not hasattr(tokenizer_class, "from_pretrained"):
        raise ValueError(f"Unknown local Transformers tokenizer class: {class_name}")
    return tokenizer_class.from_pretrained(directory, local_files_only=True)


def _load_dense(directory: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        directory,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    )
    # The dense reference is deliberately the same explicit BF16 execution cast
    # used by the encoded artifact. Source-disk dtypes are audited separately.
    model = model.to(dtype=torch.bfloat16).eval()
    return model


def _unique_parameter_count(model: torch.nn.Module) -> int:
    unique = {}
    for parameter in model.parameters():
        key = (
            str(parameter.device),
            parameter.dtype,
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        unique.setdefault(key, parameter)
    return sum(parameter.numel() for parameter in unique.values())


def _token_windows(
    tokenizer,
    data_path: Path,
    *,
    sequence_length: int,
    batches: int,
) -> List[Dict[str, torch.Tensor]]:
    if sequence_length < 4 or batches <= 0:
        raise ValueError("Token windows need sequence_length>=4 and at least one batch")
    text = data_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Calibration/validation text is empty: {data_path}")
    encoded = tokenizer(text, add_special_tokens=True)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one unbatched token sequence")
        token_ids = token_ids[0]
    if len(token_ids) < sequence_length:
        raise ValueError(
            f"{data_path} has {len(token_ids)} tokens; {sequence_length} are required"
        )
    final_start = len(token_ids) - sequence_length
    result = []
    for index in range(batches):
        start = round(index * final_start / max(1, batches - 1))
        ids = token_ids[start : start + sequence_length]
        result.append({
            "input_ids": torch.tensor([ids], dtype=torch.int64),
            "attention_mask": torch.ones((1, sequence_length), dtype=torch.int64),
        })
    return result


def _candidate_configs(codec: str) -> List[object]:
    common = dict(
        bits_per_stage=4,
        max_relative_error=None,
        max_codebook_samples=512,
        kmeans_iterations=6,
        chunk_size=8192,
        min_weight_elements=256,
        require_compression=True,
        seed=20261104,
    )
    configs: List[object] = []
    if codec in {"mixed", "rvq"}:
        configs.extend([
            QuantizationConfig(block_size=4, max_stages=4, **common),
            QuantizationConfig(block_size=8, max_stages=3, **common),
        ])
    if codec in {"mixed", "leech24"}:
        configs.append(LeechQuantizationConfig(
            max_stages=3,
            max_relative_error=None,
            gain_bits=1,
            rotation_reflections=2,
            adaptive_stages=True,
            max_gain_samples=256,
            gain_grid_size=8,
            chunk_size=64,
            min_weight_elements=256,
            require_compression=True,
            seed=20261104,
        ))
    if not configs:
        raise ValueError(f"Unsupported codec family {codec!r}")
    return configs


def _quality_thresholds(args) -> QualityGateThresholds:
    return QualityGateThresholds(
        max_relative_logits_l2_error=args.max_relative_logits,
        max_absolute_logits_error=args.max_absolute_logits,
        max_mean_kl=args.max_kl,
        min_top1_agreement=args.min_top1,
        max_mean_nll_delta=args.max_nll_delta,
    )


def _warm_worker_imports() -> None:
    # Both modes pay the same import cost before the pre-model baseline.
    from accelerate import init_empty_weights  # noqa: F401
    from safetensors.torch import load_file  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401


def _runtime_worker(args) -> Dict[str, object]:
    torch.set_num_threads(args.num_threads)
    _warm_worker_imports()
    directory = Path(args.artifact).resolve()
    runtime_input_path = Path(args.runtime_input).resolve()
    pre_model = {
        "current_rss_bytes": _current_rss_bytes(),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
    }
    started = time.perf_counter()
    with _RSSMonitor() as monitor:
        if args.mode == "dense":
            model = _load_dense(directory)
        else:
            model = load_quantized_pretrained(
                directory,
                load_tokenizer=False,
                runtime_profile=args.profile,
            )
        set_runtime_profile(model, args.profile)
        leech_layers = set_leech_decode_backend(
            model,
            args.leech_backend if args.mode == "quantized" else "python",
        )
    load_result = monitor.result
    load_seconds = time.perf_counter() - started
    payload = json.loads(runtime_input_path.read_text(encoding="utf-8"))
    inputs = {
        "input_ids": torch.tensor(payload["input_ids"], dtype=torch.int64),
        "attention_mask": torch.tensor(payload["attention_mask"], dtype=torch.int64),
    }
    prefill_times = []
    warmup_times = []
    logits = None
    with _RSSMonitor() as monitor, torch.inference_mode():
        for _ in range(args.prefill_warmup):
            started = time.perf_counter()
            logits = model(**inputs, use_cache=False).logits
            warmup_times.append(time.perf_counter() - started)
        for _ in range(args.prefill_iterations):
            started = time.perf_counter()
            logits = model(**inputs, use_cache=False).logits
            prefill_times.append(time.perf_counter() - started)
    prefill_result = monitor.result
    if logits is None or not torch.isfinite(logits).all().item():
        raise FloatingPointError("Dense/quantized worker produced nonfinite prefill logits")
    prefill_logits_sha256 = _tensor_sha256(logits)
    sequence = dict(inputs)
    generated = []
    decode_times = []
    with _RSSMonitor() as monitor, torch.inference_mode():
        for _ in range(args.decode_tokens):
            started = time.perf_counter()
            logits = model(**sequence, use_cache=False).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            decode_times.append(time.perf_counter() - started)
            generated.append(int(next_token[0, 0].item()))
            sequence = {
                "input_ids": torch.cat((sequence["input_ids"], next_token), dim=1),
                "attention_mask": torch.cat(
                    (sequence["attention_mask"], torch.ones_like(next_token)),
                    dim=1,
                ),
            }
    decode_result = monitor.result
    storage = runtime_storage_report(model)
    from .leech24 import initialized_shared_table_bytes

    native_status = None
    if args.mode == "quantized" and args.leech_backend != "python":
        from .leech_native import native_backend_status

        native_status = native_backend_status()
    return {
        "mode": args.mode,
        "profile": args.profile,
        "leech_decode_backend": args.leech_backend if args.mode == "quantized" else "python",
        "leech_layers": leech_layers,
        "native_backend": native_status,
        "artifact": str(directory),
        "runtime_input": str(runtime_input_path),
        "pre_model": pre_model,
        "load": {"seconds": load_seconds, **load_result},
        "prefill": {
            "warmup_seconds": warmup_times,
            "timed_seconds": prefill_times,
            "cold_seconds": (warmup_times + prefill_times)[0],
            "median_seconds": statistics.median(prefill_times),
            "logits_sha256": prefill_logits_sha256,
            **prefill_result,
        },
        "decode_without_kv_cache": {
            "tokens": args.decode_tokens,
            "timed_seconds": decode_times,
            "median_seconds_per_recomputed_token": statistics.median(decode_times),
            "generated_token_ids": generated,
            **decode_result,
        },
        "persistent_storage_after_run": storage,
        "initialized_python_shared_table_bytes": initialized_shared_table_bytes(),
        "source_dense_checkpoint_passed_to_worker": args.mode == "dense",
        "num_threads": args.num_threads,
        "scope": (
            "Fresh CPython process with common imports paid before the baseline. "
            "Sampled phase RSS and ru_maxrss are process measurements, distinct "
            "from persistent tensor bytes. Decode recomputes context without KV cache."
        ),
    }


def _run_worker(
    artifact: Path,
    runtime_input: Path,
    mode: str,
    args,
) -> Dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "ojasx_vector_quantization_runner.larger_bf16_qualify",
        "worker",
        "--artifact",
        str(artifact),
        "--runtime-input",
        str(runtime_input),
        "--mode",
        mode,
        "--profile",
        "memory",
        "--leech-backend",
        args.leech_backend,
        "--num-threads",
        str(args.num_threads),
        "--prefill-warmup",
        str(args.prefill_warmup),
        "--prefill-iterations",
        str(args.prefill_iterations),
        "--decode-tokens",
        str(args.decode_tokens),
    ]
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(args.num_threads),
        "MKL_NUM_THREADS": str(args.num_threads),
    })
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
            f"Runtime worker failed ({mode}):\n{completed.stdout}\n{completed.stderr}"
        )
    line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(_WORKER_PREFIX)),
        None,
    )
    if line is None:
        raise RuntimeError(f"Worker returned no JSON: {completed.stdout}\n{completed.stderr}")
    return json.loads(line[len(_WORKER_PREFIX) :])


def _median_int(values: Sequence[int]) -> int:
    return int(statistics.median(values))


def summarize_runtime_workers(
    dense_workers: Sequence[Dict[str, object]],
    quantized_workers: Sequence[Dict[str, object]],
    *,
    dense_persistent_bytes: int,
    quantized_persistent_bytes: int,
    require_rss_reduction: bool,
    require_weight_dominated_load: bool,
    min_rss_savings_bytes: int,
    min_rss_savings_fraction: float,
    require_nonoverlapping_rss: bool,
) -> Dict[str, object]:
    """Aggregate repeated workers and evaluate explicit, separate RAM gates."""

    if not dense_workers or len(dense_workers) != len(quantized_workers):
        raise ValueError("Dense and quantized runtime repetitions must be nonempty and paired")
    if dense_persistent_bytes <= 0 or quantized_persistent_bytes < 0:
        raise ValueError("Persistent byte totals must be positive for dense and nonnegative for quantized")
    if min_rss_savings_bytes < 0 or not 0 <= min_rss_savings_fraction <= 1:
        raise ValueError("Invalid RSS savings thresholds")
    phases = {}
    for phase in _PHASES:
        dense_values = [int(worker[phase]["sampled_phase_peak_rss_bytes"]) for worker in dense_workers]
        quantized_values = [
            int(worker[phase]["sampled_phase_peak_rss_bytes"]) for worker in quantized_workers
        ]
        dense_median = _median_int(dense_values)
        quantized_median = _median_int(quantized_values)
        savings = dense_median - quantized_median
        fraction = savings / dense_median if dense_median else 0.0
        nonoverlapping = min(dense_values) > max(quantized_values)
        passed = (
            savings > 0
            and savings >= min_rss_savings_bytes
            and fraction >= min_rss_savings_fraction
            and (not require_nonoverlapping_rss or nonoverlapping)
        )
        phases[phase] = {
            "dense_repetitions_bytes": dense_values,
            "quantized_repetitions_bytes": quantized_values,
            "dense_median_bytes": dense_median,
            "quantized_median_bytes": quantized_median,
            "median_savings_bytes": savings,
            "median_savings_fraction": fraction,
            "every_paired_repetition_reduced": all(
                quantized < dense for dense, quantized in zip(dense_values, quantized_values)
            ),
            "ranges_nonoverlapping": nonoverlapping,
            "gate_passed": passed,
        }
    dense_baseline = _median_int([
        int(worker["pre_model"]["current_rss_bytes"]) for worker in dense_workers
    ])
    quantized_baseline = _median_int([
        int(worker["pre_model"]["current_rss_bytes"]) for worker in quantized_workers
    ])
    dense_load = phases["load"]["dense_median_bytes"]
    load_increment = dense_load - dense_baseline
    load_measurement_valid = load_increment > 0
    incremental_nonweight = max(0, load_increment - dense_persistent_bytes)
    dominates_incremental = (
        load_measurement_valid and dense_persistent_bytes > incremental_nonweight
    )
    dominates_whole_baseline = dense_persistent_bytes > dense_baseline
    persistent_savings = dense_persistent_bytes - quantized_persistent_bytes
    dense_hashes = {worker["prefill"]["logits_sha256"] for worker in dense_workers}
    quantized_hashes = {worker["prefill"]["logits_sha256"] for worker in quantized_workers}
    dense_generations = {
        tuple(worker["decode_without_kv_cache"]["generated_token_ids"])
        for worker in dense_workers
    }
    quantized_generations = {
        tuple(worker["decode_without_kv_cache"]["generated_token_ids"])
        for worker in quantized_workers
    }
    exact_persistent = all(
        int(worker["persistent_storage_after_run"]["model_unique_state_tensor_bytes"])
        == quantized_persistent_bytes
        for worker in quantized_workers
    ) and all(
        int(worker["persistent_storage_after_run"]["model_unique_state_tensor_bytes"])
        == dense_persistent_bytes
        for worker in dense_workers
    )
    zero_cache = all(
        int(worker["persistent_storage_after_run"]["decoded_cache_bytes"]) == 0
        for worker in quantized_workers
    )
    no_dense_source = all(
        not worker["source_dense_checkpoint_passed_to_worker"] for worker in quantized_workers
    )
    dense_source = all(
        worker["source_dense_checkpoint_passed_to_worker"] for worker in dense_workers
    )
    dense_artifacts = {worker["artifact"] for worker in dense_workers}
    quantized_artifacts = {worker["artifact"] for worker in quantized_workers}
    runtime_inputs = {worker["runtime_input"] for worker in (*dense_workers, *quantized_workers)}
    worker_contract = (
        all(worker["mode"] == "dense" for worker in dense_workers)
        and all(worker["mode"] == "quantized" for worker in quantized_workers)
        and all(worker["profile"] == "memory" for worker in (*dense_workers, *quantized_workers))
        and len(dense_artifacts) == 1
        and len(quantized_artifacts) == 1
        and dense_artifacts != quantized_artifacts
        and len(runtime_inputs) == 1
        and dense_source
        and no_dense_source
    )
    reproducible = (
        worker_contract
        and len(dense_hashes) == 1
        and len(quantized_hashes) == 1
        and len(dense_generations) == 1
        and len(quantized_generations) == 1
        and exact_persistent
        and zero_cache
        and no_dense_source
    )
    rss_passed = all(phases[phase]["gate_passed"] for phase in _PHASES)
    measured_reduction = all(phases[phase]["median_savings_bytes"] > 0 for phase in _PHASES)
    acceptance = {
        "reproducibility_and_accounting_passed": reproducible,
        "rss_reduction_required": require_rss_reduction,
        "rss_reduction_passed": rss_passed,
        "measured_median_rss_reduction_in_all_phases": measured_reduction,
        "weight_dominated_incremental_load_required": require_weight_dominated_load,
        "weight_dominated_incremental_load_passed": dominates_incremental,
        "passed": (
            reproducible
            and persistent_savings > 0
            and (not require_rss_reduction or rss_passed)
            and (not require_weight_dominated_load or dominates_incremental)
        ),
    }
    return {
        "repetitions_per_mode": len(dense_workers),
        "persistent_savings_bytes": persistent_savings,
        "persistent_savings_fraction": persistent_savings / dense_persistent_bytes,
        "rss": phases,
        "latency": {
            "dense_prefill_median_of_worker_medians_seconds": statistics.median(
                float(worker["prefill"]["median_seconds"]) for worker in dense_workers
            ),
            "quantized_prefill_median_of_worker_medians_seconds": statistics.median(
                float(worker["prefill"]["median_seconds"]) for worker in quantized_workers
            ),
            "dense_decode_median_of_worker_medians_seconds": statistics.median(
                float(worker["decode_without_kv_cache"]["median_seconds_per_recomputed_token"])
                for worker in dense_workers
            ),
            "quantized_decode_median_of_worker_medians_seconds": statistics.median(
                float(worker["decode_without_kv_cache"]["median_seconds_per_recomputed_token"])
                for worker in quantized_workers
            ),
            "scope": "Latency is reported separately; no speedup is inferred from RSS savings",
        },
        "weight_residency": {
            "dense_pre_model_median_rss_bytes": dense_baseline,
            "quantized_pre_model_median_rss_bytes": quantized_baseline,
            "dense_persistent_bytes": dense_persistent_bytes,
            "dense_load_increment_bytes": load_increment,
            "dense_load_measurement_valid": load_measurement_valid,
            "incremental_nonweight_load_allocation_bytes": incremental_nonweight,
            "dominates_incremental_nonweight_load_allocation": dominates_incremental,
            "dominates_entire_pre_model_process_baseline": dominates_whole_baseline,
            "dense_persistent_fraction_of_post_load_rss": (
                dense_persistent_bytes / dense_load if dense_load else 0.0
            ),
            "scope": (
                "Incremental-load dominance and dominance over the whole Python/CUDA "
                "import baseline are different claims and are deliberately not merged"
            ),
        },
        "correctness": {
            "paired_worker_contract_valid": worker_contract,
            "dense_workers_used_dense_source": dense_source,
            "dense_prefill_hash_consistent_across_repetitions": len(dense_hashes) == 1,
            "quantized_prefill_hash_consistent_across_repetitions": len(quantized_hashes) == 1,
            "dense_generation_consistent_across_repetitions": len(dense_generations) == 1,
            "quantized_generation_consistent_across_repetitions": len(quantized_generations) == 1,
            "dense_and_quantized_prefill_bitwise_equal": dense_hashes == quantized_hashes,
            "dense_and_quantized_generated_tokens_equal": dense_generations == quantized_generations,
            "quantized_persistent_bytes_exact": exact_persistent,
            "quantized_decoded_cache_bytes_zero": zero_cache,
            "quantized_workers_had_no_dense_source": no_dense_source,
        },
        "acceptance": acceptance,
        "ram_claim": (
            "Qualified median fresh-process RSS reduction under the declared gates"
            if (
                require_rss_reduction
                and require_weight_dominated_load
                and acceptance["passed"]
            )
            else "No qualified dense-RAM reduction claim"
        ),
    }


def _artifact_file_report(output: Path) -> Dict[str, object]:
    files = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    return {
        "path": str(output),
        "files": files,
        "total_file_bytes": sum(entry["bytes"] for entry in files.values()),
    }


def qualify(args) -> int:
    torch.set_num_threads(args.num_threads)
    source = Path(args.source).resolve()
    provenance_path = Path(args.provenance).resolve()
    calibration_path = Path(args.calibration_data).resolve()
    validation_path = Path(args.validation_data).resolve()
    workspace = Path(args.workspace).resolve()
    evidence_path = Path(args.evidence).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = validate_source_provenance(
        source,
        provenance_path,
        minimum_bf16_fraction=args.min_source_bf16_fraction,
    )
    calibration_sha = _sha256(calibration_path)
    validation_sha = _sha256(validation_path)
    if calibration_sha != args.calibration_sha256 or validation_sha != args.validation_sha256:
        raise ValueError("Calibration/validation data SHA-256 does not match the command pin")
    if calibration_sha == validation_sha:
        raise ValueError("Calibration and validation content must be independent, not identical")
    tokenizer = _load_source_tokenizer(source, provenance["declaration"])
    calibration = _token_windows(
        tokenizer,
        calibration_path,
        sequence_length=args.sequence_length,
        batches=args.calibration_batches,
    )
    validation = _token_windows(
        tokenizer,
        validation_path,
        sequence_length=args.sequence_length,
        batches=args.validation_batches,
    )
    runtime_input = {
        key: value.tolist()
        for key, value in validation[0].items()
    }
    runtime_input_path = workspace / "runtime-input.json"
    _json_write(runtime_input_path, runtime_input)

    dense = _load_dense(source)
    unique_parameters = _unique_parameter_count(dense)
    if unique_parameters < args.min_unique_parameters:
        raise ValueError(
            f"Checkpoint has {unique_parameters} unique parameters, below the larger-model "
            f"minimum {args.min_unique_parameters}"
        )
    declaration = provenance["declaration"]
    if declaration["unique_parameters"] != unique_parameters:
        raise ValueError("Loaded unique parameter count differs from source provenance")
    if "model_type" in declaration and declaration["model_type"] != dense.config.model_type:
        raise ValueError("Loaded model_type differs from source provenance")
    dense_storage = runtime_storage_report(dense)
    reference = copy.deepcopy(dense)
    candidate = copy.deepcopy(dense)
    constraints = PlanningConstraints(
        target_state_ratio=args.target_state_ratio,
        max_tensor_estimated_error=args.max_tensor_estimated_error,
        runtime_profile="memory",
        decode_row_block_size=args.decode_row_block_size,
        max_frontier_states=args.max_frontier_states,
    )
    thresholds = _quality_thresholds(args)
    result = quantize_model_generalized(
        candidate,
        constraints=constraints,
        calibration_inputs=calibration,
        candidate_configs=_candidate_configs(args.codec),
        include=(args.include,),
        reference_model=reference,
        quality_thresholds=thresholds,
        validation_inputs=validation,
        max_activation_rows=args.max_activation_rows,
        max_invocations_per_module=args.max_invocations_per_module,
        max_positions_per_batch=args.max_positions_per_batch,
    )
    eligible_bytes = sum(spec.original_weight_bytes for spec in result.plan.specs if spec.quantizable)
    selected = result.plan.selected_candidates
    selected_codecs = {}
    for candidate_ in selected.values():
        if not candidate_.is_dense:
            selected_codecs[candidate_.codec] = selected_codecs.get(candidate_.codec, 0) + 1
    evidence = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "unvalidated",
        "source_provenance": provenance,
        "data_provenance": {
            "calibration": {"path": str(calibration_path), "sha256": calibration_sha},
            "validation": {"path": str(validation_path), "sha256": validation_sha},
            "independent_content": True,
            "sequence_length": args.sequence_length,
            "calibration_batches": args.calibration_batches,
            "validation_batches": args.validation_batches,
            "scope": "Only the supplied token windows; not the original training holdout unless declared separately",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(),
        },
        "model": {
            "model_type": dense.config.model_type,
            "unique_parameters": unique_parameters,
            "minimum_unique_parameters": args.min_unique_parameters,
            "quantizable_original_weight_bytes": eligible_bytes,
            "selected_codecs": selected_codecs,
        },
        "dense_bf16_storage": dense_storage,
        "generalized": result.to_dict(include_alternatives=False),
        "scope": (
            "One pinned checkpoint and supplied data on one host. Quality gates are "
            "not broad downstream certification; RSS, persistent bytes, and latency "
            "are separate measurements."
        ),
    }
    quality_failures = list(result.quality.failures)
    reference_nll = result.quality.metrics.get("reference_mean_nll")
    if reference_nll is None or reference_nll > args.max_dense_reference_nll:
        quality_failures.append(
            f"dense_reference_nll={reference_nll} exceeds {args.max_dense_reference_nll}"
        )
    if eligible_bytes < args.min_quantizable_weight_bytes:
        quality_failures.append(
            f"quantizable_weight_bytes={eligible_bytes} below {args.min_quantizable_weight_bytes}"
        )
    persistent_savings = int(result.plan.memory["predicted_weight_savings_bytes"])
    if persistent_savings < args.min_persistent_savings_bytes:
        quality_failures.append(
            f"persistent_savings={persistent_savings} below {args.min_persistent_savings_bytes}"
        )
    if quality_failures:
        evidence["status"] = "failed"
        evidence["qualification_error"] = quality_failures
        _json_write(evidence_path, evidence)
        print(json.dumps({"status": "failed", "errors": quality_failures}))
        return 2

    output = workspace / "encoded"
    save_quantized_pretrained(
        result.model,
        output,
        tokenizer=tokenizer,
        report=result.report,
        plan=result.plan,
        max_shard_size=args.max_shard_size,
    )
    _json_write(output / "source-provenance.json", provenance)
    _json_write(output / "quality-before-save.json", result.quality.to_dict())
    _json_write(output / "runtime-input.json", runtime_input)
    reloaded = load_quantized_pretrained(
        output,
        load_tokenizer=False,
        runtime_profile="memory",
    )
    reload_storage = runtime_storage_report(reloaded)
    predicted = int(result.plan.memory["predicted_model_state_bytes"])
    if reload_storage["model_unique_state_tensor_bytes"] != predicted:
        evidence["status"] = "failed"
        evidence["qualification_error"] = [
            "Meta-only reloaded persistent state differs from the plan: "
            f"predicted={predicted}, actual={reload_storage['model_unique_state_tensor_bytes']}"
        ]
        _json_write(evidence_path, evidence)
        print(json.dumps({"status": "failed", "errors": evidence["qualification_error"]}))
        return 2
    reload_quality = evaluate_quality_gates(
        reference,
        reloaded,
        validation,
        thresholds,
        max_positions_per_batch=args.max_positions_per_batch,
    )
    _json_write(output / "quality-after-meta-reload.json", reload_quality.to_dict())
    evidence["quality_after_meta_reload"] = reload_quality.to_dict()
    evidence["artifact"] = {
        **_artifact_file_report(output),
        "predicted_persistent_bytes": predicted,
        "actual_reloaded_persistent_bytes": reload_storage["model_unique_state_tensor_bytes"],
        "dense_source_materialized_on_quantized_reload": False,
    }
    if not reload_quality.passed:
        evidence["status"] = "failed"
        evidence["qualification_error"] = [
            "Post-save meta-reload quality gates failed: " + "; ".join(reload_quality.failures)
        ]
        _json_write(evidence_path, evidence)
        print(json.dumps({"status": "failed", "errors": evidence["qualification_error"]}))
        return 2

    del reloaded, candidate, dense, reference, result
    gc.collect()
    dense_workers = []
    quantized_workers = []
    for repetition in range(args.runtime_repeats):
        order = (("dense", source), ("quantized", output))
        if repetition % 2:
            order = tuple(reversed(order))
        for mode, artifact in order:
            worker = _run_worker(artifact, runtime_input_path, mode, args)
            worker["repetition"] = repetition
            (dense_workers if mode == "dense" else quantized_workers).append(worker)
    comparison = summarize_runtime_workers(
        dense_workers,
        quantized_workers,
        dense_persistent_bytes=int(dense_storage["model_unique_state_tensor_bytes"]),
        quantized_persistent_bytes=predicted,
        require_rss_reduction=args.require_rss_reduction,
        require_weight_dominated_load=args.require_weight_dominated_load,
        min_rss_savings_bytes=args.min_rss_savings_bytes,
        min_rss_savings_fraction=args.min_rss_savings_fraction,
        require_nonoverlapping_rss=args.require_nonoverlapping_rss,
    )
    evidence["runtime_workers"] = {
        "dense": dense_workers,
        "quantized_memory": quantized_workers,
    }
    evidence["runtime_comparison"] = comparison
    runtime_passed = comparison["acceptance"]["passed"]
    full_ram_qualification = args.require_rss_reduction and args.require_weight_dominated_load
    evidence["status"] = (
        "passed"
        if runtime_passed and full_ram_qualification
        else "passed_without_full_ram_qualification"
        if runtime_passed
        else "failed"
    )
    evidence["dense_ram_reduction_claim"] = comparison["ram_claim"]
    _json_write(evidence_path, evidence)
    quality = reload_quality.metrics
    print(
        "Larger BF16 qualification: "
        f"status={evidence['status']}, parameters={unique_parameters}, "
        f"persistent-savings={persistent_savings} bytes, "
        f"decode-RSS-savings={comparison['rss']['decode_without_kv_cache']['median_savings_bytes']} bytes, "
        f"relative-logits={quality['relative_logits_l2_error']:.6f}, "
        f"NLL-delta={quality['mean_nll_delta']:.6f}"
    )
    return 0 if runtime_passed else 2


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("qualify")
    command.add_argument("--source", required=True)
    command.add_argument("--provenance", required=True)
    command.add_argument("--calibration-data", required=True)
    command.add_argument("--calibration-sha256", required=True)
    command.add_argument("--validation-data", required=True)
    command.add_argument("--validation-sha256", required=True)
    command.add_argument("--workspace", required=True)
    command.add_argument("--evidence", required=True)
    command.add_argument("--codec", choices=("mixed", "rvq", "leech24"), default="mixed")
    command.add_argument("--include", default="*")
    command.add_argument("--target-state-ratio", type=float, default=0.80)
    command.add_argument("--max-tensor-estimated-error", type=float, default=2.0)
    command.add_argument("--decode-row-block-size", type=_positive, default=64)
    command.add_argument("--max-frontier-states", type=int, default=1024)
    command.add_argument("--max-relative-logits", type=float, default=0.15)
    command.add_argument("--max-absolute-logits", type=float)
    command.add_argument("--max-kl", type=float, default=0.05)
    command.add_argument("--min-top1", type=float, default=0.85)
    command.add_argument("--max-nll-delta", type=float, default=0.10)
    command.add_argument("--max-dense-reference-nll", type=float, default=8.0)
    command.add_argument("--min-unique-parameters", type=_positive, default=25_000_000)
    command.add_argument("--min-quantizable-weight-bytes", type=int, default=16_000_000)
    command.add_argument("--min-persistent-savings-bytes", type=int, default=1_000_000)
    command.add_argument("--min-source-bf16-fraction", type=float, default=0.95)
    command.add_argument("--sequence-length", type=_positive, default=32)
    command.add_argument("--calibration-batches", type=_positive, default=2)
    command.add_argument("--validation-batches", type=_positive, default=3)
    command.add_argument("--max-activation-rows", type=_positive, default=64)
    command.add_argument("--max-invocations-per-module", type=_positive, default=2)
    command.add_argument("--max-positions-per-batch", type=_positive, default=256)
    command.add_argument("--max-shard-size", default="512MB")
    command.add_argument(
        "--leech-backend",
        choices=("python", "native", "auto"),
        default="auto",
    )
    command.add_argument("--num-threads", type=_positive, default=1)
    command.add_argument("--runtime-repeats", type=_positive, default=3)
    command.add_argument("--prefill-warmup", type=int, default=1)
    command.add_argument("--prefill-iterations", type=_positive, default=3)
    command.add_argument("--decode-tokens", type=_positive, default=2)
    command.add_argument(
        "--require-rss-reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument(
        "--require-weight-dominated-load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument("--min-rss-savings-bytes", type=int, default=1)
    command.add_argument("--min-rss-savings-fraction", type=float, default=0.0)
    command.add_argument("--require-nonoverlapping-rss", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--artifact", required=True)
    worker.add_argument("--runtime-input", required=True)
    worker.add_argument("--mode", choices=("dense", "quantized"), required=True)
    worker.add_argument("--profile", choices=("memory",), default="memory")
    worker.add_argument(
        "--leech-backend",
        choices=("python", "native", "auto"),
        default="auto",
    )
    worker.add_argument("--num-threads", type=_positive, default=1)
    worker.add_argument("--prefill-warmup", type=int, default=1)
    worker.add_argument("--prefill-iterations", type=_positive, default=3)
    worker.add_argument("--decode-tokens", type=_positive, default=2)
    args = parser.parse_args()
    if args.prefill_warmup < 0:
        raise ValueError("Prefill warmup must be nonnegative")
    if args.command == "worker":
        payload = _runtime_worker(args)
        print(_WORKER_PREFIX + json.dumps(payload, sort_keys=True))
        return
    if args.min_rss_savings_bytes < 0 or not 0 <= args.min_rss_savings_fraction <= 1:
        raise ValueError("Invalid RSS savings thresholds")
    raise SystemExit(qualify(args))


if __name__ == "__main__":
    main()

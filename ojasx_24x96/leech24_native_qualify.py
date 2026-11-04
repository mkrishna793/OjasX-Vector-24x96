"""Qualify the optional native CPU Leech decoder on a fixed real artifact.

Run after ``leech24_qualify qualify``. The input artifact is not modified. Python
and native workers load the same encoded shard in fresh processes. Bitwise
Python/native prefill equality is mandatory; dense-versus-quantized equality is
not expected because quantization is lossy.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import json
from pathlib import Path
import pstats
import statistics
import time
from typing import Dict

import torch

from .leech24_qualify import (
    _json_write,
    _load_quantized,
    _run_worker,
    _sha256,
    _tensor_sha256,
)
from .leech_native import native_backend_status
from .runtime import runtime_storage_report, set_leech_decode_backend, set_runtime_profile


_PROFILE_FUNCTIONS = {
    "encoded",
    "dequantize_leech_weight_rows",
    "_decode_rotated_vectors",
    "decode_indices",
    "decode_leech_rows_native",
    "apply_rotation",
    "forward",
}


def _profile_and_compare(artifact: Path, iterations: int) -> Dict[str, object]:
    torch.set_num_threads(1)
    model = _load_quantized(artifact)
    payload = json.loads((artifact / "runtime-input.json").read_text())
    input_ids = torch.tensor(payload["input_ids"], dtype=torch.int64)
    outputs = {}
    measurements = {}
    profiles = {}
    for backend in ("python", "native"):
        set_runtime_profile(model, "memory")
        set_leech_decode_backend(model, backend)
        with torch.inference_mode():
            for _ in range(2):
                model(input_ids=input_ids)
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(3):
                output = model(input_ids=input_ids).logits
            profiler.disable()
            times = []
            for _ in range(iterations):
                started = time.perf_counter()
                output = model(input_ids=input_ids).logits
                times.append(time.perf_counter() - started)
        outputs[backend] = output.clone()
        measurements[backend] = {
            "iterations": iterations,
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "logits_sha256": _tensor_sha256(output),
        }
        stats = pstats.Stats(profiler)
        functions = []
        for (filename, line, name), values in stats.stats.items():
            if name not in _PROFILE_FUNCTIONS:
                continue
            _, calls, self_seconds, cumulative_seconds, _ = values
            functions.append({
                "name": name,
                "file": Path(filename).name,
                "line": line,
                "calls": calls,
                "self_seconds": self_seconds,
                "cumulative_seconds": cumulative_seconds,
            })
        profiles[backend] = {
            "profiled_forwards": 3,
            "total_self_seconds": stats.total_tt,
            "functions": sorted(
                functions,
                key=lambda item: item["cumulative_seconds"],
                reverse=True,
            ),
        }
    exact = torch.equal(outputs["python"], outputs["native"])
    if not exact:
        difference = (outputs["python"].float() - outputs["native"].float()).abs()
        raise AssertionError(
            f"Native and Python logits differ: max={difference.max().item()}, "
            f"count={torch.count_nonzero(difference).item()}"
        )
    return {
        "bitwise_python_native_logits_equal": True,
        "logits_sha256": _tensor_sha256(outputs["native"]),
        "same_process_timings": measurements,
        "same_process_speedup": (
            measurements["python"]["median_seconds"]
            / measurements["native"]["median_seconds"]
        ),
        "function_profiles": profiles,
        "storage_after_native_memory_run": runtime_storage_report(model),
        "scope": (
            "Same-process timings isolate backend changes but share allocated tables and "
            "allocator state. Fresh workers below are the RSS evidence."
        ),
    }


def qualify(args) -> int:
    artifact = Path(args.artifact).resolve()
    status = native_backend_status()
    if not status["available"]:
        raise RuntimeError(f"Native backend unavailable: {status['reason']}")
    shard = artifact / "model.safetensors"
    workers = [
        _run_worker(artifact.parent / "source-bf16", "dense", "memory"),
        _run_worker(artifact, "quantized", "memory", "python"),
        _run_worker(artifact, "quantized", "memory", "native"),
        _run_worker(artifact, "quantized", "speed", "python"),
        _run_worker(artifact, "quantized", "speed", "native"),
    ]
    dense, python_memory, native_memory, python_speed, native_speed = workers
    if python_memory["prefill"]["logits_sha256"] != native_memory["prefill"]["logits_sha256"]:
        raise AssertionError("Fresh Python/native memory workers have different prefill logits")
    if python_speed["prefill"]["logits_sha256"] != native_speed["prefill"]["logits_sha256"]:
        raise AssertionError("Fresh Python/native speed workers have different prefill logits")
    quantized_tokens = [
        worker["decode_without_kv_cache"]["generated_token_ids"] for worker in workers[1:]
    ]
    if any(tokens != quantized_tokens[0] for tokens in quantized_tokens[1:]):
        raise AssertionError("Python/native workers generated different token IDs")
    for worker in (python_memory, native_memory):
        if worker["persistent_storage_after_run"]["decoded_cache_bytes"] != 0:
            raise AssertionError("Memory worker retained a decoded cache")
    profile = _profile_and_compare(artifact, args.iterations)

    def decode_seconds(worker):
        return float(
            worker["decode_without_kv_cache"]["median_seconds_per_recomputed_token"]
        )

    def rss_delta(worker, phase):
        return int(worker[phase]["sampled_phase_peak_rss_bytes"]) - int(
            python_memory[phase]["sampled_phase_peak_rss_bytes"]
        )

    evidence = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "passed",
        "artifact": str(artifact),
        "model_shard_sha256": _sha256(shard),
        "native_backend": status,
        "environment": {
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "correctness": {
            "fresh_memory_prefill_logits_sha256": native_memory["prefill"]["logits_sha256"],
            "fresh_speed_prefill_logits_sha256": native_speed["prefill"]["logits_sha256"],
            "python_native_memory_prefill_bitwise_equal": True,
            "python_native_speed_prefill_bitwise_equal": True,
            "all_quantized_worker_generated_token_ids_match": True,
            "generated_token_ids": quantized_tokens[0],
        },
        "runtime_workers": workers,
        "same_process_profile": profile,
        "comparison": {
            "native_memory_decode_speedup_vs_python": (
                decode_seconds(python_memory) / decode_seconds(native_memory)
            ),
            "native_memory_decode_time_ratio_to_dense": (
                decode_seconds(native_memory) / decode_seconds(dense)
            ),
            "python_memory_decode_time_ratio_to_dense": (
                decode_seconds(python_memory) / decode_seconds(dense)
            ),
            "native_speed_decode_time_ratio_to_dense": (
                decode_seconds(native_speed) / decode_seconds(dense)
            ),
            "native_minus_python_sampled_rss_bytes": {
                phase: rss_delta(native_memory, phase)
                for phase in ("load", "prefill", "decode_without_kv_cache")
            },
            "python_memory_initialized_table_bytes": python_memory[
                "initialized_python_shared_table_bytes"
            ],
            "native_memory_initialized_python_table_bytes": native_memory[
                "initialized_python_shared_table_bytes"
            ],
            "native_cpp_table_bytes": status["native_table_bytes"],
            "native_memory_decoded_cache_bytes": native_memory[
                "persistent_storage_after_run"
            ]["decoded_cache_bytes"],
            "native_speed_decoded_cache_bytes": native_speed[
                "persistent_storage_after_run"
            ]["decoded_cache_bytes"],
        },
        "scope": (
            "One fixed BF16 TinyGPT artifact on one CPU, five fresh processes, and "
            "four recomputed tokens without KV caching. Native results do not establish "
            "CUDA support, throughput, larger-model RAM savings, or dense-model quality."
        ),
    }
    _json_write(Path(args.evidence), evidence)
    print(
        "Leech24 native qualification: PASSED, "
        f"memory-decode-speedup={evidence['comparison']['native_memory_decode_speedup_vs_python']:.3f}x, "
        f"native/dense-decode-time={evidence['comparison']['native_memory_decode_time_ratio_to_dense']:.3f}x, "
        f"native-table-bytes={status['native_table_bytes']}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("Iterations must be positive")
    raise SystemExit(qualify(args))


if __name__ == "__main__":
    main()

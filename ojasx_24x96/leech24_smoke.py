"""Reproducible Leech24 math, rate-distortion, and streaming evidence."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from .codec import QuantizationConfig, quantize_weight
from .leech24 import (
    INDEX_COUNT,
    INDEX_BITS,
    apply_rotation,
    decode_indices,
    golay_codewords,
    leech_contains,
    nearest_shell2,
    pack_indices18,
    shared_table_bytes,
    unpack_indices18,
)
from .leech_codec import (
    LeechQuantizationConfig,
    dequantize_leech_weight,
    quantize_leech_weight,
)
from .module import ResidualVQLinear
from .runtime import runtime_storage_report, set_runtime_profile


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    indices = torch.arange(INDEX_COUNT, dtype=torch.int64)
    points = decode_indices(indices)
    packed = pack_indices18(indices)
    unpacked = unpack_indices18(packed, INDEX_COUNT)
    unique_count = len({row.numpy().tobytes() for row in points})
    math_evidence = {
        "golay_codewords": len(golay_codewords()),
        "golay_weight_distribution": {
            str(weight): sum(word.bit_count() == weight for word in golay_codewords())
            for weight in (0, 8, 12, 16, 24)
        },
        "decoded_indices": INDEX_COUNT,
        "unique_decoded_points": unique_count,
        "nonzero_shell_points": INDEX_COUNT - 1,
        "all_points_are_leech_members": bool(leech_contains(points).all().item()),
        "all_nonzero_points_on_shell_2": bool(
            (points[1:].to(torch.int64).square().sum(dim=1) == 32).all().item()
        ),
        "index_bits": INDEX_BITS,
        "packed_bytes_for_complete_index_domain": packed.numel(),
        "complete_pack_roundtrip_exact": bool(torch.equal(indices, unpacked)),
        "decoder_table_bytes_before_search": shared_table_bytes(),
    }
    if not (
        unique_count == INDEX_COUNT
        and math_evidence["all_points_are_leech_members"]
        and math_evidence["complete_pack_roundtrip_exact"]
    ):
        raise RuntimeError("Leech mathematical smoke failed")

    vectors = torch.randn(4, 24, dtype=torch.float64)
    nearest, dots = nearest_shell2(vectors, chunk_size=2)
    shell_points = points[1:].to(torch.float64)
    brute_force_error = 0.0
    for row in range(vectors.shape[0]):
        expected = (shell_points @ vectors[row]).max().item()
        actual = (
            decode_indices(nearest[row]).to(torch.float64) @ vectors[row]
        ).item()
        brute_force_error = max(brute_force_error, abs(expected - actual))
    rotated = apply_rotation(vectors, 4, 71)
    rotation_error = (
        apply_rotation(rotated, 4, 71, inverse=True) - vectors
    ).abs().max().item()
    math_evidence["brute_force_nearest_max_absolute_dot_error"] = brute_force_error
    math_evidence["shared_tables_after_encoder_search_bytes"] = shared_table_bytes()
    math_evidence["householder_roundtrip_max_absolute_error"] = rotation_error
    if brute_force_error > 1e-10 or rotation_error > 1e-5:
        raise RuntimeError("Leech search/rotation smoke failed")

    source = torch.randn(32, 96) * 0.1
    original_bytes = source.numel() * source.element_size()
    rate_distortion = []
    for gain_bits in (0, 1, 2):
        for reflections in (0, 2):
            for adaptive in ((True, False) if gain_bits == 1 else (True,)):
                config = LeechQuantizationConfig(
                    max_stages=3,
                    max_relative_error=None,
                    gain_bits=gain_bits,
                    rotation_reflections=reflections,
                    adaptive_stages=adaptive,
                    max_gain_samples=512,
                    gain_grid_size=16,
                    chunk_size=32,
                    min_weight_elements=1,
                    require_compression=False,
                    seed=71,
                )
                encoded, report = quantize_leech_weight(source, config)
                if encoded is None:
                    raise RuntimeError(f"Leech smoke candidate failed: {report}")
                rate_distortion.append({
                    "codec": "leech24_shell2_residual",
                    "gain_bits": gain_bits,
                    "rotation_reflections": reflections,
                    "adaptive_stages": adaptive,
                    "stages": encoded.stages,
                    "original_bytes": original_bytes,
                    "storage_bytes": encoded.storage_bytes,
                    "measured_bits_per_weight": encoded.measured_bits_per_weight,
                    "relative_error": report["relative_error"],
                    "stage_relative_errors": report["stage_relative_errors"],
                    "active_counts": report["active_counts"],
                })

    learned_configs = (
        (24, 1, 5),
        (24, 2, 2),
        (24, 4, 1),
        (16, 1, 4),
        (16, 2, 3),
        (16, 4, 2),
        (8, 2, 4),
        (8, 4, 1),
        (8, 4, 2),
    )
    for block_size, bits, stages in learned_configs:
        config = QuantizationConfig(
            block_size=block_size,
            bits_per_stage=bits,
            max_stages=stages,
            max_relative_error=None,
            max_codebook_samples=512,
            kmeans_iterations=8,
            chunk_size=64,
            min_weight_elements=1,
            require_compression=False,
            seed=71,
        )
        encoded, report = quantize_weight(source, config)
        if encoded is None:
            raise RuntimeError(f"Learned-VQ smoke candidate failed: {report}")
        rate_distortion.append({
            "codec": "adaptive_residual_vq",
            "block_size": block_size,
            "bits_per_stage": bits,
            "stages": encoded.stages,
            "original_bytes": original_bytes,
            "storage_bytes": encoded.storage_bytes,
            "measured_bits_per_weight": encoded.measured_bits_per_weight,
            "relative_error": report["relative_error"],
        })

    learned_rows = [
        row for row in rate_distortion if row["codec"] == "adaptive_residual_vq"
    ]
    leech_rows = [
        row for row in rate_distortion if row["codec"] == "leech24_shell2_residual"
    ]

    def learned_summary(row):
        return {
            key: row[key]
            for key in (
                "block_size",
                "bits_per_stage",
                "stages",
                "storage_bytes",
                "measured_bits_per_weight",
                "relative_error",
            )
        }

    for row in leech_rows:
        budget = row["storage_bytes"]
        eligible = [candidate for candidate in learned_rows if candidate["storage_bytes"] <= budget]
        exact = [candidate for candidate in eligible if candidate["storage_bytes"] == budget]
        row["best_learned_vq_at_or_below_actual_byte_budget"] = (
            learned_summary(min(eligible, key=lambda candidate: candidate["relative_error"]))
            if eligible
            else None
        )
        row["best_learned_vq_at_identical_actual_bytes"] = (
            learned_summary(min(exact, key=lambda candidate: candidate["relative_error"]))
            if exact
            else None
        )

    rotation_comparisons = []
    adaptive_comparisons = []
    for row in leech_rows:
        if row["rotation_reflections"] != 0:
            continue
        rotated = next(
            candidate
            for candidate in leech_rows
            if candidate["gain_bits"] == row["gain_bits"]
            and candidate["adaptive_stages"] == row["adaptive_stages"]
            and candidate["rotation_reflections"] == 2
        )
        rotation_comparisons.append({
            "gain_bits": row["gain_bits"],
            "adaptive_stages": row["adaptive_stages"],
            "identical_storage_bytes": row["storage_bytes"] == rotated["storage_bytes"],
            "storage_bytes": row["storage_bytes"],
            "relative_error_without_rotation": row["relative_error"],
            "relative_error_with_two_reflections": rotated["relative_error"],
        })
        if row["adaptive_stages"] and row["gain_bits"] == 1:
            nonadaptive = next(
                candidate
                for candidate in leech_rows
                if candidate["gain_bits"] == row["gain_bits"]
                and candidate["rotation_reflections"] == 0
                and not candidate["adaptive_stages"]
            )
            adaptive_comparisons.append({
                "gain_bits": row["gain_bits"],
                "rotation_reflections": 0,
                "adaptive_storage_bytes": row["storage_bytes"],
                "nonadaptive_storage_bytes": nonadaptive["storage_bytes"],
                "adaptive_relative_error": row["relative_error"],
                "nonadaptive_relative_error": nonadaptive["relative_error"],
            })

    selected, _ = quantize_leech_weight(
        source,
        LeechQuantizationConfig(
            max_stages=3,
            max_relative_error=None,
            gain_bits=1,
            rotation_reflections=2,
            max_gain_samples=512,
            gain_grid_size=16,
            chunk_size=32,
            min_weight_elements=1,
            require_compression=False,
            seed=71,
        ),
    )
    if selected is None:
        raise RuntimeError("Streaming smoke candidate failed")
    module = ResidualVQLinear.from_encoded(
        selected,
        bias=torch.randn(source.shape[0]),
        source_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    inputs = torch.randn(5, source.shape[1])
    module.set_streaming_decode(True, row_block_size=7)
    with torch.inference_mode():
        streamed = module(inputs)
    expected = F.linear(inputs, dequantize_leech_weight(selected), module.bias)
    torch.testing.assert_close(streamed, expected)
    memory_storage = runtime_storage_report(module)
    set_runtime_profile(module, "speed")
    with torch.inference_mode():
        cached = module(inputs)
    torch.testing.assert_close(cached, streamed)
    speed_storage = runtime_storage_report(module)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "mathematics": math_evidence,
        "synthetic_matrix": {
            "shape": list(source.shape),
            "dtype": "float32",
            "initialization": "torch.randn seed 20261104, multiplied by 0.1",
            "original_bytes": original_bytes,
        },
        "rate_distortion": rate_distortion,
        "rotation_comparisons_at_identical_bytes": rotation_comparisons,
        "adaptive_stage_comparisons": adaptive_comparisons,
        "comparison_accounting": (
            "All rows use the identical source matrix and measured encoded tensor bytes. "
            "Each Leech row selects the lowest-error generated learned-VQ candidate at "
            "or below that byte budget; exact-byte matches are reported separately when "
            "one exists. This is a finite generated-candidate comparison, not a global "
            "rate-distortion optimum."
        ),
        "streaming": {
            "row_block_size": 7,
            "memory_profile": memory_storage,
            "speed_profile": speed_storage,
            "full_versus_streamed_max_absolute_output_error": (
                streamed - expected
            ).abs().max().item(),
            "cached_versus_streamed_max_absolute_output_error": (
                cached - streamed
            ).abs().max().item(),
        },
        "scope": (
            "Exhaustive finite-shell math checks plus one synthetic matrix. "
            "Rate-distortion rows are not a universal codec ranking or model-quality claim."
        ),
    }
    output = Path(__file__).resolve().parent / "evidence/leech24-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    best = min(
        (row for row in rate_distortion if row["codec"] == "leech24_shell2_residual"),
        key=lambda row: row["relative_error"],
    )
    print(
        "Leech24 math/synthetic smoke: "
        f"indices={INDEX_COUNT}, pack={packed.numel()} bytes, "
        f"brute-force-error={brute_force_error:.3g}, "
        f"best-synthetic-error={best['relative_error']:.6f} at "
        f"{best['measured_bits_per_weight']:.3f} measured bits/weight; PASSED"
    )


if __name__ == "__main__":
    main()

"""Leech24 mathematics, compact indexing, planner, and streaming tests."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F

from ojasx_vector_quantization_runner import (
    LeechQuantizationConfig,
    PlanningConstraints,
    QualityGateThresholds,
    QuantizationConfig,
    ResidualVQLinear,
    decode_leech_indices,
    decode_leech_indices_native,
    default_leech_candidate_configs,
    dequantize_leech_weight,
    dequantize_leech_weight_rows,
    golay_codewords,
    leech_contains,
    native_leech_backend_status,
    nearest_shell2,
    pack_indices18,
    quantize_leech_weight,
    quantize_model_generalized,
    unpack_indices18,
)
from ojasx_vector_quantization_runner.leech24 import (
    INDEX_COUNT,
    INDEX_BITS,
    SHELL2_COUNT,
    apply_rotation,
    shell_index,
    shared_table_bytes,
    unpack_indices18_range,
)
from ojasx_vector_quantization_runner.runtime import (
    benchmark_forward,
    runtime_storage_report,
    set_runtime_profile,
)


def _config(**overrides):
    values = {
        "max_stages": 3,
        "max_relative_error": None,
        "gain_bits": 1,
        "rotation_reflections": 2,
        "adaptive_stages": True,
        "max_gain_samples": 128,
        "gain_grid_size": 8,
        "chunk_size": 16,
        "min_weight_elements": 1,
        "require_compression": False,
        "seed": 71,
    }
    values.update(overrides)
    return LeechQuantizationConfig(**values)


def test_golay_code_has_extended_golay_invariants():
    words = golay_codewords()
    assert len(words) == len(set(words)) == 4096
    distribution = {
        weight: sum(word.bit_count() == weight for word in words)
        for weight in range(25)
    }
    assert distribution == {
        **{weight: 0 for weight in range(25)},
        0: 1,
        8: 759,
        12: 2576,
        16: 759,
        24: 1,
    }
    assert all(word.bit_count() % 4 == 0 for word in words)
    # Self-duality can be checked through all pairs: 4096^2 bit operations are
    # still small, and this catches a wrong generator rather than only its weights.
    for left in words[::257]:
        for right in words[::251]:
            assert (left & right).bit_count() % 2 == 0


def test_every_shell_index_decodes_to_a_unique_verified_leech_point():
    indices = torch.arange(INDEX_COUNT, dtype=torch.int64)
    points = decode_leech_indices(indices)
    assert points.shape == (INDEX_COUNT, 24)
    assert leech_contains(points).all()
    assert points[0].abs().sum() == 0
    assert torch.all(shell_index(points[1:]) == 2)
    assert points[1:].shape[0] == SHELL2_COUNT
    # Independent duplicate check over the actual coordinate bytes.
    unique = {row.numpy().tobytes() for row in points}
    assert len(unique) == INDEX_COUNT
    assert shared_table_bytes() > 0


def test_golay_membership_rule_is_load_bearing():
    even = torch.zeros(24, dtype=torch.int64)
    even[:4] = 2  # parity and sum pass; weight-four word is not Golay.
    assert even.sum() % 8 == 0
    assert not leech_contains(even)
    odd = torch.ones(24, dtype=torch.int64)
    odd[:2] = -1
    assert odd.sum() % 8 == 4
    assert not leech_contains(odd)


def test_all_18_bit_indices_pack_roundtrip_and_bounded_ranges_match():
    indices = torch.arange(INDEX_COUNT, dtype=torch.int64)
    packed = pack_indices18(indices)
    assert packed.dtype == torch.uint8
    assert packed.numel() == (INDEX_COUNT * INDEX_BITS + 7) // 8 == 442_263
    assert torch.equal(unpack_indices18(packed, INDEX_COUNT), indices)
    for start, count in ((0, 1), (1, 3), (2, 7), (17, 100), (196_500, 61)):
        actual = unpack_indices18_range(
            packed,
            start,
            count,
            total_count=INDEX_COUNT,
        )
        assert torch.equal(actual, indices[start : start + count])
    with pytest.raises(IndexError, match="range"):
        unpack_indices18_range(packed, INDEX_COUNT, 1, total_count=INDEX_COUNT)


def test_native_every_shell_index_matches_independent_reference():
    status = native_leech_backend_status()
    if not status["available"]:
        pytest.skip(f"Native compiler/backend unavailable: {status['reason']}")
    indices = torch.arange(INDEX_COUNT, dtype=torch.int64)
    native = decode_leech_indices_native(indices)
    reference = decode_leech_indices(indices)
    assert torch.equal(native, reference)
    assert status["native_table_bytes"] > 0


def test_algebraic_shell_search_matches_full_brute_force():
    torch.manual_seed(19)
    vectors = torch.randn(5, 24, dtype=torch.float64)
    indices, dots = nearest_shell2(vectors, chunk_size=2)
    all_points = decode_leech_indices(torch.arange(1, INDEX_COUNT)).to(torch.float64)
    for row in range(vectors.shape[0]):
        scores = all_points @ vectors[row]
        expected = scores.max()
        assert dots[row] == pytest.approx(expected.item(), abs=2e-12)
        decoded = decode_leech_indices(indices[row])
        assert (decoded.to(torch.float64) @ vectors[row]) == pytest.approx(
            expected.item(), abs=2e-12
        )
        assert leech_contains(decoded)


def test_shared_householder_rotation_is_norm_preserving_and_reversible():
    torch.manual_seed(23)
    vectors = torch.randn(31, 24)
    rotated = apply_rotation(vectors, 4, 71)
    restored = apply_rotation(rotated, 4, 71, inverse=True)
    torch.testing.assert_close(restored, vectors, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        rotated.square().sum(dim=1),
        vectors.square().sum(dim=1),
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.equal(apply_rotation(vectors, 0, 71), vectors)


def test_residual_codec_is_nonexpanding_and_row_decode_is_exact():
    torch.manual_seed(29)
    source = torch.randn(12, 48) * 0.1
    encoded, report = quantize_leech_weight(source, _config(max_stages=4))
    assert encoded is not None and report["selected"]
    errors = report["stage_relative_errors"]
    assert all(right <= left + 1e-6 for left, right in zip(errors, errors[1:]))
    assert errors[-1] == pytest.approx(report["relative_error"], rel=1e-5)
    full = dequantize_leech_weight(encoded)
    for start, stop in ((0, 1), (1, 5), (5, 12), (3, 3)):
        actual = dequantize_leech_weight_rows(encoded, start, stop)
        torch.testing.assert_close(actual, full[start:stop], rtol=0, atol=0)
    expected = (
        sum(t.numel() for t in encoded.packed_indices)
        + sum(t.numel() for t in encoded.packed_gains)
        + sum(t.numel() for t in encoded.presence)
        + encoded.gain_codebooks.numel() * 4
        + encoded.row_scales.numel() * 4
    )
    assert encoded.storage_bytes == expected
    assert encoded.measured_bits_per_weight == pytest.approx(8 * expected / source.numel())


def test_shape_gain_and_rotation_expose_measured_rate_distortion_choices():
    torch.manual_seed(31)
    source = torch.randn(16, 48) * 0.1
    results = {}
    for gain_bits in (0, 1, 2):
        for reflections in (0, 2):
            encoded, report = quantize_leech_weight(
                source,
                _config(gain_bits=gain_bits, rotation_reflections=reflections),
            )
            assert encoded is not None
            results[gain_bits, reflections] = report
            assert report["measured_bits_per_weight"] > 0
            assert report["storage_bytes"] == encoded.storage_bytes
    # This fixed regression records the measured benefit of radial resolution;
    # it is not a universal ordering claim for arbitrary matrices.
    assert results[1, 2]["relative_error"] < results[0, 2]["relative_error"]
    assert results[1, 2]["storage_bytes"] > results[0, 2]["storage_bytes"]


def test_zero_error_and_expansion_fail_closed():
    zero = torch.zeros(8, 24)
    encoded, report = quantize_leech_weight(zero, _config())
    assert encoded is None
    assert report["reason"] == "no_improving_lattice_stage"

    torch.manual_seed(37)
    source = torch.randn(8, 24)
    encoded, report = quantize_leech_weight(
        source,
        _config(max_stages=1, max_relative_error=0.0),
    )
    assert encoded is None
    assert report["reason"] == "error_budget_not_met"
    encoded, report = quantize_leech_weight(
        source,
        _config(max_stages=1, require_compression=True),
    )
    if encoded is None:
        assert report["reason"] in {"not_smaller_than_original", "error_budget_not_met"}
    else:
        assert encoded.storage_bytes < source.numel() * 4


def test_native_rows_streaming_and_speed_cache_are_bitwise_exact():
    status = native_leech_backend_status()
    if not status["available"]:
        pytest.skip(f"Native compiler/backend unavailable: {status['reason']}")
    torch.manual_seed(31)
    source = torch.randn(12, 50) * 0.1
    for gain_bits in (0, 1, 2):
        encoded, report = quantize_leech_weight(
            source,
            _config(
                max_stages=3,
                gain_bits=gain_bits,
                rotation_reflections=2,
                adaptive_stages=True,
            ),
        )
        assert encoded is not None, report
        for row_start, row_stop in ((0, 12), (0, 1), (1, 7), (7, 12), (11, 12)):
            expected = dequantize_leech_weight_rows(
                encoded,
                row_start,
                row_stop,
                backend="python",
            )
            actual = dequantize_leech_weight_rows(
                encoded,
                row_start,
                row_stop,
                backend="native",
            )
            assert torch.equal(actual, expected)
        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=torch.randn(12),
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        inputs = torch.randn(3, 50)
        module.set_streaming_decode(True, row_block_size=5)
        module.set_leech_decode_backend("python")
        with torch.inference_mode():
            expected_output = module(inputs)
        module.set_leech_decode_backend("native")
        with torch.inference_mode():
            native_output = module(inputs)
        assert torch.equal(native_output, expected_output)
        assert runtime_storage_report(module)["decoded_cache_bytes"] == 0
        set_runtime_profile(module, "speed")
        module.set_leech_decode_backend("python")
        with torch.inference_mode():
            python_cached_output = module(inputs)
        module.set_leech_decode_backend("native")
        with torch.inference_mode():
            cached_output = module(inputs)
        assert torch.equal(cached_output, python_cached_output)
        # Full and chunked GEMM have their own independent rounding difference.
        torch.testing.assert_close(cached_output, expected_output)
        assert runtime_storage_report(module)["decoded_cache_bytes"] == source.numel() * 4


def test_auto_backend_falls_back_and_explicit_native_fails_closed(monkeypatch):
    from ojasx_vector_quantization_runner import leech_native

    torch.manual_seed(32)
    source = torch.randn(6, 48) * 0.1
    encoded, report = quantize_leech_weight(source, _config(max_stages=2))
    assert encoded is not None, report
    expected = dequantize_leech_weight_rows(encoded, 1, 5, backend="python")
    monkeypatch.setenv("OJASX_LEECH_NATIVE_DISABLE", "1")
    leech_native._load_library.cache_clear()
    try:
        assert not native_leech_backend_status()["available"]
        actual = dequantize_leech_weight_rows(encoded, 1, 5, backend="auto")
        assert torch.equal(actual, expected)
        with pytest.raises(RuntimeError, match="unavailable"):
            dequantize_leech_weight_rows(encoded, 1, 5, backend="native")
    finally:
        leech_native._load_library.cache_clear()


def test_leech_linear_streaming_cache_and_benchmark_are_consistent():
    torch.manual_seed(41)
    source = torch.randn(13, 48)
    encoded, _ = quantize_leech_weight(source, _config())
    assert encoded is not None
    module = ResidualVQLinear.from_encoded(
        encoded,
        bias=torch.randn(13),
        source_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    module.set_streaming_decode(True, row_block_size=5)
    inputs = torch.randn(3, 48)
    expected = F.linear(inputs, dequantize_leech_weight(encoded), module.bias)
    with torch.inference_mode():
        streamed = module(inputs)
    torch.testing.assert_close(streamed, expected)
    assert runtime_storage_report(module)["decoded_cache_bytes"] == 0
    set_runtime_profile(module, "speed")
    with torch.inference_mode():
        cached = module(inputs)
    torch.testing.assert_close(cached, streamed)
    assert runtime_storage_report(module)["decoded_cache_bytes"] == source.numel() * 4
    benchmark = benchmark_forward(module, inputs, warmup=0, iterations=1)
    assert len(benchmark["results"]) == 2


def test_leech_generalized_planner_calibrates_applies_and_serializes():
    torch.manual_seed(43)
    model = torch.nn.Sequential(
        torch.nn.Linear(48, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 48),
    ).eval()
    reference = copy.deepcopy(model)
    batches = [torch.randn(4, 48), torch.randn(4, 48)]
    dense_bytes = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.9,
            max_tensor_estimated_error=10.0,
            decode_row_block_size=7,
        ),
        calibration_inputs=batches,
        candidate_configs=default_leech_candidate_configs(_config()),
        reference_model=reference,
        quality_thresholds=QualityGateThresholds(
            max_relative_logits_l2_error=2.0,
            min_top1_agreement=0.0,
        ),
    )
    assert result.status == "passed"
    assert any(
        candidate.codec == "leech24_shell2_residual"
        for candidate in result.plan.selected_candidates.values()
    )
    actual = runtime_storage_report(model)
    assert actual["model_unique_state_tensor_bytes"] == result.plan.memory[
        "predicted_model_state_bytes"
    ]
    assert actual["model_unique_state_tensor_bytes"] < dense_bytes
    serialized = json.dumps(result.to_dict(include_alternatives=True))
    assert "leech24_shell2_residual" in serialized
    assert "gain_bits" in serialized


def test_mixed_learned_and_leech_candidates_share_the_global_planner():
    torch.manual_seed(46)
    model = torch.nn.Sequential(
        torch.nn.Linear(48, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 48),
    ).eval()
    learned = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=None,
        max_codebook_samples=256,
        kmeans_iterations=4,
        chunk_size=32,
        min_weight_elements=1,
        require_compression=True,
        seed=46,
    )
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.95,
            max_tensor_estimated_error=10.0,
            decode_row_block_size=7,
        ),
        calibration_inputs=[torch.randn(4, 48)],
        candidate_configs=[learned, _config(max_stages=2)],
    )
    assert result.status == "unvalidated"
    assert result.plan.selected_candidates
    for tensor_name, alternatives in result.plan.candidates.by_tensor.items():
        codecs = {candidate.codec for candidate in alternatives}
        assert "dense" in codecs
        assert "adaptive_residual_vq" in codecs, tensor_name
        assert "leech24_shell2_residual" in codecs, tensor_name
    assert runtime_storage_report(model)["model_unique_state_tensor_bytes"] == (
        result.plan.memory["predicted_model_state_bytes"]
    )


def test_leech_generalized_hf_meta_reload_exact(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from ojasx_vector_quantization_runner.hf import (
        load_quantized_pretrained,
        save_quantized_pretrained,
    )

    torch.manual_seed(47)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=72,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=True,
            use_cache=False,
        )
    ).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.95,
            max_tensor_estimated_error=10.0,
            decode_row_block_size=7,
        ),
        calibration_inputs=[inputs],
        candidate_configs=[_config(max_stages=3)],
        include=("*self_attn*",),
    )
    assert result.report.selected_layers
    with torch.inference_mode():
        expected = model(**inputs).logits
    output = tmp_path / "leech"
    save_quantized_pretrained(model, output, report=result.report, plan=result.plan)
    manifest = json.loads((output / "vq_manifest.json").read_text())
    assert any(
        layer.get("codec") == "leech24_shell2_residual"
        for layer in manifest["layers"].values()
    )
    loaded = load_quantized_pretrained(output, load_tokenizer=False)
    with torch.inference_mode():
        actual = loaded(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime_storage_report(loaded)["model_unique_state_tensor_bytes"] == (
        result.plan.memory["predicted_model_state_bytes"]
    )

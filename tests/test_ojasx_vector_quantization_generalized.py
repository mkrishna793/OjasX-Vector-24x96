"""Generalized tensor-IR, planner, calibration, RAM, and quality-gate tests."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from ojasx_vector_quantization_runner import (
    PlanningConstraints,
    PlanningError,
    QualityGateThresholds,
    QuantizationCandidate,
    QuantizationConfig,
    ResidualVQLinear,
    TensorLayout,
    TensorRole,
    TensorSpec,
    apply_quantization_plan,
    calibrate_candidates,
    dequantize_weight,
    dequantize_weight_rows,
    discover_tensor_specs,
    evaluate_quality_gates,
    generate_candidates,
    plan_quantization,
    quantize_model_generalized,
    quantize_weight,
)
from ojasx_vector_quantization_runner.ir import CandidateCollection
from ojasx_vector_quantization_runner.runtime import (
    benchmark_forward,
    runtime_storage_report,
    set_decode_row_block_size,
    set_runtime_profile,
)


def _config(block_size=8, stages=1, **overrides):
    values = {
        "block_size": block_size,
        "bits_per_stage": 4,
        "max_stages": stages,
        "max_relative_error": None,
        "max_codebook_samples": 64,
        "kmeans_iterations": 2,
        "chunk_size": 31,
        "min_weight_elements": 1,
        "require_compression": False,
        "seed": 71,
    }
    values.update(overrides)
    return QuantizationConfig(**values)


def _dense_model():
    torch.manual_seed(11)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 64),
    ).eval()


def _manual_spec(name):
    return TensorSpec(
        name=name,
        module_name=name,
        attribute="weight",
        role=TensorRole.OTHER,
        layout=TensorLayout.LINEAR,
        logical_shape=(50, 1),
        matrix_shape=(50, 1),
        dtype="float32",
        dtype_bytes=4,
        original_weight_bytes=200,
        decode_group=name,
    )


def _manual_candidate(spec, identifier, encoded_bytes, error):
    if identifier == "dense":
        return QuantizationCandidate.dense(spec)
    return QuantizationCandidate(
        candidate_id=identifier,
        tensor_name=spec.name,
        codec="adaptive_residual_vq",
        original_weight_bytes=200,
        encoded_weight_bytes=encoded_bytes,
        fixed_bytes=0,
        decoded_weight_bytes=200,
        weight_elements=50,
        weight_relative_error=error,
        local_output_relative_error=error,
        calibration_values=1,
        calibration_invocations=1,
        block_size=2,
        bits_per_stage=4,
        stages=1,
    )


def _manual_collection(specs):
    return CandidateCollection(
        {
            spec.name: [
                _manual_candidate(spec, "dense", 200, 0.0),
                _manual_candidate(spec, "aggressive", 50, 0.4),
                _manual_candidate(spec, "conservative", 100, 0.1),
            ]
            for spec in specs
        }
    )


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
def test_bounded_code_ranges_and_row_decode_match_full_codec(bits):
    torch.manual_seed(19)
    source = torch.randn(7, 19)
    encoded, _ = quantize_weight(source, _config(block_size=4, stages=2, bits_per_stage=bits))
    assert encoded is not None
    expected_bytes = (
        encoded.stages * ((encoded.vector_count * bits + 7) // 8)
        + encoded.codebooks.numel() * encoded.codebooks.element_size()
        + encoded.row_scales.numel() * encoded.row_scales.element_size()
    )
    assert encoded.storage_bytes == expected_bytes
    all_codes = [encoded.codes(stage) for stage in range(encoded.stages)]
    for start, count in ((0, 1), (1, 3), (3, 7), (7, 9), (encoded.vector_count - 1, 1)):
        for stage in range(encoded.stages):
            torch.testing.assert_close(
                encoded.codes_range(stage, start, count),
                all_codes[stage][start : start + count],
                rtol=0,
                atol=0,
            )
    full = dequantize_weight(encoded)
    for start, stop in ((0, 1), (1, 4), (4, 7), (3, 3)):
        actual = dequantize_weight_rows(encoded, start, stop)
        torch.testing.assert_close(actual, full[start:stop], rtol=0, atol=0)
    with pytest.raises(IndexError, match="row range"):
        dequantize_weight_rows(encoded, -1, 2)
    with pytest.raises(IndexError, match="code range"):
        encoded.codes_range(0, encoded.vector_count, 1)


def test_streaming_linear_matches_cached_output_and_has_no_memory_cache():
    source = torch.randn(13, 20)
    encoded, _ = quantize_weight(source, _config(stages=2))
    assert encoded is not None
    module = ResidualVQLinear.from_encoded(
        encoded,
        bias=torch.randn(13),
        source_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    module.set_streaming_decode(True, row_block_size=5)
    inputs = torch.randn(3, 20)
    expected = F.linear(inputs, dequantize_weight(encoded), module.bias)
    with torch.inference_mode():
        streamed = module(inputs)
    torch.testing.assert_close(streamed, expected)
    assert runtime_storage_report(module)["decoded_cache_bytes"] == 0
    set_runtime_profile(module, "speed")
    with torch.inference_mode():
        cached = module(inputs)
    torch.testing.assert_close(cached, streamed)
    assert runtime_storage_report(module)["decoded_cache_bytes"] == source.numel() * 4
    assert set_decode_row_block_size(module, 7) == 1
    benchmark = benchmark_forward(module, inputs, warmup=0, iterations=1)
    assert len(benchmark["results"]) == 2
    with pytest.raises(ValueError, match="row block"):
        set_decode_row_block_size(module, 0)


class _CustomLinear(torch.nn.Linear):
    pass


class _RoleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.q_proj = torch.nn.Linear(8, 8)
        self.shared_expert_gate = torch.nn.Linear(8, 1)
        self.visual = torch.nn.Linear(8, 8)
        self.custom = _CustomLinear(8, 8)
        self.lm_head = torch.nn.Linear(8, 16)


def test_tensor_ir_roles_source_policy_and_unknown_linear_retention():
    model = _RoleModel()
    specs = {spec.name: spec for spec in discover_tensor_specs(model)}
    assert specs["q_proj"].role == TensorRole.ATTENTION
    assert specs["q_proj"].quantizable
    assert specs["embed_tokens"].role == TensorRole.EMBEDDING
    assert not specs["embed_tokens"].quantizable
    assert specs["shared_expert_gate"].role == TensorRole.ROUTER
    assert specs["shared_expert_gate"].protection_reason == "protected_role:router"
    assert specs["visual"].role == TensorRole.MULTIMODAL
    assert not specs["visual"].quantizable
    assert specs["lm_head"].role == TensorRole.OUTPUT_HEAD
    assert specs["custom"].layout == TensorLayout.UNSUPPORTED
    assert specs["custom"].protection_reason.startswith("unsupported_linear_subclass")

    model.config = SimpleNamespace(quantization_config={"quant_method": "fp8"})
    protected = {spec.name: spec for spec in discover_tensor_specs(model)}
    assert protected["q_proj"].protection_reason == "prequantized_source:fp8"
    explicitly_allowed = {
        spec.name: spec
        for spec in discover_tensor_specs(model, allow_prequantized_source=True)
    }
    assert explicitly_allowed["q_proj"].quantizable
    assert explicitly_allowed["q_proj"].source_quantization == "fp8"


def test_candidate_generation_uses_stored_bytes_and_fails_dense_on_nonfinite():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    specs = discover_tensor_specs(model)
    candidates = generate_candidates(model, specs, [_config(4), _config(8)])
    values = candidates["0"]
    assert len(values) == 3
    assert values[0].is_dense
    for candidate in values[1:]:
        assert candidate.encoded is not None
        assert candidate.encoded_weight_bytes == candidate.encoded.storage_bytes
        assert candidate.weight_savings_bytes > 0
        assert candidate.measured_bits_per_weight > 0

    with torch.no_grad():
        model[0].weight.fill_(float("nan"))
    specs = discover_tensor_specs(model)
    rejected = generate_candidates(model, specs, [_config()])
    assert len(rejected["0"]) == 1 and rejected["0"][0].is_dense
    assert rejected.failures["0"]["rvq-0-b8-k4-s1"] == "nonfinite_weights"


def test_global_planner_optimizes_complete_plan_not_partial_budget():
    specs = [_manual_spec("a"), _manual_spec("b")]
    constraints = PlanningConstraints(
        target_state_ratio=0.5,
        max_tensor_estimated_error=1.0,
    )
    plan = plan_quantization(
        specs,
        _manual_collection(specs),
        base_model_state_bytes=400,
        constraints=constraints,
    )
    assert plan.selected == {"a": "conservative", "b": "conservative"}
    assert plan.memory["predicted_model_state_bytes"] == 200
    assert plan.quality["estimated_squared_local_error"] == pytest.approx(0.02)
    assert plan.optimizer["candidate_optimality"] == "exact_over_generated_candidates"

    tighter = plan_quantization(
        specs,
        _manual_collection(specs),
        base_model_state_bytes=400,
        constraints=PlanningConstraints(
            target_state_ratio=0.375,
            max_tensor_estimated_error=1.0,
        ),
    )
    assert tighter.memory["predicted_model_state_bytes"] == 150
    assert sorted(tighter.selected.values()) == ["aggressive", "conservative"]
    assert tighter.quality["estimated_squared_local_error"] == pytest.approx(0.17)
    with pytest.raises(PlanningError, match="no generated candidate plan"):
        plan_quantization(
            specs,
            _manual_collection(specs),
            base_model_state_bytes=400,
            constraints=PlanningConstraints(
                target_state_ratio=0.375,
                max_tensor_estimated_error=0.2,
            ),
        )


def test_frontier_truncation_is_explicitly_reported():
    specs = [_manual_spec(f"layer_{index}") for index in range(8)]
    plan = plan_quantization(
        specs,
        _manual_collection(specs),
        base_model_state_bytes=1600,
        constraints=PlanningConstraints(
            target_state_ratio=0.5,
            max_tensor_estimated_error=1.0,
            max_frontier_states=16,
        ),
    )
    assert plan.optimizer["frontier_truncated"] is True
    assert plan.optimizer["candidate_optimality"] == "bounded"
    assert plan.memory["predicted_model_state_bytes"] <= 800


def test_weight_only_planning_requires_explicit_opt_in():
    specs = [_manual_spec("a")]
    collection = CandidateCollection(
        {
            "a": [
                QuantizationCandidate.dense(specs[0]),
                QuantizationCandidate(
                    candidate_id="vq",
                    tensor_name="a",
                    codec="adaptive_residual_vq",
                    original_weight_bytes=200,
                    encoded_weight_bytes=50,
                    fixed_bytes=0,
                    decoded_weight_bytes=200,
                    weight_elements=50,
                    weight_relative_error=0.3,
                    block_size=2,
                    bits_per_stage=4,
                    stages=1,
                ),
            ]
        }
    )
    with pytest.raises(PlanningError):
        plan_quantization(
            specs,
            collection,
            base_model_state_bytes=200,
            constraints=PlanningConstraints(
                target_state_ratio=0.75,
                max_tensor_estimated_error=1.0,
            ),
        )
    plan = plan_quantization(
        specs,
        collection,
        base_model_state_bytes=200,
        constraints=PlanningConstraints(
            target_state_ratio=0.75,
            max_tensor_estimated_error=1.0,
            allow_weight_only=True,
        ),
    )
    assert plan.selected["a"] == "vq"
    assert plan.quality["weight_only_quantized_tensors"] == 1


def test_generalized_workflow_matches_ram_prediction_and_executes_quality_gates():
    model = _dense_model()
    reference = copy.deepcopy(model)
    batches = [torch.randn(4, 64), torch.randn(4, 64)]
    dense_bytes = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.8,
            max_tensor_estimated_error=10.0,
            decode_row_block_size=7,
        ),
        calibration_inputs=batches,
        candidate_configs=[_config(4), _config(8, stages=2)],
        reference_model=reference,
        quality_thresholds=QualityGateThresholds(
            max_relative_logits_l2_error=2.0,
            min_top1_agreement=0.0,
        ),
    )
    assert result.status == "passed"
    assert result.require_accepted() is result
    assert result.report.selected_layers >= 1
    assert result.calibration["captured_modules"] == 2
    assert not result.calibration["uncalibrated_tensors"]
    actual = runtime_storage_report(model)
    assert actual["model_unique_state_tensor_bytes"] == result.plan.memory[
        "predicted_model_state_bytes"
    ]
    assert actual["model_unique_state_tensor_bytes"] <= 0.8 * dense_bytes
    assert result.plan.memory["actual_matches_prediction"]
    assert 0 < result.plan.memory["max_decode_group_output_bytes_estimate"] <= 7 * 64 * 4
    assert result.plan.quality["end_to_end_gates"]["passed"]
    assert result.plan.coverage_report()["by_layout"]["linear"]["tensors"] == 2
    for module in model.modules():
        if isinstance(module, ResidualVQLinear):
            assert module.decode_row_block_size == 7
    serialized = json.dumps(result.to_dict(include_alternatives=True))
    assert "adaptive_residual_vq" in serialized


def test_generalized_api_requires_budgets_calibration_and_real_reference():
    model = _dense_model()
    with pytest.raises(ValueError, match="persistent-state or cache budget"):
        quantize_model_generalized(model, constraints=PlanningConstraints())
    with pytest.raises(ValueError, match="quality budget"):
        quantize_model_generalized(
            model,
            constraints=PlanningConstraints(target_state_ratio=0.8),
            calibration_inputs=[torch.randn(2, 64)],
        )
    with pytest.raises(ValueError, match="Calibration inputs"):
        quantize_model_generalized(
            model,
            constraints=PlanningConstraints(
                target_state_ratio=0.8,
                max_tensor_estimated_error=1.0,
            ),
        )
    with pytest.raises(ValueError, match="separate unmodified reference"):
        quantize_model_generalized(
            model,
            constraints=PlanningConstraints(
                target_state_ratio=0.8,
                max_tensor_estimated_error=1.0,
            ),
            calibration_inputs=[torch.randn(2, 64)],
            reference_model=model,
            quality_thresholds=QualityGateThresholds(max_mean_kl=1.0),
        )
    with pytest.raises(ValueError, match="At least one"):
        QualityGateThresholds()


class _BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(64, 32)
        self.unused = torch.nn.Linear(64, 32)

    def forward(self, inputs):
        return self.used(inputs)


def test_uncalled_layer_is_uncalibrated_and_fails_dense():
    torch.manual_seed(23)
    model = _BranchModel().eval()
    specs = discover_tensor_specs(model)
    candidates = generate_candidates(model, specs, [_config()])
    summary = calibrate_candidates(model, specs, candidates, [torch.randn(3, 64)])
    assert summary["uncalibrated_tensors"] == ["unused"]
    assert all(
        candidate.local_output_relative_error is None
        for candidate in candidates["unused"]
        if not candidate.is_dense
    )
    plan = plan_quantization(
        specs,
        candidates,
        base_model_state_bytes=runtime_storage_report(model)["model_unique_state_tensor_bytes"],
        constraints=PlanningConstraints(
            target_state_ratio=0.9,
            max_tensor_estimated_error=10.0,
        ),
    )
    assert plan.selected["used"] != "dense"
    assert plan.selected["unused"] == "dense"
    model, _ = apply_quantization_plan(model, plan)
    assert isinstance(model.used, ResidualVQLinear)
    assert type(model.unused) is torch.nn.Linear


def test_speed_mode_cache_is_explicit_and_can_exceed_dense_state():
    model = _dense_model()
    dense_bytes = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    inputs = torch.randn(4, 64)
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.8,
            max_tensor_estimated_error=10.0,
            runtime_profile="speed",
        ),
        calibration_inputs=[inputs],
        candidate_configs=[_config(4), _config(8, stages=2)],
    )
    with torch.inference_mode():
        model(inputs)
    actual = runtime_storage_report(model)
    assert actual["decoded_cache_bytes"] > 0
    assert actual["model_state_plus_decoded_cache_bytes"] == result.plan.memory[
        "predicted_model_state_plus_cache_bytes"
    ]
    # With a same-dtype full decoded cache, each VQ weight adds its encoded
    # payload on top of a reconstructed dense-sized cache. This is not low-RAM mode.
    assert actual["model_state_plus_decoded_cache_bytes"] > dense_bytes
    set_runtime_profile(model, "memory")
    assert runtime_storage_report(model)["decoded_cache_bytes"] == 0


def test_fused_expert_partial_plan_calibrates_applies_and_runs():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(3)
    model = Qwen2MoeForCausalLM(
        Qwen2MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            tie_word_embeddings=True,
        )
    ).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.8,
            max_tensor_estimated_error=10.0,
        ),
        calibration_inputs=[inputs],
        candidate_configs=[_config(4), _config(8)],
        include=("*experts*",),
        max_activation_rows=16,
    )
    selected_experts = [
        name for name in result.plan.selected if ".mlp.experts." in name
    ]
    assert len(selected_experts) == 2
    assert any(result.plan.selected[name] != "dense" for name in selected_experts)
    assert any(result.plan.selected[name] == "dense" for name in selected_experts)
    assert result.plan.memory["actual_matches_prediction"]
    # At most one active expert's gate/up and down matrices, not the full bank.
    assert result.plan.memory["max_decode_group_output_bytes_estimate"] <= (
        32 * 32 + 32 * 16
    ) * 4
    with torch.inference_mode():
        logits = model(**inputs).logits
    assert logits.shape == (1, 4, 64)
    assert torch.isfinite(logits).all()


class _ToyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def test_causal_quality_gates_nll_failure_and_nonfinite_rejection():
    torch.manual_seed(31)
    reference = _ToyCausal().eval()
    candidate = copy.deepcopy(reference)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.int64),
    }
    thresholds = QualityGateThresholds(
        max_relative_logits_l2_error=0.0,
        max_absolute_logits_error=0.0,
        max_mean_kl=0.0,
        min_top1_agreement=1.0,
        max_mean_nll_delta=0.0,
    )
    identical = evaluate_quality_gates(reference, candidate, [batch], thresholds)
    assert identical.passed
    assert identical.metrics["supervised_tokens"] == 6
    assert identical.metrics["mean_nll_delta"] == 0.0

    with torch.no_grad():
        candidate.proj.weight.mul_(1.2)
    degraded = evaluate_quality_gates(reference, candidate, [batch], thresholds)
    assert not degraded.passed
    assert degraded.metrics["mean_nll_delta"] != 0.0
    assert degraded.failures

    with torch.no_grad():
        candidate.proj.weight.fill_(float("nan"))
    nonfinite = evaluate_quality_gates(reference, candidate, [batch], thresholds)
    assert not nonfinite.passed
    assert nonfinite.metrics["finite_candidate_logits"] is False


def test_nll_gate_requires_labels_and_unvalidated_result_cannot_self_certify():
    reference = _dense_model()
    candidate = copy.deepcopy(reference)
    result = evaluate_quality_gates(
        reference,
        candidate,
        [torch.randn(2, 64)],
        QualityGateThresholds(max_mean_nll_delta=0.1),
    )
    assert not result.passed
    assert "no supervised tokens" in result.failures[0]

    model = _dense_model()
    generalized = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.8,
            max_tensor_estimated_error=10.0,
        ),
        calibration_inputs=[torch.randn(2, 64)],
        candidate_configs=[_config()],
    )
    assert generalized.status == "unvalidated"
    with pytest.raises(RuntimeError, match="unvalidated"):
        generalized.require_accepted()


def test_generalized_plan_hf_persistence_and_meta_reload(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from ojasx_vector_quantization_runner.hf import (
        PLAN_NAME,
        load_quantized_pretrained,
        save_quantized_pretrained,
    )

    torch.manual_seed(41)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
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
            target_state_ratio=0.9,
            max_tensor_estimated_error=10.0,
            decode_row_block_size=7,
        ),
        calibration_inputs=[inputs],
        candidate_configs=[_config(4), _config(8)],
        include=("*self_attn*",),
    )
    assert result.report.selected_layers
    with torch.inference_mode():
        expected = model(**inputs).logits
    output = tmp_path / "encoded"
    save_quantized_pretrained(
        model,
        output,
        report=result.report,
        plan=result.plan,
    )
    saved_plan = json.loads((output / PLAN_NAME).read_text())
    assert saved_plan["format"] == "ojasx-generalized-vq-plan"
    assert saved_plan["memory"]["actual_matches_prediction"] is True
    assert saved_plan["quality"]["calibration"]["captured_modules"] > 0
    loaded = load_quantized_pretrained(output, load_tokenizer=False)
    assert all(
        module.decode_row_block_size == 7
        for module in loaded.modules()
        if isinstance(module, ResidualVQLinear)
    )
    with torch.inference_mode():
        actual = loaded(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    result.plan.memory["predicted_model_state_bytes"] += 1
    stale_output = tmp_path / "stale"
    with pytest.raises(ValueError, match="stale or unrelated"):
        save_quantized_pretrained(
            model,
            stale_output,
            report=result.report,
            plan=result.plan,
        )
    assert not stale_output.exists()


def test_outer_inference_mode_cannot_create_unsafe_versionless_cache():
    model = _dense_model()
    inputs = torch.randn(2, 64)
    with torch.inference_mode():
        result = quantize_model_generalized(
            model,
            constraints=PlanningConstraints(
                target_state_ratio=0.8,
                max_tensor_estimated_error=10.0,
                runtime_profile="speed",
            ),
            calibration_inputs=[inputs],
            candidate_configs=[_config()],
        )
        output = model(inputs)
        assert torch.isfinite(output).all()
        # Inference tensors have no mutation version. The runtime must not reuse
        # an unverifiable cache merely because speed mode was requested.
        assert runtime_storage_report(model)["decoded_cache_bytes"] == 0
    assert result.plan.memory["actual_matches_prediction"]

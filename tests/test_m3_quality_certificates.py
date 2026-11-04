"""Unit and integration tests for M3 Quality Certificates and KV-aware Pareto planning."""

import copy
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ojasx_24x96.adapters import discover_tensor_specs
from ojasx_24x96.calibration import compute_layer_sensitivity_profile
from ojasx_24x96.certificates import (
    LayerSensitivityEntry,
    QualityCertificate,
    QualityCertificateThresholds,
    generate_quality_certificate,
)
from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.contract import ArchitectureProfile, introspect_architecture
from ojasx_24x96.generalized import quantize_model_generalized
from ojasx_24x96.ir import CandidateCollection, TensorLayout, TensorRole, TensorSpec
from ojasx_24x96.planner import (
    PlanningConstraints,
    PlanningError,
    QuantizationCandidate,
    generate_candidates,
    plan_quantization,
)


def _tiny_llama_model():
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_cache=False,
    )
    return LlamaForCausalLM(cfg).eval()


def _mock_spec(name: str, elements: int = 128) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_name=name,
        attribute="weight",
        role=TensorRole.ATTENTION,
        layout=TensorLayout.LINEAR,
        logical_shape=(elements // 16, 16),
        matrix_shape=(elements // 16, 16),
        dtype="float32",
        dtype_bytes=4,
        original_weight_bytes=elements * 4,
        decode_group=name,
    )


def _mock_candidate(spec: TensorSpec, cid: str, bytes_saved: int, error: float) -> QuantizationCandidate:
    if cid == "dense":
        return QuantizationCandidate.dense(spec)
    return QuantizationCandidate(
        candidate_id=cid,
        tensor_name=spec.name,
        codec="adaptive_residual_vq",
        original_weight_bytes=spec.original_weight_bytes,
        encoded_weight_bytes=spec.original_weight_bytes - bytes_saved,
        fixed_bytes=0,
        decoded_weight_bytes=spec.original_weight_bytes,
        weight_elements=spec.weight_elements,
        weight_relative_error=error,
        local_output_relative_error=error,
        calibration_values=10,
        calibration_invocations=1,
        block_size=8,
        bits_per_stage=4,
        stages=1,
    )


# -----------------------------------------------------------------------------
# 1. KV-Aware Planning Constraints Tests
# -----------------------------------------------------------------------------

def test_planning_constraints_kv_fields_and_validation():
    # Default values
    constraints = PlanningConstraints()
    assert constraints.context_length == 2048
    assert constraints.batch_size == 1
    assert constraints.kv_dtype_bytes == 2
    assert constraints.max_total_runtime_bytes is None
    assert constraints.architecture_profile is None
    assert not constraints.has_memory_budget

    # Custom valid values
    custom = PlanningConstraints(
        max_total_runtime_bytes=100_000_000,
        context_length=4096,
        batch_size=2,
        kv_dtype_bytes=4,
    )
    assert custom.has_memory_budget
    d = custom.to_dict()
    assert d["max_total_runtime_bytes"] == 100_000_000
    assert d["context_length"] == 4096
    assert d["batch_size"] == 2
    assert d["kv_dtype_bytes"] == 4

    # Validation errors
    with pytest.raises(ValueError, match="Memory constraints must be nonnegative"):
        PlanningConstraints(max_total_runtime_bytes=-10)

    with pytest.raises(ValueError, match="context_length must be positive"):
        PlanningConstraints(context_length=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        PlanningConstraints(batch_size=-1)

    with pytest.raises(ValueError, match="kv_dtype_bytes must be positive"):
        PlanningConstraints(kv_dtype_bytes=0)


def test_plan_quantization_impossible_runtime_bytes_raises_planning_error():
    model = _tiny_llama_model()
    arch = introspect_architecture(model)

    specs = [_mock_spec("layer_0"), _mock_spec("layer_1")]
    candidates = CandidateCollection({
        s.name: [
            _mock_candidate(s, "dense", 0, 0.0),
            _mock_candidate(s, "cand_vq", 256, 0.1),
        ]
        for s in specs
    })

    # Total runtime overhead (workspace + overhead + shared tables) is > 16 MB.
    # An impossible limit of 1000 bytes must raise PlanningError immediately.
    constraints = PlanningConstraints(
        max_total_runtime_bytes=1000,
        architecture_profile=arch,
        context_length=128,
    )

    with pytest.raises(PlanningError, match="Non-weight runtime footprint alone"):
        plan_quantization(
            specs,
            candidates,
            base_model_state_bytes=1024,
            constraints=constraints,
        )


def test_plan_quantization_context_length_tightens_weight_budget():
    model = _tiny_llama_model()
    arch = introspect_architecture(model)

    specs = [_mock_spec("attn_proj", elements=512)]
    candidates = CandidateCollection({
        specs[0].name: [
            _mock_candidate(specs[0], "dense", 0, 0.0),
            _mock_candidate(specs[0], "cand_compressed", 1024, 0.05),
        ]
    })
    base_bytes = specs[0].original_weight_bytes  # 2048 bytes

    # Compute overhead with short context
    short_ctx = 64
    long_ctx = 32768
    from ojasx_24x96.contract import evaluate_memory_admission

    adm_short = evaluate_memory_admission(
        arch,
        encoded_weight_bytes=0,
        dense_weight_bytes=0,
        context_length=short_ctx,
    )
    adm_long = evaluate_memory_admission(
        arch,
        encoded_weight_bytes=0,
        dense_weight_bytes=0,
        context_length=long_ctx,
    )
    assert adm_long.kv_cache_bytes > adm_short.kv_cache_bytes
    assert adm_long.total_required_bytes > adm_short.total_required_bytes

    # Set budget enough for short_ctx + compressed weights, but too small for long_ctx
    fixed_budget = adm_short.total_required_bytes + 1500

    plan_short = plan_quantization(
        specs,
        candidates,
        base_model_state_bytes=base_bytes,
        constraints=PlanningConstraints(
            max_total_runtime_bytes=fixed_budget,
            architecture_profile=arch,
            context_length=short_ctx,
        ),
    )
    assert plan_short.selected["attn_proj"] == "cand_compressed"
    assert "predicted_total_runtime_bytes" in plan_short.memory
    assert plan_short.memory["predicted_total_runtime_bytes"] <= fixed_budget

    # With long_ctx, the KV cache grows so large that total overhead exceeds fixed_budget
    with pytest.raises(PlanningError):
        plan_quantization(
            specs,
            candidates,
            base_model_state_bytes=base_bytes,
            constraints=PlanningConstraints(
                max_total_runtime_bytes=fixed_budget,
                architecture_profile=arch,
                context_length=long_ctx,
            ),
        )


# -----------------------------------------------------------------------------
# 2. Layer Sensitivity Profiling Tests
# -----------------------------------------------------------------------------

def test_compute_layer_sensitivity_profile():
    model = _tiny_llama_model()
    specs = discover_tensor_specs(model, include=('*self_attn*', '*mlp*'))
    c_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=1,
        max_relative_error=None,
        max_codebook_samples=32,
        kmeans_iterations=2,
        require_compression=False,
        min_weight_elements=1,
    )
    candidates = generate_candidates(model, specs, [c_cfg])
    inputs = [{"input_ids": torch.tensor([[1, 4, 15, 23, 2]])}]

    profile = compute_layer_sensitivity_profile(
        specs, candidates.by_tensor, model=model, calibration_inputs=inputs
    )
    assert len(profile) == len(specs)

    for name, metrics in profile.items():
        assert "hessian_trace_proxy" in metrics
        assert "activation_amplification" in metrics
        assert "min_candidate_distortion" in metrics
        assert "max_candidate_distortion" in metrics
        assert "recommended_min_bits" in metrics
        assert metrics["hessian_trace_proxy"] > 0
        assert metrics["activation_amplification"] > 0
        assert metrics["recommended_min_bits"] >= 2.0

    # Verify sensitive layers (o_proj and down_proj) receive high sensitivity ranking
    o_proj_amp = profile["model.layers.0.self_attn.o_proj"]["activation_amplification"]
    q_proj_amp = profile["model.layers.0.self_attn.q_proj"]["activation_amplification"]
    k_proj_amp = profile["model.layers.0.self_attn.k_proj"]["activation_amplification"]
    assert o_proj_amp > q_proj_amp
    assert o_proj_amp > k_proj_amp

    down_proj_amp = profile["model.layers.0.mlp.down_proj"]["activation_amplification"]
    up_proj_amp = profile["model.layers.0.mlp.up_proj"]["activation_amplification"]
    gate_proj_amp = profile["model.layers.0.mlp.gate_proj"]["activation_amplification"]
    assert down_proj_amp > up_proj_amp
    assert down_proj_amp > gate_proj_amp


# -----------------------------------------------------------------------------
# 3. Quality Certificates Tests
# -----------------------------------------------------------------------------

def test_generate_quality_certificate_passed_and_failed():
    base_model = _tiny_llama_model()
    quant_model = copy.deepcopy(base_model)
    arch = introspect_architecture(base_model)
    inputs = [{"input_ids": torch.tensor([[1, 5, 12, 19, 2], [3, 7, 14, 21, 4]])}]

    c_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=1,
        max_relative_error=None,
        max_codebook_samples=32,
        kmeans_iterations=2,
        require_compression=False,
        min_weight_elements=1,
    )

    res = quantize_model_generalized(
        quant_model,
        constraints=PlanningConstraints(
            target_state_ratio=0.9,
            max_tensor_estimated_error=10.0,
            architecture_profile=arch,
        ),
        calibration_inputs=inputs,
        candidate_configs=[c_cfg],
        include=('*self_attn*',),
    )

    # 1. Certificate passes with reasonable / relaxed thresholds
    cert_pass = generate_quality_certificate(
        base_model,
        quant_model,
        inputs,
        plan=res.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=5.0,
            max_kl_divergence=0.5,
            min_top1_agreement=0.5,
            min_top5_agreement=0.7,
        ),
    )
    assert cert_pass.passed is True
    assert len(cert_pass.failed_checks) == 0
    assert cert_pass.certificate_version == "m3.certificate.v1"
    assert cert_pass.baseline_perplexity > 0
    assert cert_pass.quantized_perplexity > 0
    assert cert_pass.compression_ratio > 1.0
    assert cert_pass.top1_agreement >= 0.5
    assert cert_pass.top5_agreement >= 0.7
    assert len(cert_pass.layer_sensitivities) > 0

    # Serialization test
    d = cert_pass.to_dict()
    assert d["passed"] is True
    assert d["certificate_version"] == "m3.certificate.v1"
    assert isinstance(d["layer_sensitivities"], list)

    # 2. Certificate fails with impossibly strict thresholds
    cert_fail = generate_quality_certificate(
        base_model,
        quant_model,
        inputs,
        plan=res.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=1e-8,
            max_kl_divergence=1e-8,
            min_top1_agreement=0.999999,
        ),
    )
    assert cert_fail.passed is False
    assert len(cert_fail.failed_checks) > 0
    assert any("max_kl_divergence" in f or "min_top1_agreement" in f for f in cert_fail.failed_checks)


def test_quality_certificate_with_long_context_and_drift():
    base_model = _tiny_llama_model()
    quant_model = copy.deepcopy(base_model)
    arch = introspect_architecture(base_model)
    inputs = [{"input_ids": torch.tensor([[1, 5, 12, 19, 2]])}]
    long_inputs = [{"input_ids": torch.randint(0, 64, (1, 32), dtype=torch.long)}]

    c_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=1,
        max_relative_error=None,
        max_codebook_samples=32,
        kmeans_iterations=2,
        require_compression=False,
        min_weight_elements=1,
    )

    res = quantize_model_generalized(
        quant_model,
        constraints=PlanningConstraints(
            target_state_ratio=0.9,
            max_tensor_estimated_error=10.0,
            architecture_profile=arch,
        ),
        calibration_inputs=inputs,
        candidate_configs=[c_cfg],
        include=('*self_attn*',),
    )

    cert = generate_quality_certificate(
        base_model,
        quant_model,
        inputs,
        long_context_batches=long_inputs,
        plan=res.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=5.0,
            max_kl_divergence=1.0,
            min_top1_agreement=0.1,
            min_top5_agreement=0.2,
            max_long_context_kl_drift=5.0,
        ),
    )
    assert cert.long_context_kl_divergence is not None
    assert cert.long_context_length == 32
    assert cert.passed is True

    # When drift threshold is zero, it must fail the long-context drift check
    cert_drift_fail = generate_quality_certificate(
        base_model,
        quant_model,
        inputs,
        long_context_batches=long_inputs,
        plan=res.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=100.0,
            max_kl_divergence=100.0,
            min_top1_agreement=0.0,
            min_top5_agreement=0.0,
            max_long_context_kl_drift=0.0,
        ),
    )
    assert cert_drift_fail.passed is False
    assert any("max_long_context_kl_drift" in f for f in cert_drift_fail.failed_checks)


def test_quality_certificate_nonfinite_handling():
    base_model = _tiny_llama_model()
    quant_model = copy.deepcopy(base_model)
    inputs = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}]

    # Intentionally corrupt weight to NaN
    with torch.no_grad():
        quant_model.model.layers[0].self_attn.q_proj.weight.fill_(float("nan"))

    cert = generate_quality_certificate(
        base_model,
        quant_model,
        inputs,
        thresholds=QualityCertificateThresholds(require_all_finite=True),
    )
    assert cert.passed is False
    assert any("require_all_finite" in f for f in cert.failed_checks)

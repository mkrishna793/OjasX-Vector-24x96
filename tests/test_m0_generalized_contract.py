"""Milestone M0 Verification Test Suite:
Generalized IR, Custom Module Adapter Registry & Total Memory Admission.

Verifies:
1. Architecture introspection & memory admission across modern HF architectures:
   - Qwen3_5TextConfig / Qwen3_5ForCausalLM (Qwen3.8 hybrid Gated-DeltaNet + full attention)
   - Gemma4TextConfig / Gemma4ForCausalLM (Gemma 4 architecture)
   - GlmMoeDsaConfig / GlmMoeDsaForCausalLM (GLM-5.3 MLA + MoE architecture)
   - DeepseekV4Config / DeepseekV4ForCausalLM (DeepSeek-V4 MLA architecture)
2. Total memory admission 7-bucket accounting:
   (encoded_weight_bytes + dense_weight_bytes + codebooks_and_scales_bytes +
    kv_cache_bytes + activation_bytes + workspace_bytes + runtime_overhead_bytes)
   - AdmissionError when exceeding budget
   - max_admissible_context_length monotonic binary search
3. Generalized Custom Module Adapter Registry & Duck-Typed Linear Projections:
   - register_custom_module_adapter / unregister_custom_module_adapter / list_registered_custom_adapters
   - Custom module adapter discovery, quantization, and forward execution
   - Duck-typed linear discovery (module_type="custom_linear"), quantization, and forward execution
   - Preservation of embeddings, norms, and unregistered nn.Linear subclasses
4. QuantizationPlan metadata extension:
   - architecture_profile and memory_admission fields preserved and serialized in to_dict()
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pytest
import torch
import torch.nn as nn
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from ojasx_24x96.adapters import (
    apply_quantization_plan,
    discover_tensor_specs,
    list_registered_custom_adapters,
    register_custom_module_adapter,
    unregister_custom_module_adapter,
)
from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.contract import (
    AdmissionError,
    ArchitectureProfile,
    LayerKVSpec,
    MemoryAdmissionReport,
    evaluate_memory_admission,
    introspect_architecture,
    max_admissible_context_length,
)
from ojasx_24x96.generalized import PlanningConstraints, quantize_model_generalized
from ojasx_24x96.ir import (
    CandidateCollection,
    QuantizationCandidate,
    QuantizationPlan,
    TensorLayout,
    TensorRole,
    TensorSpec,
)
from ojasx_24x96.module import ResidualVQLinear


def _quick_codec_config(block_size: int = 8, stages: int = 1) -> QuantizationConfig:
    return QuantizationConfig(
        block_size=block_size,
        bits_per_stage=4,
        max_stages=stages,
        max_relative_error=None,
        kmeans_iterations=2,
        max_codebook_samples=64,
        min_weight_elements=1,
        require_compression=False,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Group 1: Modern HF Architecture Introspection & Memory Admission
# ---------------------------------------------------------------------------


def test_qwen3_5_hybrid_architecture_contract():
    """Verify Qwen3.5/3.8 hybrid Gated-DeltaNet + full attention introspection and memory."""
    cfg = Qwen3_5TextConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    profile = introspect_architecture(cfg)

    assert profile.model_type == "qwen3_5_text"
    assert profile.num_hidden_layers == 4
    assert profile.hidden_size == 64
    assert profile.is_hybrid_attention is True
    assert profile.is_moe is False

    kinds = {spec.attention_kind for spec in profile.layer_kv_specs}
    assert "linear_recurrent" in kinds
    assert "full_attention" in kinds

    # Linear recurrent layers have fixed state size; full attention scales with context length
    recurrent_spec = next(s for s in profile.layer_kv_specs if s.attention_kind == "linear_recurrent")
    full_spec = next(s for s in profile.layer_kv_specs if s.attention_kind == "full_attention")
    b1_ctx1k = recurrent_spec.kv_bytes_for_context(1, 1024)
    b1_ctx2k = recurrent_spec.kv_bytes_for_context(1, 2048)
    assert b1_ctx1k == b1_ctx2k  # O(1) recurrent memory invariant

    f1_ctx1k = full_spec.kv_bytes_for_context(1, 1024)
    f1_ctx2k = full_spec.kv_bytes_for_context(1, 2048)
    assert f1_ctx2k == 2 * f1_ctx1k  # O(S) full attention scaling

    # Model instance introspection
    model = Qwen3_5ForCausalLM(cfg)
    model_profile = introspect_architecture(model)
    assert model_profile.model_type == profile.model_type
    assert model_profile.num_hidden_layers == profile.num_hidden_layers

    # Admission check
    report = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=50_000,
        dense_weight_bytes=20_000,
        context_length=2048,
        budget_bytes=50_000_000,
    )
    assert report.admitted is True
    assert report.kv_cache_bytes > 0


def test_gemma4_architecture_contract():
    """Verify Gemma 4 architecture introspection and sliding window layer specs."""
    cfg = Gemma4TextConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    if hasattr(cfg, "allow_global_per_layer_attribute_access"):
        cfg.allow_global_per_layer_attribute_access = True

    profile = introspect_architecture(cfg)
    assert profile.model_type == "gemma4_text"
    assert profile.num_hidden_layers == 4
    assert profile.head_dim == 16

    kinds = {spec.attention_kind for spec in profile.layer_kv_specs}
    assert "sliding_attention" in kinds or "full_attention" in kinds

    model = Gemma4ForCausalLM(cfg)
    model_profile = introspect_architecture(model)
    assert model_profile.model_type == profile.model_type

    report = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=40_000,
        dense_weight_bytes=10_000,
        context_length=1024,
        budget_bytes=40_000_000,
    )
    assert report.admitted is True


def test_glm_moe_dsa_architecture_contract():
    """Verify GLM-5.3 MLA + MoE architecture introspection."""
    cfg = GlmMoeDsaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
    )
    profile = introspect_architecture(cfg)
    assert profile.model_type == "glm_moe_dsa"
    assert profile.is_moe is True
    assert profile.num_experts == 4
    assert any(s.attention_kind == "mla_attention" for s in profile.layer_kv_specs)

    model = GlmMoeDsaForCausalLM(cfg)
    model_profile = introspect_architecture(model)
    assert model_profile.model_type == "glm_moe_dsa"
    assert model_profile.is_moe is True

    report = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=80_000,
        dense_weight_bytes=30_000,
        context_length=512,
        budget_bytes=40_000_000,
    )
    assert report.admitted is True


def test_deepseek_v4_architecture_contract():
    """Verify DeepSeek-V4 MLA architecture introspection."""
    cfg = DeepseekV4Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
    )
    profile = introspect_architecture(cfg)
    assert profile.model_type == "deepseek_v4"
    assert profile.is_moe is True
    assert profile.num_experts == 4

    model = DeepseekV4ForCausalLM(cfg)
    model_profile = introspect_architecture(model)
    assert model_profile.model_type == "deepseek_v4"

    report = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=60_000,
        dense_weight_bytes=25_000,
        context_length=1024,
        budget_bytes=30_000_000,
    )
    assert report.admitted is True


# ---------------------------------------------------------------------------
# Group 2: Total Memory Admission 7-Bucket Accounting, AdmissionError & Search
# ---------------------------------------------------------------------------


def test_evaluate_memory_admission_seven_bucket_accounting():
    """Verify exact 7-bucket memory accounting and AdmissionError on budget violations."""
    profile = ArchitectureProfile(
        model_type="custom_test",
        architecture_class="CustomTestModel",
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=1,
        is_hybrid_attention=False,
        is_multimodal_composite=False,
        layer_kv_specs=(
            LayerKVSpec(layer_index=0, attention_kind="full_attention", num_kv_heads=2, head_dim=32),
            LayerKVSpec(layer_index=1, attention_kind="full_attention", num_kv_heads=2, head_dim=32),
        ),
    )

    encoded = 2_000_000
    dense = 1_000_000
    codebooks = 50_000
    overhead = 16 * 1024 * 1024

    report = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        codebooks_and_scales_bytes=codebooks,
        batch_size=2,
        context_length=1024,
        runtime_overhead_bytes=overhead,
        budget_bytes=50_000_000,
    )

    # 1. Encoded weights
    assert report.encoded_weight_bytes == encoded
    # 2. Dense weights
    assert report.dense_weight_bytes == dense
    # 3. Codebooks & scales (+ shared tables)
    assert report.codebooks_and_scales_bytes == codebooks
    assert report.shared_table_bytes > 0
    # 4. KV cache
    assert report.kv_cache_bytes > 0
    # 5. Activations (prefill & decode)
    assert report.activation_prefill_bytes > 0
    assert report.activation_decode_bytes > 0
    # 6. Workspace bytes
    assert report.workspace_bytes > 0
    # 7. Runtime overhead
    assert report.runtime_overhead_bytes == overhead

    # Verify peak formula consistency
    assert report.persistent_model_bytes == (
        encoded + dense + codebooks + report.shared_table_bytes + report.decoded_weight_cache_bytes
    )
    assert report.total_prefill_peak_bytes == (
        report.persistent_model_bytes
        + report.kv_cache_bytes
        + report.activation_prefill_bytes
        + report.workspace_bytes
        + report.runtime_overhead_bytes
    )
    assert report.total_decode_peak_bytes == (
        report.persistent_model_bytes
        + report.kv_cache_bytes
        + report.activation_decode_bytes
        + report.workspace_bytes
        + report.runtime_overhead_bytes
    )
    assert report.total_required_bytes == max(
        report.total_prefill_peak_bytes, report.total_decode_peak_bytes
    )
    assert report.admitted is True

    # Test serialization
    as_dict = report.to_dict()
    assert as_dict["admitted"] is True
    assert as_dict["total_required_bytes"] == report.total_required_bytes

    # Budget violation triggers admission failure and AdmissionError
    tight_budget = report.total_required_bytes - 1
    rejected = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        codebooks_and_scales_bytes=codebooks,
        batch_size=2,
        context_length=1024,
        runtime_overhead_bytes=overhead,
        budget_bytes=tight_budget,
    )
    assert rejected.admitted is False
    assert any("budget_bytes" in reason for reason in rejected.rejection_reasons)

    with pytest.raises(AdmissionError) as exc_info:
        raise AdmissionError(rejected)
    assert exc_info.value.report is rejected
    assert "budget_bytes" in str(exc_info.value)


def test_max_admissible_context_length_monotonicity():
    """Verify max_admissible_context_length binary search is monotonic and exact."""
    profile = ArchitectureProfile(
        model_type="custom_search",
        architecture_class="SearchModel",
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=1,
        is_hybrid_attention=False,
        is_multimodal_composite=False,
        layer_kv_specs=tuple(
            LayerKVSpec(layer_index=i, attention_kind="full_attention", num_kv_heads=2, head_dim=16)
            for i in range(4)
        ),
    )

    encoded = 1_000_000
    dense = 500_000
    # Determine the required bytes for context=1024
    rep_1k = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        batch_size=1,
        context_length=1024,
        budget_bytes=None,
    )
    target_budget = rep_1k.total_required_bytes

    best_ctx = max_admissible_context_length(
        profile,
        budget_bytes=target_budget,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        batch_size=1,
        upper_bound=4096,
    )

    # best_ctx must admit target_budget
    rep_best = evaluate_memory_admission(
        profile,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        batch_size=1,
        context_length=best_ctx,
        budget_bytes=target_budget,
    )
    assert rep_best.admitted is True

    # best_ctx + 1 must exceed target_budget
    if best_ctx < 4096:
        rep_over = evaluate_memory_admission(
            profile,
            encoded_weight_bytes=encoded,
            dense_weight_bytes=dense,
            batch_size=1,
            context_length=best_ctx + 1,
            budget_bytes=target_budget,
        )
        assert rep_over.admitted is False

    # Budget smaller than persistent model state admits 0 context length
    impossible_budget = 100
    best_zero = max_admissible_context_length(
        profile,
        budget_bytes=impossible_budget,
        encoded_weight_bytes=encoded,
        dense_weight_bytes=dense,
        batch_size=1,
    )
    assert best_zero == 0


# ---------------------------------------------------------------------------
# Group 3: Custom Module Adapter Registry & Duck-Typed Linear Projections
# ---------------------------------------------------------------------------


class _CustomRegisteredLinear(nn.Module):
    """Custom projection layer with nonstandard attribute names."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.proj_weight, self.bias)


class _DuckTypedLinear(nn.Module):
    """User-defined projection class with 2D weight and 1D bias, no submodules."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class _ContainerModel(nn.Module):
    def __init__(self, custom_layer: nn.Module):
        super().__init__()
        self.layer = custom_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_custom_module_adapter_registry_lifecycle():
    """Verify registration, listing, type enforcement, and unregistration."""
    assert _CustomRegisteredLinear.__name__ not in list_registered_custom_adapters()

    with pytest.raises(TypeError, match="subclass of nn.Module"):
        register_custom_module_adapter(int, lambda name, m: None)

    def extractor(name: str, module: nn.Module):
        return module.proj_weight, module.bias, TensorLayout.LINEAR, "proj_weight"

    register_custom_module_adapter(_CustomRegisteredLinear, extractor)
    assert _CustomRegisteredLinear.__name__ in list_registered_custom_adapters()

    unregister_custom_module_adapter(_CustomRegisteredLinear)
    assert _CustomRegisteredLinear.__name__ not in list_registered_custom_adapters()


def test_custom_module_adapter_discovery_quantization_and_forward():
    """Verify custom adapter registration, generalized quantization, and forward execution."""
    replacer_called = []

    def extractor(name: str, module: nn.Module):
        return module.proj_weight, module.bias, TensorLayout.LINEAR, "proj_weight"

    def replacer(module: nn.Module, candidate: QuantizationCandidate, spec: Any) -> nn.Module:
        replacer_called.append(True)
        return ResidualVQLinear.from_encoded(
            candidate.encoded,
            bias=module.bias,
            source_dtype=module.proj_weight.dtype,
            device=module.proj_weight.device,
        )

    register_custom_module_adapter(_CustomRegisteredLinear, extractor, replacer)
    try:
        model = _ContainerModel(_CustomRegisteredLinear(48, 48))
        specs = discover_tensor_specs(model)
        assert len(specs) == 1
        assert specs[0].name == "layer"
        assert specs[0].attribute == "proj_weight"
        assert specs[0].layout == TensorLayout.LINEAR
        assert specs[0].quantizable is True
        assert specs[0].module_type == "_CustomRegisteredLinear"

        conf = _quick_codec_config(block_size=8, stages=1)
        inputs = [torch.randn(2, 48)]
        result = quantize_model_generalized(
            model,
            constraints=PlanningConstraints(target_state_ratio=0.8, max_tensor_estimated_error=10.0),
            candidate_configs=[conf],
            calibration_inputs=inputs,
        )

        assert len(replacer_called) == 1
        assert isinstance(result.model.layer, ResidualVQLinear)
        assert result.report.to_dict()["module_types"]["_CustomRegisteredLinear"]["selected"] == 1

        # Verify forward execution
        x = torch.randn(3, 48)
        out = result.model(x)
        assert out.shape == (3, 48)
        assert torch.isfinite(out).all()
    finally:
        unregister_custom_module_adapter(_CustomRegisteredLinear)


def test_duck_typed_linear_discovery_quantization_and_forward():
    """Verify duck-typed linear projections are automatically discovered and quantized."""
    model = _ContainerModel(_DuckTypedLinear(48, 48, bias=True))

    # With allow_duck_typed_linear=True (default)
    specs = discover_tensor_specs(model, allow_duck_typed_linear=True)
    assert len(specs) == 1
    assert specs[0].name == "layer"
    assert specs[0].layout == TensorLayout.LINEAR
    assert specs[0].module_type == "custom_linear"
    assert specs[0].quantizable is True

    # With allow_duck_typed_linear=False, duck-typed linears are skipped
    specs_disabled = discover_tensor_specs(model, allow_duck_typed_linear=False)
    assert len(specs_disabled) == 0

    # Generalized quantization
    conf = _quick_codec_config(block_size=8, stages=1)
    inputs = [torch.randn(2, 48)]
    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(target_state_ratio=0.8, max_tensor_estimated_error=10.0),
        candidate_configs=[conf],
        calibration_inputs=inputs,
    )

    assert isinstance(result.model.layer, ResidualVQLinear)
    assert result.report.to_dict()["module_types"]["custom_linear"]["selected"] == 1

    # Forward execution
    x = torch.randn(4, 48)
    out = result.model(x)
    assert out.shape == (4, 48)
    assert torch.isfinite(out).all()


def test_duck_typed_linear_exclusions_preserve_embeddings_and_norms():
    """Verify embeddings, normalizations, and unregistered nn.Linear subclasses are preserved."""
    class _SubclassLinear(nn.Linear):
        pass

    class _CompositeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 48)
            self.norm = nn.LayerNorm(48)
            self.subclass_linear = _SubclassLinear(48, 48)
            self.duck_linear = _DuckTypedLinear(48, 48)

    model = _CompositeModel()
    specs = {spec.name: spec for spec in discover_tensor_specs(model)}

    # Embedding is discovered as embedding, not duck-typed linear
    assert specs["embed"].layout == TensorLayout.EMBEDDING
    assert not specs["embed"].quantizable

    # Norm layer has weight but is excluded from duck-typed linear discovery
    assert "norm" not in specs

    # Unregistered nn.Linear subclass is retained as unsupported
    assert specs["subclass_linear"].layout == TensorLayout.UNSUPPORTED
    assert specs["subclass_linear"].protection_reason.startswith("unsupported_linear_subclass")

    # Duck linear is discovered as quantizable custom_linear
    assert specs["duck_linear"].layout == TensorLayout.LINEAR
    assert specs["duck_linear"].module_type == "custom_linear"
    assert specs["duck_linear"].quantizable is True


# ---------------------------------------------------------------------------
# Group 4: QuantizationPlan Optional Metadata Serialization
# ---------------------------------------------------------------------------


def test_quantization_plan_optional_metadata():
    """Verify QuantizationPlan optional architecture_profile and memory_admission fields."""
    spec = TensorSpec(
        name="test_proj",
        module_name="test_proj",
        attribute="weight",
        role=TensorRole.ATTENTION,
        layout=TensorLayout.LINEAR,
        logical_shape=(48, 48),
        matrix_shape=(48, 48),
        dtype="float32",
        dtype_bytes=4,
        original_weight_bytes=9216,
    )
    cand = QuantizationCandidate.dense(spec)
    collection = CandidateCollection({spec.name: [cand]})

    arch_meta = {"model_type": "test_arch", "num_layers": 4}
    mem_meta = {"budget_bytes": 10_000_000, "admitted": True}

    plan = QuantizationPlan(
        specs=[spec],
        candidates=collection,
        selected={spec.name: "dense"},
        base_model_state_bytes=9216,
        constraints={"target_ratio": 1.0},
        memory={"predicted_model_state_bytes": 9216},
        quality={"quantized_tensors": 0},
        optimizer={"objective": "noop"},
        architecture_profile=arch_meta,
        memory_admission=mem_meta,
    )

    plan_dict = plan.to_dict()
    assert plan_dict["architecture_profile"] == arch_meta
    assert plan_dict["memory_admission"] == mem_meta

    # When None, metadata keys are omitted
    plan_none = QuantizationPlan(
        specs=[spec],
        candidates=collection,
        selected={spec.name: "dense"},
        base_model_state_bytes=9216,
        constraints={"target_ratio": 1.0},
        memory={"predicted_model_state_bytes": 9216},
        quality={"quantized_tensors": 0},
        optimizer={"objective": "noop"},
    )
    plan_none_dict = plan_none.to_dict()
    assert "architecture_profile" not in plan_none_dict
    assert "memory_admission" not in plan_none_dict

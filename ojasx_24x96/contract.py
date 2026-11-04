"""Generalized Model Contract & Total Memory Admission Engine (Milestone M0).

Provides architecture-neutral introspection for any PyTorch ``nn.Module`` or
Hugging Face ``PreTrainedModel`` / ``PretrainedConfig`` (including modern dense,
GQA, MLA, MoE, multimodal composite, and hybrid Gated-DeltaNet/full-attention
models such as Qwen3.8, Gemma 4, GLM-5.3, and DeepSeek-V4) and enforces the
complete production inference memory equation:

    Encoded weights + remaining dense weights + codebooks/scales
    + KV cache + activations + workspaces + runtime overhead <= Budget
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .leech24 import shared_table_bytes


CODEC_SCHEMA_VERSION = "ojasx-24x96-v1.0"


def _dtype_bytes(dtype: Union[torch.dtype, str]) -> int:
    if isinstance(dtype, str):
        normalized = dtype.lower().replace("torch.", "")
        table = {
            "float16": 2,
            "fp16": 2,
            "half": 2,
            "bfloat16": 2,
            "bf16": 2,
            "float32": 4,
            "fp32": 4,
            "float": 4,
            "float64": 8,
            "fp64": 8,
            "int8": 1,
            "uint8": 1,
        }
        if normalized not in table:
            raise ValueError(f"Unsupported dtype specification: {dtype}")
        return table[normalized]
    return int(torch.empty((), dtype=dtype).element_size())


def _extract_text_config(config_or_model: Any) -> Tuple[Any, bool]:
    """Unwrap composite multimodal configs (Qwen3.5/3.8, Gemma4, GLM5-Next) to their text config."""
    cfg = getattr(config_or_model, "config", config_or_model)
    if cfg is None:
        return None, False
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return text_cfg, True
    return cfg, False


@dataclass(frozen=True)
class LayerKVSpec:
    """Per-layer KV cache or recurrent state memory contract."""

    layer_index: int
    attention_kind: str  # "full_attention", "sliding_attention", "mla_attention", "linear_recurrent"
    num_kv_heads: int
    head_dim: int
    sliding_window: Optional[int] = None
    mla_latent_dim: Optional[int] = None
    recurrent_state_elements: int = 0

    def kv_bytes_for_context(
        self,
        batch_size: int,
        context_length: int,
        kv_dtype: Union[torch.dtype, str] = torch.bfloat16,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if context_length < 0:
            raise ValueError("context_length must be nonnegative")
        elem_bytes = _dtype_bytes(kv_dtype)
        if self.attention_kind == "linear_recurrent":
            # Linear attention / Gated DeltaNet / SSM uses fixed O(1) state size
            return int(batch_size * self.recurrent_state_elements * elem_bytes)
        effective_seq = context_length
        if self.sliding_window is not None and self.sliding_window > 0:
            effective_seq = min(context_length, int(self.sliding_window))
        if self.attention_kind == "mla_attention" and self.mla_latent_dim is not None:
            # Compressed KV latent cache (e.g., DeepSeek MLA: kv_lora_rank + qk_rope_head_dim)
            return int(batch_size * effective_seq * self.mla_latent_dim * elem_bytes)
        # Standard K + V cache: 2 * B * N_kv * d_head * S
        return int(2 * batch_size * self.num_kv_heads * self.head_dim * effective_seq * elem_bytes)


@dataclass(frozen=True)
class ArchitectureProfile:
    """Architecture-neutral structural profile for any HF or custom model."""

    model_type: str
    architecture_class: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    is_moe: bool
    num_experts: int
    num_experts_per_tok: int
    is_hybrid_attention: bool
    is_multimodal_composite: bool
    layer_kv_specs: Tuple[LayerKVSpec, ...]

    def total_kv_cache_bytes(
        self,
        batch_size: int,
        context_length: int,
        kv_dtype: Union[torch.dtype, str] = torch.bfloat16,
    ) -> int:
        return sum(
            spec.kv_bytes_for_context(batch_size, context_length, kv_dtype)
            for spec in self.layer_kv_specs
        )

    def estimate_activation_peak_bytes(
        self,
        batch_size: int,
        seq_len: int,
        activation_dtype: Union[torch.dtype, str] = torch.bfloat16,
    ) -> int:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        elem_bytes = _dtype_bytes(activation_dtype)
        # Peak in-flight per-layer activation buffer: residual + QKV + MLP gate/up/down intermediates
        active_ffn = self.intermediate_size * max(1, self.num_experts_per_tok if self.is_moe else 1)
        qkv_dim = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
        per_token_elements = 2 * self.hidden_size + qkv_dim + 2 * active_ffn
        return int(batch_size * seq_len * per_token_elements * elem_bytes)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["layer_kv_specs"] = [asdict(s) for s in self.layer_kv_specs]
        return payload


def _scalar_int(val: Any, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return int(max((int(x) for x in val if x is not None), default=default))
    return int(val)


def introspect_architecture(model_or_config: Any) -> ArchitectureProfile:
    """Introspect any Hugging Face model/config or custom PyTorch nn.Module."""

    text_cfg, is_composite = _extract_text_config(model_or_config)
    arch_class = (
        model_or_config.__class__.__name__
        if isinstance(model_or_config, nn.Module)
        else getattr(text_cfg, "architectures", ["CustomModel"])[0]
        if getattr(text_cfg, "architectures", None)
        else text_cfg.__class__.__name__
    )

    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        model_type = str(getattr(text_cfg, "model_type", "custom_hf"))
        num_layers = _scalar_int(getattr(text_cfg, "num_hidden_layers", getattr(text_cfg, "n_layer", 1)), 1)
        hidden_size = _scalar_int(getattr(text_cfg, "hidden_size", getattr(text_cfg, "n_embd", 64)), 64)
        intermediate_size = _scalar_int(
            getattr(
                text_cfg,
                "intermediate_size",
                getattr(text_cfg, "moe_intermediate_size", hidden_size * 4),
            ),
            hidden_size * 4,
        )
        vocab_size = _scalar_int(getattr(text_cfg, "vocab_size", 32000), 32000)
        num_heads = _scalar_int(getattr(text_cfg, "num_attention_heads", getattr(text_cfg, "n_head", 4)), 4)
        num_kv_heads = _scalar_int(
            getattr(text_cfg, "num_key_value_heads", getattr(text_cfg, "num_kv_heads", num_heads)),
            num_heads,
        )
        head_dim = _scalar_int(getattr(text_cfg, "head_dim", hidden_size // max(1, num_heads)), hidden_size // max(1, num_heads))
        num_experts = _scalar_int(
            getattr(
                text_cfg,
                "num_experts",
                getattr(text_cfg, "num_local_experts", getattr(text_cfg, "n_routed_experts", 0)),
            ),
            0,
        )
        num_experts_per_tok = _scalar_int(
            getattr(text_cfg, "num_experts_per_tok", getattr(text_cfg, "num_selected_experts", 1 if num_experts == 0 else 2)),
            1 if num_experts == 0 else 2,
        )
        is_moe = num_experts > 1
        sliding_window = _scalar_int(getattr(text_cfg, "sliding_window", None), 0) or None
        kv_lora_rank = _scalar_int(getattr(text_cfg, "kv_lora_rank", None), 0) or None
        qk_rope_head_dim = _scalar_int(getattr(text_cfg, "qk_rope_head_dim", 0), 0)

        layer_types = getattr(text_cfg, "layer_types", None) or getattr(text_cfg, "attn_layer_types", None)
        specs: List[LayerKVSpec] = []
        is_hybrid = False
        for idx in range(num_layers):
            ltype = str(layer_types[idx]).lower() if layer_types and idx < len(layer_types) else "full_attention"
            if any(tag in ltype for tag in ("linear", "deltanet", "mamba", "recurrent", "ssm", " indexed")):
                is_hybrid = True
                # Linear/recurrent state: fixed size proportional to num_heads * head_dim * head_dim
                state_elems = num_heads * head_dim * head_dim
                specs.append(
                    LayerKVSpec(
                        layer_index=idx,
                        attention_kind="linear_recurrent",
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        recurrent_state_elements=state_elems,
                    )
                )
            elif kv_lora_rank is not None and int(kv_lora_rank) > 0:
                specs.append(
                    LayerKVSpec(
                        layer_index=idx,
                        attention_kind="mla_attention",
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        mla_latent_dim=int(kv_lora_rank) + qk_rope_head_dim,
                    )
                )
            elif "sliding" in ltype or (sliding_window is not None and int(sliding_window) > 0 and "full" not in ltype):
                specs.append(
                    LayerKVSpec(
                        layer_index=idx,
                        attention_kind="sliding_attention",
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        sliding_window=int(sliding_window) if sliding_window else 4096,
                    )
                )
            else:
                specs.append(
                    LayerKVSpec(
                        layer_index=idx,
                        attention_kind="full_attention",
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                    )
                )

        return ArchitectureProfile(
            model_type=model_type,
            architecture_class=arch_class,
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            is_moe=is_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            is_hybrid_attention=is_hybrid,
            is_multimodal_composite=is_composite,
            layer_kv_specs=tuple(specs),
        )

    # Fallback: arbitrary custom PyTorch nn.Module without HF config
    if isinstance(model_or_config, nn.Module):
        linears = [
            m for m in model_or_config.modules()
            if isinstance(m, nn.Linear) or (hasattr(m, "in_features") and hasattr(m, "out_features"))
        ]
        embeddings = [m for m in model_or_config.modules() if isinstance(m, nn.Embedding)]
        max_hidden = max((int(getattr(m, "in_features", 64)) for m in linears), default=64)
        max_inter = max((int(getattr(m, "out_features", 64)) for m in linears), default=max_hidden * 2)
        vocab = max((int(m.num_embeddings) for m in embeddings), default=1024)
        est_layers = max(1, len(linears) // 4)
        num_heads = max(1, min(8, max_hidden // 8))
        head_dim = max(1, max_hidden // num_heads)
        specs = tuple(
            LayerKVSpec(
                layer_index=i,
                attention_kind="full_attention",
                num_kv_heads=num_heads,
                head_dim=head_dim,
            )
            for i in range(est_layers)
        )
        return ArchitectureProfile(
            model_type="custom_pytorch_module",
            architecture_class=arch_class,
            num_hidden_layers=est_layers,
            hidden_size=max_hidden,
            intermediate_size=max_inter,
            vocab_size=vocab,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=head_dim,
            is_moe=False,
            num_experts=0,
            num_experts_per_tok=1,
            is_hybrid_attention=False,
            is_multimodal_composite=False,
            layer_kv_specs=specs,
        )

    raise TypeError(f"Cannot introspect architecture from object of type {type(model_or_config).__name__}")


@dataclass(frozen=True)
class MemoryAdmissionReport:
    """Complete production memory accounting across all 7 mandatory buckets."""

    encoded_weight_bytes: int
    dense_weight_bytes: int
    codebooks_and_scales_bytes: int
    shared_table_bytes: int
    kv_cache_bytes: int
    activation_prefill_bytes: int
    activation_decode_bytes: int
    workspace_bytes: int
    runtime_overhead_bytes: int
    decoded_weight_cache_bytes: int
    batch_size: int
    context_length: int
    prefill_chunk_size: int
    kv_dtype: str
    budget_bytes: Optional[int]
    admitted: bool
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def persistent_model_bytes(self) -> int:
        return (
            self.encoded_weight_bytes
            + self.dense_weight_bytes
            + self.codebooks_and_scales_bytes
            + self.shared_table_bytes
            + self.decoded_weight_cache_bytes
        )

    @property
    def total_prefill_peak_bytes(self) -> int:
        return (
            self.persistent_model_bytes
            + self.kv_cache_bytes
            + self.activation_prefill_bytes
            + self.workspace_bytes
            + self.runtime_overhead_bytes
        )

    @property
    def total_decode_peak_bytes(self) -> int:
        return (
            self.persistent_model_bytes
            + self.kv_cache_bytes
            + self.activation_decode_bytes
            + self.workspace_bytes
            + self.runtime_overhead_bytes
        )

    @property
    def total_required_bytes(self) -> int:
        return max(self.total_prefill_peak_bytes, self.total_decode_peak_bytes)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["persistent_model_bytes"] = self.persistent_model_bytes
        payload["total_prefill_peak_bytes"] = self.total_prefill_peak_bytes
        payload["total_decode_peak_bytes"] = self.total_decode_peak_bytes
        payload["total_required_bytes"] = self.total_required_bytes
        return payload


class AdmissionError(RuntimeError):
    """Raised when a model + KV-cache + runtime configuration exceeds the hardware memory budget."""

    def __init__(self, report: MemoryAdmissionReport):
        self.report = report
        reasons = "; ".join(report.rejection_reasons) or "memory budget exceeded"
        super().__init__(f"Memory admission rejected ({reasons})")


def evaluate_memory_admission(
    profile: ArchitectureProfile,
    *,
    encoded_weight_bytes: int,
    dense_weight_bytes: int,
    codebooks_and_scales_bytes: int = 0,
    decoded_weight_cache_bytes: int = 0,
    batch_size: int = 1,
    context_length: int = 2048,
    prefill_chunk_size: int = 512,
    kv_dtype: Union[torch.dtype, str] = torch.bfloat16,
    activation_dtype: Union[torch.dtype, str] = torch.bfloat16,
    decode_row_block_size: int = 64,
    max_matrix_cols: Optional[int] = None,
    runtime_overhead_bytes: int = 16 * 1024 * 1024,
    budget_bytes: Optional[int] = None,
    max_kv_cache_bytes: Optional[int] = None,
    fused_register_decode: bool = False,
) -> MemoryAdmissionReport:
    """Compute exact 7-bucket memory footprint and verify admission against hardware constraints."""

    if encoded_weight_bytes < 0 or dense_weight_bytes < 0 or codebooks_and_scales_bytes < 0:
        raise ValueError("Weight and codebook byte counts must be nonnegative")
    if batch_size <= 0 or context_length <= 0 or prefill_chunk_size <= 0:
        raise ValueError("batch_size, context_length, and prefill_chunk_size must be positive")

    effective_prefill = min(context_length, prefill_chunk_size)
    kv_bytes = profile.total_kv_cache_bytes(batch_size, context_length, kv_dtype)
    act_prefill = profile.estimate_activation_peak_bytes(batch_size, effective_prefill, activation_dtype)
    act_decode = profile.estimate_activation_peak_bytes(batch_size, 1, activation_dtype)

    table_bytes = int(shared_table_bytes())
    max_cols = max_matrix_cols or max(profile.hidden_size, profile.intermediate_size)
    padded_cols = ((max_cols + 23) // 24) * 24
    workspace = 0 if fused_register_decode else int(decode_row_block_size * padded_cols * 4)

    reasons: List[str] = []
    persistent = (
        encoded_weight_bytes
        + dense_weight_bytes
        + codebooks_and_scales_bytes
        + table_bytes
        + decoded_weight_cache_bytes
    )
    total_prefill = persistent + kv_bytes + act_prefill + workspace + runtime_overhead_bytes
    total_decode = persistent + kv_bytes + act_decode + workspace + runtime_overhead_bytes
    total_req = max(total_prefill, total_decode)

    if max_kv_cache_bytes is not None and kv_bytes > max_kv_cache_bytes:
        reasons.append(f"kv_cache_bytes ({kv_bytes}) > max_kv_cache_bytes ({max_kv_cache_bytes})")
    if budget_bytes is not None and total_req > budget_bytes:
        reasons.append(f"total_required_bytes ({total_req}) > budget_bytes ({budget_bytes})")

    return MemoryAdmissionReport(
        encoded_weight_bytes=int(encoded_weight_bytes),
        dense_weight_bytes=int(dense_weight_bytes),
        codebooks_and_scales_bytes=int(codebooks_and_scales_bytes),
        shared_table_bytes=table_bytes,
        kv_cache_bytes=int(kv_bytes),
        activation_prefill_bytes=int(act_prefill),
        activation_decode_bytes=int(act_decode),
        workspace_bytes=int(workspace),
        runtime_overhead_bytes=int(runtime_overhead_bytes),
        decoded_weight_cache_bytes=int(decoded_weight_cache_bytes),
        batch_size=int(batch_size),
        context_length=int(context_length),
        prefill_chunk_size=int(effective_prefill),
        kv_dtype=str(kv_dtype).replace("torch.", ""),
        budget_bytes=int(budget_bytes) if budget_bytes is not None else None,
        admitted=len(reasons) == 0,
        rejection_reasons=tuple(reasons),
    )


def max_admissible_context_length(
    profile: ArchitectureProfile,
    *,
    budget_bytes: int,
    encoded_weight_bytes: int,
    dense_weight_bytes: int,
    codebooks_and_scales_bytes: int = 0,
    decoded_weight_cache_bytes: int = 0,
    batch_size: int = 1,
    prefill_chunk_size: int = 512,
    kv_dtype: Union[torch.dtype, str] = torch.bfloat16,
    activation_dtype: Union[torch.dtype, str] = torch.bfloat16,
    decode_row_block_size: int = 64,
    runtime_overhead_bytes: int = 16 * 1024 * 1024,
    fused_register_decode: bool = False,
    upper_bound: int = 1_048_576,
) -> int:
    """Binary-search the maximum context length S_ctx that satisfies the total memory budget."""

    low = 0
    high = upper_bound
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        report = evaluate_memory_admission(
            profile,
            encoded_weight_bytes=encoded_weight_bytes,
            dense_weight_bytes=dense_weight_bytes,
            codebooks_and_scales_bytes=codebooks_and_scales_bytes,
            decoded_weight_cache_bytes=decoded_weight_cache_bytes,
            batch_size=batch_size,
            context_length=mid,
            prefill_chunk_size=prefill_chunk_size,
            kv_dtype=kv_dtype,
            activation_dtype=activation_dtype,
            decode_row_block_size=decode_row_block_size,
            runtime_overhead_bytes=runtime_overhead_bytes,
            budget_bytes=budget_bytes,
            fused_register_decode=fused_register_decode,
        )
        if report.admitted:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


@dataclass(frozen=True)
class QualificationContract:
    """Explicit production qualification record scoped to an exact model + hardware + context tuple."""

    checkpoint_id: str
    checkpoint_sha256: str
    model_type: str
    architecture_class: str
    codec_version: str
    backend: str
    os_platform: str
    hardware_target: str
    max_context_length: int
    kv_dtype: str
    workload: str
    quality_gate_passed: bool
    kv_cache_parity_verified: bool
    admission_report: Dict[str, Any]

    @classmethod
    def create_for_host(
        cls,
        *,
        checkpoint_id: str,
        checkpoint_sha256: str,
        profile: ArchitectureProfile,
        admission: MemoryAdmissionReport,
        backend: str = "cpu",
        workload: str = "causal_lm_cached_generation",
        quality_gate_passed: bool = True,
        kv_cache_parity_verified: bool = True,
    ) -> "QualificationContract":
        hw = f"{platform.system()}-{platform.machine()}"
        if backend == "cuda" and torch.cuda.is_available():
            hw = torch.cuda.get_device_name(0)
        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            model_type=profile.model_type,
            architecture_class=profile.architecture_class,
            codec_version=CODEC_SCHEMA_VERSION,
            backend=backend,
            os_platform=platform.platform(),
            hardware_target=hw,
            max_context_length=admission.context_length,
            kv_dtype=admission.kv_dtype,
            workload=workload,
            quality_gate_passed=bool(quality_gate_passed and admission.admitted),
            kv_cache_parity_verified=bool(kv_cache_parity_verified),
            admission_report=admission.to_dict(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

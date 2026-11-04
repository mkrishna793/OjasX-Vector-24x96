"""Inference-only modules and architecture-neutral HF layer replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .codec import (
    QuantizationConfig,
    QuantizedWeight,
    dequantize_weight,
    dequantize_weight_rows,
    quantize_weight,
)
from .leech_codec import (
    LeechQuantizedWeight,
    dequantize_leech_weight,
    dequantize_leech_weight_rows,
)


@dataclass
class LayerQuantizationReport:
    name: str
    shape: Tuple[int, int]
    selected: bool
    reason: str
    original_bytes: int
    storage_bytes: int
    relative_error: Optional[float] = None
    selected_stages: int = 0
    nominal_bits_per_weight: float = 0.0
    measured_bits_per_weight: float = 0.0
    module_type: str = "linear"

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


@dataclass
class ModelQuantizationReport:
    layers: List[LayerQuantizationReport]
    configuration: Dict[str, object] = field(default_factory=dict)
    include_globs: Tuple[str, ...] = ()
    exclude_globs: Tuple[str, ...] = ()
    preserve_shared_weights: bool = True

    @property
    def selected_layers(self) -> int:
        return sum(layer.selected for layer in self.layers)

    @property
    def original_bytes(self) -> int:
        return sum(layer.original_bytes for layer in self.layers)

    @property
    def selected_original_bytes(self) -> int:
        return sum(layer.original_bytes for layer in self.layers if layer.selected)

    @property
    def stored_bytes(self) -> int:
        return sum(
            layer.storage_bytes if layer.selected else layer.original_bytes
            for layer in self.layers
        )

    def to_dict(self) -> Dict[str, object]:
        original = self.selected_original_bytes
        type_counts: Dict[str, Dict[str, int]] = {}
        for layer in self.layers:
            counts = type_counts.setdefault(layer.module_type, {"eligible": 0, "selected": 0})
            counts["eligible"] += 1
            counts["selected"] += int(layer.selected)
        return {
            "schema_version": 1,
            "codec": "ojasx_adaptive_residual_vq",
            "configuration": self.configuration,
            "include_globs": list(self.include_globs),
            "exclude_globs": list(self.exclude_globs),
            "preserve_shared_weights": self.preserve_shared_weights,
            "selected_layers": self.selected_layers,
            "eligible_layers": len(self.layers),
            "module_types": type_counts,
            "selected_original_weight_bytes": original,
            "selected_stored_weight_bytes": sum(
                layer.storage_bytes for layer in self.layers if layer.selected
            ),
            "selected_weight_compression_ratio": (
                original / sum(layer.storage_bytes for layer in self.layers if layer.selected)
                if original and self.selected_layers
                else 1.0
            ),
            "scope": (
                "Per-tensor weight measurements only; excludes tokenizer, KV cache, "
                "activations, runtime overhead, and model-quality guarantees"
            ),
            "layers": [layer.to_dict() for layer in self.layers],
        }


class ResidualVQLinear(nn.Module):
    """An inference-only ``nn.Linear`` replacement with packed residual codes.

    Memory mode retains no dense matrix and decodes bounded row chunks. Optional
    speed mode reuses a full decoded matrix and reports its storage explicitly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        block_size: int,
        bits_per_stage: int,
        stages: int,
        source_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        codec: str = "adaptive_residual_vq",
        gain_bits: int = 0,
        rotation_reflections: int = 0,
        rotation_seed: int = 20261104,
        adaptive_stages: bool = True,
        active_counts: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or stages <= 0:
            raise ValueError("ResidualVQLinear dimensions and stages must be positive")
        if codec not in {"adaptive_residual_vq", "leech24_shell2_residual"}:
            raise ValueError(f"Unsupported residual codec {codec!r}")
        if codec == "leech24_shell2_residual" and (
            block_size != 24
            or bits_per_stage != 18 + gain_bits
            or gain_bits not in {0, 1, 2}
            or not 0 <= rotation_reflections <= 8
        ):
            raise ValueError("Invalid Leech24 linear codec dimensions")
        if active_counts is not None and len(active_counts) != stages:
            raise ValueError("active_counts must have one entry per residual stage")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = int(block_size)
        self.bits_per_stage = int(bits_per_stage)
        self.stages = int(stages)
        self.codec = codec
        self.gain_bits = int(gain_bits)
        self.rotation_reflections = int(rotation_reflections)
        self.rotation_seed = int(rotation_seed)
        self.adaptive_stages = bool(adaptive_stages)
        if not source_dtype.is_floating_point:
            raise TypeError("Residual VQ source dtype must be floating point")
        self.source_dtype = source_dtype
        self.weight_cache = False
        self.streaming_decode = True
        self.decode_row_block_size = 64
        self.leech_decode_backend = "auto"
        self.fused_native_gemm = False
        self._cached_weight_key: Optional[Tuple[object, ...]] = None
        self.register_buffer("_cached_weight", None, persistent=False)
        blocks_per_row = (in_features + block_size - 1) // block_size
        vectors = out_features * blocks_per_row
        if codec == "adaptive_residual_vq":
            packed_bytes = (vectors * bits_per_stage + 7) // 8
            self.register_buffer(
                "packed_codes",
                torch.zeros(stages, packed_bytes, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "codebooks",
                torch.zeros(
                    stages,
                    1 << bits_per_stage,
                    block_size,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        else:
            counts = active_counts or tuple(vectors for _ in range(stages))
            for stage, active in enumerate(counts):
                if not 0 <= active <= vectors or (stage == 0 and active != vectors):
                    raise ValueError("Invalid Leech active-vector count")
                if not adaptive_stages and active != vectors:
                    raise ValueError("Fixed Leech stages must contain every vector")
                self.register_buffer(
                    f"leech_indices_{stage}",
                    torch.zeros((active * 18 + 7) // 8, dtype=torch.uint8, device=device),
                )
                self.register_buffer(
                    f"leech_gains_{stage}",
                    torch.zeros(
                        (active * gain_bits + 7) // 8,
                        dtype=torch.uint8,
                        device=device,
                    ),
                )
                presence_bytes = (vectors + 7) // 8 if adaptive_stages and stage > 0 else 0
                self.register_buffer(
                    f"leech_presence_{stage}",
                    torch.zeros(presence_bytes, dtype=torch.uint8, device=device),
                )
            self.register_buffer(
                "gain_codebooks",
                torch.zeros(stages, 1 << gain_bits, dtype=torch.float32, device=device),
            )
        self.register_buffer(
            "row_scales",
            torch.zeros(out_features, dtype=torch.float32, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_encoded(
        cls,
        encoded,
        *,
        bias: Optional[torch.Tensor],
        source_dtype: torch.dtype,
        device: torch.device,
    ) -> "ResidualVQLinear":
        rows, columns = encoded.original_shape
        leech = isinstance(encoded, LeechQuantizedWeight)
        module = cls(
            columns,
            rows,
            bias is not None,
            block_size=encoded.block_size,
            bits_per_stage=encoded.bits_per_stage,
            stages=encoded.stages,
            source_dtype=source_dtype,
            device=device,
            codec="leech24_shell2_residual" if leech else "adaptive_residual_vq",
            gain_bits=encoded.gain_bits if leech else 0,
            rotation_reflections=encoded.rotation_reflections if leech else 0,
            rotation_seed=encoded.rotation_seed if leech else 20261104,
            adaptive_stages=encoded.adaptive_stages if leech else True,
            active_counts=(
                tuple(encoded.active_count(stage) for stage in range(encoded.stages))
                if leech
                else None
            ),
        )
        if leech:
            for stage in range(encoded.stages):
                getattr(module, f"leech_indices_{stage}").copy_(
                    encoded.packed_indices[stage].to(device)
                )
                getattr(module, f"leech_gains_{stage}").copy_(
                    encoded.packed_gains[stage].to(device)
                )
                getattr(module, f"leech_presence_{stage}").copy_(
                    encoded.presence[stage].to(device)
                )
            module.gain_codebooks.copy_(encoded.gain_codebooks.to(device))
        else:
            module.packed_codes.copy_(encoded.packed_codes.to(device))
            module.codebooks.copy_(encoded.codebooks.to(device))
        module.row_scales.copy_(encoded.row_scales.to(device))
        if bias is not None:
            module.bias.data = bias.detach().to(
                device=device,
                dtype=source_dtype,
            ).reshape(module.bias.shape).clone()
        return module

    def encoded(self):
        if self.codec == "leech24_shell2_residual":
            return LeechQuantizedWeight(
                original_shape=(self.out_features, self.in_features),
                packed_indices=tuple(
                    getattr(self, f"leech_indices_{stage}") for stage in range(self.stages)
                ),
                packed_gains=tuple(
                    getattr(self, f"leech_gains_{stage}") for stage in range(self.stages)
                ),
                presence=tuple(
                    getattr(self, f"leech_presence_{stage}") for stage in range(self.stages)
                ),
                gain_codebooks=self.gain_codebooks,
                row_scales=self.row_scales,
                gain_bits=self.gain_bits,
                rotation_reflections=self.rotation_reflections,
                rotation_seed=self.rotation_seed,
                adaptive_stages=self.adaptive_stages,
                metrics={},
            )
        return QuantizedWeight(
            original_shape=(self.out_features, self.in_features),
            block_size=self.block_size,
            bits_per_stage=self.bits_per_stage,
            packed_codes=self.packed_codes,
            codebooks=self.codebooks,
            row_scales=self.row_scales,
            metrics={},
        )

    @property
    def weight(self) -> torch.Tensor:
        """Decoded compatibility view for HF code that reads ``linear.weight``.

        This is a temporary tensor in memory mode, not a persistent dense weight.
        Direct reads use the original source dtype; ordinary forwards use the
        activation dtype as before.
        """

        return self.dequantize_weight(dtype=self.source_dtype)

    def clear_weight_cache(self) -> None:
        self._cached_weight = None
        self._cached_weight_key = None

    def set_weight_cache(self, enabled: bool) -> None:
        self.weight_cache = bool(enabled)
        self.clear_weight_cache()

    def set_streaming_decode(self, enabled: bool, *, row_block_size: int = 64) -> None:
        if row_block_size <= 0:
            raise ValueError("Streaming decode row block size must be positive")
        self.streaming_decode = bool(enabled)
        self.decode_row_block_size = int(row_block_size)

    def set_leech_decode_backend(self, backend: str) -> None:
        if self.codec != "leech24_shell2_residual":
            raise ValueError("Leech decode backends apply only to Leech modules")
        if backend not in {"python", "native", "auto"}:
            raise ValueError("Leech decode backend must be python, native, or auto")
        self.leech_decode_backend = backend
        self.clear_weight_cache()

    def set_fused_native_gemm(self, enabled: bool) -> None:
        self.fused_native_gemm = bool(enabled)

    def _decode_encoded_rows(
        self,
        encoded,
        row_start: int,
        row_stop: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self.codec == "leech24_shell2_residual":
            return dequantize_leech_weight_rows(
                encoded,
                row_start,
                row_stop,
                dtype=dtype,
                backend=self.leech_decode_backend,
            )
        return dequantize_weight_rows(
            encoded,
            row_start,
            row_stop,
            dtype=dtype,
        )

    def dequantize_weight_rows(
        self,
        row_start: int,
        row_stop: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self._decode_encoded_rows(
            self.encoded(),
            row_start,
            row_stop,
            dtype=dtype,
        )

    def _payload_tensors(self) -> Tuple[torch.Tensor, ...]:
        if self.codec == "leech24_shell2_residual":
            return (
                *(getattr(self, f"leech_indices_{stage}") for stage in range(self.stages)),
                *(getattr(self, f"leech_gains_{stage}") for stage in range(self.stages)),
                *(getattr(self, f"leech_presence_{stage}") for stage in range(self.stages)),
                self.gain_codebooks,
                self.row_scales,
            )
        return self.packed_codes, self.codebooks, self.row_scales

    def dequantize_weight(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype or torch.float32
        key = None
        versioned = self.weight_cache
        if self.weight_cache:
            versions = []
            payloads = self._payload_tensors()
            for tensor in payloads:
                try:
                    versions.append(tensor._version)
                except RuntimeError:
                    # Inference tensors do not track mutation versions. Reusing a
                    # cache after an untracked mutation would be unsafe, so such
                    # modules deliberately decode every time even in speed mode.
                    versioned = False
                    break
            if versioned:
                key = (
                    self.row_scales.device,
                    target_dtype,
                    *(id(tensor) for tensor in payloads),
                    *versions,
                )
                if self._cached_weight is not None and self._cached_weight_key == key:
                    return self._cached_weight
            else:
                self.clear_weight_cache()
        if self.codec == "leech24_shell2_residual":
            weight = dequantize_leech_weight(
                self.encoded(),
                dtype=target_dtype,
                backend=self.leech_decode_backend,
            )
        else:
            weight = dequantize_weight(self.encoded(), dtype=target_dtype)
        if self.weight_cache and versioned:
            self._cached_weight = weight
            self._cached_weight_key = key
        return weight

    @property
    def cached_weight_bytes(self) -> int:
        if self._cached_weight is None:
            return 0
        return self._cached_weight.numel() * self._cached_weight.element_size()

    @property
    def storage_bytes(self) -> int:
        result = self.encoded().storage_bytes
        if self.bias is not None:
            result += self.bias.numel() * self.bias.element_size()
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected final dimension {self.in_features}, received {inputs.shape[-1]}"
            )
        if self.fused_native_gemm and not self.weight_cache and inputs.device.type == "cpu":
            encoded = self.encoded()
            bias = self.bias.to(dtype=torch.float32) if self.bias is not None else None
            flat_inputs = inputs.reshape(-1, self.in_features).to(dtype=torch.float32)
            if self.codec == "leech24_shell2_residual":
                from .leech_codec import rotate_activation_for_leech
                from .leech_native import fused_leech_gemm_native

                rotated = rotate_activation_for_leech(
                    flat_inputs,
                    self.in_features,
                    self.rotation_reflections,
                    self.rotation_seed,
                )
                output = fused_leech_gemm_native(
                    encoded,
                    rotated,
                    0,
                    self.out_features,
                    bias=bias,
                )
                if output is not None:
                    return output.to(dtype=inputs.dtype).reshape(*inputs.shape[:-1], self.out_features)
            elif self.codec == "adaptive_residual_vq":
                from .leech_native import fused_rvq_gemm_native

                output = fused_rvq_gemm_native(
                    encoded,
                    flat_inputs,
                    0,
                    self.out_features,
                    bias=bias,
                )
                if output is not None:
                    return output.to(dtype=inputs.dtype).reshape(*inputs.shape[:-1], self.out_features)

        if self.streaming_decode and not self.weight_cache:
            flat_inputs = inputs.reshape(-1, self.in_features)
            flat_output = torch.empty(
                flat_inputs.shape[0],
                self.out_features,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            # The lightweight encoded view is constructed once per forward. Each
            # decode remains bounded and no dense weight is retained afterwards.
            encoded = self.encoded()
            for row_start in range(0, self.out_features, self.decode_row_block_size):
                row_stop = min(row_start + self.decode_row_block_size, self.out_features)
                weight = self._decode_encoded_rows(
                    encoded,
                    row_start,
                    row_stop,
                    dtype=inputs.dtype,
                )
                bias = (
                    self.bias[row_start:row_stop].to(dtype=inputs.dtype)
                    if self.bias is not None
                    else None
                )
                flat_output[:, row_start:row_stop] = F.linear(flat_inputs, weight, bias)
            return flat_output.reshape(*inputs.shape[:-1], self.out_features)
        weight = self.dequantize_weight(dtype=inputs.dtype)
        bias = self.bias.to(dtype=inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, codec={self.codec}, block_size={self.block_size}, "
            f"bits_per_stage={self.bits_per_stage}, stages={self.stages}, "
            f"weight_cache={self.weight_cache}, streaming_decode={self.streaming_decode}, "
            f"decode_row_block_size={self.decode_row_block_size}, "
            f"leech_decode_backend={self.leech_decode_backend}, inference_only=True"
        )


class ResidualVQConv1D(ResidualVQLinear):
    """HF ``Conv1D`` adapter; its source weight is stored as ``[in, out]``."""

    @property
    def weight(self) -> torch.Tensor:
        return super().weight.T

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The encoder transposed HF's [in, out] source into [out, in]. The base
        # streaming/cached F.linear paths therefore both compute inputs @ source.
        return super().forward(inputs)


class ResidualVQFusedExperts(nn.Module):
    """Adapter for HF's fused ``gate_up_proj``/``down_proj`` expert banks.

    Each projection is flattened only for codec row processing, then restored to
    its logical ``[expert, out, in]`` shape for the original expert computation.
    A rejected projection remains a dense parameter inside the replacement.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        gate_up_proj,
        down_proj,
        act_fn: nn.Module,
        gate_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        if min(num_experts, hidden_dim, intermediate_dim) <= 0:
            raise ValueError("Expert-bank dimensions must be positive")
        if gate_limit is not None and (
            not math.isfinite(gate_limit) or gate_limit <= 0
        ):
            raise ValueError("Expert gate limit must be finite and positive")
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn
        self.gate_limit = float(gate_limit) if gate_limit is not None else None
        self.streaming_decode = True

    def _projection_weight(
        self,
        projection,
        projection_name: str,
        dtype: torch.dtype,
        expert_idx: Optional[int] = None,
    ):
        if isinstance(projection, ResidualVQLinear):
            if self.streaming_decode and expert_idx is not None:
                rows_per_expert = (
                    2 * self.intermediate_dim
                    if projection_name == "gate_up_proj"
                    else self.hidden_dim
                )
                row_start = int(expert_idx) * rows_per_expert
                return projection.dequantize_weight_rows(
                    row_start,
                    row_start + rows_per_expert,
                    dtype=dtype,
                )
            weight = projection.dequantize_weight(dtype=dtype)
            if projection_name == "gate_up_proj":
                return weight.reshape(
                    self.num_experts,
                    2 * self.intermediate_dim,
                    self.hidden_dim,
                )
            return weight.reshape(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            )
        return projection[expert_idx] if expert_idx is not None else projection

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.gate_limit is not None:
            gate = gate.clamp(max=self.gate_limit)
            up = up.clamp(min=-self.gate_limit, max=self.gate_limit)
        return self.act_fn(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Speed mode decodes each bank once and caches it. Memory mode instead
        # decodes only each active expert inside the loop, bounding dense weight
        # workspace by the expert rather than by the complete expert bank.
        gate_up_bank = None
        down_bank = None
        if not self.streaming_decode:
            gate_up_bank = self._projection_weight(
                self.gate_up_proj, "gate_up_proj", hidden_states.dtype
            )
            down_bank = self._projection_weight(
                self.down_proj, "down_proj", hidden_states.dtype
            )
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if self.streaming_decode:
                gate_up_weight = self._projection_weight(
                    self.gate_up_proj,
                    "gate_up_proj",
                    hidden_states.dtype,
                    int(expert_idx),
                )
                down_weight = self._projection_weight(
                    self.down_proj,
                    "down_proj",
                    hidden_states.dtype,
                    int(expert_idx),
                )
            else:
                gate_up_weight = gate_up_bank[expert_idx]
                down_weight = down_bank[expert_idx]
            gate_up = F.linear(current_state, gate_up_weight)
            current_hidden_states = self._apply_gate(gate_up)
            current_hidden_states = F.linear(current_hidden_states, down_weight)
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


def _conv1d_type():
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return None
    return Conv1D


# Exact HF forward contracts verified by source inspection and roundtrip tests.
# Values name the optional clamp-limit attribute; None means ordinary SwiGLU.
_FUSED_EXPERT_CONTRACTS = {
    ("transformers.models.qwen2_moe.modeling_qwen2_moe", "Qwen2MoeExperts"): None,
    ("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts"): None,
    ("transformers.models.mixtral.modeling_mixtral", "MixtralExperts"): None,
    ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextExperts"): None,
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeExperts"): None,
    ("transformers.models.gemma4.modeling_gemma4", "Gemma4TextExperts"): None,
    ("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa", "GlmMoeDsaExperts"): None,
    ("transformers.models.deepseek_v4.modeling_deepseek_v4", "DeepseekV4Experts"): "limit",
    ("transformers.models.glm5_next.modeling_glm5_next", "Glm5NextTextExperts"): "swiglu_limit",
}


def _expert_contract_key(module: nn.Module) -> Tuple[str, str]:
    return module.__class__.__module__, module.__class__.__name__


def _expert_gate_limit(module: nn.Module) -> Optional[float]:
    attribute = _FUSED_EXPERT_CONTRACTS.get(_expert_contract_key(module))
    value = getattr(module, attribute, None) if attribute is not None else None
    return float(value) if value is not None else None


def _expert_activation(module: nn.Module) -> nn.Module:
    activation = getattr(module, "act_fn", None)
    # GLM5-Next's verified clamped contract calls F.silu directly.
    return activation if isinstance(activation, nn.Module) else nn.SiLU()


def _is_fused_expert_bank(module: nn.Module) -> bool:
    # Keep an explicit allowlist: equal attribute names in custom remote code are
    # not sufficient evidence that replacing its forward method is semantics-safe.
    key = _expert_contract_key(module)
    if key not in _FUSED_EXPERT_CONTRACTS:
        return False
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        return False
    experts = int(getattr(module, "num_experts", 0))
    gate_limit = _expert_gate_limit(module)
    return (
        experts > 0
        and gate_up.ndim == 3
        and down.ndim == 3
        and gate_up.shape[0] == down.shape[0] == experts
        and gate_up.shape[2] == int(getattr(module, "hidden_dim", -1))
        and down.shape[1] == gate_up.shape[2]
        and gate_up.shape[1] == 2 * int(getattr(module, "intermediate_dim", -2))
        and down.shape[2] == int(getattr(module, "intermediate_dim", -1))
        and (
            isinstance(getattr(module, "act_fn", None), nn.Module)
            or key[1] == "Glm5NextTextExperts"
        )
        and (
            _FUSED_EXPERT_CONTRACTS[key] is None
            or (gate_limit is not None and math.isfinite(gate_limit) and gate_limit > 0)
        )
    )


def _layer_kind(module: nn.Module) -> Optional[str]:
    # Unknown Linear subclasses can override forward or reshape their inputs
    # (for example DeepseekV4GroupedLinear). Fail closed instead of replacing
    # them with ordinary F.linear semantics.
    if type(module) is nn.Linear:
        return "linear"
    conv1d = _conv1d_type()
    if conv1d is not None and isinstance(module, conv1d):
        return "conv1d"
    if _is_fused_expert_bank(module):
        return "fused_experts"
    return None


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _stable_seed(name: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for byte in name.encode("utf-8"):
        value = (value * 1664525 + byte + 1013904223) & 0xFFFFFFFF
    return value


def _shared_weight_ids(model: nn.Module) -> set[int]:
    counts: Dict[int, int] = {}
    for module in model.modules():
        if isinstance(module, ResidualVQLinear):
            continue  # Do not materialize the decoded compatibility property.
        for attribute in ("weight", "gate_up_proj", "down_proj"):
            weight = module._parameters.get(attribute)
            if isinstance(weight, torch.Tensor):
                key = id(weight)
                counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _quantize_fused_experts(
    child: nn.Module,
    name: str,
    config: QuantizationConfig,
    include: Tuple[str, ...],
    exclude: Tuple[str, ...],
    shared: set[int],
):
    encoded_projections: Dict[str, Optional[QuantizedWeight]] = {}
    reports: List[LayerQuantizationReport] = []
    for attribute, module_type in (
        ("gate_up_proj", "fused_expert_gate_up"),
        ("down_proj", "fused_expert_down"),
    ):
        parameter = getattr(child, attribute)
        projection_name = f"{name}.{attribute}"
        encoded_projections[attribute] = None
        eligible = (
            (_matches(name, include) or _matches(projection_name, include))
            and not _matches(name, exclude)
            and not _matches(projection_name, exclude)
        )
        if not eligible:
            continue
        source_weight = parameter.reshape(-1, parameter.shape[-1])
        shape = tuple(map(int, source_weight.shape))
        original_bytes = parameter.numel() * parameter.element_size()
        if id(parameter) in shared:
            reports.append(
                LayerQuantizationReport(
                    name=projection_name,
                    shape=shape,
                    selected=False,
                    reason="shared_or_tied_weight",
                    original_bytes=original_bytes,
                    storage_bytes=original_bytes,
                    module_type=module_type,
                )
            )
            continue
        encoded, measurement = quantize_weight(
            source_weight,
            config,
            seed=_stable_seed(projection_name, config.seed),
        )
        if encoded is None:
            reports.append(
                LayerQuantizationReport(
                    name=projection_name,
                    shape=shape,
                    selected=False,
                    reason=str(measurement["reason"]),
                    original_bytes=original_bytes,
                    storage_bytes=original_bytes,
                    relative_error=measurement.get("relative_error"),
                    selected_stages=int(measurement.get("selected_stages", 0)),
                    module_type=module_type,
                )
            )
            continue
        encoded_projections[attribute] = encoded
        reports.append(
            LayerQuantizationReport(
                name=projection_name,
                shape=shape,
                selected=True,
                reason="admitted",
                original_bytes=original_bytes,
                storage_bytes=encoded.storage_bytes,
                relative_error=float(encoded.metrics["relative_error"]),
                selected_stages=encoded.stages,
                nominal_bits_per_weight=encoded.nominal_bits_per_weight,
                measured_bits_per_weight=encoded.measured_bits_per_weight,
                module_type=module_type,
            )
        )

    if all(value is None for value in encoded_projections.values()):
        return None, reports

    def projection(attribute: str):
        encoded = encoded_projections[attribute]
        if encoded is None:
            return getattr(child, attribute)
        parameter = getattr(child, attribute)
        return ResidualVQLinear.from_encoded(
            encoded,
            bias=None,
            source_dtype=parameter.dtype,
            device=parameter.device,
        )

    replacement = ResidualVQFusedExperts(
        int(child.num_experts),
        int(child.hidden_dim),
        int(child.intermediate_dim),
        gate_up_proj=projection("gate_up_proj"),
        down_proj=projection("down_proj"),
        act_fn=_expert_activation(child),
        gate_limit=_expert_gate_limit(child),
    )
    return replacement, reports


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig = QuantizationConfig(),
    *,
    include: Iterable[str] = ("*",),
    exclude: Iterable[str] = ("lm_head", "*.lm_head"),
    preserve_shared_weights: bool = True,
) -> Tuple[nn.Module, ModelQuantizationReport]:
    """Replace exact Linear, Conv1D, and allowlisted fused-expert descendants.

    Unknown Linear subclasses remain dense to preserve custom forward contracts.
    Shared/tied weights remain untouched by default. This prevents an encoded
    embedding or output head from being silently detached from its counterpart.
    Fused expert projections have independent admission and dense fallback.
    """

    include = tuple(include)
    exclude = tuple(exclude)
    reports: List[LayerQuantizationReport] = []
    shared = _shared_weight_ids(model) if preserve_shared_weights else set()

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (ResidualVQLinear, ResidualVQFusedExperts)):
                continue
            kind = _layer_kind(child)
            if kind is None:
                visit(child, name)
                continue
            if kind == "fused_experts":
                replacement, fused_reports = _quantize_fused_experts(
                    child, name, config, include, exclude, shared
                )
                reports.extend(fused_reports)
                if replacement is not None:
                    setattr(parent, child_name, replacement)
                continue
            eligible = _matches(name, include) and not _matches(name, exclude)
            if not eligible:
                continue
            source_weight = child.weight if kind == "linear" else child.weight.T
            shape = tuple(map(int, source_weight.shape))
            original_bytes = child.weight.numel() * child.weight.element_size()
            if id(child.weight) in shared:
                reports.append(
                    LayerQuantizationReport(
                        name=name,
                        shape=shape,
                        selected=False,
                        reason="shared_or_tied_weight",
                        original_bytes=original_bytes,
                        storage_bytes=original_bytes,
                        module_type=kind,
                    )
                )
                continue
            encoded, measurement = quantize_weight(
                source_weight,
                config,
                seed=_stable_seed(name, config.seed),
            )
            if encoded is None:
                reports.append(
                    LayerQuantizationReport(
                        name=name,
                        shape=shape,
                        selected=False,
                        reason=str(measurement["reason"]),
                        original_bytes=original_bytes,
                        storage_bytes=original_bytes,
                        relative_error=measurement.get("relative_error"),
                        selected_stages=int(measurement.get("selected_stages", 0)),
                        module_type=kind,
                    )
                )
                continue
            replacement_type = ResidualVQLinear if kind == "linear" else ResidualVQConv1D
            replacement = replacement_type.from_encoded(
                encoded,
                bias=child.bias,
                source_dtype=child.weight.dtype,
                device=child.weight.device,
            )
            setattr(parent, child_name, replacement)
            reports.append(
                LayerQuantizationReport(
                    name=name,
                    shape=shape,
                    selected=True,
                    reason="admitted",
                    original_bytes=original_bytes,
                    storage_bytes=encoded.storage_bytes,
                    relative_error=float(encoded.metrics["relative_error"]),
                    selected_stages=encoded.stages,
                    nominal_bits_per_weight=encoded.nominal_bits_per_weight,
                    measured_bits_per_weight=encoded.measured_bits_per_weight,
                    module_type=kind,
                )
            )

    visit(model, "")
    return model, ModelQuantizationReport(
        reports,
        configuration=config.to_dict(),
        include_globs=include,
        exclude_globs=exclude,
        preserve_shared_weights=preserve_shared_weights,
    )


def _linear_manifest(module: ResidualVQLinear) -> Dict[str, object]:
    return {
        "type": (
            "residual_vq_conv1d"
            if isinstance(module, ResidualVQConv1D)
            else "residual_vq_linear"
        ),
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
        "block_size": module.block_size,
        "bits_per_stage": module.bits_per_stage,
        "stages": module.stages,
        "codec": module.codec,
        "source_dtype": str(module.source_dtype).replace("torch.", ""),
        **(
            {
                "gain_bits": module.gain_bits,
                "rotation_reflections": module.rotation_reflections,
                "rotation_seed": module.rotation_seed,
                "adaptive_stages": module.adaptive_stages,
                "active_counts": [
                    module.encoded().active_count(stage) for stage in range(module.stages)
                ],
            }
            if module.codec == "leech24_shell2_residual"
            else {}
        ),
    }


def _expert_projection_manifest(projection) -> Dict[str, object]:
    if isinstance(projection, ResidualVQLinear):
        return _linear_manifest(projection)
    if isinstance(projection, torch.Tensor):
        return {"type": "dense_parameter", "shape": list(projection.shape)}
    raise TypeError("Unsupported fused-expert projection type")


def quantization_manifest(model: nn.Module) -> Dict[str, object]:
    """Describe every replacement needed to reconstruct a meta HF model."""

    layers = {}
    fused_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in fused_prefixes):
            continue
        if isinstance(module, ResidualVQFusedExperts):
            fused_prefixes.append(f"{name}.")
            layers[name] = {
                "type": "residual_vq_fused_experts",
                "num_experts": module.num_experts,
                "hidden_dim": module.hidden_dim,
                "intermediate_dim": module.intermediate_dim,
                "gate_limit": module.gate_limit,
                "gate_up_proj": _expert_projection_manifest(module.gate_up_proj),
                "down_proj": _expert_projection_manifest(module.down_proj),
            }
        elif isinstance(module, ResidualVQLinear):
            layers[name] = _linear_manifest(module)
    return {
        "format": "ojasx-vector-quantization-runner",
        "schema_version": 1,
        "codec": "adaptive_multi_stage_residual_vq",
        "layers": layers,
    }


def _validated_source_dtype(specification, weight: torch.Tensor, name: str) -> torch.dtype:
    expected = specification.get("source_dtype")
    actual = str(weight.dtype).replace("torch.", "")
    if expected is not None and expected != actual:
        raise ValueError(f"Source dtype mismatch for {name!r}: {expected} != {actual}")
    return weight.dtype


def _codec_restore_options(specification: Dict[str, object]) -> Dict[str, object]:
    codec = str(specification.get("codec", "adaptive_residual_vq"))
    return {
        "codec": codec,
        "gain_bits": int(specification.get("gain_bits", 0)),
        "rotation_reflections": int(specification.get("rotation_reflections", 0)),
        "rotation_seed": int(specification.get("rotation_seed", 20261104)),
        "adaptive_stages": bool(specification.get("adaptive_stages", True)),
        "active_counts": (
            tuple(map(int, specification["active_counts"]))
            if "active_counts" in specification
            else None
        ),
    }


def restore_vq_modules(model: nn.Module, manifest: Dict[str, object]) -> nn.Module:
    """Instantiate encoded modules inside a meta model before shard loading."""

    if manifest.get("format") != "ojasx-vector-quantization-runner":
        raise ValueError("Not an OJASX Vector Quantization Runner manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported quantization manifest schema: {manifest.get('schema_version')}")
    layers = manifest.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Quantization manifest has no layer mapping")
    existing = dict(model.named_modules())

    def resolve_parent(path: str) -> Tuple[nn.Module, str]:
        if "." not in path:
            return model, path
        parent_path, child_name = path.rsplit(".", 1)
        if parent_path not in existing:
            raise ValueError(f"Missing module parent {parent_path!r}")
        parent = existing[parent_path]
        if not isinstance(parent, nn.Module):
            raise TypeError(f"Invalid parent module {parent_path!r}")
        return parent, child_name

    for name, specification in layers.items():
        if name not in existing:
            raise ValueError(f"Model architecture does not contain {name!r}")
        source = existing[name]
        encoded_type = specification.get("type")
        if encoded_type == "residual_vq_fused_experts":
            if _layer_kind(source) != "fused_experts":
                raise TypeError(f"Manifest target {name!r} is not a supported expert bank")
            dimensions = (
                int(specification["num_experts"]),
                int(specification["hidden_dim"]),
                int(specification["intermediate_dim"]),
            )
            if dimensions != (
                int(source.num_experts),
                int(source.hidden_dim),
                int(source.intermediate_dim),
            ):
                raise ValueError(f"Expert-bank shape mismatch for {name!r}")
            source_gate_limit = _expert_gate_limit(source)
            manifest_gate_limit = specification.get("gate_limit")
            if source_gate_limit != manifest_gate_limit:
                raise ValueError(f"Expert-gate contract mismatch for {name!r}")

            def restore_projection(attribute: str):
                projection_spec = specification[attribute]
                source_parameter = getattr(source, attribute)
                projection_type = projection_spec.get("type")
                if projection_type == "dense_parameter":
                    if tuple(projection_spec["shape"]) != tuple(source_parameter.shape):
                        raise ValueError(f"Dense expert shape mismatch for {name}.{attribute}")
                    return source_parameter
                if projection_type != "residual_vq_linear" or projection_spec.get("bias"):
                    raise ValueError(f"Invalid encoded expert projection for {name}.{attribute}")
                rows_per_expert = (
                    2 * int(source.intermediate_dim)
                    if attribute == "gate_up_proj"
                    else int(source.hidden_dim)
                )
                columns = (
                    int(source.hidden_dim)
                    if attribute == "gate_up_proj"
                    else int(source.intermediate_dim)
                )
                expected = (int(source.num_experts) * rows_per_expert, columns)
                actual = (
                    int(projection_spec["out_features"]),
                    int(projection_spec["in_features"]),
                )
                if actual != expected:
                    raise ValueError(f"Encoded expert shape mismatch for {name}.{attribute}")
                return ResidualVQLinear(
                    columns,
                    expected[0],
                    False,
                    block_size=int(projection_spec["block_size"]),
                    bits_per_stage=int(projection_spec["bits_per_stage"]),
                    stages=int(projection_spec["stages"]),
                    source_dtype=_validated_source_dtype(
                        projection_spec, source_parameter, f"{name}.{attribute}"
                    ),
                    device=source_parameter.device,
                    **_codec_restore_options(projection_spec),
                )

            replacement = ResidualVQFusedExperts(
                *dimensions,
                gate_up_proj=restore_projection("gate_up_proj"),
                down_proj=restore_projection("down_proj"),
                act_fn=_expert_activation(source),
                gate_limit=source_gate_limit,
            )
        else:
            expected_kind = {
                "residual_vq_linear": "linear",
                "residual_vq_conv1d": "conv1d",
            }.get(encoded_type)
            if expected_kind is None:
                raise ValueError(f"Unknown encoded layer type for {name!r}")
            if _layer_kind(source) != expected_kind:
                raise TypeError(f"Manifest target {name!r} has an incompatible module type")
            expected = (int(specification["out_features"]), int(specification["in_features"]))
            if expected_kind == "linear":
                actual = (int(source.weight.shape[0]), int(source.weight.shape[1]))
            else:
                actual = (int(source.weight.shape[1]), int(source.weight.shape[0]))
            if expected != actual:
                raise ValueError(f"Shape mismatch for {name!r}: {expected} != {actual}")
            replacement_type = (
                ResidualVQLinear if expected_kind == "linear" else ResidualVQConv1D
            )
            replacement = replacement_type(
                int(specification["in_features"]),
                int(specification["out_features"]),
                source.bias is not None,
                block_size=int(specification["block_size"]),
                bits_per_stage=int(specification["bits_per_stage"]),
                stages=int(specification["stages"]),
                source_dtype=_validated_source_dtype(specification, source.weight, name),
                device=source.weight.device,
                **_codec_restore_options(specification),
            )
        parent, child_name = resolve_parent(name)
        setattr(parent, child_name, replacement)
        existing[name] = replacement
    return model

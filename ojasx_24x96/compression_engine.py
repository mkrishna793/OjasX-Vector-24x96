"""M11: Universal Compress-Once Engine (Engine 1) & Ring-Buffer OQPack Compiler.

Provides a unified, model-agnostic compression engine that converts any model
(in-memory nn.Module, local Hugging Face checkpoint directory, or remote HF Hub
repository) into a self-contained, memory-mapped `.oqpack` container or streamed
checkpoint directory:

1. Supports 2D linear projections, Conv1D projections, and 3D routed MoE expert
   banks (`[E, 2*I, H]` / `[E, H, I]`) via structural duck-typing without any
   model-specific hardcoding.
2. Applies role-aware dual-codec routing:
   - Sensitive attention, recurrent/DeltaNet, vision, and shared-expert layers
     can use 24D Leech Lattice (Lambda_24) shell quantization (~1.71 bpw).
   - Large FFN and routed MoE expert banks use multi-stage Adaptive RVQ (~1.0 bpw).
3. Supports ring-buffer shard-by-shard streaming so 100B+ models compile into
   real binary `.safetensors` shards on disk with bounded RAM and disk usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import safetensors.torch
from safetensors import safe_open
import torch
import torch.nn as nn

from .adapters import classify_tensor_role
from .codec import QuantizationConfig, quantize_weight
from .contract import CODEC_SCHEMA_VERSION, introspect_architecture
from .ingest import (
    StreamingConversionReport,
    discover_checkpoint_safetensors,
    should_quantize_tensor_name,
    stream_convert_checkpoint,
)
from .ir import TensorRole
from .leech_codec import LeechQuantizationConfig, quantize_leech_weight
from .math_engine import SmoothChannelConditioner
from .module import (
    ModelQuantizationReport,
    ResidualVQConv1D,
    ResidualVQFusedExperts,
    ResidualVQLinear,
    _expert_activation,
    _expert_gate_limit,
    _infer_fused_expert_dims,
    _layer_kind,
    _shared_weight_ids,
    _stable_seed,
    quantize_model,
)
from .oqpack import (
    MANIFEST_FILENAME,
    OQPACK_FORMAT_VERSION,
    OQPackManifest,
    OQPackShardEntry,
    OQPackTensorRecord,
    save_oqpack,
)


@dataclass
class UniversalCompressionConfig:
    """Configuration for the Universal Compress-Once Engine."""

    mode: str = "hybrid_leech_rvq"  # "adaptive_residual_vq", "leech24_shell2_residual", "hybrid_leech_rvq"
    vq_config: QuantizationConfig = field(
        default_factory=lambda: QuantizationConfig(
            block_size=24,
            bits_per_stage=8,
            max_stages=3,
            max_relative_error=None,
            max_codebook_samples=2048,
            kmeans_iterations=8,
            require_compression=False,
        )
    )
    leech_config: LeechQuantizationConfig = field(
        default_factory=lambda: LeechQuantizationConfig(
            max_stages=2,
            gain_bits=2,
            rotation_reflections=2,
            adaptive_stages=True,
            max_relative_error=None,
            require_compression=False,
        )
    )
    smooth_channels: bool = False
    smooth_alpha: float = 0.5
    include_globs: Tuple[str, ...] = ("*",)
    exclude_globs: Tuple[str, ...] = (
        "*embed*",
        "*lm_head*",
        "*norm*",
        "*.bias",
        "*.gate.weight",
        "*.shared_expert_gate.weight",
    )
    max_shard_bytes: int = 512 * 1024 * 1024
    allow_3d_experts: bool = True
    device: str = "cpu"


@dataclass
class UniversalCompressionResult:
    """Summary artifact produced by UniversalCompressionEngine."""

    model_id: str
    output_dir: str
    total_tensors: int
    quantized_tensors: int
    passthrough_tensors: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    effective_bpw: float
    elapsed_sec: float
    manifest_path: str
    layer_summaries: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "output_dir": self.output_dir,
            "total_tensors": self.total_tensors,
            "quantized_tensors": self.quantized_tensors,
            "passthrough_tensors": self.passthrough_tensors,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": round(self.compression_ratio, 4),
            "effective_bpw": round(self.effective_bpw, 4),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "manifest_path": self.manifest_path,
            "layer_summaries": self.layer_summaries,
        }


def select_codec_for_tensor(name: str, shape: Tuple[int, ...], mode: str) -> str:
    """Selects Leech24 vs Adaptive RVQ based on structural tensor role."""
    if mode in {"adaptive_residual_vq", "leech24_shell2_residual"}:
        return mode
    lowered = name.lower()
    # In hybrid mode, sensitive attention / linear_attn / shared_expert projections use Leech24
    # while massive routed MoE 3D expert banks and standard FFN use Adaptive RVQ
    if len(shape) == 3:
        return "adaptive_residual_vq"
    if any(
        tok in lowered
        for tok in (
            "self_attn",
            "attention",
            "attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "linear_attn",
            "gated_delta",
            "shared_expert",
            "vision",
            "audio",
        )
    ):
        return "leech24_shell2_residual"
    return "adaptive_residual_vq"


class UniversalCompressionEngine:
    """Engine 1: Universal Compress-Once Engine for any PyTorch or Hugging Face model."""

    def __init__(self, config: Optional[UniversalCompressionConfig] = None) -> None:
        self.config = config or UniversalCompressionConfig()

    def compress_module_in_place(
        self,
        model: nn.Module,
        *,
        calibration_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[nn.Module, ModelQuantizationReport]:
        """Quantize any in-memory nn.Module (2D Linear, Conv1D, 3D Fused Experts) in-place."""
        cfg = self.config
        shared = _shared_weight_ids(model)
        reports = []

        def _encode_matrix(
            mat: torch.Tensor,
            tensor_name: str,
            orig_shape: Tuple[int, ...],
        ):
            codec_choice = select_codec_for_tensor(tensor_name, orig_shape, cfg.mode)
            work_mat = mat
            if cfg.smooth_channels:
                cond = SmoothChannelConditioner(mat.shape[-1], alpha=cfg.smooth_alpha).to(
                    mat.device
                )
                act = (
                    calibration_activations.get(tensor_name)
                    if calibration_activations
                    else None
                )
                cond.fit(mat, act)
                work_mat = cond.smooth_weight(mat)
            if codec_choice == "leech24_shell2_residual":
                leech_cfg = replace(
                    cfg.leech_config,
                    seed=_stable_seed(tensor_name, cfg.leech_config.seed),
                )
                return quantize_leech_weight(
                    work_mat,
                    leech_cfg,
                ), codec_choice
            return quantize_weight(
                work_mat,
                cfg.vq_config,
                seed=_stable_seed(tensor_name, cfg.vq_config.seed),
            ), codec_choice

        def _visit(parent: nn.Module, prefix: str) -> None:
            for child_name, child in list(parent.named_children()):
                name = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, (ResidualVQLinear, ResidualVQFusedExperts)):
                    continue
                kind = _layer_kind(child)
                if kind is None:
                    # Also support duck-typed 2D linear layers
                    from .adapters import _is_duck_typed_linear

                    if _is_duck_typed_linear(child):
                        kind = "linear"
                    else:
                        _visit(child, name)
                        continue

                if kind == "fused_experts":
                    dims = _infer_fused_expert_dims(child)
                    if dims is None:
                        continue
                    encoded_projs = {}
                    for attr, mtype in (
                        ("gate_up_proj", "fused_expert_gate_up"),
                        ("down_proj", "fused_expert_down"),
                    ):
                        param = getattr(child, attr)
                        proj_name = f"{name}.{attr}"
                        encoded_projs[attr] = None
                        if not should_quantize_tensor_name(
                            proj_name,
                            tuple(param.shape),
                            include_globs=cfg.include_globs,
                            exclude_globs=cfg.exclude_globs,
                            allow_3d_experts=cfg.allow_3d_experts,
                        ):
                            continue
                        if id(param) in shared:
                            continue
                        flat_w = param.reshape(-1, param.shape[-1])
                        (enc, meas), _ = _encode_matrix(
                            flat_w, proj_name, tuple(param.shape)
                        )
                        if enc is not None:
                            encoded_projs[attr] = ResidualVQLinear.from_encoded(
                                enc,
                                bias=None,
                                source_dtype=param.dtype,
                                device=param.device,
                            )
                            from .module import LayerQuantizationReport

                            reports.append(
                                LayerQuantizationReport(
                                    name=proj_name,
                                    shape=tuple(flat_w.shape),
                                    selected=True,
                                    reason="admitted",
                                    original_bytes=param.numel() * param.element_size(),
                                    storage_bytes=enc.storage_bytes,
                                    relative_error=float(enc.metrics.get("relative_error", 0.0)),
                                    selected_stages=enc.stages,
                                    nominal_bits_per_weight=enc.nominal_bits_per_weight,
                                    measured_bits_per_weight=enc.measured_bits_per_weight,
                                    module_type=mtype,
                                )
                            )
                    if any(v is not None for v in encoded_projs.values()):
                        rep = ResidualVQFusedExperts(
                            dims[0],
                            dims[1],
                            dims[2],
                            gate_up_proj=encoded_projs["gate_up_proj"]
                            or getattr(child, "gate_up_proj"),
                            down_proj=encoded_projs["down_proj"]
                            or getattr(child, "down_proj"),
                            act_fn=_expert_activation(child),
                            gate_limit=_expert_gate_limit(child),
                        )
                        setattr(parent, child_name, rep)
                    continue

                w_name = f"{name}.weight"
                weight = child.weight
                if not should_quantize_tensor_name(
                    w_name,
                    tuple(weight.shape),
                    include_globs=cfg.include_globs,
                    exclude_globs=cfg.exclude_globs,
                    allow_3d_experts=False,
                ):
                    continue
                if id(weight) in shared:
                    continue
                source_w = weight if kind == "linear" else weight.T
                (enc, meas), _ = _encode_matrix(source_w, w_name, tuple(source_w.shape))
                if enc is not None:
                    rep_cls = ResidualVQLinear if kind == "linear" else ResidualVQConv1D
                    rep = rep_cls.from_encoded(
                        enc,
                        bias=getattr(child, "bias", None),
                        source_dtype=weight.dtype,
                        device=weight.device,
                    )
                    setattr(parent, child_name, rep)
                    from .module import LayerQuantizationReport

                    reports.append(
                        LayerQuantizationReport(
                            name=name,
                            shape=tuple(source_w.shape),
                            selected=True,
                            reason="admitted",
                            original_bytes=weight.numel() * weight.element_size(),
                            storage_bytes=enc.storage_bytes,
                            relative_error=float(enc.metrics.get("relative_error", 0.0)),
                            selected_stages=enc.stages,
                            nominal_bits_per_weight=enc.nominal_bits_per_weight,
                            measured_bits_per_weight=enc.measured_bits_per_weight,
                            module_type=kind,
                        )
                    )

        _visit(model, "")
        return model, ModelQuantizationReport(
            reports,
            configuration={"mode": cfg.mode},
            include_globs=cfg.include_globs,
            exclude_globs=cfg.exclude_globs,
            preserve_shared_weights=True,
        )

    def compress_model_to_oqpack(
        self,
        model: nn.Module,
        output_dir: Union[str, Path],
        *,
        model_id: str = "universal-model",
        tokenizer_dir: Optional[Union[str, Path]] = None,
        chat_template: Optional[str] = None,
    ) -> Tuple[nn.Module, OQPackManifest, UniversalCompressionResult]:
        """Quantize an in-memory model and persist it into a verified `.oqpack` container."""
        t0 = time.perf_counter()
        orig_bytes = sum(
            t.numel() * t.element_size()
            for t in list(model.parameters()) + list(model.buffers())
        )
        total_tensors = len(model.state_dict())

        q_model, q_report = self.compress_module_in_place(model)
        manifest = save_oqpack(
            q_model,
            output_dir,
            model_id=model_id,
            max_shard_bytes=self.config.max_shard_bytes,
            tokenizer_dir=tokenizer_dir,
            chat_template=chat_template,
        )
        comp_bytes = sum(s.size_bytes for s in manifest.shards)
        elapsed = time.perf_counter() - t0
        ratio = orig_bytes / max(comp_bytes, 1)
        total_params = max(orig_bytes // 2, 1)
        bpw = (comp_bytes * 8.0) / total_params

        result = UniversalCompressionResult(
            model_id=model_id,
            output_dir=str(output_dir),
            total_tensors=total_tensors,
            quantized_tensors=q_report.selected_layers,
            passthrough_tensors=max(0, total_tensors - q_report.selected_layers),
            original_bytes=orig_bytes,
            compressed_bytes=comp_bytes,
            compression_ratio=ratio,
            effective_bpw=bpw,
            elapsed_sec=elapsed,
            manifest_path=str(Path(output_dir) / MANIFEST_FILENAME),
            layer_summaries=[r.to_dict() for r in q_report.layers],
        )
        return q_model, manifest, result

    def stream_compress_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        resume: bool = True,
    ) -> StreamingConversionReport:
        """Stream-compress a local safetensors checkpoint (2D + 3D MoE experts) with bounded RAM."""
        codec = (
            "leech24_shell2_residual"
            if self.config.mode == "leech24_shell2_residual"
            else "adaptive_residual_vq"
        )
        return stream_convert_checkpoint(
            checkpoint_dir,
            output_dir,
            codec=codec,
            vq_config=self.config.vq_config,
            leech_config=self.config.leech_config,
            include_globs=self.config.include_globs,
            exclude_globs=self.config.exclude_globs,
            resume=resume,
            allow_3d_experts=self.config.allow_3d_experts,
        )


__all__ = [
    "UniversalCompressionConfig",
    "UniversalCompressionEngine",
    "UniversalCompressionResult",
    "select_codec_for_tensor",
]

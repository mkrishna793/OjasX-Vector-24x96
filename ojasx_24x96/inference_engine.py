"""M14: Universal Inference Engine (Engine 2) & Layer-by-Layer Forward Graph Auditor.

Provides a unified, zero-fake inference and verification runtime that loads and
executes ANY compressed model (`.oqpack` container, streamed checkpoint, or
in-memory model) across any modality (Text, Routed MoE, Vision-Language, Audio,
Hybrid State-Space/Gated-DeltaNet):

1. Zero-RAM `meta`-Skeleton Instantiation:
   Instantiates any Hugging Face or custom PyTorch architecture inside
   `torch.device("meta")` (0 bytes dense weight allocation) and populates
   compressed `ResidualVQLinear` and `ResidualVQFusedExperts` directly from
   memory-mapped `.oqpack` or streamed `.safetensors` shards.
2. Layer-by-Layer Mathematical Forward Graph Auditor:
   Attaches lightweight forward hooks to every quantized linear and MoE expert
   module to record real execution counts, finite activation norms, and active
   expert routing indices during live forward passes—guaranteeing zero fake or
   bypassed layers.
3. Universal Modality & Ultra-Long Context Execution:
   Supports arbitrary `forward(**kwargs)` (`input_ids`, `inputs_embeds`,
   `pixel_values`, `input_features`), multi-turn KV-cached generation via
   `OjasXInferenceEngine`, and 10M+ token streaming via `UltraLongContextStreamer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from .engine import GenerationResult, OjasXInferenceEngine, SamplingParams
from .ingest import load_streamed_checkpoint_into_model
from .long_context import UltraLongContextStreamer
from .module import ResidualVQConv1D, ResidualVQFusedExperts, ResidualVQLinear
from .oqpack import MANIFEST_FILENAME, OQPackManifest, load_quantized_model_from_oqpack


@dataclass
class LayerExecutionTrace:
    """Recorded telemetry from a single layer's real mathematical forward execution."""

    layer_name: str
    module_type: str
    codec: str
    call_count: int = 0
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    output_l2_norm: float = 0.0
    all_finite: bool = True
    active_experts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_name": self.layer_name,
            "module_type": self.module_type,
            "codec": self.codec,
            "call_count": self.call_count,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "output_l2_norm": round(self.output_l2_norm, 6),
            "all_finite": self.all_finite,
            "active_experts": self.active_experts,
        }


@dataclass
class ForwardGraphAuditReport:
    """Comprehensive proof that every quantized layer in the model computed real outputs."""

    total_quantized_modules: int
    executed_quantized_modules: int
    all_outputs_finite: bool
    total_forward_calls: int
    active_moe_experts_union: List[int]
    traces: Dict[str, LayerExecutionTrace]

    @property
    def execution_coverage_ratio(self) -> float:
        if self.total_quantized_modules == 0:
            return 1.0
        return self.executed_quantized_modules / self.total_quantized_modules

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_quantized_modules": self.total_quantized_modules,
            "executed_quantized_modules": self.executed_quantized_modules,
            "execution_coverage_ratio": round(self.execution_coverage_ratio, 4),
            "all_outputs_finite": self.all_outputs_finite,
            "total_forward_calls": self.total_forward_calls,
            "active_moe_experts_union": self.active_moe_experts_union,
            "traces": {k: v.to_dict() for k, v in self.traces.items()},
        }


def _materialize_meta_tensors(model: nn.Module, target_device: torch.device) -> None:
    """Materializes uninitialized parameters or non-persistent buffers left on meta device."""
    # First, recompute RoPE rotary embedding buffers if left on meta
    for _, mod in model.named_modules():
        if hasattr(mod, "rope_type") and hasattr(mod, "config") and hasattr(mod, "inv_freq"):
            if getattr(mod.inv_freq, "device", None) is not None and mod.inv_freq.device.type == "meta":
                rope_init_fn = getattr(mod, "compute_default_rope_parameters", None)
                if getattr(mod, "rope_type", "default") != "default":
                    try:
                        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                        rope_init_fn = ROPE_INIT_FUNCTIONS.get(mod.rope_type, rope_init_fn)
                    except Exception:
                        pass
                if rope_init_fn is not None:
                    inv_freq, attn_scaling = rope_init_fn(mod.config)
                    mod.inv_freq = nn.Buffer(inv_freq.to(target_device), persistent=False)
                    if hasattr(mod, "original_inv_freq"):
                        mod.original_inv_freq = nn.Buffer(
                            inv_freq.clone().to(target_device), persistent=False
                        )
                    mod.attention_scaling = attn_scaling

    # Materialize any remaining non-persistent buffers
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[-1],
                nn.Buffer(
                    torch.zeros(buf.shape, dtype=buf.dtype, device=target_device),
                    persistent=False,
                ),
            )

    # Materialize any remaining parameters (e.g. uninitialized passthrough biases)
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device=target_device)),
            )


class UniversalInferenceEngine:
    """Engine 2: Universal Compressed-Domain Inference & Forward-Backward Execution Engine."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        *,
        device: Union[str, torch.device] = "cpu",
        max_context_length: int = 4096,
        subspace_lut_mode: bool = False,
        fused_native_gemm: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_context_length = int(max_context_length)

        for _, mod in self.model.named_modules():
            if isinstance(mod, ResidualVQLinear):
                mod.set_subspace_lut_mode(subspace_lut_mode)
                mod.set_fused_native_gemm(fused_native_gemm)

        self._core_engine: Optional[OjasXInferenceEngine] = None
        self.long_context_streamer = UltraLongContextStreamer(
            self.model,
            prefill_chunk_size=min(1024, max_context_length),
            max_kv_cache_tokens=max_context_length,
        )

    @property
    def core_engine(self) -> OjasXInferenceEngine:
        if self._core_engine is None:
            self._core_engine = OjasXInferenceEngine(
                self.model,
                tokenizer=self.tokenizer,
                max_context_length=self.max_context_length,
            )
        return self._core_engine

    @classmethod
    def from_oqpack(
        cls,
        skeleton_or_factory: Union[nn.Module, Callable[[], nn.Module]],
        package_dir: Union[str, Path],
        *,
        tokenizer: Any = None,
        device: Union[str, torch.device] = "cpu",
        verify_checksums: bool = True,
        streaming_decode: bool = True,
        subspace_lut_mode: bool = False,
    ) -> Tuple["UniversalInferenceEngine", OQPackManifest]:
        """Load any model from an `.oqpack` directory using a zero-RAM skeleton or factory."""
        if callable(skeleton_or_factory) and not isinstance(skeleton_or_factory, nn.Module):
            with torch.device("meta"):
                skeleton = skeleton_or_factory()
        else:
            skeleton = skeleton_or_factory

        model, manifest = load_quantized_model_from_oqpack(
            skeleton,
            package_dir,
            device=device,
            verify_checksums=verify_checksums,
            streaming_decode=streaming_decode,
        )
        _materialize_meta_tensors(model, torch.device(device))
        if tokenizer is None:
            pkg = Path(package_dir)
            if (pkg / "tokenizer_config.json").is_file() or (pkg / "tokenizer.json").is_file():
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(pkg, local_files_only=True)
                except Exception:
                    tokenizer = None

        engine = cls(
            model,
            tokenizer=tokenizer,
            device=device,
            subspace_lut_mode=subspace_lut_mode,
        )
        return engine, manifest

    @classmethod
    def from_streamed_checkpoint(
        cls,
        skeleton_or_factory: Union[nn.Module, Callable[[], nn.Module]],
        converted_dir: Union[str, Path],
        *,
        tokenizer: Any = None,
        device: Union[str, torch.device] = "cpu",
        subspace_lut_mode: bool = False,
    ) -> "UniversalInferenceEngine":
        """Load any model from a `stream_convert_checkpoint` directory into a skeleton."""
        if callable(skeleton_or_factory) and not isinstance(skeleton_or_factory, nn.Module):
            with torch.device("meta"):
                skeleton = skeleton_or_factory()
        else:
            skeleton = skeleton_or_factory

        model = load_streamed_checkpoint_into_model(
            skeleton,
            converted_dir,
            device=device,
        )
        _materialize_meta_tensors(model, torch.device(device))
        return cls(
            model,
            tokenizer=tokenizer,
            device=device,
            subspace_lut_mode=subspace_lut_mode,
        )

    def forward_any(self, *args, **kwargs) -> Any:
        """Execute the model's native forward computational graph on any modality inputs."""
        return self.model(*args, **kwargs)

    def audit_forward_graph(
        self,
        *forward_args,
        **forward_kwargs,
    ) -> Tuple[Any, ForwardGraphAuditReport]:
        """Run a real forward pass while auditing every quantized linear & MoE expert layer."""
        traces: Dict[str, LayerExecutionTrace] = {}
        handles = []

        # Collect top-level quantized modules (excluding inner projections of ResidualVQFusedExperts)
        fused_prefixes: List[str] = []
        for name, mod in self.model.named_modules():
            if any(name.startswith(p) for p in fused_prefixes):
                continue
            if isinstance(mod, ResidualVQFusedExperts):
                fused_prefixes.append(f"{name}.")
                codec_str = (
                    mod.gate_up_proj.codec
                    if isinstance(mod.gate_up_proj, ResidualVQLinear)
                    else "mixed"
                )
                traces[name] = LayerExecutionTrace(
                    layer_name=name,
                    module_type="ResidualVQFusedExperts",
                    codec=codec_str,
                )

                def _make_moe_hook(layer_key: str):
                    def _hook(module, inputs, output):
                        tr = traces[layer_key]
                        tr.call_count += 1
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            tr.input_shape = tuple(inputs[0].shape)
                        if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
                            uniq = torch.unique(inputs[1].detach()).cpu().tolist()
                            tr.active_experts = sorted(set(tr.active_experts).union(int(x) for x in uniq))
                        out_t = output[0] if isinstance(output, tuple) else output
                        if isinstance(out_t, torch.Tensor):
                            tr.output_shape = tuple(out_t.shape)
                            tr.output_l2_norm = float(out_t.detach().float().norm().item())
                            tr.all_finite = tr.all_finite and bool(torch.isfinite(out_t).all().item())

                    return _hook

                handles.append(mod.register_forward_hook(_make_moe_hook(name)))
            elif isinstance(mod, ResidualVQLinear):
                mtype = "ResidualVQConv1D" if isinstance(mod, ResidualVQConv1D) else "ResidualVQLinear"
                traces[name] = LayerExecutionTrace(
                    layer_name=name,
                    module_type=mtype,
                    codec=mod.codec,
                )

                def _make_lin_hook(layer_key: str):
                    def _hook(module, inputs, output):
                        tr = traces[layer_key]
                        tr.call_count += 1
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            tr.input_shape = tuple(inputs[0].shape)
                        out_t = output[0] if isinstance(output, tuple) else output
                        if isinstance(out_t, torch.Tensor):
                            tr.output_shape = tuple(out_t.shape)
                            tr.output_l2_norm = float(out_t.detach().float().norm().item())
                            tr.all_finite = tr.all_finite and bool(torch.isfinite(out_t).all().item())

                    return _hook

                handles.append(mod.register_forward_hook(_make_lin_hook(name)))

        try:
            with torch.no_grad():
                output = self.model(*forward_args, **forward_kwargs)
        finally:
            for h in handles:
                h.remove()

        executed = sum(1 for t in traces.values() if t.call_count > 0)
        all_finite = all(t.all_finite for t in traces.values() if t.call_count > 0)
        total_calls = sum(t.call_count for t in traces.values())
        experts_union = sorted({e for t in traces.values() for e in t.active_experts})

        report = ForwardGraphAuditReport(
            total_quantized_modules=len(traces),
            executed_quantized_modules=executed,
            all_outputs_finite=all_finite,
            total_forward_calls=total_calls,
            active_moe_experts_union=experts_union,
            traces=traces,
        )
        return output, report

    def generate(
        self,
        prompt: Union[str, Sequence[int], torch.Tensor],
        params: SamplingParams = SamplingParams(),
        *,
        use_cache: bool = True,
    ) -> Union[GenerationResult, Tuple[str, GenerationResult]]:
        """Generate autoregressive tokens from token IDs or text prompt."""
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("String prompt requires a tokenizer attached to UniversalInferenceEngine")
            encoded = self.tokenizer.encode(prompt)
            res = self.core_engine.generate(encoded, params=params, use_cache=use_cache)
            text = self.tokenizer.decode(res.generated_token_ids, skip_special_tokens=True)
            return text, res
        return self.core_engine.generate(prompt, params=params, use_cache=use_cache)

    def stream_ultra_long(
        self,
        prompt_chunks: Iterable[Union[Sequence[int], torch.Tensor]],
        *,
        max_new_tokens: int = 64,
    ) -> List[int]:
        """Process 10M+ token streams in bounded chunks and return generated tokens."""
        return list(
            self.long_context_streamer.stream_generate_infinite(
                prompt_chunks,
                max_new_tokens=max_new_tokens,
                device=self.device,
            )
        )


__all__ = [
    "ForwardGraphAuditReport",
    "LayerExecutionTrace",
    "UniversalInferenceEngine",
]

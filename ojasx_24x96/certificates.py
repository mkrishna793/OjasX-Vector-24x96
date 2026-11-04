"""M3 Quality Certificates: Perplexity, forward KL, Top-1/Top-5 agreement, and long-context drift."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from .calibration import _disable_hf_cache, _forward, _model_device, _move_to_device
from .gates import _logits
from .ir import QuantizationPlan, TensorSpec
from .module import ResidualVQLinear


@dataclass(frozen=True)
class QualityCertificateThresholds:
    """Declared acceptance thresholds for M3 Quality Certificates."""

    max_perplexity_delta: float = 0.5
    max_kl_divergence: float = 0.08
    min_top1_agreement: float = 0.75
    min_top5_agreement: float = 0.90
    max_long_context_kl_drift: float = 0.12
    require_all_finite: bool = True

    def __post_init__(self) -> None:
        for val, name in (
            (self.max_perplexity_delta, "max_perplexity_delta"),
            (self.max_kl_divergence, "max_kl_divergence"),
            (self.min_top1_agreement, "min_top1_agreement"),
            (self.min_top5_agreement, "min_top5_agreement"),
            (self.max_long_context_kl_drift, "max_long_context_kl_drift"),
        ):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not (0 <= self.min_top1_agreement <= 1):
            raise ValueError("min_top1_agreement must be in [0, 1]")
        if not (0 <= self.min_top5_agreement <= 1):
            raise ValueError("min_top5_agreement must be in [0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_perplexity_delta": self.max_perplexity_delta,
            "max_kl_divergence": self.max_kl_divergence,
            "min_top1_agreement": self.min_top1_agreement,
            "min_top5_agreement": self.min_top5_agreement,
            "max_long_context_kl_drift": self.max_long_context_kl_drift,
            "require_all_finite": self.require_all_finite,
        }


@dataclass
class LayerSensitivityEntry:
    """Per-layer sensitivity and quantization summary."""

    name: str
    role: str
    selected_codec: str
    bits_per_weight: float
    relative_error: float
    empirical_activation_mse: Optional[float] = None

    def __init__(
        self,
        name: str,
        role: str,
        selected_codec: str,
        bits_per_weight: float,
        relative_error: float,
        empirical_activation_mse: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.role = role
        self.selected_codec = selected_codec
        self.bits_per_weight = bits_per_weight
        self.relative_error = relative_error
        if empirical_activation_mse is None and "Empirical_activation_mse" in kwargs:
            empirical_activation_mse = kwargs["Empirical_activation_mse"]
        self.empirical_activation_mse = empirical_activation_mse

    @property
    def Empirical_activation_mse(self) -> Optional[float]:
        return self.empirical_activation_mse

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "selected_codec": self.selected_codec,
            "bits_per_weight": self.bits_per_weight,
            "relative_error": self.relative_error,
            "empirical_activation_mse": self.empirical_activation_mse,
        }


@dataclass
class QualityCertificate:
    """M3 formal quality certification bundle."""

    certificate_version: str = "m3.certificate.v1"
    passed: bool = False
    failed_checks: List[str] = field(default_factory=list)
    baseline_perplexity: float = 1.0
    quantized_perplexity: float = 1.0
    perplexity_delta: float = 0.0
    mean_kl_divergence: float = 0.0
    max_kl_divergence: float = 0.0
    top1_agreement: float = 1.0
    top5_agreement: float = 1.0
    long_context_kl_divergence: Optional[float] = None
    long_context_length: Optional[int] = None
    compression_ratio: float = 1.0
    layer_sensitivities: List[LayerSensitivityEntry] = field(default_factory=list)
    thresholds: QualityCertificateThresholds = field(
        default_factory=QualityCertificateThresholds
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "certificate_version": self.certificate_version,
            "passed": self.passed,
            "failed_checks": list(self.failed_checks),
            "baseline_perplexity": self.baseline_perplexity,
            "quantized_perplexity": self.quantized_perplexity,
            "perplexity_delta": self.perplexity_delta,
            "mean_kl_divergence": self.mean_kl_divergence,
            "max_kl_divergence": self.max_kl_divergence,
            "top1_agreement": self.top1_agreement,
            "top5_agreement": self.top5_agreement,
            "long_context_kl_divergence": self.long_context_kl_divergence,
            "long_context_length": self.long_context_length,
            "compression_ratio": self.compression_ratio,
            "layer_sensitivities": [entry.to_dict() for entry in self.layer_sensitivities],
            "thresholds": self.thresholds.to_dict(),
        }


def _evaluate_batch_pair(
    baseline_model: nn.Module,
    quantized_model: nn.Module,
    batch: Union[torch.Tensor, Mapping[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
    base_device = _model_device(baseline_model)
    quant_device = _model_device(quantized_model)
    base_batch = _disable_hf_cache(baseline_model, _move_to_device(batch, base_device))
    quant_batch = _disable_hf_cache(quantized_model, _move_to_device(batch, quant_device))

    base_out = _forward(baseline_model, base_batch)
    quant_out = _forward(quantized_model, quant_batch)

    base_logits = _logits(base_out).float()
    quant_logits = _logits(quant_out).float().to(base_logits.device)

    input_ids = None
    attention_mask = None
    if isinstance(batch, Mapping):
        if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            input_ids = batch["input_ids"]
        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            attention_mask = batch["attention_mask"]
    elif isinstance(batch, torch.Tensor) and batch.dtype in (torch.int64, torch.int32, torch.long):
        input_ids = batch

    # Check for next-token CausalLM structure
    if (
        input_ids is not None
        and input_ids.ndim == 2
        and base_logits.ndim == 3
        and base_logits.shape[:2] == input_ids.shape
        and input_ids.shape[1] >= 2
    ):
        base_shift = base_logits[:, :-1, :].contiguous()
        quant_shift = quant_logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous().to(base_logits.device)
        if attention_mask is not None and attention_mask.shape == input_ids.shape:
            valid_mask = attention_mask[:, 1:].bool().to(base_logits.device)
        else:
            valid_mask = torch.ones(targets.shape, dtype=torch.bool, device=base_logits.device)

        base_flat = base_shift[valid_mask]
        quant_flat = quant_shift[valid_mask]
        target_flat = targets[valid_mask]
        return base_flat, quant_flat, target_flat, None, input_ids.shape[1]

    # Non-causal or general sequence/tensor
    base_flat = base_logits.reshape(-1, base_logits.shape[-1])
    quant_flat = quant_logits.reshape(-1, quant_logits.shape[-1])
    seq_len = base_logits.shape[1] if base_logits.ndim >= 2 else base_logits.shape[0]
    return base_flat, quant_flat, None, None, seq_len


def generate_quality_certificate(
    baseline_model: nn.Module,
    quantized_model: nn.Module,
    evaluation_batches: Sequence[Union[torch.Tensor, Mapping[str, torch.Tensor]]],
    *,
    long_context_batches: Optional[Sequence[Union[torch.Tensor, Mapping[str, torch.Tensor]]]] = None,
    plan: Optional[QuantizationPlan] = None,
    thresholds: QualityCertificateThresholds = QualityCertificateThresholds(),
) -> QualityCertificate:
    """Generate an M3 Quality Certificate comparing baseline and quantized models."""
    eval_batches = tuple(evaluation_batches)
    if not eval_batches:
        raise ValueError("generate_quality_certificate requires at least one evaluation batch")

    base_training = baseline_model.training
    quant_training = quantized_model.training
    baseline_model.eval()
    quantized_model.eval()

    total_tokens = 0
    total_base_nll = 0.0
    total_quant_nll = 0.0
    total_kl = 0.0
    max_kl = 0.0
    total_top1 = 0
    total_top5 = 0
    all_finite = True

    try:
        with torch.inference_mode():
            for batch in eval_batches:
                base_flat, quant_flat, target_flat, _, _ = _evaluate_batch_pair(
                    baseline_model, quantized_model, batch
                )
                if not (torch.isfinite(base_flat).all().item() and torch.isfinite(quant_flat).all().item()):
                    all_finite = False

                n_tokens = base_flat.shape[0]
                if n_tokens == 0:
                    continue
                total_tokens += n_tokens

                # NLL / Perplexity
                if target_flat is not None:
                    loss_base = F.cross_entropy(base_flat, target_flat, reduction="sum").item()
                    loss_quant = F.cross_entropy(quant_flat, target_flat, reduction="sum").item()
                else:
                    # Soft cross entropy / entropy proxy
                    p_base = F.softmax(base_flat, dim=-1)
                    log_p_base = F.log_softmax(base_flat, dim=-1)
                    log_p_quant = F.log_softmax(quant_flat, dim=-1)
                    loss_base = -(p_base * log_p_base).sum().item()
                    loss_quant = -(p_base * log_p_quant).sum().item()

                total_base_nll += loss_base
                total_quant_nll += loss_quant

                # Forward KL: D_KL(P_base || P_quant)
                log_p_base = F.log_softmax(base_flat, dim=-1)
                log_p_quant = F.log_softmax(quant_flat, dim=-1)
                kl_elem = (log_p_base.exp() * (log_p_base - log_p_quant)).sum(dim=-1).clamp(min=0.0)
                total_kl += kl_elem.sum().item()
                curr_max = kl_elem.max().item() if kl_elem.numel() else 0.0
                max_kl = max(max_kl, curr_max)

                # Top-1 and Top-5 agreement
                base_top1 = base_flat.argmax(dim=-1)
                quant_top1 = quant_flat.argmax(dim=-1)
                total_top1 += (base_top1 == quant_top1).sum().item()

                k = min(5, base_flat.shape[-1])
                quant_topk = torch.topk(quant_flat, k=k, dim=-1).indices
                total_top5 += (base_top1.unsqueeze(-1) == quant_topk).any(dim=-1).sum().item()

            # Long-context evaluation if provided
            long_context_kl: Optional[float] = None
            max_long_len: Optional[int] = None
            if long_context_batches:
                long_batches = tuple(long_context_batches)
                long_kl_sum = 0.0
                long_tok_count = 0
                max_seq = 0
                for batch in long_batches:
                    base_l, quant_l, _, _, seq_len = _evaluate_batch_pair(
                        baseline_model, quantized_model, batch
                    )
                    max_seq = max(max_seq, seq_len)
                    if base_l.shape[0] == 0:
                        continue
                    log_pb = F.log_softmax(base_l, dim=-1)
                    log_pq = F.log_softmax(quant_l, dim=-1)
                    kl_l = (log_pb.exp() * (log_pb - log_pq)).sum(dim=-1).clamp(min=0.0)
                    long_kl_sum += kl_l.sum().item()
                    long_tok_count += base_l.shape[0]

                if long_tok_count > 0:
                    long_context_kl = float(long_kl_sum / long_tok_count)
                    max_long_len = max_seq
    finally:
        baseline_model.train(base_training)
        quantized_model.train(quant_training)

    safe_toks = max(1, total_tokens)
    mean_base_nll = total_base_nll / safe_toks
    mean_quant_nll = total_quant_nll / safe_toks

    base_ppl = math.exp(min(20.0, mean_base_nll))
    quant_ppl = math.exp(min(20.0, mean_quant_nll))
    ppl_delta = quant_ppl - base_ppl
    mean_kl = total_kl / safe_toks
    top1 = total_top1 / safe_toks
    top5 = total_top5 / safe_toks

    # Compression ratio
    if plan is not None:
        dense_bytes = plan.base_model_state_bytes
        predicted_state = plan.memory.get("predicted_model_state_bytes", dense_bytes)
        compression_ratio = float(dense_bytes / max(1, predicted_state))
    else:
        base_bytes = sum(p.numel() * p.element_size() for p in baseline_model.parameters()) + sum(
            b.numel() * b.element_size() for b in baseline_model.buffers()
        )
        quant_bytes = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) + sum(
            b.numel() * b.element_size() for b in quantized_model.buffers()
        )
        compression_ratio = float(base_bytes / max(1, quant_bytes))

    # Layer sensitivities
    layer_sensitivities: List[LayerSensitivityEntry] = []
    if plan is not None:
        for spec in plan.specs:
            cand_id = plan.selected.get(spec.name, "dense")
            cand = next(
                (c for c in plan.candidates[spec.name] if c.candidate_id == cand_id),
                None,
            )
            if cand is not None:
                layer_sensitivities.append(
                    LayerSensitivityEntry(
                        name=spec.name,
                        role=spec.role.value if hasattr(spec.role, "value") else str(spec.role),
                        selected_codec=cand.codec,
                        bits_per_weight=float(cand.measured_bits_per_weight),
                        relative_error=float(cand.estimated_error),
                        empirical_activation_mse=(
                            float(cand.local_output_relative_error)
                            if cand.local_output_relative_error is not None
                            else None
                        ),
                    )
                )
    else:
        for name, module in quantized_model.named_modules():
            if isinstance(module, ResidualVQLinear) or hasattr(module, "codebooks"):
                bits = 0.0
                if hasattr(module, "stages") and hasattr(module, "bits_per_stage") and hasattr(module, "vector_count"):
                    total_el = (
                        module.out_features * module.in_features
                        if hasattr(module, "out_features")
                        else 1
                    )
                    bits = (module.stages * module.bits_per_stage * getattr(module, "vector_count", 0)) / max(1, total_el)
                layer_sensitivities.append(
                    LayerSensitivityEntry(
                        name=name,
                        role="attention" if "attn" in name or "self" in name else ("feed_forward" if "mlp" in name else "linear"),
                        selected_codec=getattr(module, "codec", "adaptive_residual_vq"),
                        bits_per_weight=float(bits),
                        relative_error=0.0,
                        empirical_activation_mse=None,
                    )
                )

    # Acceptance checks
    failed_checks: List[str] = []
    if thresholds.require_all_finite:
        if (
            not all_finite
            or not math.isfinite(base_ppl)
            or not math.isfinite(quant_ppl)
            or not math.isfinite(mean_kl)
            or not math.isfinite(max_kl)
        ):
            failed_checks.append("require_all_finite: non-finite outputs or metrics encountered")

    if ppl_delta > thresholds.max_perplexity_delta:
        failed_checks.append(
            f"max_perplexity_delta: {ppl_delta:.4f} > {thresholds.max_perplexity_delta:.4f}"
        )
    if mean_kl > thresholds.max_kl_divergence:
        failed_checks.append(
            f"max_kl_divergence: {mean_kl:.4f} > {thresholds.max_kl_divergence:.4f}"
        )
    if top1 < thresholds.min_top1_agreement:
        failed_checks.append(
            f"min_top1_agreement: {top1:.4f} < {thresholds.min_top1_agreement:.4f}"
        )
    if top5 < thresholds.min_top5_agreement:
        failed_checks.append(
            f"min_top5_agreement: {top5:.4f} < {thresholds.min_top5_agreement:.4f}"
        )
    if long_context_kl is not None and long_context_kl > thresholds.max_long_context_kl_drift:
        failed_checks.append(
            f"max_long_context_kl_drift: {long_context_kl:.4f} > {thresholds.max_long_context_kl_drift:.4f}"
        )

    return QualityCertificate(
        certificate_version="m3.certificate.v1",
        passed=len(failed_checks) == 0,
        failed_checks=failed_checks,
        baseline_perplexity=float(base_ppl),
        quantized_perplexity=float(quant_ppl),
        perplexity_delta=float(ppl_delta),
        mean_kl_divergence=float(mean_kl),
        max_kl_divergence=float(max_kl),
        top1_agreement=float(top1),
        top5_agreement=float(top5),
        long_context_kl_divergence=long_context_kl,
        long_context_length=max_long_len,
        compression_ratio=compression_ratio,
        layer_sensitivities=layer_sensitivities,
        thresholds=thresholds,
    )

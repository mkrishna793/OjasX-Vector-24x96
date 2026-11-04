"""End-to-end output quality gates for generalized quantization plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F

from .calibration import _disable_hf_cache, _forward, _model_device, _move_to_device


@dataclass(frozen=True)
class QualityGateThresholds:
    """Declared acceptance thresholds; at least one is required."""

    max_relative_logits_l2_error: Optional[float] = None
    max_absolute_logits_error: Optional[float] = None
    max_mean_kl: Optional[float] = None
    min_top1_agreement: Optional[float] = None
    max_mean_nll_delta: Optional[float] = None

    def __post_init__(self) -> None:
        values = (
            self.max_relative_logits_l2_error,
            self.max_absolute_logits_error,
            self.max_mean_kl,
            self.min_top1_agreement,
            self.max_mean_nll_delta,
        )
        if all(value is None for value in values):
            raise ValueError("At least one end-to-end quality threshold is required")
        for value in values:
            if value is not None and not math.isfinite(value):
                raise ValueError("Quality thresholds must be finite")
        for value in (
            self.max_relative_logits_l2_error,
            self.max_absolute_logits_error,
            self.max_mean_kl,
        ):
            if value is not None and value < 0:
                raise ValueError("Logit and KL thresholds must be nonnegative")
        if self.min_top1_agreement is not None and not 0 <= self.min_top1_agreement <= 1:
            raise ValueError("min_top1_agreement must be in [0, 1]")

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "max_relative_logits_l2_error": self.max_relative_logits_l2_error,
            "max_absolute_logits_error": self.max_absolute_logits_error,
            "max_mean_kl": self.max_mean_kl,
            "min_top1_agreement": self.min_top1_agreement,
            "max_mean_nll_delta": self.max_mean_nll_delta,
        }


@dataclass
class QualityGateResult:
    passed: bool
    metrics: Dict[str, object]
    failures: List[str]
    thresholds: QualityGateThresholds

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "failures": self.failures,
            "thresholds": self.thresholds.to_dict(),
        }


class QualityGateError(RuntimeError):
    def __init__(self, result: QualityGateResult):
        self.result = result
        super().__init__("Quality gates failed: " + "; ".join(result.failures))


def _logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Quality gates require a tensor output or an object with .logits")


def _selected_positions(
    batch,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    maximum: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Logit shape changed: {reference.shape} vs {candidate.shape}")
    if reference.ndim < 2:
        raise ValueError("Quality gates require batch/position and vocabulary dimensions")
    labels = None
    valid = torch.ones(reference.shape[:-1], dtype=torch.bool, device=reference.device)
    if isinstance(batch, Mapping) and isinstance(batch.get("input_ids"), torch.Tensor):
        input_ids = batch["input_ids"]
        if (
            reference.ndim == 3
            and input_ids.ndim == 2
            and reference.shape[:2] == input_ids.shape
            and input_ids.shape[1] >= 2
        ):
            valid = valid[:, 1:]
            reference = reference[:, :-1]
            candidate = candidate[:, :-1]
            labels = input_ids[:, 1:]
            attention = batch.get("attention_mask")
            if isinstance(attention, torch.Tensor) and attention.shape == input_ids.shape:
                valid = attention[:, 1:].bool().to(valid.device)
    reference = reference[valid]
    candidate = candidate[valid.to(candidate.device)]
    candidate = candidate.to(reference.device)
    if labels is not None:
        labels = labels[valid.to(labels.device)]
    if reference.shape[0] == 0:
        raise ValueError("No valid logit positions were available for quality gates")
    if maximum is not None and reference.shape[0] > maximum:
        indices = (
            torch.arange(maximum, dtype=torch.int64) * reference.shape[0]
        ) // maximum
        device_indices = indices.to(reference.device)
        reference = reference[device_indices]
        candidate = candidate[device_indices]
        if labels is not None:
            labels = labels[indices.to(labels.device)]
    return reference.float(), candidate.float(), labels, int(reference.shape[0])


@torch.inference_mode()
def evaluate_quality_gates(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    batches: Iterable[object],
    thresholds: QualityGateThresholds,
    *,
    max_positions_per_batch: Optional[int] = 256,
) -> QualityGateResult:
    """Compare models on supplied inputs and evaluate every declared threshold.

    With causal ``input_ids``, metrics use supervised next-token positions. Other
    models use all output positions. NLL is reported only when labels are available.
    """

    if max_positions_per_batch is not None and max_positions_per_batch <= 0:
        raise ValueError("max_positions_per_batch must be positive or None")
    batches = tuple(batches)
    if not batches:
        raise ValueError("Quality gates require at least one input batch")
    reference_training = reference_model.training
    candidate_training = candidate_model.training
    reference_model.eval()
    candidate_model.eval()
    totals = {
        "difference_squared": 0.0,
        "reference_squared": 0.0,
        "maximum_absolute": 0.0,
        "kl": 0.0,
        "positions": 0,
        "top1": 0,
        "reference_nll": 0.0,
        "candidate_nll": 0.0,
        "supervised_tokens": 0,
    }
    per_batch: List[Dict[str, object]] = []
    try:
        for index, original_batch in enumerate(batches):
            reference_batch = _disable_hf_cache(
                reference_model,
                _move_to_device(original_batch, _model_device(reference_model)),
            )
            candidate_batch = _disable_hf_cache(
                candidate_model,
                _move_to_device(original_batch, _model_device(candidate_model)),
            )
            reference_output = _logits(_forward(reference_model, reference_batch))
            candidate_output = _logits(_forward(candidate_model, candidate_batch))
            reference, candidate, labels, positions = _selected_positions(
                candidate_batch,
                reference_output,
                candidate_output,
                max_positions_per_batch,
            )
            if not torch.isfinite(reference).all().item():
                raise ValueError("Reference model produced nonfinite logits")
            if not torch.isfinite(candidate).all().item():
                return QualityGateResult(
                    passed=False,
                    metrics={
                        "finite_candidate_logits": False,
                        "failed_batch": index,
                        "per_batch": per_batch,
                    },
                    failures=[f"Candidate produced nonfinite logits in batch {index}"],
                    thresholds=thresholds,
                )
            difference = candidate.double() - reference.double()
            difference_squared = difference.square().sum().item()
            reference_squared = reference.double().square().sum().item()
            maximum = difference.abs().max().item()
            reference_log_probs = F.log_softmax(reference, dim=-1)
            candidate_log_probs = F.log_softmax(candidate, dim=-1)
            kl_values = (
                reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)
            ).sum(dim=-1)
            kl = max(0.0, kl_values.sum().item())
            top1 = int((reference.argmax(-1) == candidate.argmax(-1)).sum().item())
            batch_result: Dict[str, object] = {
                "batch": index,
                "positions": positions,
                "relative_logits_l2_error": math.sqrt(
                    difference_squared / max(reference_squared, 1e-30)
                ),
                "maximum_absolute_logits_error": maximum,
                "mean_kl": kl / positions,
                "top1_agreement": top1 / positions,
            }
            totals["difference_squared"] += difference_squared
            totals["reference_squared"] += reference_squared
            totals["maximum_absolute"] = max(totals["maximum_absolute"], maximum)
            totals["kl"] += kl
            totals["positions"] += positions
            totals["top1"] += top1
            if labels is not None:
                target = labels.to(reference.device).reshape(-1, 1)
                reference_nll = -reference_log_probs.gather(-1, target).squeeze(-1)
                candidate_nll = -candidate_log_probs.gather(-1, target).squeeze(-1)
                reference_sum = reference_nll.sum().item()
                candidate_sum = candidate_nll.sum().item()
                totals["reference_nll"] += reference_sum
                totals["candidate_nll"] += candidate_sum
                totals["supervised_tokens"] += positions
                batch_result.update(
                    reference_mean_nll=reference_sum / positions,
                    candidate_mean_nll=candidate_sum / positions,
                    mean_nll_delta=(candidate_sum - reference_sum) / positions,
                )
            per_batch.append(batch_result)
    finally:
        reference_model.train(reference_training)
        candidate_model.train(candidate_training)

    positions = totals["positions"]
    supervised = totals["supervised_tokens"]
    metrics: Dict[str, object] = {
        "finite_candidate_logits": True,
        "batches": len(batches),
        "positions": positions,
        "relative_logits_l2_error": math.sqrt(
            totals["difference_squared"] / max(totals["reference_squared"], 1e-30)
        ),
        "maximum_absolute_logits_error": totals["maximum_absolute"],
        "mean_kl": totals["kl"] / positions,
        "top1_agreement": totals["top1"] / positions,
        "supervised_tokens": supervised,
        "per_batch": per_batch,
        "scope": (
            "End-to-end gates only for supplied inputs and optional position sampling; "
            "passing is not a universal accuracy or downstream-task guarantee"
        ),
    }
    if supervised:
        metrics.update(
            reference_mean_nll=totals["reference_nll"] / supervised,
            candidate_mean_nll=totals["candidate_nll"] / supervised,
            mean_nll_delta=(
                totals["candidate_nll"] - totals["reference_nll"]
            )
            / supervised,
        )
    failures: List[str] = []

    def check(value: float, threshold: Optional[float], name: str, mode: str = "max") -> None:
        if threshold is None:
            return
        failed = value > threshold if mode == "max" else value < threshold
        if failed:
            failures.append(f"{name}={value} violates {mode} {threshold}")

    check(
        metrics["relative_logits_l2_error"],
        thresholds.max_relative_logits_l2_error,
        "relative_logits_l2_error",
    )
    check(
        metrics["maximum_absolute_logits_error"],
        thresholds.max_absolute_logits_error,
        "maximum_absolute_logits_error",
    )
    check(metrics["mean_kl"], thresholds.max_mean_kl, "mean_kl")
    check(
        metrics["top1_agreement"],
        thresholds.min_top1_agreement,
        "top1_agreement",
        mode="min",
    )
    if thresholds.max_mean_nll_delta is not None:
        if not supervised:
            failures.append("max_mean_nll_delta was requested but no supervised tokens exist")
        else:
            check(
                metrics["mean_nll_delta"],
                thresholds.max_mean_nll_delta,
                "mean_nll_delta",
            )
    return QualityGateResult(not failures, metrics, failures, thresholds)


def assert_quality_gates(result: QualityGateResult) -> QualityGateResult:
    if not result.passed:
        raise QualityGateError(result)
    return result

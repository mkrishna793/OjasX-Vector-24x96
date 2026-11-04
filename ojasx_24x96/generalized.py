"""End-to-end generalized discover/calibrate/plan/apply/validate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Union

import torch

from .adapters import DEFAULT_PROTECTED_ROLES, apply_quantization_plan, discover_tensor_specs
from .calibration import calibrate_candidates
from .codec import QuantizationConfig
from .gates import QualityGateError, QualityGateResult, QualityGateThresholds, evaluate_quality_gates
from .ir import QuantizationPlan, TensorRole
from .leech_codec import LeechQuantizationConfig
from .module import ModelQuantizationReport
from .planner import PlanningConstraints, generate_candidates, plan_quantization
from .runtime import runtime_storage_report


@dataclass
class GeneralizedQuantizationResult:
    model: torch.nn.Module
    plan: QuantizationPlan
    report: ModelQuantizationReport
    calibration: Optional[Dict[str, object]]
    quality: Optional[QualityGateResult] = None

    @property
    def status(self) -> str:
        if self.quality is None:
            return "unvalidated"
        return "passed" if self.quality.passed else "failed"

    def require_accepted(self) -> "GeneralizedQuantizationResult":
        """Require executed end-to-end gates, not merely local planning budgets."""

        if self.quality is None:
            raise RuntimeError(
                "Generalized quantization is unvalidated; execute end-to-end quality gates"
            )
        if not self.quality.passed:
            raise QualityGateError(self.quality)
        return self

    def to_dict(self, *, include_alternatives: bool = False) -> Dict[str, object]:
        return {
            "status": self.status,
            "plan": self.plan.to_dict(include_alternatives=include_alternatives),
            "report": self.report.to_dict(),
            "calibration": self.calibration,
            "end_to_end_quality": self.quality.to_dict() if self.quality is not None else None,
        }


def quantize_model_generalized(
    model: torch.nn.Module,
    *,
    constraints: PlanningConstraints,
    calibration_inputs: Optional[Iterable[object]] = None,
    candidate_configs: Optional[
        Sequence[Union[QuantizationConfig, LeechQuantizationConfig]]
    ] = None,
    include: Sequence[str] = ("*",),
    exclude: Sequence[str] = (),
    protected_roles: Iterable[TensorRole] = DEFAULT_PROTECTED_ROLES,
    allow_prequantized_source: bool = False,
    max_activation_rows: int = 128,
    max_invocations_per_module: int = 8,
    reference_model: Optional[torch.nn.Module] = None,
    quality_thresholds: Optional[QualityGateThresholds] = None,
    validation_inputs: Optional[Iterable[object]] = None,
    max_positions_per_batch: Optional[int] = 256,
) -> GeneralizedQuantizationResult:
    """Quantize in place using explicit RAM and quality constraints.

    The caller must declare at least one persistent-state/cache budget and at
    least one local or end-to-end quality budget. Calibration is mandatory unless
    ``allow_weight_only=True`` deliberately permits the weaker weight-error proxy.
    End-to-end validation requires a separate, unmodified reference model.
    """

    if not constraints.has_memory_budget:
        raise ValueError(
            "Generalized quantization requires an explicit persistent-state or cache budget"
        )
    if not constraints.has_quality_budget and quality_thresholds is None:
        raise ValueError(
            "Declare a quality budget: per-tensor/total local error or end-to-end thresholds"
        )
    if calibration_inputs is None and not constraints.allow_weight_only:
        raise ValueError(
            "Calibration inputs are required unless allow_weight_only=True is explicit"
        )
    if quality_thresholds is not None:
        if reference_model is None or reference_model is model:
            raise ValueError("End-to-end gates require a separate unmodified reference model")
        if validation_inputs is None and calibration_inputs is None:
            raise ValueError("Quality gates need validation or calibration inputs")
    elif reference_model is not None or validation_inputs is not None:
        raise ValueError("Reference/validation inputs require quality_thresholds")

    specs = discover_tensor_specs(
        model,
        include=include,
        exclude=exclude,
        protected_roles=protected_roles,
        allow_prequantized_source=allow_prequantized_source,
    )
    candidates = generate_candidates(model, specs, candidate_configs)
    calibration_summary = None
    if calibration_inputs is not None:
        calibration_inputs = tuple(calibration_inputs)
        calibration_summary = calibrate_candidates(
            model,
            specs,
            candidates,
            calibration_inputs,
            max_activation_rows=max_activation_rows,
            max_invocations_per_module=max_invocations_per_module,
        )
    base_bytes = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    plan = plan_quantization(
        specs,
        candidates,
        base_model_state_bytes=base_bytes,
        constraints=constraints,
    )
    if calibration_summary is not None:
        plan.quality["calibration"] = calibration_summary
    model, report = apply_quantization_plan(
        model,
        plan,
        runtime_profile=constraints.runtime_profile,
    )
    quality = None
    if quality_thresholds is not None:
        inputs = validation_inputs if validation_inputs is not None else calibration_inputs
        quality = evaluate_quality_gates(
            reference_model,
            model,
            inputs,
            quality_thresholds,
            max_positions_per_batch=max_positions_per_batch,
        )
        plan.quality["end_to_end_gates"] = quality.to_dict()
    return GeneralizedQuantizationResult(
        model=model,
        plan=plan,
        report=report,
        calibration=calibration_summary,
        quality=quality,
    )

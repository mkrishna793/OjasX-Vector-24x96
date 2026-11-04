"""Candidate generation and memory-constrained multiple-choice VQ planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .adapters import tensor_weight
from .codec import QuantizationConfig, quantize_weight
from .contract import ArchitectureProfile, evaluate_memory_admission
from .leech_codec import LeechQuantizationConfig, quantize_leech_weight
from .ir import (
    CandidateCollection,
    QuantizationCandidate,
    QuantizationPlan,
    TensorLayout,
    TensorSpec,
)
from .module import _stable_seed


class PlanningError(ValueError):
    """No candidate plan satisfies the declared hard constraints."""


@dataclass(frozen=True)
class PlanningConstraints:
    """Hard candidate-admission, persistent-state, and decode-output limits.

    ``target_state_ratio`` refers to unique persistent model state tensors, not
    RSS. In speed mode, use ``max_state_plus_cache_bytes`` as an additional hard
    constraint. ``max_decode_group_bytes`` limits decoded output matrices, not the
    codec's entire temporary workspace or allocator high-water mark.
    """

    target_state_ratio: Optional[float] = None
    max_model_state_bytes: Optional[int] = None
    max_state_plus_cache_bytes: Optional[int] = None
    max_decode_group_bytes: Optional[int] = None
    max_tensor_estimated_error: Optional[float] = None
    max_tensor_weight_error: Optional[float] = None
    max_total_estimated_squared_error: Optional[float] = None
    runtime_profile: str = "memory"
    decode_row_block_size: int = 64
    allow_weight_only: bool = False
    max_frontier_states: int = 2048
    max_total_runtime_bytes: Optional[int] = None
    architecture_profile: Optional[ArchitectureProfile] = None
    context_length: int = 2048
    batch_size: int = 1
    kv_dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if self.target_state_ratio is not None and not (
            math.isfinite(self.target_state_ratio) and 0 < self.target_state_ratio <= 1
        ):
            raise ValueError("target_state_ratio must be in (0, 1]")
        for value in (
            self.max_model_state_bytes,
            self.max_state_plus_cache_bytes,
            self.max_decode_group_bytes,
            self.max_total_runtime_bytes,
        ):
            if value is not None and value < 0:
                raise ValueError("Memory constraints must be nonnegative")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.kv_dtype_bytes <= 0:
            raise ValueError("kv_dtype_bytes must be positive")
        for value in (
            self.max_tensor_estimated_error,
            self.max_tensor_weight_error,
            self.max_total_estimated_squared_error,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Error constraints must be finite and nonnegative")
        if self.runtime_profile not in {"memory", "speed"}:
            raise ValueError("runtime_profile must be 'memory' or 'speed'")
        if self.decode_row_block_size <= 0:
            raise ValueError("decode_row_block_size must be positive")
        if self.max_frontier_states < 16:
            raise ValueError("max_frontier_states must be at least 16")

    @property
    def has_memory_budget(self) -> bool:
        return any(
            value is not None
            for value in (
                self.target_state_ratio,
                self.max_model_state_bytes,
                self.max_state_plus_cache_bytes,
                self.max_total_runtime_bytes,
            )
        )

    @property
    def has_quality_budget(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_tensor_estimated_error,
                self.max_tensor_weight_error,
                self.max_total_estimated_squared_error,
            )
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_state_ratio": self.target_state_ratio,
            "max_model_state_bytes": self.max_model_state_bytes,
            "max_state_plus_cache_bytes": self.max_state_plus_cache_bytes,
            "max_decode_group_bytes": self.max_decode_group_bytes,
            "max_tensor_estimated_error": self.max_tensor_estimated_error,
            "max_tensor_weight_error": self.max_tensor_weight_error,
            "max_total_estimated_squared_error": self.max_total_estimated_squared_error,
            "runtime_profile": self.runtime_profile,
            "decode_row_block_size": self.decode_row_block_size,
            "allow_weight_only": self.allow_weight_only,
            "max_frontier_states": self.max_frontier_states,
            "max_total_runtime_bytes": self.max_total_runtime_bytes,
            "architecture_profile": (
                self.architecture_profile.to_dict()
                if self.architecture_profile is not None
                else None
            ),
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "kv_dtype_bytes": self.kv_dtype_bytes,
        }


def default_candidate_configs(base: Optional[QuantizationConfig] = None) -> List[QuantizationConfig]:
    """Return conservative-to-aggressive candidates using actual stored bytes."""

    common = base or QuantizationConfig(
        max_relative_error=None,
        max_codebook_samples=1024,
        kmeans_iterations=8,
        require_compression=True,
    )
    profiles = ((4, 4, 4), (8, 4, 4), (8, 4, 2), (16, 4, 2))
    return [
        replace(
            common,
            block_size=block,
            bits_per_stage=bits,
            max_stages=stages,
            max_relative_error=None,
            require_compression=True,
        )
        for block, bits, stages in profiles
    ]


def default_leech_candidate_configs(
    base: Optional[LeechQuantizationConfig] = None,
) -> List[LeechQuantizationConfig]:
    """Return measured-byte Leech candidates with radial/rotation diversity."""

    common = base or LeechQuantizationConfig(
        max_stages=3,
        max_relative_error=None,
        gain_bits=1,
        rotation_reflections=2,
        adaptive_stages=True,
        max_gain_samples=1024,
        gain_grid_size=16,
        require_compression=True,
    )
    profiles = (
        (1, 1, 0, False),
        (2, 1, 0, True),
        (3, 1, 2, True),
        (3, 2, 2, True),
    )
    return [
        replace(
            common,
            max_stages=stages,
            gain_bits=gain_bits,
            rotation_reflections=reflections,
            adaptive_stages=adaptive,
            max_relative_error=None,
            require_compression=True,
        )
        for stages, gain_bits, reflections, adaptive in profiles
    ]


def generate_candidates(
    model,
    specs: Iterable[TensorSpec],
    configs: Optional[
        Sequence[Union[QuantizationConfig, LeechQuantizationConfig]]
    ] = None,
) -> CandidateCollection:
    """Generate dense fallback and multiple measured VQ candidates per tensor."""

    configs = tuple(configs or default_candidate_configs())
    if not configs:
        raise ValueError("Candidate generation requires at least one codec configuration")
    by_tensor: Dict[str, List[QuantizationCandidate]] = {}
    failures: Dict[str, Dict[str, str]] = {}
    for spec in specs:
        candidates = [QuantizationCandidate.dense(spec)]
        by_tensor[spec.name] = candidates
        if not spec.quantizable:
            continue
        weight = tensor_weight(model, spec)
        for index, config in enumerate(configs):
            if isinstance(config, LeechQuantizationConfig):
                identifier = (
                    f"leech24-{index}-s{config.max_stages}-g{config.gain_bits}"
                    f"-r{config.rotation_reflections}-a{int(config.adaptive_stages)}"
                )
                encoded, measurement = quantize_leech_weight(weight, config)
                codec_name = "leech24_shell2_residual"
            else:
                identifier = (
                    f"rvq-{index}-b{config.block_size}-k{config.bits_per_stage}"
                    f"-s{config.max_stages}"
                )
                encoded, measurement = quantize_weight(
                    weight,
                    config,
                    seed=_stable_seed(f"{spec.name}:{index}", config.seed),
                )
                codec_name = "adaptive_residual_vq"
            if encoded is None:
                failures.setdefault(spec.name, {})[identifier] = str(measurement["reason"])
                continue
            if encoded.storage_bytes >= spec.original_weight_bytes:
                # The planner's RAM contract never silently accepts expansion,
                # even when a low-level codec config disabled its own guard.
                failures.setdefault(spec.name, {})[identifier] = "not_smaller_than_original"
                continue
            candidates.append(
                QuantizationCandidate(
                    candidate_id=identifier,
                    tensor_name=spec.name,
                    codec=codec_name,
                    original_weight_bytes=spec.original_weight_bytes,
                    encoded_weight_bytes=encoded.storage_bytes,
                    fixed_bytes=spec.fixed_bytes,
                    decoded_weight_bytes=weight.numel() * spec.dtype_bytes,
                    weight_elements=spec.weight_elements,
                    weight_relative_error=float(encoded.metrics["relative_error"]),
                    block_size=encoded.block_size,
                    bits_per_stage=encoded.bits_per_stage,
                    stages=encoded.stages,
                    config=config,
                    encoded=encoded,
                )
            )
    return CandidateCollection(by_tensor, failures)


@dataclass(frozen=True)
class _GroupOption:
    identifiers: Tuple[str, ...]
    savings: int
    cache: int
    decode: int
    error: float
    quantized: int


@dataclass(frozen=True)
class _FrontierEntry:
    savings: int
    cache: int
    error: float
    max_decode: int
    quantized: int
    choices: Tuple[int, ...]


def _dominates_group(left: _GroupOption, right: _GroupOption) -> bool:
    return (
        left.savings >= right.savings
        and left.cache <= right.cache
        and left.decode <= right.decode
        and left.error <= right.error
        and (
            left.savings > right.savings
            or left.cache < right.cache
            or left.decode < right.decode
            or left.error < right.error
        )
    )


def _dominates_entry(left: _FrontierEntry, right: _FrontierEntry) -> bool:
    return (
        left.savings >= right.savings
        and left.cache <= right.cache
        and left.max_decode <= right.max_decode
        and left.error <= right.error
        and (
            left.savings > right.savings
            or left.cache < right.cache
            or left.max_decode < right.max_decode
            or left.error < right.error
        )
    )


def _prune_groups(options: List[_GroupOption]) -> List[_GroupOption]:
    ordered = sorted(
        options,
        key=lambda option: (
            option.error,
            -option.savings,
            option.cache,
            option.decode,
            option.identifiers,
        ),
    )
    kept: List[_GroupOption] = []
    for option in ordered:
        if any(_dominates_group(other, option) for other in kept):
            continue
        kept.append(option)
    return kept


def _prune_frontier(entries: List[_FrontierEntry]) -> List[_FrontierEntry]:
    ordered = sorted(
        entries,
        key=lambda entry: (
            entry.error,
            -entry.savings,
            entry.cache,
            entry.max_decode,
            entry.choices,
        ),
    )
    kept: List[_FrontierEntry] = []
    for entry in ordered:
        if any(_dominates_entry(other, entry) for other in kept):
            continue
        kept.append(entry)
    return kept


def _truncate_frontier(
    entries: List[_FrontierEntry], maximum: int
) -> List[_FrontierEntry]:
    """Retain byte-diverse and error-extreme states when a beam limit is hit."""

    if len(entries) <= maximum:
        return entries
    by_bytes = sorted(entries, key=lambda entry: (-entry.savings + entry.cache, entry.error))
    by_error = sorted(entries, key=lambda entry: (entry.error, -entry.savings))
    selected: List[_FrontierEntry] = []
    seen = set()

    def add(entry: _FrontierEntry) -> None:
        if entry.choices not in seen and len(selected) < maximum:
            selected.append(entry)
            seen.add(entry.choices)

    add(by_error[0])
    add(by_bytes[0])
    add(by_bytes[-1])
    for rank in range(maximum):
        index = round(rank * (len(by_bytes) - 1) / max(1, maximum - 1))
        add(by_bytes[index])
    for entry in by_error:
        add(entry)
        if len(selected) == maximum:
            break
    return selected


def _candidate_decode_bytes(
    spec: TensorSpec,
    candidate: QuantizationCandidate,
    constraints: PlanningConstraints,
) -> int:
    if candidate.is_dense:
        return 0
    if constraints.runtime_profile == "speed":
        return candidate.decoded_weight_bytes
    if spec.layout in {
        TensorLayout.FUSED_EXPERT_GATE_UP,
        TensorLayout.FUSED_EXPERT_DOWN,
    }:
        rows = int(spec.logical_shape[1])
    else:
        rows = min(spec.matrix_shape[0], constraints.decode_row_block_size)
    return rows * spec.matrix_shape[1] * spec.dtype_bytes


def _group_options(
    specs: Sequence[TensorSpec],
    candidates: CandidateCollection,
    constraints: PlanningConstraints,
) -> List[_GroupOption]:
    options: List[_GroupOption] = []
    for choices in product(*(candidates[spec.name] for spec in specs)):
        if any(
            not candidate.is_dense and not constraints.allow_weight_only and not candidate.calibrated
            for candidate in choices
        ):
            continue
        if any(candidate.weight_savings_bytes < 0 for candidate in choices):
            continue
        if constraints.max_tensor_estimated_error is not None and any(
            candidate.estimated_error > constraints.max_tensor_estimated_error
            for candidate in choices
        ):
            continue
        if constraints.max_tensor_weight_error is not None and any(
            candidate.weight_relative_error > constraints.max_tensor_weight_error
            for candidate in choices
        ):
            continue
        decode = sum(
            _candidate_decode_bytes(spec, candidate, constraints)
            for spec, candidate in zip(specs, choices)
        )
        if constraints.max_decode_group_bytes is not None and (
            decode > constraints.max_decode_group_bytes
        ):
            continue
        options.append(
            _GroupOption(
                identifiers=tuple(candidate.candidate_id for candidate in choices),
                savings=sum(candidate.weight_savings_bytes for candidate in choices),
                cache=(
                    sum(candidate.decoded_weight_bytes for candidate in choices)
                    if constraints.runtime_profile == "speed"
                    else 0
                ),
                decode=decode,
                error=sum(candidate.estimated_error**2 for candidate in choices),
                quantized=sum(not candidate.is_dense for candidate in choices),
            )
        )
    return _prune_groups(options)


def plan_quantization(
    specs: Sequence[TensorSpec],
    candidates: CandidateCollection,
    *,
    base_model_state_bytes: int,
    constraints: PlanningConstraints,
) -> QuantizationPlan:
    """Choose minimum additive local error among plans fitting hard RAM limits.

    The search is exact over generated candidates unless the deterministic
    frontier beam is reached. The objective is a local heuristic, not a bound on
    final model NLL or task accuracy; end-to-end gates remain necessary.
    """

    if base_model_state_bytes < 0:
        raise ValueError("base_model_state_bytes must be nonnegative")
    if not specs:
        raise PlanningError("No IR tensors were discovered")
    spec_names = [spec.name for spec in specs]
    if set(spec_names) != set(candidates.by_tensor):
        raise ValueError("Candidate collection does not exactly match the IR specs")
    groups: List[List[TensorSpec]] = []
    group_indices: Dict[str, int] = {}
    for spec in specs:
        key = spec.decode_group or spec.module_name
        if key not in group_indices:
            group_indices[key] = len(groups)
            groups.append([])
        groups[group_indices[key]].append(spec)
    options = [_group_options(group, candidates, constraints) for group in groups]
    for group, group_options in zip(groups, options):
        if not group_options:
            names = [spec.name for spec in group]
            raise PlanningError(f"No admissible candidates for decode group {names}")

    state_limit = base_model_state_bytes
    if constraints.target_state_ratio is not None:
        state_limit = min(
            state_limit,
            math.floor(base_model_state_bytes * constraints.target_state_ratio),
        )
    if constraints.max_model_state_bytes is not None:
        state_limit = min(state_limit, constraints.max_model_state_bytes)
    total_limit = constraints.max_state_plus_cache_bytes

    non_weight_overhead = 0
    if constraints.max_total_runtime_bytes is not None and constraints.architecture_profile is not None:
        kv_dtype_map = {1: "int8", 2: "bfloat16", 4: "float32", 8: "float64"}
        kv_dtype = kv_dtype_map.get(constraints.kv_dtype_bytes, "bfloat16")
        admission_report = evaluate_memory_admission(
            constraints.architecture_profile,
            encoded_weight_bytes=0,
            dense_weight_bytes=0,
            codebooks_and_scales_bytes=0,
            decoded_weight_cache_bytes=0,
            batch_size=constraints.batch_size,
            context_length=constraints.context_length,
            decode_row_block_size=constraints.decode_row_block_size,
            kv_dtype=kv_dtype,
        )
        non_weight_overhead = admission_report.total_required_bytes
        if non_weight_overhead >= constraints.max_total_runtime_bytes:
            raise PlanningError(
                f"Non-weight runtime footprint alone ({non_weight_overhead} bytes: "
                f"KV {admission_report.kv_cache_bytes}, "
                f"activation {max(admission_report.activation_prefill_bytes, admission_report.activation_decode_bytes)}, "
                f"workspace {admission_report.workspace_bytes}, overhead {admission_report.runtime_overhead_bytes}) "
                f"exceeds or meets max_total_runtime_bytes ({constraints.max_total_runtime_bytes} bytes)"
            )
        allowed_weight_budget = constraints.max_total_runtime_bytes - non_weight_overhead
        state_limit = min(state_limit, allowed_weight_budget)
        if total_limit is None:
            total_limit = allowed_weight_budget
        else:
            total_limit = min(total_limit, allowed_weight_budget)
    frontier = [_FrontierEntry(0, 0, 0.0, 0, 0, ())]
    truncated = False
    peak_frontier = 1
    for group_options in options:
        combined: List[_FrontierEntry] = []
        for entry in frontier:
            for option_index, option in enumerate(group_options):
                savings = entry.savings + option.savings
                cache = entry.cache + option.cache
                # Byte limits apply only to complete plans. Future groups add
                # savings, so rejecting an over-budget partial plan is unsound.
                combined.append(
                    _FrontierEntry(
                        savings=savings,
                        cache=cache,
                        error=entry.error + option.error,
                        max_decode=max(entry.max_decode, option.decode),
                        quantized=entry.quantized + option.quantized,
                        choices=entry.choices + (option_index,),
                    )
                )
        if not combined:
            raise PlanningError("Candidate frontier became empty while combining decode groups")
        frontier = _prune_frontier(combined)
        peak_frontier = max(peak_frontier, len(frontier))
        if len(frontier) > constraints.max_frontier_states:
            frontier = _truncate_frontier(frontier, constraints.max_frontier_states)
            truncated = True

    feasible = [
        entry
        for entry in frontier
        if base_model_state_bytes - entry.savings <= state_limit
        and (
            total_limit is None
            or base_model_state_bytes - entry.savings + entry.cache <= total_limit
        )
    ]
    if not feasible:
        detail = (
            "no plan was retained within the byte constraints after frontier truncation; "
            "increase max_frontier_states"
            if truncated
            else "no generated candidate plan fits the persistent-state/cache constraints"
        )
        raise PlanningError(detail)
    best = min(
        feasible,
        key=lambda entry: (
            entry.error,
            -entry.savings,
            entry.cache,
            entry.max_decode,
            entry.choices,
        ),
    )
    if constraints.max_total_estimated_squared_error is not None and (
        best.error > constraints.max_total_estimated_squared_error
    ):
        detail = (
            "no retained feasible plan satisfies the total local-error budget after "
            "frontier truncation; increase max_frontier_states"
            if truncated
            else f"best feasible plan's local error {best.error} exceeds the declared "
            f"total budget {constraints.max_total_estimated_squared_error}"
        )
        raise PlanningError(detail)
    selected: Dict[str, str] = {}
    for group, group_options, option_index in zip(groups, options, best.choices):
        option = group_options[option_index]
        for spec, identifier in zip(group, option.identifiers):
            selected[spec.name] = identifier
    state = base_model_state_bytes - best.savings
    quantized_candidates = [
        candidate
        for name in spec_names
        for candidate in candidates[name]
        if candidate.candidate_id == selected[name] and not candidate.is_dense
    ]
    max_tensor_error = max(
        (candidate.estimated_error for candidate in quantized_candidates),
        default=0.0,
    )
    constraint_dict = constraints.to_dict()
    constraint_dict["derived_max_model_state_bytes"] = state_limit
    memory = {
        "dense_model_unique_state_bytes": base_model_state_bytes,
        "predicted_model_state_bytes": state,
        "predicted_weight_savings_bytes": best.savings,
        "predicted_state_ratio": state / base_model_state_bytes if base_model_state_bytes else 1.0,
        "runtime_profile": constraints.runtime_profile,
        "predicted_full_decoded_cache_bytes": best.cache,
        "predicted_model_state_plus_cache_bytes": state + best.cache,
        "max_decode_group_output_bytes_estimate": best.max_decode,
        **({"predicted_total_runtime_bytes": state + best.cache + non_weight_overhead, "non_weight_runtime_bytes": non_weight_overhead} if non_weight_overhead > 0 else {}),
        "scope": (
            "Persistent bytes are tensor-inventory accounting, not RSS. Cache assumes "
            "every selected VQ projection is populated. Memory mode estimates one "
            "streaming row block or active-expert decode group; speed mode estimates "
            "full decoded banks. Values use source-dtype bytes and exclude padded "
            "float32 intermediates, activations, and KV cache."
        ),
    }
    quality = {
        "estimated_squared_local_error": best.error,
        "estimated_rms_local_error_over_quantized_tensors": (
            math.sqrt(best.error / best.quantized) if best.quantized else 0.0
        ),
        "max_selected_tensor_estimated_error": max_tensor_error,
        "quantized_tensors": best.quantized,
        "calibrated_quantized_tensors": sum(
            candidate.calibrated for candidate in quantized_candidates
        ),
        "weight_only_quantized_tensors": sum(
            not candidate.calibrated for candidate in quantized_candidates
        ),
        "scope": (
            "Additive squared local errors guide selection only. They are not an "
            "end-to-end NLL, KL, task-accuracy, or cross-layer interaction bound."
        ),
    }
    optimizer = {
        "algorithm": "multiple_choice_pareto_frontier",
        "decode_groups": len(groups),
        "peak_frontier_states": peak_frontier,
        "frontier_truncated": truncated,
        "candidate_optimality": "bounded" if truncated else "exact_over_generated_candidates",
        "objective": "minimum_sum_squared_local_error_then_maximum_weight_savings",
    }
    plan = QuantizationPlan(
        specs=list(specs),
        candidates=candidates,
        selected=selected,
        base_model_state_bytes=base_model_state_bytes,
        constraints=constraint_dict,
        memory=memory,
        quality=quality,
        optimizer=optimizer,
    )
    return plan

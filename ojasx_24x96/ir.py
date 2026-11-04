"""Model-independent tensor, candidate, and plan intermediate representations.

The IR deliberately contains no Hugging Face model classes.  Adapters describe
execution semantics; codecs provide candidates; planners select candidates; and
runtime adapters apply the resulting plan.  Quality estimates in this file are
measurements, never universal model-accuracy guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .codec import QuantizationConfig, QuantizedWeight
from .leech_codec import LeechQuantizationConfig, LeechQuantizedWeight


class TensorRole(str, Enum):
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    ROUTED_EXPERT = "routed_expert"
    SHARED_EXPERT = "shared_expert"
    ROUTER = "router"
    EMBEDDING = "embedding"
    OUTPUT_HEAD = "output_head"
    MULTIMODAL = "multimodal"
    RECURRENT = "recurrent"
    OTHER = "other"


class TensorLayout(str, Enum):
    LINEAR = "linear"
    CONV1D = "conv1d"
    FUSED_EXPERT_GATE_UP = "fused_expert_gate_up"
    FUSED_EXPERT_DOWN = "fused_expert_down"
    EMBEDDING = "embedding"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class TensorSpec:
    """Serializable description of one logical weight and its execution contract."""

    name: str
    module_name: str
    attribute: str
    role: TensorRole
    layout: TensorLayout
    logical_shape: Tuple[int, ...]
    matrix_shape: Tuple[int, int]
    dtype: str
    dtype_bytes: int
    original_weight_bytes: int
    fixed_bytes: int = 0
    quantizable: bool = True
    protection_reason: Optional[str] = None
    shared_group: Optional[str] = None
    source_quantization: str = "dense"
    decode_group: str = ""
    module_type: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TensorRole(self.role))
        object.__setattr__(self, "layout", TensorLayout(self.layout))
        object.__setattr__(self, "logical_shape", tuple(map(int, self.logical_shape)))
        object.__setattr__(self, "matrix_shape", tuple(map(int, self.matrix_shape)))
        if not self.name:
            raise ValueError("Tensor name must be nonempty")
        if not self.attribute:
            raise ValueError("Tensor attribute must be nonempty")
        if len(self.matrix_shape) != 2 or any(value < 0 for value in self.logical_shape):
            raise ValueError("IR weights require a rank-two matrix view")
        if any(value <= 0 for value in self.matrix_shape):
            raise ValueError("IR matrix dimensions must be positive")
        if self.dtype_bytes <= 0 or self.original_weight_bytes < 0 or self.fixed_bytes < 0:
            raise ValueError("IR byte counts must be nonnegative and dtype size positive")
        if not self.quantizable and not self.protection_reason:
            raise ValueError("A non-quantizable IR tensor needs a protection reason")

    @property
    def weight_elements(self) -> int:
        return self.matrix_shape[0] * self.matrix_shape[1]

    @property
    def original_resident_bytes(self) -> int:
        return self.original_weight_bytes + self.fixed_bytes

    def to_dict(self) -> Dict[str, object]:
        result = {
            "name": self.name,
            "module_name": self.module_name,
            "attribute": self.attribute,
            "role": self.role.value,
            "layout": self.layout.value,
            "logical_shape": list(self.logical_shape),
            "matrix_shape": list(self.matrix_shape),
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "original_weight_bytes": self.original_weight_bytes,
            "fixed_bytes": self.fixed_bytes,
            "quantizable": self.quantizable,
            "protection_reason": self.protection_reason,
            "shared_group": self.shared_group,
            "source_quantization": self.source_quantization,
            "decode_group": self.decode_group or self.module_name,
        }
        if self.module_type is not None:
            result["module_type"] = self.module_type
        return result


@dataclass
class QuantizationCandidate:
    """One dense or encoded choice for a :class:`TensorSpec`.

    ``encoded`` is runtime-only payload and is intentionally excluded from JSON.
    Unselected payloads can be released after planning.  Local output error is
    measured on captured inputs; when absent, the planner can explicitly opt in
    to the weaker weight-relative-error proxy.
    """

    candidate_id: str
    tensor_name: str
    codec: str
    original_weight_bytes: int
    encoded_weight_bytes: int
    fixed_bytes: int
    decoded_weight_bytes: int
    weight_elements: int = 0
    weight_relative_error: float = 0.0
    local_output_relative_error: Optional[float] = None
    local_output_max_absolute_error: Optional[float] = None
    calibration_values: int = 0
    calibration_invocations: int = 0
    block_size: Optional[int] = None
    bits_per_stage: Optional[int] = None
    stages: int = 0
    config: Optional[Union[QuantizationConfig, LeechQuantizationConfig]] = field(
        default=None, compare=False
    )
    encoded: Optional[Union[QuantizedWeight, LeechQuantizedWeight]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.tensor_name or not self.codec:
            raise ValueError("Candidate identity, tensor, and codec must be nonempty")
        if min(
            self.original_weight_bytes,
            self.encoded_weight_bytes,
            self.fixed_bytes,
            self.decoded_weight_bytes,
            self.weight_elements,
        ) < 0:
            raise ValueError("Candidate byte counts must be nonnegative")
        if not math.isfinite(self.weight_relative_error) or self.weight_relative_error < 0:
            raise ValueError("Candidate weight error must be finite and nonnegative")
        if self.local_output_relative_error is not None and (
            not math.isfinite(self.local_output_relative_error)
            or self.local_output_relative_error < 0
        ):
            raise ValueError("Candidate local output error must be finite and nonnegative")
        if self.local_output_max_absolute_error is not None and (
            not math.isfinite(self.local_output_max_absolute_error)
            or self.local_output_max_absolute_error < 0
        ):
            raise ValueError("Candidate local absolute error must be finite and nonnegative")
        if self.calibration_values < 0 or self.calibration_invocations < 0:
            raise ValueError("Calibration counts must be nonnegative")
        if self.codec == "dense":
            if self.encoded is not None or self.stages != 0:
                raise ValueError("Dense candidates cannot contain VQ payload or stages")
            if self.encoded_weight_bytes != self.original_weight_bytes:
                raise ValueError("Dense candidates must retain original weight bytes")
            if self.decoded_weight_bytes != 0:
                raise ValueError("Dense candidates do not allocate a decoded VQ weight")
        else:
            if self.stages <= 0 or self.block_size is None or self.bits_per_stage is None:
                raise ValueError("Encoded candidates need positive codec dimensions")

    @classmethod
    def dense(cls, spec: TensorSpec) -> "QuantizationCandidate":
        return cls(
            candidate_id="dense",
            tensor_name=spec.name,
            codec="dense",
            original_weight_bytes=spec.original_weight_bytes,
            encoded_weight_bytes=spec.original_weight_bytes,
            fixed_bytes=spec.fixed_bytes,
            decoded_weight_bytes=0,
            weight_elements=spec.weight_elements,
        )

    @property
    def is_dense(self) -> bool:
        return self.codec == "dense"

    @property
    def calibrated(self) -> bool:
        return (
            self.local_output_relative_error is not None
            and self.calibration_values > 0
            and self.calibration_invocations > 0
        )

    @property
    def estimated_error(self) -> float:
        if self.calibrated:
            return float(self.local_output_relative_error)
        return self.weight_relative_error

    @property
    def weight_savings_bytes(self) -> int:
        return self.original_weight_bytes - self.encoded_weight_bytes

    @property
    def resident_bytes(self) -> int:
        return self.encoded_weight_bytes + self.fixed_bytes

    @property
    def measured_bits_per_weight(self) -> float:
        return (
            8.0 * self.encoded_weight_bytes / self.weight_elements
            if self.weight_elements
            else 0.0
        )

    def release_payload(self) -> None:
        self.encoded = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "tensor_name": self.tensor_name,
            "codec": self.codec,
            "original_weight_bytes": self.original_weight_bytes,
            "encoded_weight_bytes": self.encoded_weight_bytes,
            "fixed_bytes": self.fixed_bytes,
            "weight_savings_bytes": self.weight_savings_bytes,
            "weight_elements": self.weight_elements,
            "measured_bits_per_weight": self.measured_bits_per_weight,
            "decoded_weight_bytes_estimate": self.decoded_weight_bytes,
            "weight_relative_error": self.weight_relative_error,
            "local_output_relative_error": self.local_output_relative_error,
            "local_output_max_absolute_error": self.local_output_max_absolute_error,
            "estimated_error_for_planning": self.estimated_error,
            "calibrated": self.calibrated,
            "calibration_values": self.calibration_values,
            "calibration_invocations": self.calibration_invocations,
            "block_size": self.block_size,
            "bits_per_stage": self.bits_per_stage,
            "stages": self.stages,
            "config": self.config.to_dict() if self.config is not None else None,
            "payload_resident": self.encoded is not None,
        }


@dataclass
class CandidateCollection:
    by_tensor: Dict[str, List[QuantizationCandidate]]
    failures: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, candidates in self.by_tensor.items():
            if not candidates:
                raise ValueError(f"No candidates for {name!r}")
            identifiers = [candidate.candidate_id for candidate in candidates]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"Duplicate candidate identifiers for {name!r}")
            if sum(candidate.is_dense for candidate in candidates) != 1:
                raise ValueError(f"Exactly one dense fallback is required for {name!r}")
            if any(candidate.tensor_name != name for candidate in candidates):
                raise ValueError(f"Candidate tensor mismatch for {name!r}")

    def __getitem__(self, name: str) -> List[QuantizationCandidate]:
        return self.by_tensor[name]


@dataclass
class QuantizationPlan:
    """A selected candidate per IR tensor plus auditable memory/quality math."""

    specs: List[TensorSpec]
    candidates: CandidateCollection
    selected: Dict[str, str]
    base_model_state_bytes: int
    constraints: Dict[str, object]
    memory: Dict[str, object]
    quality: Dict[str, object]
    optimizer: Dict[str, object]
    schema_version: int = 1
    architecture_profile: Optional[Dict[str, object]] = None
    memory_admission: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate tensor names in quantization plan")
        if set(self.selected) != set(names):
            raise ValueError("Plan must select exactly one candidate for every IR tensor")
        for spec in self.specs:
            self.candidate_for(spec.name)
        if self.base_model_state_bytes < 0:
            raise ValueError("Base model state bytes must be nonnegative")

    @property
    def spec_map(self) -> Dict[str, TensorSpec]:
        return {spec.name: spec for spec in self.specs}

    def candidate_for(self, name: str) -> QuantizationCandidate:
        identifier = self.selected[name]
        for candidate in self.candidates[name]:
            if candidate.candidate_id == identifier:
                return candidate
        raise ValueError(f"Plan selects missing candidate {identifier!r} for {name!r}")

    @property
    def selected_candidates(self) -> Dict[str, QuantizationCandidate]:
        return {spec.name: self.candidate_for(spec.name) for spec in self.specs}

    def compact_unselected_payloads(self) -> None:
        """Release search payloads that are not needed to apply this plan."""

        for name, candidates in self.candidates.by_tensor.items():
            selected = self.selected[name]
            for candidate in candidates:
                if candidate.candidate_id != selected:
                    candidate.release_payload()

    def coverage_report(self) -> Dict[str, object]:
        roles: Dict[str, Dict[str, int]] = {}
        layouts: Dict[str, Dict[str, int]] = {}
        for spec in self.specs:
            candidate = self.candidate_for(spec.name)
            for mapping, key in ((roles, spec.role.value), (layouts, spec.layout.value)):
                row = mapping.setdefault(
                    key,
                    {
                        "tensors": 0,
                        "quantized_tensors": 0,
                        "original_weight_bytes": 0,
                        "selected_weight_bytes": 0,
                    },
                )
                row["tensors"] += 1
                row["quantized_tensors"] += int(not candidate.is_dense)
                row["original_weight_bytes"] += spec.original_weight_bytes
                row["selected_weight_bytes"] += candidate.encoded_weight_bytes
        return {"by_role": roles, "by_layout": layouts}

    def to_dict(self, *, include_alternatives: bool = False) -> Dict[str, object]:
        result: Dict[str, object] = {
            "schema_version": self.schema_version,
            "format": "ojasx-generalized-vq-plan",
            "base_model_state_bytes": self.base_model_state_bytes,
            "constraints": self.constraints,
            "memory": self.memory,
            "quality": self.quality,
            "optimizer": self.optimizer,
            "coverage": self.coverage_report(),
            "generation_failures": self.candidates.failures,
            "tensors": [spec.to_dict() for spec in self.specs],
            "selected": {
                spec.name: self.candidate_for(spec.name).to_dict() for spec in self.specs
            },
            "scope": (
                "Persistent tensor-byte accounting is exact for the represented dense "
                "state and selected payloads. Decode bytes are estimates; local errors "
                "are not end-to-end accuracy guarantees."
            ),
        }
        if self.architecture_profile is not None:
            result["architecture_profile"] = self.architecture_profile
        if self.memory_admission is not None:
            result["memory_admission"] = self.memory_admission
        if include_alternatives:
            result["alternatives"] = {
                name: [candidate.to_dict() for candidate in candidates]
                for name, candidates in self.candidates.by_tensor.items()
            }
        return result


def selected_quantized_count(plan: QuantizationPlan) -> int:
    return sum(not candidate.is_dense for candidate in plan.selected_candidates.values())


def candidate_map(
    specs: Iterable[TensorSpec], candidates: Mapping[str, List[QuantizationCandidate]]
) -> CandidateCollection:
    return CandidateCollection({spec.name: list(candidates[spec.name]) for spec in specs})

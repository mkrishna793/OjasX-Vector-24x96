"""Execution adapters between dense models and the model-independent VQ IR."""

from __future__ import annotations

import fnmatch
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn

from .ir import (
    QuantizationCandidate,
    QuantizationPlan,
    TensorLayout,
    TensorRole,
    TensorSpec,
    selected_quantized_count,
)
from .module import (
    LayerQuantizationReport,
    ModelQuantizationReport,
    ResidualVQConv1D,
    ResidualVQFusedExperts,
    ResidualVQLinear,
    _expert_activation,
    _expert_gate_limit,
    _layer_kind,
    _shared_weight_ids,
)
from .runtime import runtime_storage_report, set_decode_row_block_size, set_runtime_profile


DEFAULT_PROTECTED_ROLES = frozenset(
    {
        TensorRole.EMBEDDING,
        TensorRole.OUTPUT_HEAD,
        TensorRole.ROUTER,
        TensorRole.MULTIMODAL,
    }
)


_CUSTOM_ADAPTERS: Dict[
    type,
    Tuple[
        Callable[[str, nn.Module], Optional[Tuple[torch.Tensor, Optional[torch.Tensor], TensorLayout, str]]],
        Optional[Callable[[nn.Module, QuantizationCandidate, Any], nn.Module]],
    ],
] = {}


def register_custom_module_adapter(
    module_type: type,
    extractor: Callable[[str, nn.Module], Optional[Tuple[torch.Tensor, Optional[torch.Tensor], TensorLayout, str]]],
    replacer: Optional[Callable[[nn.Module, QuantizationCandidate, Any], nn.Module]] = None,
) -> None:
    """Register a custom module adapter with extractor and optional replacer."""
    if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
        raise TypeError(f"module_type must be a subclass of nn.Module, got {module_type}")
    _CUSTOM_ADAPTERS[module_type] = (extractor, replacer)


def unregister_custom_module_adapter(module_type: type) -> None:
    """Unregister a custom module adapter."""
    _CUSTOM_ADAPTERS.pop(module_type, None)


def list_registered_custom_adapters() -> List[str]:
    """List class names of all registered custom module adapters."""
    return [cls.__name__ for cls in _CUSTOM_ADAPTERS]


def _find_custom_adapter(module: nn.Module) -> Optional[Tuple[type, Tuple[Callable, Optional[Callable]]]]:
    mod_type = type(module)
    if mod_type in _CUSTOM_ADAPTERS:
        return mod_type, _CUSTOM_ADAPTERS[mod_type]
    for reg_type, entry in _CUSTOM_ADAPTERS.items():
        if isinstance(module, reg_type):
            return reg_type, entry
    return None


def _is_duck_typed_linear(module: nn.Module) -> bool:
    if any(True for _ in module.children()):
        return False
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
        return False
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return False
    name = type(module).__name__.lower()
    if "norm" in name:
        return False
    norm_types = [
        getattr(nn.modules.normalization, "LayerNorm", None),
        getattr(nn.modules.normalization, "GroupNorm", None),
        getattr(nn.modules.normalization, "LocalResponseNorm", None),
        getattr(nn.modules.batchnorm, "_BatchNorm", None),
        getattr(nn.modules.instancenorm, "_InstanceNorm", None),
        getattr(nn, "RMSNorm", None),
    ]
    actual_norm_types = tuple(t for t in norm_types if isinstance(t, type))
    if actual_norm_types and isinstance(module, actual_norm_types):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor):
        return False
    if weight.ndim != 2 or not weight.dtype.is_floating_point:
        return False
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        return False
    bias = getattr(module, "bias", None)
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            return False
    return True


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _source_quantization(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    quantization = getattr(config, "quantization_config", None)
    if not quantization:
        return "dense"
    if isinstance(quantization, dict):
        method = (
            quantization.get("quant_method")
            or quantization.get("method")
            or quantization.get("format")
            or "unspecified"
        )
    else:
        method = getattr(quantization, "quant_method", None) or str(quantization)
    return str(method)


def classify_tensor_role(name: str, module: nn.Module) -> TensorRole:
    """Conservative name/interface role classification for planning policy.

    Roles control default protection and reports; they do not establish that an
    unknown module is safe to replace. Layout safety is determined independently.
    """

    if isinstance(module, nn.Embedding):
        return TensorRole.EMBEDDING
    lowered = name.lower()
    leaf = lowered.rsplit(".", 1)[-1]
    segments = set(lowered.split("."))
    if leaf in {"lm_head", "output_head", "output_projection", "score"}:
        return TensorRole.OUTPUT_HEAD
    if leaf in {
        "gate",
        "router",
        "router_proj",
        "gate_score",
        "shared_expert_gate",
        "expert_gate",
    } and "gate_proj" not in lowered:
        return TensorRole.ROUTER
    if segments.intersection(
        {
            "vision",
            "visual",
            "vision_tower",
            "audio",
            "audio_tower",
            "multi_modal_projector",
            "multimodal_projector",
        }
    ) or "projector" in leaf:
        return TensorRole.MULTIMODAL
    if "shared_expert" in lowered or "shared_experts" in lowered:
        return TensorRole.SHARED_EXPERT
    if "expert" in lowered:
        return TensorRole.ROUTED_EXPERT
    if segments.intersection({"self_attn", "attention", "attn", "indexer"}) or leaf in {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "kv_b_proj",
        "out_proj",
    }:
        return TensorRole.ATTENTION
    if segments.intersection({"mlp", "ffn", "feed_forward"}) or leaf in {
        "gate_proj",
        "up_proj",
        "down_proj",
    }:
        return TensorRole.FEED_FORWARD
    if segments.intersection({"recurrent", "linear_attn", "gated_delta"}) or leaf in {
        "dt_proj",
        "in_proj",
        "a_proj",
        "ba_proj",
    }:
        return TensorRole.RECURRENT
    return TensorRole.OTHER


def _protection_reason(
    *,
    name: str,
    module_name: str,
    role: TensorRole,
    initially_supported: bool,
    unsupported_reason: Optional[str],
    shared: bool,
    source_quantization: str,
    allow_prequantized_source: bool,
    include: Sequence[str],
    exclude: Sequence[str],
    protected_roles: Set[TensorRole],
) -> Optional[str]:
    if not initially_supported:
        return unsupported_reason or "unsupported_tensor_contract"
    if shared:
        return "shared_or_tied_weight"
    if source_quantization != "dense" and not allow_prequantized_source:
        return f"prequantized_source:{source_quantization}"
    if role in protected_roles:
        return f"protected_role:{role.value}"
    matched_include = _matches(name, include) or _matches(module_name, include)
    if not matched_include:
        return "not_selected_by_include_globs"
    if _matches(name, exclude) or _matches(module_name, exclude):
        return "excluded_by_globs"
    return None


def discover_tensor_specs(
    model: nn.Module,
    *,
    include: Sequence[str] = ("*",),
    exclude: Sequence[str] = (),
    protected_roles: Iterable[TensorRole] = DEFAULT_PROTECTED_ROLES,
    allow_prequantized_source: bool = False,
    allow_duck_typed_linear: bool = True,
) -> List[TensorSpec]:
    """Describe supported tensors and explicit dense/protected coverage gaps.

    Only exact ``nn.Linear``, HF ``Conv1D``, allowlisted fused expert banks,
    registered custom module adapters, and duck-typed linear modules are quantizable.
    Unknown ``nn.Linear`` subclasses remain unsupported unless registered.
    """

    include = tuple(include)
    exclude = tuple(exclude)
    if not include:
        raise ValueError("Tensor discovery requires at least one include glob")
    protected = {TensorRole(role) for role in protected_roles}
    shared_ids = _shared_weight_ids(model)
    shared_groups: Dict[int, str] = {}
    source_quantization = _source_quantization(model)
    specs: List[TensorSpec] = []

    def add_spec(
        *,
        name: str,
        module_name: str,
        attribute: str,
        module: nn.Module,
        weight: torch.Tensor,
        layout: TensorLayout,
        matrix_shape: Tuple[int, int],
        fixed_bytes: int = 0,
        supported: bool = True,
        unsupported_reason: Optional[str] = None,
        module_type: Optional[str] = None,
    ) -> None:
        role = classify_tensor_role(name, module)
        key = id(weight)
        shared = key in shared_ids
        if shared and key not in shared_groups:
            shared_groups[key] = f"shared-{len(shared_groups) + 1}"
        reason = _protection_reason(
            name=name,
            module_name=module_name,
            role=role,
            initially_supported=supported and weight.dtype.is_floating_point,
            unsupported_reason=(
                unsupported_reason
                if supported
                else unsupported_reason or "unsupported_tensor_contract"
            ),
            shared=shared,
            source_quantization=source_quantization,
            allow_prequantized_source=allow_prequantized_source,
            include=include,
            exclude=exclude,
            protected_roles=protected,
        )
        specs.append(
            TensorSpec(
                name=name,
                module_name=module_name,
                attribute=attribute,
                role=role,
                layout=layout,
                logical_shape=tuple(map(int, weight.shape)),
                matrix_shape=matrix_shape,
                dtype=str(weight.dtype).replace("torch.", ""),
                dtype_bytes=weight.element_size(),
                original_weight_bytes=weight.numel() * weight.element_size(),
                fixed_bytes=fixed_bytes,
                quantizable=reason is None,
                protection_reason=reason,
                shared_group=shared_groups.get(key),
                source_quantization=source_quantization,
                decode_group=module_name or "<root>",
                module_type=module_type,
            )
        )

    for module_name, module in model.named_modules():
        if isinstance(module, ResidualVQLinear):
            raise ValueError("Generalized planning expects a dense model, not existing VQ modules")

        # 1. Check custom module adapter registry
        custom_adapter = _find_custom_adapter(module)
        if custom_adapter is not None:
            reg_type, (extractor, replacer) = custom_adapter
            extracted = extractor(module_name, module)
            if extracted is not None:
                weight, bias, layout, attr = extracted
                lay = TensorLayout(layout)
                matrix_shape = (
                    tuple(map(int, weight.shape))
                    if lay == TensorLayout.LINEAR
                    else (int(weight.shape[1]), int(weight.shape[0]))
                    if lay == TensorLayout.CONV1D
                    else (weight.numel() // weight.shape[-1], weight.shape[-1])
                    if lay in {TensorLayout.FUSED_EXPERT_GATE_UP, TensorLayout.FUSED_EXPERT_DOWN}
                    else tuple(map(int, weight.shape))
                )
                add_spec(
                    name=module_name or "<root>",
                    module_name=module_name,
                    attribute=attr,
                    module=module,
                    weight=weight,
                    layout=lay,
                    matrix_shape=matrix_shape,
                    fixed_bytes=(
                        bias.numel() * bias.element_size()
                        if bias is not None
                        else 0
                    ),
                    supported=bool(module_name),
                    unsupported_reason="root_module_replacement_unsupported",
                    module_type=reg_type.__name__,
                )
                continue

        # 2. Known layer kinds (fused experts, exact linear, conv1d)
        kind = _layer_kind(module)
        if kind == "fused_experts":
            for attribute, layout in (
                ("gate_up_proj", TensorLayout.FUSED_EXPERT_GATE_UP),
                ("down_proj", TensorLayout.FUSED_EXPERT_DOWN),
            ):
                weight = getattr(module, attribute)
                add_spec(
                    name=f"{module_name}.{attribute}",
                    module_name=module_name,
                    attribute=attribute,
                    module=module,
                    weight=weight,
                    layout=layout,
                    matrix_shape=(weight.numel() // weight.shape[-1], weight.shape[-1]),
                    module_type="fused_experts",
                )
            continue
        if kind in {"linear", "conv1d"}:
            weight = module.weight
            matrix_shape = (
                tuple(map(int, weight.shape))
                if kind == "linear"
                else (int(weight.shape[1]), int(weight.shape[0]))
            )
            add_spec(
                name=module_name or "<root>",
                module_name=module_name,
                attribute="weight",
                module=module,
                weight=weight,
                layout=TensorLayout.LINEAR if kind == "linear" else TensorLayout.CONV1D,
                matrix_shape=matrix_shape,
                fixed_bytes=(
                    module.bias.numel() * module.bias.element_size()
                    if module.bias is not None
                    else 0
                ),
                supported=bool(module_name),
                unsupported_reason="root_module_replacement_unsupported",
                module_type=kind,
            )
            continue
        if isinstance(module, nn.Embedding):
            add_spec(
                name=module_name or "<root>",
                module_name=module_name,
                attribute="weight",
                module=module,
                weight=module.weight,
                layout=TensorLayout.EMBEDDING,
                matrix_shape=tuple(map(int, module.weight.shape)),
                supported=False,
                unsupported_reason="embedding_codec_not_implemented",
                module_type="embedding",
            )
            continue

        # 3. Duck-typed linear projections
        if allow_duck_typed_linear and _is_duck_typed_linear(module):
            weight = module.weight
            bias = getattr(module, "bias", None)
            matrix_shape = tuple(map(int, weight.shape))
            add_spec(
                name=module_name or "<root>",
                module_name=module_name,
                attribute="weight",
                module=module,
                weight=weight,
                layout=TensorLayout.LINEAR,
                matrix_shape=matrix_shape,
                fixed_bytes=(
                    bias.numel() * bias.element_size()
                    if bias is not None
                    else 0
                ),
                supported=bool(module_name),
                unsupported_reason="root_module_replacement_unsupported",
                module_type="custom_linear",
            )
            continue

        # 4. Fallback for unregistered nn.Linear subclasses
        if isinstance(module, nn.Linear):
            # _layer_kind accepts exact nn.Linear only. This branch therefore
            # describes, but never silently replaces, a custom subclass.
            add_spec(
                name=module_name or "<root>",
                module_name=module_name,
                attribute="weight",
                module=module,
                weight=module.weight,
                layout=TensorLayout.UNSUPPORTED,
                matrix_shape=tuple(map(int, module.weight.shape)),
                fixed_bytes=(
                    module.bias.numel() * module.bias.element_size()
                    if module.bias is not None
                    else 0
                ),
                supported=False,
                unsupported_reason=f"unsupported_linear_subclass:{type(module).__name__}",
            )
    return specs


def tensor_weight(model: nn.Module, spec: TensorSpec) -> torch.Tensor:
    """Return the rank-two source view described by a tensor spec."""

    module = model.get_submodule(spec.module_name)
    weight = getattr(module, spec.attribute)
    if spec.layout == TensorLayout.CONV1D:
        return weight.T
    if spec.layout in {
        TensorLayout.FUSED_EXPERT_GATE_UP,
        TensorLayout.FUSED_EXPERT_DOWN,
    }:
        return weight.reshape(-1, weight.shape[-1])
    return weight


def _validate_dense_model(model: nn.Module, plan: QuantizationPlan) -> None:
    actual_base = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    if actual_base != plan.base_model_state_bytes:
        raise ValueError(
            f"Model state changed since planning: {actual_base} != {plan.base_model_state_bytes}"
        )
    for spec in plan.specs:
        try:
            weight = tensor_weight(model, spec)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"Cannot resolve planned tensor {spec.name!r}") from error
        if tuple(weight.shape) != spec.matrix_shape:
            raise ValueError(f"Planned matrix shape mismatch for {spec.name!r}")
        if str(weight.dtype).replace("torch.", "") != spec.dtype:
            raise ValueError(f"Planned dtype mismatch for {spec.name!r}")


def _ordinary_report(
    spec: TensorSpec,
    candidate: QuantizationCandidate,
    module_type: Optional[str] = None,
) -> LayerQuantizationReport:
    selected = not candidate.is_dense
    return LayerQuantizationReport(
        name=spec.name,
        shape=spec.matrix_shape,
        selected=selected,
        reason=(
            "admitted_by_generalized_planner"
            if selected
            else spec.protection_reason or "planner_selected_dense"
        ),
        original_bytes=spec.original_weight_bytes,
        storage_bytes=(
            candidate.encoded_weight_bytes if selected else spec.original_weight_bytes
        ),
        relative_error=candidate.weight_relative_error if selected else None,
        selected_stages=candidate.stages,
        nominal_bits_per_weight=(
            candidate.stages * candidate.bits_per_stage / candidate.block_size
            if selected
            else 0.0
        ),
        measured_bits_per_weight=(
            8.0 * candidate.encoded_weight_bytes / spec.weight_elements if selected else 0.0
        ),
        module_type=module_type or spec.module_type or spec.layout.value,
    )


def apply_quantization_plan(
    model: nn.Module,
    plan: QuantizationPlan,
    *,
    runtime_profile: str = "memory",
    allow_duck_typed_linear: bool = True,
) -> Tuple[nn.Module, ModelQuantizationReport]:
    """Apply a validated plan and verify the predicted persistent-state bytes."""

    if any(isinstance(module, ResidualVQLinear) for module in model.modules()):
        raise ValueError("Cannot apply a generalized plan to an already quantized model")
    _validate_dense_model(model, plan)
    selected = plan.selected_candidates

    # Ordinary projections can be replaced independently. Root modules are always
    # protected by discovery because Python cannot replace the root object in place.
    for spec in plan.specs:
        candidate = selected[spec.name]
        if candidate.is_dense:
            continue
        module = model.get_submodule(spec.module_name)
        custom_adapter = _find_custom_adapter(module)
        is_custom_with_replacer = custom_adapter is not None and custom_adapter[1][1] is not None
        if not is_custom_with_replacer and spec.layout not in {TensorLayout.LINEAR, TensorLayout.CONV1D}:
            continue
        if not spec.quantizable or candidate.encoded is None:
            raise ValueError(f"Selected payload is unavailable or protected for {spec.name!r}")
        if not allow_duck_typed_linear and spec.module_type == "custom_linear" and custom_adapter is None:
            raise ValueError(f"Duck-typed linear replacement disabled for {spec.name!r}")

        if is_custom_with_replacer:
            replacer = custom_adapter[1][1]
            try:
                replacement = replacer(module, candidate, spec)
            except TypeError:
                replacement = replacer(module, candidate)
        else:
            replacement_type = (
                ResidualVQLinear if spec.layout == TensorLayout.LINEAR else ResidualVQConv1D
            )
            weight = getattr(module, spec.attribute)
            replacement = replacement_type.from_encoded(
                candidate.encoded,
                bias=getattr(module, "bias", None),
                source_dtype=weight.dtype,
                device=weight.device,
            )
        parent_name, separator, child_name = spec.module_name.rpartition(".")
        if not separator:
            child_name = spec.module_name
        setattr(model.get_submodule(parent_name), child_name, replacement)

    # A fused bank is one decode group: both projections may be dense, both VQ,
    # or mixed. Rebuild it once so partial plans preserve the original contract.
    fused_groups: Dict[str, List[TensorSpec]] = {}
    for spec in plan.specs:
        if spec.layout in {
            TensorLayout.FUSED_EXPERT_GATE_UP,
            TensorLayout.FUSED_EXPERT_DOWN,
        }:
            fused_groups.setdefault(spec.module_name, []).append(spec)
    for module_name, group_specs in fused_groups.items():
        if {spec.attribute for spec in group_specs} != {"gate_up_proj", "down_proj"}:
            raise ValueError(f"Fused plan for {module_name!r} needs both expert projections")
        if all(selected[spec.name].is_dense for spec in group_specs):
            continue
        module = model.get_submodule(module_name)
        if _layer_kind(module) != "fused_experts":
            raise TypeError(f"Planned expert bank {module_name!r} changed contract")

        def projection(attribute: str):
            spec = next(spec for spec in group_specs if spec.attribute == attribute)
            candidate = selected[spec.name]
            parameter = getattr(module, attribute)
            if candidate.is_dense:
                return parameter
            if not spec.quantizable or candidate.encoded is None:
                raise ValueError(f"Selected expert payload is unavailable for {spec.name!r}")
            return ResidualVQLinear.from_encoded(
                candidate.encoded,
                bias=None,
                source_dtype=parameter.dtype,
                device=parameter.device,
            )

        replacement = ResidualVQFusedExperts(
            int(module.num_experts),
            int(module.hidden_dim),
            int(module.intermediate_dim),
            gate_up_proj=projection("gate_up_proj"),
            down_proj=projection("down_proj"),
            act_fn=_expert_activation(module),
            gate_limit=_expert_gate_limit(module),
        )
        parent_name, separator, child_name = module_name.rpartition(".")
        if not separator:
            child_name = module_name
        setattr(model.get_submodule(parent_name), child_name, replacement)

    set_runtime_profile(model, runtime_profile)
    set_decode_row_block_size(model, int(plan.constraints.get("decode_row_block_size", 64)))
    reports = [_ordinary_report(spec, selected[spec.name]) for spec in plan.specs]
    report = ModelQuantizationReport(
        reports,
        configuration={
            "planner": "bounded_pareto",
            "constraints": plan.constraints,
        },
        preserve_shared_weights=True,
    )
    actual = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
    predicted = int(plan.memory["predicted_model_state_bytes"])
    if actual != predicted:
        raise RuntimeError(f"Persistent-state prediction mismatch: actual {actual}, planned {predicted}")
    plan.memory["actual_model_state_bytes_after_apply"] = actual
    plan.memory["actual_matches_prediction"] = True
    plan.quality["quantized_tensors"] = selected_quantized_count(plan)
    plan.compact_unselected_payloads()
    model.eval()
    return model, report

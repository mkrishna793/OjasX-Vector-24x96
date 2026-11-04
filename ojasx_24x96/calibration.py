"""Bounded activation capture and per-candidate local output calibration."""

from __future__ import annotations

from collections import defaultdict
import inspect
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.nn import functional as F

from .codec import dequantize_weight
from .leech_codec import LeechQuantizedWeight, dequantize_leech_weight
from .ir import (
    CandidateCollection,
    QuantizationCandidate,
    TensorLayout,
    TensorRole,
    TensorSpec,
)
from .module import (
    ResidualVQFusedExperts,
    ResidualVQLinear,
    _expert_activation,
    _expert_gate_limit,
)


Batch = object


def _model_device(model: torch.nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.device.type != "meta"
    }
    if len(devices) > 1:
        raise NotImplementedError(
            "Activation calibration currently requires one execution device; "
            "device-map-specific calibration must be explicit"
        )
    return next(iter(devices), torch.device("cpu"))


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _disable_hf_cache(model, batch):
    if not isinstance(batch, Mapping) or "use_cache" in batch:
        return batch
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return batch
    if "use_cache" in parameters:
        updated = dict(batch)
        updated["use_cache"] = False
        return updated
    return batch


def _forward(model, batch):
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


def _sample_rows(tensor: torch.Tensor, maximum: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = tensor.shape[0]
    if rows <= maximum:
        indices = torch.arange(rows, dtype=torch.int64)
    else:
        indices = (torch.arange(maximum, dtype=torch.int64) * rows) // maximum
    return indices, indices.to(tensor.device)


def _capture_args(
    kind: str,
    args: Tuple[object, ...],
    max_rows: int,
) -> Tuple[torch.Tensor, ...]:
    if kind == "ordinary":
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("Ordinary projection calibration needs a tensor input")
        inputs = args[0]
        if inputs.ndim < 1:
            raise ValueError("Projection input must have a feature dimension")
        rows = inputs.reshape(-1, inputs.shape[-1])
        _, device_indices = _sample_rows(rows, max_rows)
        return (rows[device_indices].detach().clone(),)
    if len(args) < 3 or not all(isinstance(value, torch.Tensor) for value in args[:3]):
        raise ValueError("Fused-expert calibration needs hidden/index/weight tensors")
    hidden, top_k_index, top_k_weights = args[:3]
    if hidden.ndim != 2 or top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        raise ValueError("Fused-expert calibration currently expects rank-two token tensors")
    if not (
        top_k_index.shape[0] == top_k_weights.shape[0] == hidden.shape[0]
        and top_k_index.shape == top_k_weights.shape
    ):
        raise ValueError("Fused-expert token routing shapes are inconsistent")
    _, device_indices = _sample_rows(hidden, max_rows)
    return (
        hidden[device_indices].detach().clone(),
        top_k_index[device_indices].detach().clone(),
        top_k_weights[device_indices].detach().clone(),
    )


def _accumulate_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> Tuple[float, float, float, int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Calibration output shape changed: {reference.shape} vs {candidate.shape}")
    if not torch.isfinite(reference).all().item():
        raise ValueError("Reference module produced nonfinite calibration outputs")
    if not torch.isfinite(candidate).all().item():
        raise FloatingPointError("Candidate produced nonfinite calibration outputs")
    reference64 = reference.detach().double()
    difference = candidate.detach().double() - reference64
    error_squared = difference.square().sum().item()
    reference_squared = reference64.square().sum().item()
    maximum = difference.abs().max().item() if difference.numel() else 0.0
    return error_squared, reference_squared, maximum, reference.numel()


def _finish_error(
    candidate: QuantizationCandidate,
    error_squared: float,
    reference_squared: float,
    maximum: float,
    values: int,
    invocations: int,
) -> None:
    relative = math.sqrt(error_squared / max(reference_squared, 1e-30))
    if not math.isfinite(relative) or not math.isfinite(maximum):
        raise FloatingPointError("Nonfinite local calibration metric")
    candidate.local_output_relative_error = relative
    candidate.local_output_max_absolute_error = maximum
    candidate.calibration_values = values
    candidate.calibration_invocations = invocations


@torch.no_grad()
def calibrate_candidates(
    model: torch.nn.Module,
    specs: Sequence[TensorSpec],
    candidates: CandidateCollection,
    batches: Iterable[Batch],
    *,
    max_activation_rows: int = 128,
    max_invocations_per_module: int = 8,
) -> Dict[str, object]:
    """Measure candidate outputs on bounded captured inputs before model mutation.

    For a fused bank, each projection is calibrated independently while the other
    remains dense. This intentionally does not claim to capture cross-projection
    or cross-layer interactions; final model-level gates are still required.
    """

    if max_activation_rows <= 0 or max_invocations_per_module <= 0:
        raise ValueError("Calibration row and invocation bounds must be positive")
    batches = tuple(batches)
    if not batches:
        raise ValueError("Calibration requires at least one input batch")
    module_specs: Dict[str, List[TensorSpec]] = defaultdict(list)
    for spec in specs:
        if any(not candidate.is_dense for candidate in candidates[spec.name]):
            module_specs[spec.module_name].append(spec)
    if not module_specs:
        return {
            "batches": len(batches),
            "captured_modules": 0,
            "uncalibrated_tensors": [],
            "capture_errors": {},
            "scope": "No VQ candidates required calibration",
        }

    device = _model_device(model)
    captures: Dict[str, List[Tuple[torch.Tensor, ...]]] = defaultdict(list)
    capture_errors: Dict[str, str] = {}
    handles = []
    was_training = model.training
    model.eval()
    try:
        for module_name, group_specs in module_specs.items():
            module = model.get_submodule(module_name)
            kind = (
                "fused"
                if any(
                    spec.layout in {
                        TensorLayout.FUSED_EXPERT_GATE_UP,
                        TensorLayout.FUSED_EXPERT_DOWN,
                    }
                    for spec in group_specs
                )
                else "ordinary"
            )

            def hook(_module, args, _output, *, name=module_name, capture_kind=kind):
                if len(captures[name]) >= max_invocations_per_module or name in capture_errors:
                    return
                try:
                    captures[name].append(_capture_args(capture_kind, args, max_activation_rows))
                except (TypeError, ValueError) as error:
                    capture_errors[name] = str(error)

            handles.append(module.register_forward_hook(hook))
        for original_batch in batches:
            batch = _move_to_device(original_batch, device)
            batch = _disable_hf_cache(model, batch)
            _forward(model, batch)
    except Exception:
        model.train(was_training)
        raise
    finally:
        for handle in handles:
            handle.remove()

    baseline_outputs: Dict[str, List[torch.Tensor]] = {}
    try:
        for module_name, invocations in captures.items():
            module = model.get_submodule(module_name)
            baseline_outputs[module_name] = [module(*args) for args in invocations]

        for module_name, group_specs in module_specs.items():
            invocations = captures.get(module_name, [])
            references = baseline_outputs.get(module_name, [])
            if not invocations:
                continue
            module = model.get_submodule(module_name)
            for spec in group_specs:
                for candidate in candidates[spec.name]:
                    if candidate.is_dense:
                        values = sum(reference.numel() for reference in references)
                        _finish_error(candidate, 0.0, 1.0, 0.0, values, len(references))
                        continue
                    if candidate.encoded is None:
                        raise ValueError(
                            f"Candidate payload for {spec.name!r} was released before calibration"
                        )
                    error_squared = 0.0
                    reference_squared = 0.0
                    maximum = 0.0
                    values = 0
                    try:
                        if spec.layout in {TensorLayout.LINEAR, TensorLayout.CONV1D}:
                            for args, reference in zip(invocations, references):
                                inputs = args[0]
                                decoder = (
                                    dequantize_leech_weight
                                    if isinstance(candidate.encoded, LeechQuantizedWeight)
                                    else dequantize_weight
                                )
                                decoded = decoder(
                                    candidate.encoded,
                                    dtype=inputs.dtype,
                                ).to(inputs.device)
                                bias = (
                                    module.bias.to(inputs.dtype)
                                    if module.bias is not None
                                    else None
                                )
                                output = F.linear(inputs, decoded, bias)
                                error, reference_norm, current_max, count = _accumulate_error(
                                    reference, output
                                )
                                error_squared += error
                                reference_squared += reference_norm
                                maximum = max(maximum, current_max)
                                values += count
                        else:
                            parameter = getattr(module, spec.attribute)
                            projection = ResidualVQLinear.from_encoded(
                                candidate.encoded,
                                bias=None,
                                source_dtype=parameter.dtype,
                                device=parameter.device,
                            )
                            replacement = ResidualVQFusedExperts(
                                int(module.num_experts),
                                int(module.hidden_dim),
                                int(module.intermediate_dim),
                                gate_up_proj=(
                                    projection
                                    if spec.attribute == "gate_up_proj"
                                    else module.gate_up_proj
                                ),
                                down_proj=(
                                    projection
                                    if spec.attribute == "down_proj"
                                    else module.down_proj
                                ),
                                act_fn=_expert_activation(module),
                                gate_limit=_expert_gate_limit(module),
                            )
                            for args, reference in zip(invocations, references):
                                output = replacement(*args)
                                error, reference_norm, current_max, count = _accumulate_error(
                                    reference, output
                                )
                                error_squared += error
                                reference_squared += reference_norm
                                maximum = max(maximum, current_max)
                                values += count
                        _finish_error(
                            candidate,
                            error_squared,
                            reference_squared,
                            maximum,
                            values,
                            len(references),
                        )
                    except FloatingPointError as error:
                        capture_errors[f"{spec.name}:{candidate.candidate_id}"] = str(error)
                        candidate.local_output_relative_error = None
                        candidate.local_output_max_absolute_error = None
                        candidate.calibration_values = 0
                        candidate.calibration_invocations = 0
    finally:
        model.train(was_training)

    uncalibrated = sorted(
        {
            spec.name
            for spec in specs
            for candidate in candidates[spec.name]
            if not candidate.is_dense and not candidate.calibrated
        }
    )
    return {
        "batches": len(batches),
        "captured_modules": len(captures),
        "captured_invocations": sum(len(values) for values in captures.values()),
        "max_activation_rows": max_activation_rows,
        "uncalibrated_tensors": uncalibrated,
        "capture_errors": capture_errors,
        "scope": (
            "Bounded local-output measurements on supplied inputs only; fused "
            "projections are isolated and cross-layer/final-task effects are not bounded"
        ),
    }


def compute_layer_sensitivity_profile(
    specs: Sequence[TensorSpec],
    candidates_by_tensor: Mapping[str, Sequence[QuantizationCandidate]],
    model: Optional[torch.nn.Module] = None,
    calibration_inputs: Optional[Sequence[object]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-layer sensitivity scores for quantization ranking.

    Scores include hessian_trace_proxy, activation_amplification,
    min_candidate_distortion, max_candidate_distortion, and recommended_min_bits.
    Attention output (o_proj) and feedforward down-projections (down_proj) that
    project directly into the residual stream receive higher sensitivity weighting.
    """
    captures_in: Dict[str, List[torch.Tensor]] = defaultdict(list)
    captures_out: Dict[str, List[torch.Tensor]] = defaultdict(list)

    if model is not None and calibration_inputs is not None:
        batches = tuple(calibration_inputs)
        if batches:
            device = _model_device(model)
            handles = []
            was_training = model.training
            model.eval()
            try:
                for spec in specs:
                    try:
                        submodule = model.get_submodule(spec.module_name)
                    except (AttributeError, KeyError):
                        continue

                    def make_hook(name=spec.module_name):
                        def hook(_m, args, out):
                            if len(captures_in[name]) < 8:
                                if args and isinstance(args[0], torch.Tensor):
                                    captures_in[name].append(args[0].detach().float().cpu())
                                if isinstance(out, torch.Tensor):
                                    captures_out[name].append(out.detach().float().cpu())
                                elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                                    captures_out[name].append(out[0].detach().float().cpu())
                        return hook

                    handles.append(submodule.register_forward_hook(make_hook()))

                with torch.inference_mode():
                    for batch in batches:
                        b = _move_to_device(batch, device)
                        b = _disable_hf_cache(model, b)
                        _forward(model, b)
            except Exception:
                pass
            finally:
                for h in handles:
                    h.remove()
                model.train(was_training)

    profile: Dict[str, Dict[str, float]] = {}
    for spec in specs:
        cands = [c for c in candidates_by_tensor.get(spec.name, []) if not c.is_dense]
        if cands:
            min_dist = float(min(c.estimated_error for c in cands))
            max_dist = float(max(c.estimated_error for c in cands))
        else:
            min_dist = 0.0
            max_dist = 0.0

        name_lower = spec.name.lower()
        if "lm_head" in name_lower or spec.role == TensorRole.OUTPUT_HEAD:
            role_mult = 2.5
            target_bits = 4.0
        elif "o_proj" in name_lower or "out_proj" in name_lower or (spec.role == TensorRole.ATTENTION and "down" in name_lower):
            role_mult = 2.0
            target_bits = 4.0
        elif "down_proj" in name_lower or (spec.role == TensorRole.FEED_FORWARD and "down" in name_lower) or "fc2" in name_lower:
            role_mult = 1.8
            target_bits = 3.5
        elif "v_proj" in name_lower:
            role_mult = 1.4
            target_bits = 3.0
        elif "q_proj" in name_lower or "k_proj" in name_lower or spec.role == TensorRole.ATTENTION:
            role_mult = 1.2
            target_bits = 2.5
        elif "gate_proj" in name_lower or "up_proj" in name_lower or spec.role == TensorRole.FEED_FORWARD:
            role_mult = 1.1
            target_bits = 2.5
        elif spec.role == TensorRole.ROUTER:
            role_mult = 2.2
            target_bits = 4.0
        else:
            role_mult = 1.0
            target_bits = 2.0

        in_acts = captures_in.get(spec.module_name, [])
        out_acts = captures_out.get(spec.module_name, [])

        if in_acts:
            mean_sq = float(sum(t.pow(2).sum(dim=-1).mean().item() for t in in_acts) / len(in_acts))
            w_norm = 1.0
            if model is not None:
                try:
                    param = getattr(model.get_submodule(spec.module_name), spec.attribute)
                    w_norm = float(param.float().norm().item())
                except Exception:
                    pass
            hessian_trace_proxy = (mean_sq + w_norm * (spec.matrix_shape[1] / 32.0)) * role_mult
        elif model is not None:
            try:
                submodule = model.get_submodule(spec.module_name)
                param = getattr(submodule, spec.attribute)
                hessian_trace_proxy = float((param.float() ** 2).mean().item() * spec.matrix_shape[1]) * role_mult
            except Exception:
                hessian_trace_proxy = float(math.sqrt(spec.matrix_shape[1])) * role_mult
        else:
            hessian_trace_proxy = float(math.sqrt(spec.matrix_shape[1])) * role_mult

        if in_acts and out_acts:
            in_norm = sum(t.norm().item() for t in in_acts)
            out_norm = sum(t.norm().item() for t in out_acts)
            gain = out_norm / max(in_norm, 1e-12)
            activation_amplification = float(gain) * role_mult
        elif model is not None:
            try:
                submodule = model.get_submodule(spec.module_name)
                param = getattr(submodule, spec.attribute)
                gain = param.float().norm().item() / max(1e-6, math.sqrt(spec.matrix_shape[1]))
                activation_amplification = float(gain) * role_mult
            except Exception:
                gain = math.sqrt(spec.matrix_shape[0] / max(1, spec.matrix_shape[1]))
                activation_amplification = float(gain) * role_mult
        else:
            gain = math.sqrt(spec.matrix_shape[0] / max(1, spec.matrix_shape[1]))
            activation_amplification = float(gain) * role_mult

        rec_bits = target_bits
        if cands:
            cand_bits = sorted([c.measured_bits_per_weight for c in cands if c.measured_bits_per_weight > 0])
            if cand_bits:
                eligible = [b for b in cand_bits if b >= target_bits - 0.5]
                rec_bits = eligible[0] if eligible else cand_bits[-1]

        profile[spec.name] = {
            "hessian_trace_proxy": float(hessian_trace_proxy),
            "activation_amplification": float(activation_amplification),
            "min_candidate_distortion": float(min_dist),
            "max_candidate_distortion": float(max_dist),
            "recommended_min_bits": float(rec_bits),
        }

    return profile

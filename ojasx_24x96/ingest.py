"""Bounded-RAM streaming Hugging Face checkpoint ingestion and resumable conversion (Milestone M2).

Converts HF checkpoints (.safetensors single or sharded with model.safetensors.index.json)
or standalone state_dict files into residual-VQ packages tensor-by-tensor with bounded RAM
(never loading more than 1 dense weight tensor into RAM at a time) and resumable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import safetensors.torch
from safetensors import safe_open
import torch
import torch.nn as nn

from .codec import QuantizationConfig, quantize_weight
from .leech_codec import LeechQuantizationConfig, quantize_leech_weight
from .module import ResidualVQFusedExperts, ResidualVQLinear, _expert_activation, _expert_gate_limit, _infer_fused_expert_dims


class CheckpointIngestionError(ValueError):
    """Raised when checkpoint discovery, sharding, or tensor mappings are invalid."""


@dataclass
class StreamingTensorRecord:
    """Record describing the conversion status and location of a single tensor."""

    name: str
    status: str
    codec: str
    original_shape: Tuple[int, ...]
    original_bytes: int
    stored_bytes: int
    shard_file: str
    sha256: str
    metrics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "codec": self.codec,
            "original_shape": list(self.original_shape),
            "original_bytes": self.original_bytes,
            "stored_bytes": self.stored_bytes,
            "shard_file": self.shard_file,
            "sha256": self.sha256,
            "metrics": dict(self.metrics),
        }


@dataclass
class StreamingConversionReport:
    """Summary report of streaming checkpoint conversion."""

    source_dir: str
    output_dir: str
    total_tensors: int
    quantized_tensors: int
    passthrough_tensors: int
    resumed_tensors: int
    original_bytes: int
    stored_bytes: int
    peak_single_tensor_bytes: int
    records: Dict[str, StreamingTensorRecord] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.stored_bytes <= 0:
            return 1.0
        return float(self.original_bytes) / float(self.stored_bytes)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "total_tensors": self.total_tensors,
            "quantized_tensors": self.quantized_tensors,
            "passthrough_tensors": self.passthrough_tensors,
            "resumed_tensors": self.resumed_tensors,
            "original_bytes": self.original_bytes,
            "stored_bytes": self.stored_bytes,
            "peak_single_tensor_bytes": self.peak_single_tensor_bytes,
            "compression_ratio": self.compression_ratio,
            "records": {k: v.to_dict() for k, v in self.records.items()},
        }


def _atomic_write_json(path: Path, data: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_checkpoint_safetensors(
    checkpoint_dir: Union[str, Path],
) -> Tuple[List[Path], Optional[Dict[str, str]]]:
    """Finds model.safetensors or sharded *.safetensors + model.safetensors.index.json.

    Validates that every weight in weight_map exists in the shard files and there
    are no orphan/missing shards. Raises CheckpointIngestionError if invalid.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise CheckpointIngestionError(f"Checkpoint path does not exist: {ckpt_path}")

    if ckpt_path.is_file():
        if ckpt_path.suffix != ".safetensors":
            raise CheckpointIngestionError(
                f"Checkpoint file must be a .safetensors file, got: {ckpt_path.name}"
            )
        try:
            with safe_open(str(ckpt_path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as e:
            raise CheckpointIngestionError(f"Failed to open safetensors file {ckpt_path}: {e}") from e
        weight_map = {k: ckpt_path.name for k in keys}
        return [ckpt_path], weight_map

    # Directory checkpoint
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise CheckpointIngestionError(f"Malformed model.safetensors.index.json: {e}") from e

        if not isinstance(index_data, dict) or "weight_map" not in index_data:
            raise CheckpointIngestionError("model.safetensors.index.json missing 'weight_map' dictionary")

        weight_map = index_data["weight_map"]
        if not isinstance(weight_map, dict):
            raise CheckpointIngestionError("'weight_map' in model.safetensors.index.json must be a dict")

        referenced_shards = sorted(set(weight_map.values()))
        shard_paths: List[Path] = []
        for shard_name in referenced_shards:
            s_path = ckpt_path / shard_name
            if not s_path.exists():
                raise CheckpointIngestionError(
                    f"Missing shard file referenced in weight_map: {shard_name}"
                )
            shard_paths.append(s_path)

        # Check for orphan shards
        existing_safetensors = {p.name for p in ckpt_path.glob("*.safetensors")}
        orphan_shards = existing_safetensors - set(referenced_shards)
        if orphan_shards:
            raise CheckpointIngestionError(
                f"Orphan shard(s) found in {ckpt_path}: {sorted(orphan_shards)}"
            )

        # Validate that every weight in weight_map exists in the shard files
        shard_keys_map: Dict[str, Set[str]] = {}
        for s_path in shard_paths:
            try:
                with safe_open(str(s_path), framework="pt", device="cpu") as handle:
                    shard_keys_map[s_path.name] = set(handle.keys())
            except Exception as e:
                raise CheckpointIngestionError(f"Failed to read shard {s_path.name}: {e}") from e

        for weight_name, target_shard in weight_map.items():
            if target_shard not in shard_keys_map:
                raise CheckpointIngestionError(
                    f"Weight {weight_name!r} mapped to nonexistent shard {target_shard!r}"
                )
            if weight_name not in shard_keys_map[target_shard]:
                raise CheckpointIngestionError(
                    f"Weight {weight_name!r} mapped to {target_shard!r} but not found in file"
                )

        # Validate that shard files do not have orphan weights not in weight_map
        all_mapped_weights = set(weight_map.keys())
        for s_name, keys in shard_keys_map.items():
            unmapped = keys - all_mapped_weights
            if unmapped:
                raise CheckpointIngestionError(
                    f"Shard {s_name!r} contains unmapped weight(s) not in weight_map: {sorted(unmapped)}"
                )

        return shard_paths, weight_map

    # Check for single model.safetensors
    single_file = ckpt_path / "model.safetensors"
    if single_file.exists():
        try:
            with safe_open(str(single_file), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as e:
            raise CheckpointIngestionError(f"Failed to read model.safetensors: {e}") from e
        weight_map = {k: single_file.name for k in keys}
        return [single_file], weight_map

    # Check for any *.safetensors files
    all_shards = sorted(ckpt_path.glob("*.safetensors"))
    if len(all_shards) == 1:
        single_file = all_shards[0]
        try:
            with safe_open(str(single_file), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as e:
            raise CheckpointIngestionError(f"Failed to read {single_file.name}: {e}") from e
        weight_map = {k: single_file.name for k in keys}
        return [single_file], weight_map
    elif len(all_shards) > 1:
        raise CheckpointIngestionError(
            f"Multiple .safetensors shards found in {ckpt_path} but model.safetensors.index.json is missing"
        )
    else:
        raise CheckpointIngestionError(f"No .safetensors checkpoint found in {ckpt_path}")


def should_quantize_tensor_name(
    name: str,
    shape: Tuple[int, ...],
    *,
    include_globs: Sequence[str] = ("*",),
    exclude_globs: Sequence[str] = ("*embed*", "*lm_head*", "*norm*", "*.bias"),
    allow_3d_experts: bool = False,
) -> bool:
    """Determines whether a 2D (or 3D expert) weight tensor should be quantized vs stored as passthrough dense tensor."""
    if len(shape) == 3 and allow_3d_experts:
        if not (name.endswith("gate_up_proj") or name.endswith("down_proj")):
            return False
    elif len(shape) != 2:
        return False
    if any(fnmatch.fnmatch(name, pat) for pat in exclude_globs):
        return False
    if not any(fnmatch.fnmatch(name, pat) for pat in include_globs):
        return False
    return True


def stream_convert_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    codec: str = "adaptive_residual_vq",
    vq_config: Optional[QuantizationConfig] = None,
    leech_config: Optional[LeechQuantizationConfig] = None,
    include_globs: Sequence[str] = ("*",),
    exclude_globs: Sequence[str] = ("*embed*", "*lm_head*", "*norm*", "*.bias"),
    resume: bool = True,
    fail_after_tensors: Optional[int] = None,
    allow_3d_experts: bool = True,
) -> StreamingConversionReport:
    """Streamingly converts a checkpoint tensor-by-tensor with bounded RAM and resumability."""
    if codec not in {"adaptive_residual_vq", "leech24_shell2_residual"}:
        raise ValueError(f"Unsupported residual codec: {codec!r}")

    shards, weight_map = discover_checkpoint_safetensors(checkpoint_dir)
    ckpt_path = Path(checkpoint_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-safetensors metadata files (config.json, generation_config.json, tokenizer, etc.)
    if ckpt_path.is_dir():
        for item in ckpt_path.iterdir():
            if (
                item.is_file()
                and not item.name.endswith(".safetensors")
                and item.name not in {"model.safetensors.index.json", "conversion_state.json"}
            ):
                dest = out_dir / item.name
                if not dest.exists():
                    try:
                        shutil.copy2(item, dest)
                    except Exception:
                        pass

    state_path = out_dir / "conversion_state.json"
    records: Dict[str, StreamingTensorRecord] = {}
    resumed_tensors = 0

    if resume and state_path.exists():
        try:
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            existing_records = state_data.get("records", {})
            for t_name, r_info in existing_records.items():
                shard_rel = r_info.get("shard_file", "")
                shard_path = out_dir / shard_rel
                if shard_path.exists():
                    disk_sha = _compute_file_sha256(shard_path)
                    if disk_sha == r_info.get("sha256"):
                        records[t_name] = StreamingTensorRecord(
                            name=r_info["name"],
                            status=r_info["status"],
                            codec=r_info["codec"],
                            original_shape=tuple(r_info["original_shape"]),
                            original_bytes=int(r_info["original_bytes"]),
                            stored_bytes=int(r_info["stored_bytes"]),
                            shard_file=shard_rel,
                            sha256=disk_sha,
                            metrics=dict(r_info.get("metrics", {})),
                        )
        except Exception:
            records = {}

    # Build queue of (tensor_name, shard_path)
    if weight_map is not None:
        base_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        tensor_queue = [(k, base_dir / v) for k, v in weight_map.items()]
    else:
        tensor_queue = []
        for s_path in shards:
            with safe_open(str(s_path), framework="pt", device="cpu") as handle:
                for k in handle.keys():
                    tensor_queue.append((k, s_path))

    peak_single_tensor_bytes = 0
    if records:
        peak_single_tensor_bytes = max((r.original_bytes for r in records.values()), default=0)

    newly_converted = 0

    for tensor_name, shard_path in tensor_queue:
        if resume and tensor_name in records:
            resumed_tensors += 1
            continue

        # Slices ONLY that single tensor from safe_open
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)

        tensor_bytes = tensor.numel() * tensor.element_size()
        peak_single_tensor_bytes = max(peak_single_tensor_bytes, tensor_bytes)

        orig_tensor_shape = tuple(tensor.shape)
        should_q = should_quantize_tensor_name(
            tensor_name,
            orig_tensor_shape,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            allow_3d_experts=allow_3d_experts,
        )

        safe_name = tensor_name.replace("/", "_")
        out_shard_filename = f"{safe_name}.safetensors"
        out_shard_path = out_dir / out_shard_filename

        prefix = tensor_name[:-len(".weight")] if tensor_name.endswith(".weight") else tensor_name
        prefix_dot = f"{prefix}." if prefix else ""

        if not should_q:
            status = "passthrough"
            codec_used = "passthrough"
            metrics: Dict[str, object] = {"reason": "excluded_or_ineligible"}
            tensors_to_save = {tensor_name: tensor}
            stored_bytes = tensor_bytes
        else:
            q_tensor = tensor.reshape(-1, tensor.shape[-1]) if tensor.ndim == 3 else tensor
            if codec == "adaptive_residual_vq":
                cfg = vq_config if vq_config is not None else QuantizationConfig()
                encoded, measurement = quantize_weight(q_tensor, config=cfg)
            else:
                cfg = leech_config if leech_config is not None else LeechQuantizationConfig()
                encoded, measurement = quantize_leech_weight(q_tensor, config=cfg)

            if encoded is None:
                status = "passthrough"
                codec_used = "passthrough"
                metrics = dict(measurement)
                tensors_to_save = {tensor_name: tensor}
                stored_bytes = tensor_bytes
            else:
                status = "quantized"
                codec_used = codec
                metrics = dict(measurement)
                metrics["block_size"] = encoded.block_size
                metrics["bits_per_stage"] = encoded.bits_per_stage
                metrics["stages"] = encoded.stages
                metrics["selected_stages"] = encoded.stages
                metrics["original_shape"] = list(encoded.original_shape)
                metrics["logical_shape"] = list(orig_tensor_shape)
                metrics["is_3d_expert"] = len(orig_tensor_shape) == 3
                metrics["codec"] = codec
                stored_bytes = encoded.storage_bytes

                if codec == "adaptive_residual_vq":
                    tensors_to_save = {
                        f"{prefix_dot}packed_codes": encoded.packed_codes,
                        f"{prefix_dot}codebooks": encoded.codebooks,
                        f"{prefix_dot}row_scales": encoded.row_scales,
                    }
                else:
                    metrics["gain_bits"] = encoded.gain_bits
                    metrics["rotation_reflections"] = encoded.rotation_reflections
                    metrics["rotation_seed"] = encoded.rotation_seed
                    metrics["adaptive_stages"] = encoded.adaptive_stages
                    metrics["active_counts"] = [
                        encoded.active_count(s) for s in range(encoded.stages)
                    ]
                    tensors_to_save = {
                        f"{prefix_dot}gain_codebooks": encoded.gain_codebooks,
                        f"{prefix_dot}row_scales": encoded.row_scales,
                    }
                    for s in range(encoded.stages):
                        tensors_to_save[f"{prefix_dot}leech_indices_{s}"] = encoded.packed_indices[s]
                        tensors_to_save[f"{prefix_dot}leech_gains_{s}"] = encoded.packed_gains[s]
                        tensors_to_save[f"{prefix_dot}leech_presence_{s}"] = encoded.presence[s]

        # Save output shard atomically
        temp_shard = out_shard_path.with_suffix(".tmp")
        safetensors.torch.save_file(tensors_to_save, str(temp_shard))
        temp_shard.replace(out_shard_path)

        stored_sha256 = _compute_file_sha256(out_shard_path)

        rec = StreamingTensorRecord(
            name=tensor_name,
            status=status,
            codec=codec_used,
            original_shape=tuple(tensor.shape),
            original_bytes=tensor_bytes,
            stored_bytes=stored_bytes,
            shard_file=out_shard_filename,
            sha256=stored_sha256,
            metrics=metrics,
        )
        records[tensor_name] = rec

        del tensor
        del tensors_to_save

        # Atomically update conversion_state.json
        _atomic_write_json(
            state_path,
            {
                "source_dir": str(ckpt_path),
                "output_dir": str(out_dir),
                "codec": codec,
                "total_tensors": len(records),
                "resumed_tensors": resumed_tensors,
                "peak_single_tensor_bytes": peak_single_tensor_bytes,
                "records": {k: v.to_dict() for k, v in records.items()},
            },
        )

        newly_converted += 1
        if fail_after_tensors is not None and newly_converted >= fail_after_tensors:
            raise RuntimeError("Simulated interruption")

    total_tensors = len(records)
    quantized_tensors = sum(1 for r in records.values() if r.status == "quantized")
    passthrough_tensors = sum(1 for r in records.values() if r.status == "passthrough")
    original_bytes = sum(r.original_bytes for r in records.values())
    stored_bytes = sum(r.stored_bytes for r in records.values())

    report = StreamingConversionReport(
        source_dir=str(ckpt_path),
        output_dir=str(out_dir),
        total_tensors=total_tensors,
        quantized_tensors=quantized_tensors,
        passthrough_tensors=passthrough_tensors,
        resumed_tensors=resumed_tensors,
        original_bytes=original_bytes,
        stored_bytes=stored_bytes,
        peak_single_tensor_bytes=peak_single_tensor_bytes,
        records=records,
    )

    # Write final conversion_state.json
    _atomic_write_json(state_path, report.to_dict())

    # Write model.safetensors.index.json
    out_weight_map: Dict[str, str] = {}
    for r in records.values():
        if r.status != "quantized":
            out_weight_map[r.name] = r.shard_file
        else:
            prefix = r.name[:-len(".weight")] if r.name.endswith(".weight") else r.name
            prefix_dot = f"{prefix}." if prefix else ""
            if r.codec == "adaptive_residual_vq":
                out_weight_map[f"{prefix_dot}packed_codes"] = r.shard_file
                out_weight_map[f"{prefix_dot}codebooks"] = r.shard_file
                out_weight_map[f"{prefix_dot}row_scales"] = r.shard_file
            else:
                out_weight_map[f"{prefix_dot}gain_codebooks"] = r.shard_file
                out_weight_map[f"{prefix_dot}row_scales"] = r.shard_file
                stages_count = int(r.metrics.get("selected_stages", r.metrics.get("stages", 1)))
                for s in range(stages_count):
                    out_weight_map[f"{prefix_dot}leech_indices_{s}"] = r.shard_file
                    out_weight_map[f"{prefix_dot}leech_gains_{s}"] = r.shard_file
                    out_weight_map[f"{prefix_dot}leech_presence_{s}"] = r.shard_file

    _atomic_write_json(
        out_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": stored_bytes}, "weight_map": out_weight_map},
    )

    # Write vq_manifest.json
    layers_manifest: Dict[str, object] = {}
    for r in records.values():
        if r.status == "quantized":
            prefix = r.name[:-len(".weight")] if r.name.endswith(".weight") else r.name
            eff_shape = r.metrics.get("original_shape", list(r.original_shape))
            if len(eff_shape) == 3:
                eff_shape = [eff_shape[0] * eff_shape[1], eff_shape[2]]
            layers_manifest[prefix] = {
                "type": "residual_vq_linear",
                "in_features": int(eff_shape[1]),
                "out_features": int(eff_shape[0]),
                "is_3d_expert": bool(r.metrics.get("is_3d_expert", len(r.original_shape) == 3)),
                "logical_shape": list(r.metrics.get("logical_shape", r.original_shape)),
                "bias": False,
                "block_size": int(
                    r.metrics.get("block_size", 8 if r.codec == "adaptive_residual_vq" else 24)
                ),
                "bits_per_stage": int(
                    r.metrics.get("bits_per_stage", 4 if r.codec == "adaptive_residual_vq" else 20)
                ),
                "stages": int(r.metrics.get("selected_stages", r.metrics.get("stages", 1))),
                "codec": r.codec,
                "source_dtype": "float32",
                **(
                    {
                        "gain_bits": int(r.metrics.get("gain_bits", 2)),
                        "rotation_reflections": int(r.metrics.get("rotation_reflections", 2)),
                        "rotation_seed": int(r.metrics.get("rotation_seed", 20261104)),
                        "adaptive_stages": bool(r.metrics.get("adaptive_stages", True)),
                        "active_counts": list(r.metrics.get("active_counts", [])),
                    }
                    if r.codec == "leech24_shell2_residual"
                    else {}
                ),
            }
    for r in records.values():
        if r.name.endswith(".bias"):
            prefix = r.name[:-len(".bias")]
            if prefix in layers_manifest:
                layers_manifest[prefix]["bias"] = True

    _atomic_write_json(
        out_dir / "vq_manifest.json",
        {
            "format": "ojasx-vector-quantization-runner",
            "schema_version": 1,
            "codec": codec,
            "layers": layers_manifest,
        },
    )

    return report


def _get_submodule(root: nn.Module, target: str) -> Optional[nn.Module]:
    if not target:
        return root
    atoms = target.split(".")
    curr = root
    for atom in atoms:
        if isinstance(curr, (nn.ModuleList, nn.Sequential)):
            try:
                idx = int(atom)
                curr = curr[idx]
            except (ValueError, IndexError):
                return None
        elif isinstance(curr, nn.ModuleDict):
            if atom in curr:
                curr = curr[atom]
            else:
                return None
        elif hasattr(curr, atom):
            curr = getattr(curr, atom)
        else:
            return None
    return curr


def _set_submodule(root: nn.Module, target: str, new_module: nn.Module) -> None:
    if not target:
        raise ValueError("Cannot replace root module directly via _set_submodule")
    if "." in target:
        parent_path, child_name = target.rsplit(".", 1)
        parent = _get_submodule(root, parent_path)
    else:
        parent = root
        child_name = target

    if parent is None:
        raise ValueError(f"Parent for submodule {target!r} not found in model")

    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = new_module
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = new_module
    else:
        setattr(parent, child_name, new_module)


def _get_first_match(shard: Dict[str, torch.Tensor], candidate_keys: Sequence[str]) -> torch.Tensor:
    for k in candidate_keys:
        if k in shard:
            return shard[k]
    # Suffix check
    for k, v in shard.items():
        for cand in candidate_keys:
            if k == cand or k.endswith(f".{cand}") or cand.endswith(f".{k}"):
                return v
    raise KeyError(f"None of {candidate_keys} found in shard keys: {list(shard.keys())}")


def _assign_param_or_buffer(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    device: Union[str, torch.device],
) -> None:
    named_params = dict(model.named_parameters())
    if name in named_params:
        target = named_params[name]
        if target.device.type == "meta":
            target.data = tensor.to(device=device, dtype=target.dtype)
        else:
            target.data.copy_(tensor.to(device=device, dtype=target.dtype))
        return

    named_bufs = dict(model.named_buffers())
    if name in named_bufs:
        target = named_bufs[name]
        if target.device.type == "meta":
            target.data = tensor.to(device=device, dtype=target.dtype)
        else:
            target.copy_(tensor.to(device=device, dtype=target.dtype))
        return

    # Check parent module
    if "." in name:
        submod_name, attr_name = name.rsplit(".", 1)
        submod = _get_submodule(model, submod_name)
        if submod is not None and hasattr(submod, attr_name):
            attr = getattr(submod, attr_name)
            if isinstance(attr, nn.Parameter):
                if attr.device.type == "meta":
                    attr.data = tensor.to(device=device, dtype=attr.dtype)
                else:
                    attr.data.copy_(tensor.to(device=device, dtype=attr.dtype))
                return
            elif isinstance(attr, torch.Tensor):
                attr.copy_(tensor.to(device=device, dtype=attr.dtype))
                return


def load_streamed_checkpoint_into_model(
    model: nn.Module,
    converted_dir: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Replaces quantized nn.Linear layers with ResidualVQLinear populated from shards and loads passthrough weights."""
    conv_path = Path(converted_dir)
    state_file = conv_path / "conversion_state.json"
    if not state_file.exists():
        raise CheckpointIngestionError(f"conversion_state.json not found in {conv_path}")

    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    records = state_data.get("records", {})

    # 1. Replace quantized layers with ResidualVQLinear or ResidualVQFusedExperts
    expert_bank_projections: Dict[str, Dict[str, ResidualVQLinear]] = {}

    for r_name, r_info in records.items():
        if r_info.get("status") == "quantized":
            prefix = r_name[:-len(".weight")] if r_name.endswith(".weight") else r_name
            orig_shape = tuple(r_info["original_shape"])
            metrics = r_info.get("metrics", {})
            codec = r_info.get("codec", "adaptive_residual_vq")
            is_3d_expert = bool(metrics.get("is_3d_expert", len(orig_shape) == 3))

            if is_3d_expert:
                parent_bank_path, proj_attr = prefix.rsplit(".", 1)
                bank_sub = _get_submodule(model, parent_bank_path)
                if bank_sub is None:
                    continue
                param_obj = getattr(bank_sub, proj_attr, None)
                source_dtype = getattr(param_obj, "dtype", torch.float32)
                has_bias = False
                bias_data = None
                eff_rows = orig_shape[0] * orig_shape[1] if len(orig_shape) == 3 else orig_shape[0]
                eff_cols = orig_shape[2] if len(orig_shape) == 3 else orig_shape[1]
            else:
                target_sub = _get_submodule(model, prefix)
                if target_sub is None:
                    continue
                has_bias = getattr(target_sub, "bias", None) is not None
                bias_data = (
                    target_sub.bias.data.clone()
                    if (has_bias and target_sub.bias.device.type != "meta")
                    else None
                )
                source_dtype = getattr(getattr(target_sub, "weight", None), "dtype", torch.float32)
                eff_rows, eff_cols = orig_shape[0], orig_shape[1]

            if codec == "adaptive_residual_vq":
                vq_linear = ResidualVQLinear(
                    in_features=eff_cols,
                    out_features=eff_rows,
                    bias=has_bias,
                    block_size=int(metrics.get("block_size", 8)),
                    bits_per_stage=int(metrics.get("bits_per_stage", 4)),
                    stages=int(metrics.get("selected_stages", metrics.get("stages", 1))),
                    source_dtype=source_dtype,
                    device=device,
                    codec="adaptive_residual_vq",
                )
            elif codec == "leech24_shell2_residual":
                active_counts = (
                    tuple(map(int, metrics["active_counts"]))
                    if "active_counts" in metrics
                    else None
                )
                vq_linear = ResidualVQLinear(
                    in_features=eff_cols,
                    out_features=eff_rows,
                    bias=has_bias,
                    block_size=int(metrics.get("block_size", 24)),
                    gain_bits=int(metrics.get("gain_bits", 2)),
                    bits_per_stage=18 + int(metrics.get("gain_bits", 2)),
                    stages=int(
                        metrics.get("selected_stages", len(active_counts) if active_counts else 1)
                    ),
                    source_dtype=source_dtype,
                    device=device,
                    codec="leech24_shell2_residual",
                    rotation_reflections=int(metrics.get("rotation_reflections", 2)),
                    rotation_seed=int(metrics.get("rotation_seed", 20261104)),
                    adaptive_stages=bool(metrics.get("adaptive_stages", True)),
                    active_counts=active_counts,
                )
            else:
                raise ValueError(f"Unknown codec: {codec!r}")

            # Populate buffers from shard
            shard_path = conv_path / r_info["shard_file"]
            shard_tensors = safetensors.torch.load_file(str(shard_path), device=str(device))
            prefix_dot = f"{prefix}." if prefix else ""

            if codec == "adaptive_residual_vq":
                vq_linear.packed_codes.copy_(
                    _get_first_match(shard_tensors, [f"{prefix_dot}packed_codes", "packed_codes"])
                )
                vq_linear.codebooks.copy_(
                    _get_first_match(shard_tensors, [f"{prefix_dot}codebooks", "codebooks"])
                )
                vq_linear.row_scales.copy_(
                    _get_first_match(shard_tensors, [f"{prefix_dot}row_scales", "row_scales"])
                )
            else:
                vq_linear.gain_codebooks.copy_(
                    _get_first_match(shard_tensors, [f"{prefix_dot}gain_codebooks", "gain_codebooks"])
                )
                vq_linear.row_scales.copy_(
                    _get_first_match(shard_tensors, [f"{prefix_dot}row_scales", "row_scales"])
                )
                for s in range(vq_linear.stages):
                    getattr(vq_linear, f"leech_indices_{s}").copy_(
                        _get_first_match(
                            shard_tensors,
                            [f"{prefix_dot}leech_indices_{s}", f"leech_indices_{s}"],
                        )
                    )
                    getattr(vq_linear, f"leech_gains_{s}").copy_(
                        _get_first_match(
                            shard_tensors,
                            [f"{prefix_dot}leech_gains_{s}", f"leech_gains_{s}"],
                        )
                    )
                    getattr(vq_linear, f"leech_presence_{s}").copy_(
                        _get_first_match(
                            shard_tensors,
                            [f"{prefix_dot}leech_presence_{s}", f"leech_presence_{s}"],
                        )
                    )

            if has_bias and bias_data is not None:
                vq_linear.bias.data.copy_(bias_data.to(device=device))

            if is_3d_expert:
                parent_bank_path, proj_attr = prefix.rsplit(".", 1)
                expert_bank_projections.setdefault(parent_bank_path, {})[proj_attr] = vq_linear
            else:
                _set_submodule(model, prefix, vq_linear)

    # 1b. Replace any 3D fused expert banks collected from streaming records
    for bank_path, proj_map in expert_bank_projections.items():
        bank_sub = _get_submodule(model, bank_path)
        if bank_sub is None:
            continue
        dims = _infer_fused_expert_dims(bank_sub)
        if dims is None:
            # Infer from ResidualVQLinear shapes if already partially replaced
            gu = proj_map.get("gate_up_proj", getattr(bank_sub, "gate_up_proj", None))
            dn = proj_map.get("down_proj", getattr(bank_sub, "down_proj", None))
            num_exp = int(getattr(bank_sub, "num_experts", 1))
            hid = int(getattr(bank_sub, "hidden_dim", getattr(gu, "in_features", 1)))
            inter = int(getattr(bank_sub, "intermediate_dim", getattr(dn, "in_features", 1)))
            dims = (num_exp, hid, inter)
        gate_up_mod = proj_map.get("gate_up_proj", getattr(bank_sub, "gate_up_proj"))
        down_mod = proj_map.get("down_proj", getattr(bank_sub, "down_proj"))
        fused_rep = ResidualVQFusedExperts(
            dims[0],
            dims[1],
            dims[2],
            gate_up_proj=gate_up_mod,
            down_proj=down_mod,
            act_fn=_expert_activation(bank_sub),
            gate_limit=_expert_gate_limit(bank_sub),
        )
        _set_submodule(model, bank_path, fused_rep)

    # 2. Load passthrough parameters/buffers
    for r_name, r_info in records.items():
        if r_info.get("status") == "passthrough":
            shard_path = conv_path / r_info["shard_file"]
            shard_tensors = safetensors.torch.load_file(str(shard_path), device=str(device))
            loaded_tensor = _get_first_match(
                shard_tensors, [r_name, "weight", "bias", Path(r_info["shard_file"]).stem]
            )
            _assign_param_or_buffer(model, r_name, loaded_tensor, device=device)

    model.to(device=device)
    model.eval()
    return model

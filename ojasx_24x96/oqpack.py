"""Canonical M1 OQPack Container and Zero-Copy mmap Artifact Format (oqpack.v1).

Provides architecture-neutral packaging, cryptographic integrity verification,
multi-shard safetensors serialization, zero-copy memory-mapped loading, and
instant skeleton model reconstruction without materializing dense weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import safetensors.torch
from safetensors import SafetensorError, safe_open
import torch
import torch.nn as nn

from .adapters import classify_tensor_role
from .contract import (
    CODEC_SCHEMA_VERSION,
    ArchitectureProfile,
    MemoryAdmissionReport,
    introspect_architecture,
)
from .ir import QuantizationPlan, TensorRole
from .module import (
    ResidualVQConv1D,
    ResidualVQFusedExperts,
    ResidualVQLinear,
)


OQPACK_FORMAT_VERSION = "oqpack.v1"
MANIFEST_FILENAME = "oqpack_manifest.json"


class OQPackValidationError(ValueError):
    """Raised when an OQPack artifact fails integrity, checksum, or schema validation."""

    pass


@dataclass(frozen=True)
class OQPackShardEntry:
    """Metadata and cryptographic checksum for a single safetensors shard."""

    filename: str
    sha256: str
    size_bytes: int
    tensor_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "tensor_names": list(self.tensor_names),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OQPackShardEntry":
        return cls(
            filename=str(data["filename"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            tensor_names=tuple(data.get("tensor_names", ())),
        )


@dataclass(frozen=True)
class OQPackTensorRecord:
    """Exact location, dtype, shape, byte count, and architectural role for one tensor."""

    name: str
    shard: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shard": self.shard,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OQPackTensorRecord":
        return cls(
            name=str(data["name"]),
            shard=str(data["shard"]),
            dtype=str(data["dtype"]),
            shape=tuple(data.get("shape", ())),
            nbytes=int(data["nbytes"]),
            role=str(data.get("role", "other")),
        )


@dataclass
class OQPackManifest:
    """Complete manifest describing the canonical OQPack v1 package."""

    format_version: str
    codec_schema_version: str
    model_id: str
    source_checkpoint_sha256: str
    architecture_profile: Dict[str, object]
    memory_admission: Dict[str, object]
    quantization_plan: Dict[str, object]
    quality_certificate: Optional[Dict[str, object]]
    shards: List[OQPackShardEntry]
    tensors: Dict[str, OQPackTensorRecord]
    layer_specs: Dict[str, Dict[str, object]]
    tokenizer_files: List[str]
    chat_template: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "codec_schema_version": self.codec_schema_version,
            "model_id": self.model_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "architecture_profile": self.architecture_profile,
            "memory_admission": self.memory_admission,
            "quantization_plan": self.quantization_plan,
            "quality_certificate": self.quality_certificate,
            "shards": [s.to_dict() for s in self.shards],
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "layer_specs": self.layer_specs,
            "tokenizer_files": list(self.tokenizer_files),
            "chat_template": self.chat_template,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OQPackManifest":
        shards = [OQPackShardEntry.from_dict(s) for s in data.get("shards", [])]
        tensors = {k: OQPackTensorRecord.from_dict(v) for k, v in data.get("tensors", {}).items()}
        return cls(
            format_version=str(data.get("format_version", "")),
            codec_schema_version=str(data.get("codec_schema_version", "")),
            model_id=str(data.get("model_id", "unknown")),
            source_checkpoint_sha256=str(data.get("source_checkpoint_sha256", "")),
            architecture_profile=dict(data.get("architecture_profile", {})),
            memory_admission=dict(data.get("memory_admission", {})),
            quantization_plan=dict(data.get("quantization_plan", {})),
            quality_certificate=data.get("quality_certificate"),
            shards=shards,
            tensors=tensors,
            layer_specs=dict(data.get("layer_specs", {})),
            tokenizer_files=list(data.get("tokenizer_files", [])),
            chat_template=data.get("chat_template"),
        )


def _to_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower().replace("torch.", "")
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "bool": torch.bool,
    }
    if normalized not in table:
        raise ValueError(f"Unsupported dtype specification: {dtype}")
    return table[normalized]


def _determine_tensor_role(
    name: str,
    model: nn.Module,
    plan: Optional[QuantizationPlan],
) -> str:
    if plan is not None and name in plan.spec_map:
        return plan.spec_map[name].role.value
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        candidate_weight_name = f"{module_name}.weight"
        if plan is not None and candidate_weight_name in plan.spec_map:
            return plan.spec_map[candidate_weight_name].role.value
    try:
        mod = model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        return classify_tensor_role(name, mod).value
    except Exception:
        pass
    try:
        return classify_tensor_role(name, model).value
    except Exception:
        return TensorRole.OTHER.value


def _linear_layer_spec(module: ResidualVQLinear) -> Dict[str, object]:
    spec: Dict[str, object] = {
        "module_type": "conv1d" if isinstance(module, ResidualVQConv1D) else "linear",
        "type": "residual_vq_conv1d" if isinstance(module, ResidualVQConv1D) else "residual_vq_linear",
        "codec": module.codec,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "block_size": module.block_size,
        "bits_per_stage": module.bits_per_stage,
        "stages": module.stages,
        "gain_bits": getattr(module, "gain_bits", 0),
        "rotation_reflections": getattr(module, "rotation_reflections", 0),
        "rotation_seed": getattr(module, "rotation_seed", 20261104),
        "adaptive_stages": getattr(module, "adaptive_stages", True),
        "source_dtype": str(module.source_dtype).replace("torch.", ""),
        "has_bias": module.bias is not None,
        "bias": module.bias is not None,
    }
    if module.codec == "leech24_shell2_residual":
        spec["active_counts"] = [
            module.encoded().active_count(stage) for stage in range(module.stages)
        ]
    else:
        spec["active_counts"] = None
    return spec


def extract_layer_specs(model: nn.Module) -> Dict[str, Dict[str, object]]:
    """Extract metadata for every quantized layer in the model."""
    layer_specs: Dict[str, Dict[str, object]] = {}
    fused_prefixes: List[str] = []

    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in fused_prefixes):
            continue
        if isinstance(module, ResidualVQFusedExperts):
            fused_prefixes.append(f"{name}.")

            def _proj_manifest(proj):
                if isinstance(proj, ResidualVQLinear):
                    return _linear_layer_spec(proj)
                elif isinstance(proj, torch.Tensor):
                    return {"type": "dense_parameter", "shape": list(proj.shape)}
                return {"type": "unknown"}

            layer_specs[name] = {
                "module_type": "fused_experts",
                "type": "residual_vq_fused_experts",
                "num_experts": module.num_experts,
                "hidden_dim": module.hidden_dim,
                "intermediate_dim": module.intermediate_dim,
                "gate_limit": module.gate_limit,
                "gate_up_proj": _proj_manifest(module.gate_up_proj),
                "down_proj": _proj_manifest(module.down_proj),
            }
        elif isinstance(module, ResidualVQLinear):
            layer_specs[name] = _linear_layer_spec(module)

    return layer_specs


def _get_child(module: nn.Module, key: str) -> nn.Module:
    if key.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
        return module[int(key)]
    return getattr(module, key)


def _set_child(module: nn.Module, key: str, value: nn.Module) -> None:
    if key.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
        module[int(key)] = value
    else:
        setattr(module, key, value)


def _resolve_parent_and_attr(root: nn.Module, path: str) -> Tuple[nn.Module, str, nn.Module]:
    parts = path.split(".")
    curr = root
    for part in parts[:-1]:
        curr = _get_child(curr, part)
    attr = parts[-1]
    target = _get_child(curr, attr)
    return curr, attr, target


def _build_replacement(spec: Dict[str, object], target: Optional[nn.Module], device: str) -> nn.Module:
    mtype = spec.get("module_type") or spec.get("type")
    if mtype in {"linear", "residual_vq_linear"}:
        cls = ResidualVQLinear
    elif mtype in {"conv1d", "residual_vq_conv1d"}:
        cls = ResidualVQConv1D
    elif mtype in {"fused_experts", "residual_vq_fused_experts"}:
        num_experts = int(spec["num_experts"])
        hidden_dim = int(spec["hidden_dim"])
        intermediate_dim = int(spec["intermediate_dim"])
        gate_limit = spec.get("gate_limit")
        gate_up_spec = spec["gate_up_proj"]
        down_spec = spec["down_proj"]

        if gate_up_spec.get("type") in {"residual_vq_linear", "linear", "residual_vq_conv1d"}:
            target_gate_up = getattr(target, "gate_up_proj", target) if target else None
            gate_up = _build_replacement(gate_up_spec, target_gate_up, device)
        else:
            gate_up = getattr(target, "gate_up_proj")

        if down_spec.get("type") in {"residual_vq_linear", "linear", "residual_vq_conv1d"}:
            target_down = getattr(target, "down_proj", target) if target else None
            down = _build_replacement(down_spec, target_down, device)
        else:
            down = getattr(target, "down_proj")

        act_fn = getattr(target, "act_fn", nn.SiLU()) if target else nn.SiLU()
        if not isinstance(act_fn, nn.Module):
            act_fn = nn.SiLU()

        rep = ResidualVQFusedExperts(
            num_experts,
            hidden_dim,
            intermediate_dim,
            gate_up_proj=gate_up,
            down_proj=down,
            act_fn=act_fn,
            gate_limit=float(gate_limit) if gate_limit is not None else None,
        )
        return rep
    else:
        raise OQPackValidationError(f"Unknown quantized module type: {mtype}")

    codec = str(spec.get("codec", "adaptive_residual_vq"))
    in_features = int(spec["in_features"])
    out_features = int(spec["out_features"])
    bias = bool(spec.get("has_bias", spec.get("bias", False)))
    block_size = int(spec["block_size"])
    bits_per_stage = int(spec["bits_per_stage"])
    stages = int(spec["stages"])
    source_dtype_str = spec.get("source_dtype", "float32")
    source_dtype = _to_torch_dtype(source_dtype_str)
    gain_bits = int(spec.get("gain_bits", 0))
    rotation_reflections = int(spec.get("rotation_reflections", 0))
    rotation_seed = int(spec.get("rotation_seed", 20261104))
    adaptive_stages = bool(spec.get("adaptive_stages", True))
    active_counts = (
        tuple(map(int, spec["active_counts"]))
        if spec.get("active_counts") is not None
        else None
    )

    mod = cls(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        block_size=block_size,
        bits_per_stage=bits_per_stage,
        stages=stages,
        source_dtype=source_dtype,
        device=torch.device(device),
        codec=codec,
        gain_bits=gain_bits,
        rotation_reflections=rotation_reflections,
        rotation_seed=rotation_seed,
        adaptive_stages=adaptive_stages,
        active_counts=active_counts,
    )
    return mod


def save_oqpack(
    model: nn.Module,
    output_dir: Union[str, Path],
    *,
    plan: Optional[QuantizationPlan] = None,
    architecture_profile: Optional[ArchitectureProfile] = None,
    memory_admission: Optional[MemoryAdmissionReport] = None,
    quality_certificate: Optional[Dict[str, object]] = None,
    model_id: str = "unknown",
    source_checkpoint_sha256: str = "",
    max_shard_bytes: int = 512 * 1024 * 1024,
    tokenizer_dir: Optional[Union[str, Path]] = None,
    chat_template: Optional[str] = None,
) -> OQPackManifest:
    """Save model to canonical OQPack v1 container format with multi-shard safetensors."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract all parameters and persistent buffers cleanly
    raw_sd = model.state_dict()
    clean_sd: Dict[str, torch.Tensor] = {}
    seen_storage_ptrs: Set[int] = set()

    for name, tensor in raw_sd.items():
        t = tensor.detach().cpu().contiguous()
        try:
            ptr = t.untyped_storage().data_ptr()
            if ptr in seen_storage_ptrs:
                t = t.clone().contiguous()
            else:
                seen_storage_ptrs.add(ptr)
        except Exception:
            pass
        clean_sd[name] = t

    # Partition tensors into shards respecting max_shard_bytes
    shards_data: List[Dict[str, torch.Tensor]] = []
    cur_shard: Dict[str, torch.Tensor] = {}
    cur_bytes = 0

    for name, tensor in clean_sd.items():
        t_nbytes = tensor.numel() * tensor.element_size()
        if cur_shard and (cur_bytes + t_nbytes > max_shard_bytes):
            shards_data.append(cur_shard)
            cur_shard = {name: tensor}
            cur_bytes = t_nbytes
        else:
            cur_shard[name] = tensor
            cur_bytes += t_nbytes

    if cur_shard or not shards_data:
        shards_data.append(cur_shard)

    num_shards = len(shards_data)
    shard_entries: List[OQPackShardEntry] = []
    tensor_records: Dict[str, OQPackTensorRecord] = {}

    for idx, shard_tensors in enumerate(shards_data):
        shard_filename = f"shard-{idx + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = out_dir / shard_filename
        safetensors.torch.save_file(shard_tensors, str(shard_path))

        hasher = hashlib.sha256()
        with open(shard_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256_hex = hasher.hexdigest()
        size_bytes = shard_path.stat().st_size

        shard_entries.append(
            OQPackShardEntry(
                filename=shard_filename,
                sha256=sha256_hex,
                size_bytes=size_bytes,
                tensor_names=tuple(shard_tensors.keys()),
            )
        )

        for tname, tensor in shard_tensors.items():
            tensor_records[tname] = OQPackTensorRecord(
                name=tname,
                shard=shard_filename,
                dtype=str(tensor.dtype).replace("torch.", ""),
                shape=tuple(tensor.shape),
                nbytes=tensor.numel() * tensor.element_size(),
                role=_determine_tensor_role(tname, model, plan),
            )

    # Copy tokenizer files if tokenizer_dir is provided
    tokenizer_files: List[str] = []
    if tokenizer_dir is not None:
        tdir = Path(tokenizer_dir)
        if tdir.is_dir():
            known_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "vocab.json",
                "merges.txt",
            ]
            for kf in known_files:
                src_file = tdir / kf
                if src_file.is_file():
                    dest_file = out_dir / kf
                    shutil.copy2(src_file, dest_file)
                    tokenizer_files.append(kf)

    if chat_template is not None:
        (out_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")
        if "chat_template.jinja" not in tokenizer_files:
            tokenizer_files.append("chat_template.jinja")
    elif (out_dir / "chat_template.jinja").is_file():
        chat_template = (out_dir / "chat_template.jinja").read_text(encoding="utf-8")

    # Extract layer specifications
    layer_specs = extract_layer_specs(model)

    if architecture_profile is not None:
        arch_dict = (
            architecture_profile.to_dict()
            if hasattr(architecture_profile, "to_dict")
            else dict(architecture_profile)
        )
    else:
        try:
            arch_dict = introspect_architecture(model).to_dict()
        except Exception:
            arch_dict = {}

    if memory_admission is not None:
        mem_dict = (
            memory_admission.to_dict()
            if hasattr(memory_admission, "to_dict")
            else dict(memory_admission)
        )
    else:
        mem_dict = {}

    if plan is not None:
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
    else:
        plan_dict = {}

    manifest = OQPackManifest(
        format_version=OQPACK_FORMAT_VERSION,
        codec_schema_version=CODEC_SCHEMA_VERSION,
        model_id=str(model_id),
        source_checkpoint_sha256=str(source_checkpoint_sha256),
        architecture_profile=arch_dict,
        memory_admission=mem_dict,
        quantization_plan=plan_dict,
        quality_certificate=quality_certificate,
        shards=shard_entries,
        tensors=tensor_records,
        layer_specs=layer_specs,
        tokenizer_files=tokenizer_files,
        chat_template=chat_template,
    )

    manifest_path = out_dir / MANIFEST_FILENAME
    tmp_manifest = manifest_path.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp_manifest.replace(manifest_path)

    return manifest


def verify_oqpack_integrity(
    package_dir: Union[str, Path],
    *,
    verify_checksums: bool = True,
) -> OQPackManifest:
    """Verify package cryptographic checksums, shard structures, and tensor layouts."""
    pkg = Path(package_dir)
    if not pkg.is_dir():
        raise OQPackValidationError(f"OQPack directory does not exist: {pkg}")

    manifest_path = pkg / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise OQPackValidationError(f"Missing manifest file: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise OQPackValidationError(f"Corrupt manifest JSON: {e}") from e

    manifest = OQPackManifest.from_dict(data)

    if manifest.format_version != OQPACK_FORMAT_VERSION:
        raise OQPackValidationError(
            f"Unsupported format_version {manifest.format_version!r}, expected {OQPACK_FORMAT_VERSION!r}"
        )

    if not manifest.shards:
        raise OQPackValidationError("OQPack manifest specifies no shards")

    seen_tensors: Set[str] = set()

    for shard in manifest.shards:
        shard_path = pkg / shard.filename
        if not shard_path.is_file():
            raise OQPackValidationError(f"Shard file missing: {shard.filename}")

        actual_size = shard_path.stat().st_size
        if actual_size != shard.size_bytes:
            raise OQPackValidationError(
                f"Shard {shard.filename} size mismatch: expected {shard.size_bytes}, found {actual_size}"
            )

        if verify_checksums:
            hasher = hashlib.sha256()
            with open(shard_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()
            if actual_sha != shard.sha256:
                raise OQPackValidationError(
                    f"Shard {shard.filename} checksum mismatch: expected {shard.sha256}, found {actual_sha}"
                )

        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                expected_keys = set(shard.tensor_names)
                if actual_keys != expected_keys:
                    missing = expected_keys - actual_keys
                    extra = actual_keys - expected_keys
                    raise OQPackValidationError(
                        f"Shard {shard.filename} tensor names mismatch. Missing: {missing}, Extra: {extra}"
                    )

                dups = seen_tensors.intersection(actual_keys)
                if dups:
                    raise OQPackValidationError(
                        f"Duplicate tensor names across shards: {dups} in {shard.filename}"
                    )
                seen_tensors.update(actual_keys)

                for tname in actual_keys:
                    if tname not in manifest.tensors:
                        raise OQPackValidationError(
                            f"Tensor {tname} in shard {shard.filename} not found in manifest.tensors"
                        )
                    record = manifest.tensors[tname]
                    if record.shard != shard.filename:
                        raise OQPackValidationError(
                            f"Tensor {tname} shard mismatch: record says {record.shard}, found in {shard.filename}"
                        )
                    tensor = handle.get_tensor(tname)
                    if tuple(tensor.shape) != tuple(record.shape):
                        raise OQPackValidationError(
                            f"Tensor {tname} shape mismatch: expected {record.shape}, got {tuple(tensor.shape)}"
                        )
                    actual_dtype = str(tensor.dtype).replace("torch.", "")
                    expected_dtype = str(record.dtype).replace("torch.", "")
                    if actual_dtype != expected_dtype:
                        raise OQPackValidationError(
                            f"Tensor {tname} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                        )
                    actual_nbytes = tensor.numel() * tensor.element_size()
                    if actual_nbytes != record.nbytes:
                        raise OQPackValidationError(
                            f"Tensor {tname} nbytes mismatch: expected {record.nbytes}, got {actual_nbytes}"
                        )
        except SafetensorError as e:
            raise OQPackValidationError(f"Corrupt safetensors shard {shard.filename}: {e}") from e
        except Exception as e:
            if isinstance(e, OQPackValidationError):
                raise
            raise OQPackValidationError(f"Error inspecting shard {shard.filename}: {e}") from e

    manifest_missing = set(manifest.tensors.keys()) - seen_tensors
    if manifest_missing:
        raise OQPackValidationError(
            f"Tensors declared in manifest.tensors missing from shards: {manifest_missing}"
        )

    for tok_file in manifest.tokenizer_files:
        if not (pkg / tok_file).is_file():
            raise OQPackValidationError(f"Tokenizer file missing: {tok_file}")

    return manifest


def open_oqpack_mmap(
    package_dir: Union[str, Path],
    *,
    device: str = "cpu",
    verify_checksums: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load zero-copy memory-mapped tensors across all package shards."""
    pkg = Path(package_dir)
    manifest = verify_oqpack_integrity(pkg, verify_checksums=verify_checksums)
    tensors: Dict[str, torch.Tensor] = {}

    for shard in manifest.shards:
        shard_path = pkg / shard.filename
        with safe_open(str(shard_path), framework="pt", device=device) as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)

    return tensors


def load_quantized_model_from_oqpack(
    skeleton_model: nn.Module,
    package_dir: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    verify_checksums: bool = True,
    streaming_decode: bool = True,
    weight_cache: bool = False,
) -> Tuple[nn.Module, OQPackManifest]:
    """Reconstruct quantized model directly from OQPack manifest and zero-copy mmap shards."""
    pkg = Path(package_dir)
    manifest = verify_oqpack_integrity(pkg, verify_checksums=verify_checksums)
    device_str = str(device)

    # Replace quantized layers directly in skeleton_model
    for name, spec in manifest.layer_specs.items():
        if not name:
            skeleton_model = _build_replacement(spec, skeleton_model, device_str)
        else:
            parent, attr, target = _resolve_parent_and_attr(skeleton_model, name)
            replacement = _build_replacement(spec, target, device_str)
            _set_child(parent, attr, replacement)

    # Load tensors via mmap safe_open
    mmap_tensors = open_oqpack_mmap(pkg, device="cpu", verify_checksums=False)

    if device_str != "cpu":
        mmap_tensors = {k: v.to(device) for k, v in mmap_tensors.items()}
        skeleton_model.to(device)

    # Verify tensor completeness
    expected_keys = set(skeleton_model.state_dict().keys())
    loaded_keys = set(mmap_tensors.keys())
    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys

    ignored_patterns = []
    for attr in ("_tied_weights_keys", "_keys_to_ignore_on_load_missing"):
        val = getattr(skeleton_model, attr, None) or []
        if isinstance(val, (list, tuple, set)):
            ignored_patterns.extend(str(item) for item in val)
        elif isinstance(val, str):
            ignored_patterns.append(val)

    def is_ignored(k: str) -> bool:
        return any(p in k for p in ignored_patterns)

    unexp_missing = [k for k in missing if not is_ignored(k)]
    if unexp_missing:
        raise OQPackValidationError(f"Missing tensors for model: {unexp_missing[:8]}")
    if unexpected:
        raise OQPackValidationError(f"Unexpected tensors in package: {list(unexpected)[:8]}")

    skeleton_model.load_state_dict(mmap_tensors, strict=False, assign=True)

    if hasattr(skeleton_model, "tie_weights"):
        skeleton_model.tie_weights()

    for _, mod in skeleton_model.named_modules():
        if isinstance(mod, ResidualVQLinear):
            mod.set_streaming_decode(streaming_decode)
            mod.set_weight_cache(weight_cache)
        elif isinstance(mod, ResidualVQFusedExperts):
            mod.streaming_decode = streaming_decode

    skeleton_model.eval()
    return skeleton_model, manifest


__all__ = [
    "OQPACK_FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "OQPackValidationError",
    "OQPackShardEntry",
    "OQPackTensorRecord",
    "OQPackManifest",
    "save_oqpack",
    "verify_oqpack_integrity",
    "open_oqpack_mmap",
    "load_quantized_model_from_oqpack",
]

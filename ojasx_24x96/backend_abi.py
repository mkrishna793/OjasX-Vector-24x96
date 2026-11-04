"""OjasX-24X96 Stable Backend C ABI & llama.cpp / ggml Adapter (Milestone M6).

Provides standard C99/C++17 ctypes bindings matching ``ojasx_backend_abi.h``,
runtime descriptor builders, llama.cpp / ggml custom operator manifest export,
fused native matmul dispatch with zero dense weight caching, and a comprehensive
deterministic backend conformance suite.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import QuantizationConfig, QuantizedWeight, dequantize_weight, quantize_weight
from .leech_codec import (
    LeechQuantizationConfig,
    LeechQuantizedWeight,
    dequantize_leech_weight,
    quantize_leech_weight,
    rotate_activation_for_leech,
)
from .leech_native import (
    _load_library,
    fused_leech_gemm_native,
    fused_rvq_gemm_native,
    native_backend_status,
)
from .module import ResidualVQLinear


OJASX_BACKEND_ABI_VERSION = 1
OJASX_LEECH24_DIM = 24
OJASX_MAX_STAGES = 8


class OjasxStatus(IntEnum):
    """Status codes corresponding to ojasx_status_e in ojasx_backend_abi.h."""

    OK = 0
    INVALID_ARGUMENT = -1
    NULL_POINTER = -2
    MISSING_PRESENCE = -3
    GAIN_OUT_OF_RANGE = -5
    DECODE_POINT_FAILED = -6
    ABI_MISMATCH = -7


# C ABI structure definitions matching ojasx_backend_abi.h


class CLeech24LayerDesc(ctypes.Structure):
    """C struct descriptor matching ojasx_leech24_layer_desc_t."""

    _fields_ = [
        ("packed_indices", ctypes.c_void_p * OJASX_MAX_STAGES),
        ("packed_gains", ctypes.c_void_p * OJASX_MAX_STAGES),
        ("presence", ctypes.c_void_p * OJASX_MAX_STAGES),
        ("gain_codebooks", ctypes.c_void_p),
        ("row_scales", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("stages", ctypes.c_int32),
        ("gain_bits", ctypes.c_int32),
        ("adaptive_stages", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("columns", ctypes.c_int32),
        ("rotation_reflections", ctypes.c_int32),
        ("rotation_seed", ctypes.c_int32),
    ]


class CRVQLayerDesc(ctypes.Structure):
    """C struct descriptor matching ojasx_rvq_layer_desc_t."""

    _fields_ = [
        ("packed_codes", ctypes.c_void_p),
        ("codebooks", ctypes.c_void_p),
        ("row_scales", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("stages", ctypes.c_int32),
        ("bits_per_stage", ctypes.c_int32),
        ("block_size", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("columns", ctypes.c_int32),
    ]


class CBackendVTable(ctypes.Structure):
    """C vtable struct matching ojasx_backend_vtable_t."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend_name", ctypes.c_char_p),
        ("table_bytes", ctypes.c_void_p),
        ("decode_indices", ctypes.c_void_p),
        ("decode_rows", ctypes.c_void_p),
        ("leech24_fused_gemm", ctypes.c_void_p),
        ("rvq_fused_gemm", ctypes.c_void_p),
    ]


def abi_header_path() -> Path:
    """Return the absolute path to ojasx_backend_abi.h."""
    return Path(__file__).resolve().parent / "native" / "ojasx_backend_abi.h"


def abi_header_status() -> Dict[str, object]:
    """Verify ojasx_backend_abi.h presence, calculate SHA-256, and confirm ABI version."""
    header = abi_header_path()
    if not header.exists():
        return {
            "exists": False,
            "header_path": str(header),
            "sha256": None,
            "abi_version": None,
            "verified": False,
            "error": "Header file not found",
        }

    content = header.read_text(encoding="utf-8")
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    match = re.search(r"#define\s+OJASX_BACKEND_ABI_VERSION\s+(\d+)U?", content)
    header_version = int(match.group(1)) if match else None

    verified = bool(
        header_version == OJASX_BACKEND_ABI_VERSION and
        "ojasx_leech24_layer_desc_s" in content and
        "ojasx_rvq_layer_desc_s" in content and
        "ojasx_backend_vtable_s" in content
    )

    return {
        "exists": True,
        "header_path": str(header),
        "sha256": sha256,
        "abi_version": header_version,
        "expected_abi_version": OJASX_BACKEND_ABI_VERSION,
        "matches_python_version": header_version == OJASX_BACKEND_ABI_VERSION,
        "verified": verified,
    }


def build_c_layer_descriptor(
    module_or_encoded: Any,
) -> Tuple[Union[CLeech24LayerDesc, CRVQLayerDesc], Dict[str, object]]:
    """Build a C ABI descriptor struct holding live raw pointers to encoded buffers.

    The returned tuple contains the ctypes Structure and a metadata dictionary
    retaining references to the underlying contiguous CPU tensors, guaranteeing
    memory validity while external C/C++ runtimes access them.
    """
    bias_tensor: Optional[torch.Tensor] = None

    if isinstance(module_or_encoded, ResidualVQLinear):
        codec = module_or_encoded.codec
        if module_or_encoded.bias is not None:
            bias_tensor = module_or_encoded.bias.detach().cpu().contiguous().to(dtype=torch.float32)
        encoded = module_or_encoded.encoded()
    elif isinstance(module_or_encoded, LeechQuantizedWeight):
        codec = "leech24_shell2_residual"
        encoded = module_or_encoded
    elif isinstance(module_or_encoded, QuantizedWeight):
        codec = "adaptive_residual_vq"
        encoded = module_or_encoded
    else:
        raise TypeError(f"Unsupported module or encoded weight type: {type(module_or_encoded)}")

    retained_tensors: Dict[str, torch.Tensor] = {}

    if codec == "leech24_shell2_residual":
        desc = CLeech24LayerDesc()
        desc.stages = int(encoded.stages)
        desc.gain_bits = int(encoded.gain_bits)
        desc.adaptive_stages = int(bool(encoded.adaptive_stages))
        desc.rows = int(encoded.original_shape[0])
        desc.columns = int(encoded.original_shape[1])
        desc.rotation_reflections = int(encoded.rotation_reflections)
        desc.rotation_seed = int(encoded.rotation_seed)

        # Retain and point to contiguous CPU tensors
        packed_indices_list: List[torch.Tensor] = []
        packed_gains_list: List[torch.Tensor] = []
        presence_list: List[torch.Tensor] = []

        for s in range(OJASX_MAX_STAGES):
            if s < encoded.stages:
                pi = encoded.packed_indices[s].detach().cpu().contiguous()
                pg = encoded.packed_gains[s].detach().cpu().contiguous()
                pr = encoded.presence[s].detach().cpu().contiguous()
                packed_indices_list.append(pi)
                packed_gains_list.append(pg)
                presence_list.append(pr)

                desc.packed_indices[s] = pi.data_ptr() if pi.numel() else 0
                desc.packed_gains[s] = pg.data_ptr() if pg.numel() else 0
                desc.presence[s] = pr.data_ptr() if pr.numel() else 0
            else:
                desc.packed_indices[s] = 0
                desc.packed_gains[s] = 0
                desc.presence[s] = 0

        gc = encoded.gain_codebooks.detach().cpu().contiguous().to(dtype=torch.float32)
        rs = encoded.row_scales.detach().cpu().contiguous().to(dtype=torch.float32)
        desc.gain_codebooks = gc.data_ptr() if gc.numel() else 0
        desc.row_scales = rs.data_ptr() if rs.numel() else 0
        desc.bias = bias_tensor.data_ptr() if bias_tensor is not None and bias_tensor.numel() else 0

        retained_tensors = {
            "gain_codebooks": gc,
            "row_scales": rs,
        }
        if bias_tensor is not None:
            retained_tensors["bias"] = bias_tensor
        for i, t in enumerate(packed_indices_list):
            retained_tensors[f"packed_indices_{i}"] = t
        for i, t in enumerate(packed_gains_list):
            retained_tensors[f"packed_gains_{i}"] = t
        for i, t in enumerate(presence_list):
            retained_tensors[f"presence_{i}"] = t

        metadata: Dict[str, object] = {
            "codec": "OJASX_LEECH24",
            "abi_version": OJASX_BACKEND_ABI_VERSION,
            "shape": (desc.rows, desc.columns),
            "rows": desc.rows,
            "columns": desc.columns,
            "stages": desc.stages,
            "gain_bits": desc.gain_bits,
            "adaptive_stages": bool(desc.adaptive_stages),
            "rotation_reflections": desc.rotation_reflections,
            "rotation_seed": desc.rotation_seed,
            "block_size": OJASX_LEECH24_DIM,
            "workspace_bytes": 0,
            "has_bias": bias_tensor is not None,
            "tensors": retained_tensors,
        }
        return desc, metadata

    elif codec == "adaptive_residual_vq":
        desc = CRVQLayerDesc()
        desc.stages = int(encoded.stages)
        desc.bits_per_stage = int(encoded.bits_per_stage)
        desc.block_size = int(encoded.block_size)
        desc.rows = int(encoded.original_shape[0])
        desc.columns = int(encoded.original_shape[1])

        pc = encoded.packed_codes.detach().cpu().contiguous()
        cb = encoded.codebooks.detach().cpu().contiguous().to(dtype=torch.float32)
        rs = encoded.row_scales.detach().cpu().contiguous().to(dtype=torch.float32)

        desc.packed_codes = pc.data_ptr() if pc.numel() else 0
        desc.codebooks = cb.data_ptr() if cb.numel() else 0
        desc.row_scales = rs.data_ptr() if rs.numel() else 0
        desc.bias = bias_tensor.data_ptr() if bias_tensor is not None and bias_tensor.numel() else 0

        retained_tensors = {
            "packed_codes": pc,
            "codebooks": cb,
            "row_scales": rs,
        }
        if bias_tensor is not None:
            retained_tensors["bias"] = bias_tensor

        metadata = {
            "codec": "OJASX_RVQ",
            "abi_version": OJASX_BACKEND_ABI_VERSION,
            "shape": (desc.rows, desc.columns),
            "rows": desc.rows,
            "columns": desc.columns,
            "stages": desc.stages,
            "bits_per_stage": desc.bits_per_stage,
            "block_size": desc.block_size,
            "workspace_bytes": 0,
            "has_bias": bias_tensor is not None,
            "tensors": retained_tensors,
        }
        return desc, metadata

    else:
        raise ValueError(f"Unsupported codec: {codec}")


class LlamaCppGGMLAdapter:
    """Bridge OjasX quantized weights and OQPack containers to llama.cpp / ggml."""

    @staticmethod
    def export_ggml_adapter_manifest(
        model_or_oqpack_dir: Any,
        output_path: Union[str, Path],
    ) -> Dict[str, object]:
        """Export ggml_ojasx_adapter.json manifest for external engine integration."""
        quantized_tensors: Dict[str, Dict[str, object]] = {}
        model_id = "unknown"

        # Check if input is a path or string pointing to an OQPack directory or manifest
        if isinstance(model_or_oqpack_dir, (str, Path)):
            p = Path(model_or_oqpack_dir)
            if p.is_file() and p.suffix == ".json":
                with p.open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            elif p.is_dir() and (p / "oqpack_manifest.json").exists():
                with (p / "oqpack_manifest.json").open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            elif p.is_dir() and (p / "manifest.json").exists():
                with (p / "manifest.json").open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            else:
                raise FileNotFoundError(f"No OQPack manifest found at {p}")

            model_id = str(manifest_data.get("model_id", "unknown"))
            layer_specs = manifest_data.get("layer_specs", {})

            for name, spec in layer_specs.items():
                if spec.get("module_type") == "fused_experts":
                    for proj_key in ("gate_up_proj", "down_proj"):
                        proj_spec = spec.get(proj_key)
                        if isinstance(proj_spec, dict) and proj_spec.get("codec"):
                            tname = f"{name}.{proj_key}.weight"
                            is_leech = proj_spec.get("codec") == "leech24_shell2_residual"
                            rows = int(proj_spec.get("out_features", 0))
                            cols = int(proj_spec.get("in_features", 0))
                            entry: Dict[str, object] = {
                                "tensor_name": tname,
                                "codec": "OJASX_LEECH24" if is_leech else "OJASX_RVQ",
                                "abi_version": OJASX_BACKEND_ABI_VERSION,
                                "shape": [rows, cols],
                                "rows": rows,
                                "columns": cols,
                                "stages": int(proj_spec.get("stages", 1)),
                                "block_size": int(proj_spec.get("block_size", 24 if is_leech else 8)),
                                "workspace_bytes": 0,
                                "has_bias": bool(proj_spec.get("has_bias", proj_spec.get("bias", False))),
                            }
                            if is_leech:
                                entry["gain_bits"] = int(proj_spec.get("gain_bits", 0))
                                entry["adaptive_stages"] = bool(proj_spec.get("adaptive_stages", True))
                                entry["rotation_reflections"] = int(proj_spec.get("rotation_reflections", 0))
                                entry["rotation_seed"] = int(proj_spec.get("rotation_seed", 20261104))
                            else:
                                entry["bits_per_stage"] = int(proj_spec.get("bits_per_stage", 4))
                            quantized_tensors[tname] = entry
                else:
                    tname = f"{name}.weight" if not name.endswith(".weight") else name
                    is_leech = spec.get("codec") == "leech24_shell2_residual"
                    rows = int(spec.get("out_features", spec.get("rows", 0)))
                    cols = int(spec.get("in_features", spec.get("columns", 0)))
                    entry = {
                        "tensor_name": tname,
                        "codec": "OJASX_LEECH24" if is_leech else "OJASX_RVQ",
                        "abi_version": OJASX_BACKEND_ABI_VERSION,
                        "shape": [rows, cols],
                        "rows": rows,
                        "columns": cols,
                        "stages": int(spec.get("stages", 1)),
                        "block_size": int(spec.get("block_size", 24 if is_leech else 8)),
                        "workspace_bytes": 0,
                        "has_bias": bool(spec.get("has_bias", spec.get("bias", False))),
                    }
                    if is_leech:
                        entry["gain_bits"] = int(spec.get("gain_bits", 0))
                        entry["adaptive_stages"] = bool(spec.get("adaptive_stages", True))
                        entry["rotation_reflections"] = int(spec.get("rotation_reflections", 0))
                        entry["rotation_seed"] = int(spec.get("rotation_seed", 20261104))
                    else:
                        entry["bits_per_stage"] = int(spec.get("bits_per_stage", 4))
                    quantized_tensors[tname] = entry

        elif isinstance(model_or_oqpack_dir, dict) and "layer_specs" in model_or_oqpack_dir:
            model_id = str(model_or_oqpack_dir.get("model_id", "unknown"))
            layer_specs = model_or_oqpack_dir.get("layer_specs", {})
            for name, spec in layer_specs.items():
                tname = f"{name}.weight" if not name.endswith(".weight") else name
                is_leech = spec.get("codec") == "leech24_shell2_residual"
                rows = int(spec.get("out_features", spec.get("rows", 0)))
                cols = int(spec.get("in_features", spec.get("columns", 0)))
                entry = {
                    "tensor_name": tname,
                    "codec": "OJASX_LEECH24" if is_leech else "OJASX_RVQ",
                    "abi_version": OJASX_BACKEND_ABI_VERSION,
                    "shape": [rows, cols],
                    "rows": rows,
                    "columns": cols,
                    "stages": int(spec.get("stages", 1)),
                    "block_size": int(spec.get("block_size", 24 if is_leech else 8)),
                    "workspace_bytes": 0,
                    "has_bias": bool(spec.get("has_bias", spec.get("bias", False))),
                }
                if is_leech:
                    entry["gain_bits"] = int(spec.get("gain_bits", 0))
                    entry["adaptive_stages"] = bool(spec.get("adaptive_stages", True))
                    entry["rotation_reflections"] = int(spec.get("rotation_reflections", 0))
                    entry["rotation_seed"] = int(spec.get("rotation_seed", 20261104))
                else:
                    entry["bits_per_stage"] = int(spec.get("bits_per_stage", 4))
                quantized_tensors[tname] = entry

        elif isinstance(model_or_oqpack_dir, nn.Module):
            model_id = getattr(model_or_oqpack_dir, "model_id", "pytorch_model")
            for name, module in model_or_oqpack_dir.named_modules():
                if isinstance(module, ResidualVQLinear):
                    tname = f"{name}.weight" if name else "weight"
                    is_leech = module.codec == "leech24_shell2_residual"
                    entry = {
                        "tensor_name": tname,
                        "codec": "OJASX_LEECH24" if is_leech else "OJASX_RVQ",
                        "abi_version": OJASX_BACKEND_ABI_VERSION,
                        "shape": [module.out_features, module.in_features],
                        "rows": module.out_features,
                        "columns": module.in_features,
                        "stages": module.stages,
                        "block_size": module.block_size,
                        "workspace_bytes": 0,
                        "has_bias": module.bias is not None,
                    }
                    if is_leech:
                        entry["gain_bits"] = module.gain_bits
                        entry["adaptive_stages"] = module.adaptive_stages
                        entry["rotation_reflections"] = module.rotation_reflections
                        entry["rotation_seed"] = module.rotation_seed
                    else:
                        entry["bits_per_stage"] = module.bits_per_stage
                    quantized_tensors[tname] = entry
        else:
            raise TypeError(f"Unsupported model or artifact input type: {type(model_or_oqpack_dir)}")

        manifest: Dict[str, object] = {
            "format": "ggml_ojasx_adapter.v1",
            "abi_version": OJASX_BACKEND_ABI_VERSION,
            "adapter_type": "llama.cpp_ggml_custom_op",
            "model_id": model_id,
            "total_quantized_tensors": len(quantized_tensors),
            "default_workspace_bytes": 0,
            "quantized_tensors": quantized_tensors,
        }

        out = Path(output_path)
        if out.is_dir() or not out.name.endswith(".json"):
            target_path = out / "ggml_ojasx_adapter.json"
        else:
            target_path = out
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    @staticmethod
    def dispatch_ggml_custom_matmul(
        layer: ResidualVQLinear,
        input_tensor: torch.Tensor,
        *,
        prefer_native: bool = True,
    ) -> torch.Tensor:
        """Execute fused GEMM operator through the C ABI with zero dense weight caching."""
        if input_tensor.shape[-1] != layer.in_features:
            raise ValueError(
                f"Expected inputs with {layer.in_features} features, got {input_tensor.shape[-1]}"
            )

        batch_shape = input_tensor.shape[:-1]
        flat_inputs = input_tensor.reshape(-1, layer.in_features)
        output: Optional[torch.Tensor] = None

        if prefer_native and flat_inputs.device.type == "cpu":
            encoded = layer.encoded()
            bias = layer.bias.to(dtype=torch.float32) if layer.bias is not None else None
            flat_fp32 = flat_inputs.to(dtype=torch.float32)

            if layer.codec == "leech24_shell2_residual":
                rotated = rotate_activation_for_leech(
                    flat_fp32,
                    layer.in_features,
                    layer.rotation_reflections,
                    layer.rotation_seed,
                )
                output = fused_leech_gemm_native(
                    encoded,
                    rotated,
                    0,
                    layer.out_features,
                    bias=bias,
                )
            elif layer.codec == "adaptive_residual_vq":
                output = fused_rvq_gemm_native(
                    encoded,
                    flat_fp32,
                    0,
                    layer.out_features,
                    bias=bias,
                )

        if output is None:
            # Fallback path guarantees zero persistent dense weight cache
            original_cache = layer.weight_cache
            try:
                layer.set_weight_cache(False)
                output = layer(flat_inputs)
            finally:
                layer.set_weight_cache(original_cache)

        if layer.cached_weight_bytes != 0:
            raise RuntimeError(
                f"Unreported dense weight cache detected in ggml dispatch: {layer.cached_weight_bytes} bytes"
            )

        return output.to(dtype=input_tensor.dtype).reshape(*batch_shape, layer.out_features)


def run_backend_conformance_suite(*, include_native: bool = True) -> Dict[str, object]:
    """Run comprehensive, deterministic M6 C ABI & conformance verification suite."""
    checks: Dict[str, Dict[str, object]] = {}

    # Check 1: ABI Header & Symbol verification
    hdr_status = abi_header_status()
    symbols_verified = False
    symbols_detail: Dict[str, bool] = {}

    if include_native:
        lib, native_stat = _load_library()
        if lib is not None:
            required_symbols = [
                "ojasx_leech24_decode_indices",
                "ojasx_leech24_table_bytes",
                "ojasx_leech24_decode_rows",
                "ojasx_leech24_fused_gemm",
                "ojasx_rvq_fused_gemm",
            ]
            for sym in required_symbols:
                symbols_detail[sym] = hasattr(lib, sym)
            symbols_verified = all(symbols_detail.values())
        else:
            symbols_detail["library_load_error"] = str(native_stat.get("reason", "unknown"))

    checks["abi_header_and_symbols"] = {
        "passed": bool(hdr_status["verified"] and (not include_native or symbols_verified)),
        "header_status": hdr_status,
        "symbols": symbols_detail,
    }

    # Check 2: Leech24 multi-stage & adaptive-stage matmul conformance (columns=70, non-24-multiple)
    torch.manual_seed(20261104)
    leech_diffs: List[float] = []
    leech_cases_passed = True

    leech_configs = [
        # (stages, gain_bits, adaptive_stages, rotation_reflections, batch_size, with_bias)
        (1, 1, False, 0, 1, True),
        (2, 1, True, 2, 3, True),
        (2, 2, False, 4, 1, False),
    ]

    for stages, gain_bits, adaptive, rots, bsz, has_bias in leech_configs:
        rows, columns = 16, 70
        weight = torch.randn(rows, columns, dtype=torch.float32)
        inputs = torch.randn(bsz, columns, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32) if has_bias else None

        cfg = LeechQuantizationConfig(
            max_stages=stages,
            gain_bits=gain_bits,
            rotation_reflections=rots,
            adaptive_stages=adaptive,
            seed=20261104,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, cfg)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        ref_w = dequantize_leech_weight(encoded)
        ref_out = F.linear(inputs, ref_w, bias)

        dispatched_out = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(
            module, inputs, prefer_native=include_native
        )
        max_abs = float((dispatched_out - ref_out).abs().max().item())
        leech_diffs.append(max_abs)
        if max_abs > 1e-5:
            leech_cases_passed = False

    checks["leech24_matmul_conformance"] = {
        "passed": leech_cases_passed,
        "max_abs_diff": max(leech_diffs) if leech_diffs else 0.0,
        "threshold": 1e-5,
        "cases_evaluated": len(leech_configs),
    }

    # Check 3: RVQ 1/2/4/8-bit matmul conformance (columns=30, non-block-multiple)
    rvq_diffs: List[float] = []
    rvq_cases_passed = True
    rvq_configs = [
        # (bits_per_stage, block_size, stages, batch_size, with_bias)
        (1, 8, 2, 1, True),
        (2, 8, 2, 3, True),
        (4, 4, 2, 1, False),
        (8, 8, 2, 2, True),
    ]

    for bits, blk, stages, bsz, has_bias in rvq_configs:
        rows, columns = 14, 30
        weight = torch.randn(rows, columns, dtype=torch.float32)
        inputs = torch.randn(bsz, columns, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32) if has_bias else None

        cfg_rvq = QuantizationConfig(
            block_size=blk,
            bits_per_stage=bits,
            max_stages=stages,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded_rvq, _ = quantize_weight(weight, cfg_rvq)
        assert encoded_rvq is not None

        module_rvq = ResidualVQLinear.from_encoded(
            encoded_rvq,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        ref_w_rvq = dequantize_weight(encoded_rvq)
        ref_out_rvq = F.linear(inputs, ref_w_rvq, bias)

        dispatched_rvq = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(
            module_rvq, inputs, prefer_native=include_native
        )
        max_abs_rvq = float((dispatched_rvq - ref_out_rvq).abs().max().item())
        rvq_diffs.append(max_abs_rvq)
        if max_abs_rvq > 1e-5:
            rvq_cases_passed = False

    checks["rvq_matmul_conformance"] = {
        "passed": rvq_cases_passed,
        "max_abs_diff": max(rvq_diffs) if rvq_diffs else 0.0,
        "threshold": 1e-5,
        "cases_evaluated": len(rvq_configs),
    }

    # Check 4: Zero unreported dense-weight cache check
    checks["zero_dense_weight_cache"] = {
        "passed": module.cached_weight_bytes == 0 and module_rvq.cached_weight_bytes == 0,
        "leech_cached_bytes": module.cached_weight_bytes,
        "rvq_cached_bytes": module_rvq.cached_weight_bytes,
    }

    # Check 5: Negative boundary & error-code conformance
    negative_passed = True
    negative_details: Dict[str, object] = {}

    if include_native:
        lib, _ = _load_library()
        if lib is not None:
            # 1. Null pointers in decode_indices
            res_null_idx = lib.ojasx_leech24_decode_indices(None, 10, None)
            negative_details["null_indices_status"] = res_null_idx
            if res_null_idx != OjasxStatus.INVALID_ARGUMENT:
                negative_passed = False

            # 2. Invalid row range in decode_rows (row_stop <= row_start)
            arr_type1 = ctypes.c_void_p * 1
            dummy_buf = torch.zeros(100, dtype=torch.uint8)
            buf_ptr = ctypes.c_void_p(dummy_buf.data_ptr())
            gc_dummy = torch.zeros(1, 1, dtype=torch.float32)
            out_dummy = torch.zeros(1, 24, dtype=torch.float32)
            res_inv_rows = lib.ojasx_leech24_decode_rows(
                arr_type1(buf_ptr),
                arr_type1(buf_ptr),
                arr_type1(buf_ptr),
                ctypes.c_void_p(gc_dummy.data_ptr()),
                1,
                0,
                0,
                10,
                24,
                5,
                2,  # row_start=5, row_stop=2 (invalid)
                ctypes.c_void_p(out_dummy.data_ptr()),
            )
            negative_details["invalid_rows_status"] = res_inv_rows
            if res_inv_rows != OjasxStatus.INVALID_ARGUMENT:
                negative_passed = False

            # 3. Null stage pointer in packed_indices
            res_null_stage = lib.ojasx_leech24_decode_rows(
                arr_type1(None),
                arr_type1(buf_ptr),
                arr_type1(buf_ptr),
                ctypes.c_void_p(gc_dummy.data_ptr()),
                1,
                0,
                0,
                1,
                24,
                0,
                1,
                ctypes.c_void_p(out_dummy.data_ptr()),
            )
            negative_details["null_stage_status"] = res_null_stage
            if res_null_stage != OjasxStatus.NULL_POINTER:
                negative_passed = False

            # 4. Missing presence in adaptive stage
            arr_type2 = ctypes.c_void_p * 2
            gc_dummy2 = torch.zeros(2, 1, dtype=torch.float32)
            res_missing_pres = lib.ojasx_leech24_decode_rows(
                arr_type2(buf_ptr, buf_ptr),
                arr_type2(buf_ptr, buf_ptr),
                arr_type2(buf_ptr, None),  # presence[1] is None
                ctypes.c_void_p(gc_dummy2.data_ptr()),
                2,
                0,
                1,  # adaptive_stages = 1
                1,
                24,
                0,
                1,
                ctypes.c_void_p(out_dummy.data_ptr()),
            )
            negative_details["missing_presence_status"] = res_missing_pres
            if res_missing_pres != OjasxStatus.MISSING_PRESENCE:
                negative_passed = False

            # 5. RVQ null pointers / invalid dimensions
            res_rvq_null = lib.ojasx_rvq_fused_gemm(
                None, None, None, None, None, 1, 1, 4, 8, 10, 16, 0, 10, None
            )
            negative_details["rvq_null_status"] = res_rvq_null
            if res_rvq_null != OjasxStatus.INVALID_ARGUMENT:
                negative_passed = False
        else:
            negative_passed = False
            negative_details["error"] = "Native library could not be loaded for negative tests"

    checks["negative_boundary_error_codes"] = {
        "passed": negative_passed,
        "details": negative_details,
    }

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "passed": all_passed,
        "abi_version": OJASX_BACKEND_ABI_VERSION,
        "checks": checks,
    }

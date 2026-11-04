"""Optional native CPU Leech24 row decoder with a pure-PyTorch fallback.

The C ABI source is built locally into an excluded cache. No downloaded binary,
Python development headers, or PyTorch C++ extension is required. ``native`` mode
fails loudly when the backend is unavailable; ``auto`` returns ``None`` so the
caller can use the reference implementation.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Optional

import torch

from .leech24 import INDEX_COUNT, LEECH_DIM, apply_rotation


_SOURCE = Path(__file__).resolve().parent / "native" / "leech24_cpu.cpp"
_CUDA_SOURCE = Path(__file__).resolve().parent / "native" / "vq_kernels_cuda.cu"
_FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math")


def _cache_directory() -> Path:
    configured = os.environ.get("OJASX_LEECH_NATIVE_CACHE")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "ojasx-leech24-native"


def _compiler() -> Optional[str]:
    return shutil.which(os.environ.get("CXX", "g++"))


def _source_identity(compiler: str) -> Dict[str, str]:
    version = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    source_bytes = _SOURCE.read_bytes()
    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update("\0".join(_FLAGS).encode())
    digest.update(version.encode())
    return {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "build_identity_sha256": digest.hexdigest(),
        "compiler_version": version,
    }


@functools.lru_cache(maxsize=1)
def _load_library():
    if os.environ.get("OJASX_LEECH_NATIVE_DISABLE") == "1":
        return None, {"available": False, "reason": "disabled_by_environment"}
    compiler = _compiler()
    if compiler is None:
        return None, {"available": False, "reason": "no_cxx_compiler"}
    try:
        identity = _source_identity(compiler)
        cache = _cache_directory()
        cache.mkdir(parents=True, exist_ok=True)
        library_path = cache / f"libleech24-{identity['build_identity_sha256'][:16]}.so"
        lock_path = cache / "build.lock"
        with lock_path.open("a+b") as lock:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            if not library_path.exists():
                temporary = cache / f"build-{os.getpid()}.so"
                command = [compiler, *_FLAGS, str(_SOURCE), "-o", str(temporary)]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if completed.returncode:
                    temporary.unlink(missing_ok=True)
                    reason = f"compiler_failed: {completed.stderr.strip()[:1000]}"
                    return None, {"available": False, "reason": reason, **identity}
                os.replace(temporary, library_path)
        library = ctypes.CDLL(str(library_path))
        void_pointer_array = ctypes.POINTER(ctypes.c_void_p)
        library.ojasx_leech24_decode_indices.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        library.ojasx_leech24_decode_indices.restype = ctypes.c_int
        library.ojasx_leech24_table_bytes.argtypes = []
        library.ojasx_leech24_table_bytes.restype = ctypes.c_size_t
        library.ojasx_leech24_decode_rows.argtypes = [
            void_pointer_array,
            void_pointer_array,
            void_pointer_array,
            ctypes.c_void_p,
            *([ctypes.c_int] * 7),
            ctypes.c_void_p,
        ]
        library.ojasx_leech24_decode_rows.restype = ctypes.c_int
        library.ojasx_leech24_fused_gemm.argtypes = [
            void_pointer_array,
            void_pointer_array,
            void_pointer_array,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *([ctypes.c_int] * 8),
            ctypes.c_void_p,
        ]
        library.ojasx_leech24_fused_gemm.restype = ctypes.c_int
        library.ojasx_rvq_fused_gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *([ctypes.c_int] * 8),
            ctypes.c_void_p,
        ]
        library.ojasx_rvq_fused_gemm.restype = ctypes.c_int
        status = {
            "available": True,
            "reason": "loaded",
            "library_path": str(library_path),
            "build_flags": list(_FLAGS),
            "native_table_bytes": int(library.ojasx_leech24_table_bytes()),
            **identity,
        }
        return library, status
    except (OSError, subprocess.SubprocessError) as error:
        return None, {"available": False, "reason": f"{type(error).__name__}: {error}"}


def native_backend_status() -> Dict[str, object]:
    """Load/build if necessary and return auditable backend information."""

    _, status = _load_library()
    return dict(status)


def _pointer(tensor: torch.Tensor) -> Optional[ctypes.c_void_p]:
    return ctypes.c_void_p(tensor.data_ptr()) if tensor.numel() else None


def _require_cpu_contiguous(tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError("Native Leech decoding requires contiguous CPU tensors of the declared dtype")


def decode_leech_indices_native(indices: torch.Tensor) -> torch.Tensor:
    """Decode integer points through the independent native construction."""

    library, status = _load_library()
    if library is None:
        raise RuntimeError(f"Native Leech backend unavailable: {status['reason']}")
    if indices.device.type != "cpu" or indices.dtype not in {
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("Native index decoding expects an integer CPU tensor")
    flat = indices.to(torch.int32).reshape(-1).contiguous()
    if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= INDEX_COUNT):
        raise ValueError("A Leech shell index is outside 0 .. 196560")
    output = torch.empty((*indices.shape, LEECH_DIM), dtype=torch.int32)
    result = library.ojasx_leech24_decode_indices(
        _pointer(flat),
        flat.numel(),
        _pointer(output),
    )
    if result != 0:
        raise RuntimeError(f"Native Leech index decoding failed with status {result}")
    return output


def decode_leech_rows_unrotated_native(
    encoded, row_start: int, row_stop: int
) -> Optional[torch.Tensor]:
    """Decode scaled, padded float32 row blocks in rotated Leech coordinates."""

    library, _ = _load_library()
    if library is None:
        return None
    tensors = (
        *encoded.packed_indices,
        *encoded.packed_gains,
        *encoded.presence,
        encoded.gain_codebooks,
        encoded.row_scales,
    )
    for tensor in tensors:
        _require_cpu_contiguous(
            tensor,
            torch.float32
            if tensor is encoded.gain_codebooks or tensor is encoded.row_scales
            else torch.uint8,
        )
    rows, columns = encoded.original_shape
    if not 0 <= row_start < row_stop <= rows:
        raise IndexError("Invalid native Leech row range")
    padded_columns = encoded.blocks_per_row * LEECH_DIM
    output = torch.empty((row_stop - row_start, padded_columns), dtype=torch.float32)
    stages = encoded.stages
    array_type = ctypes.c_void_p * stages
    indices = array_type(*(_pointer(tensor) for tensor in encoded.packed_indices))
    gains = array_type(*(_pointer(tensor) for tensor in encoded.packed_gains))
    presence = array_type(*(_pointer(tensor) for tensor in encoded.presence))
    result = library.ojasx_leech24_decode_rows(
        indices,
        gains,
        presence,
        _pointer(encoded.gain_codebooks),
        stages,
        encoded.gain_bits,
        int(encoded.adaptive_stages),
        rows,
        columns,
        row_start,
        row_stop,
        _pointer(output),
    )
    if result != 0:
        raise RuntimeError(f"Native Leech row decoding failed with status {result}")
    return output * encoded.row_scales[row_start:row_stop, None]


def decode_leech_rows_native(encoded, row_start: int, row_stop: int) -> Optional[torch.Tensor]:
    """Decode a float32 row block, or return ``None`` when auto-fallback is needed."""

    library, _ = _load_library()
    if library is None:
        return None
    tensors = (
        *encoded.packed_indices,
        *encoded.packed_gains,
        *encoded.presence,
        encoded.gain_codebooks,
        encoded.row_scales,
    )
    for tensor in tensors:
        _require_cpu_contiguous(
            tensor,
            torch.float32
            if tensor is encoded.gain_codebooks or tensor is encoded.row_scales
            else torch.uint8,
        )
    rows, columns = encoded.original_shape
    if not 0 <= row_start < row_stop <= rows:
        raise IndexError("Invalid native Leech row range")
    padded_columns = encoded.blocks_per_row * LEECH_DIM
    output = torch.empty((row_stop - row_start, padded_columns), dtype=torch.float32)
    stages = encoded.stages
    array_type = ctypes.c_void_p * stages
    indices = array_type(*(_pointer(tensor) for tensor in encoded.packed_indices))
    gains = array_type(*(_pointer(tensor) for tensor in encoded.packed_gains))
    presence = array_type(*(_pointer(tensor) for tensor in encoded.presence))
    result = library.ojasx_leech24_decode_rows(
        indices,
        gains,
        presence,
        _pointer(encoded.gain_codebooks),
        stages,
        encoded.gain_bits,
        int(encoded.adaptive_stages),
        rows,
        columns,
        row_start,
        row_stop,
        _pointer(output),
    )
    if result != 0:
        raise RuntimeError(f"Native Leech row decoding failed with status {result}")
    rotated = apply_rotation(
        output.reshape(-1, LEECH_DIM),
        encoded.rotation_reflections,
        encoded.rotation_seed,
        inverse=True,
    )
    reconstructed = rotated.reshape(row_stop - row_start, padded_columns)
    reconstructed = reconstructed * encoded.row_scales[row_start:row_stop, None]
    return reconstructed[:, :columns]


def fused_leech_gemm_native(
    encoded,
    rotated_inputs: torch.Tensor,
    row_start: int,
    row_stop: int,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Fused C++ CPU matrix-matrix multiplication in rotated Leech coordinates."""
    library, _ = _load_library()
    if library is None:
        return None
    tensors = (
        *encoded.packed_indices,
        *encoded.packed_gains,
        *encoded.presence,
        encoded.gain_codebooks,
        encoded.row_scales,
    )
    for tensor in tensors:
        _require_cpu_contiguous(
            tensor,
            torch.float32
            if tensor is encoded.gain_codebooks or tensor is encoded.row_scales
            else torch.uint8,
        )
    rows, columns = encoded.original_shape
    if not 0 <= row_start <= row_stop <= rows:
        raise IndexError("Invalid native Leech row range")
    padded_columns = encoded.blocks_per_row * LEECH_DIM
    if rotated_inputs.shape[-1] != padded_columns:
        raise ValueError(
            f"Expected rotated inputs with {padded_columns} columns, got {rotated_inputs.shape[-1]}"
        )
    batch_shape = rotated_inputs.shape[:-1]
    flat_inputs = rotated_inputs.reshape(-1, padded_columns)
    batch_size = flat_inputs.shape[0]
    row_count = row_stop - row_start
    if row_count == 0 or batch_size == 0:
        return torch.empty((*batch_shape, row_count), dtype=torch.float32, device=rotated_inputs.device)
    if flat_inputs.device.type != "cpu" or flat_inputs.dtype != torch.float32:
        return None
    if not flat_inputs.is_contiguous():
        flat_inputs = flat_inputs.contiguous()

    bias_ptr = None
    if bias is not None:
        _require_cpu_contiguous(bias, torch.float32)
        if bias.numel() == rows:
            bias_ptr = _pointer(bias)
        elif bias.numel() == row_count:
            full_bias = torch.zeros(rows, dtype=torch.float32)
            full_bias[row_start:row_stop] = bias
            bias_ptr = _pointer(full_bias)
        else:
            raise ValueError(f"Bias must have {rows} or {row_count} elements, got {bias.numel()}")

    output = torch.empty((batch_size, row_count), dtype=torch.float32)
    stages = encoded.stages
    array_type = ctypes.c_void_p * stages
    indices = array_type(*(_pointer(tensor) for tensor in encoded.packed_indices))
    gains = array_type(*(_pointer(tensor) for tensor in encoded.packed_gains))
    presence = array_type(*(_pointer(tensor) for tensor in encoded.presence))

    result = library.ojasx_leech24_fused_gemm(
        indices,
        gains,
        presence,
        _pointer(encoded.gain_codebooks),
        _pointer(encoded.row_scales),
        bias_ptr,
        _pointer(flat_inputs),
        batch_size,
        stages,
        encoded.gain_bits,
        int(encoded.adaptive_stages),
        rows,
        columns,
        row_start,
        row_stop,
        _pointer(output),
    )
    if result != 0:
        raise RuntimeError(f"Native Leech fused GEMM failed with status {result}")
    return output.reshape(*batch_shape, row_count)


def fused_rvq_gemm_native(
    encoded,
    inputs: torch.Tensor,
    row_start: int,
    row_stop: int,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Fused C++ CPU matrix-matrix multiplication for adaptive residual VQ."""
    library, _ = _load_library()
    if library is None:
        return None
    _require_cpu_contiguous(encoded.packed_codes, torch.uint8)
    _require_cpu_contiguous(encoded.codebooks, torch.float32)
    _require_cpu_contiguous(encoded.row_scales, torch.float32)
    rows, columns = encoded.original_shape
    if not 0 <= row_start <= row_stop <= rows:
        raise IndexError("Invalid native RVQ row range")
    if inputs.shape[-1] != columns:
        raise ValueError(
            f"Expected inputs with {columns} columns, got {inputs.shape[-1]}"
        )
    batch_shape = inputs.shape[:-1]
    flat_inputs = inputs.reshape(-1, columns)
    batch_size = flat_inputs.shape[0]
    row_count = row_stop - row_start
    if row_count == 0 or batch_size == 0:
        return torch.empty((*batch_shape, row_count), dtype=torch.float32, device=inputs.device)
    if flat_inputs.device.type != "cpu" or flat_inputs.dtype != torch.float32:
        return None
    if not flat_inputs.is_contiguous():
        flat_inputs = flat_inputs.contiguous()

    bias_ptr = None
    if bias is not None:
        _require_cpu_contiguous(bias, torch.float32)
        if bias.numel() == rows:
            bias_ptr = _pointer(bias)
        elif bias.numel() == row_count:
            full_bias = torch.zeros(rows, dtype=torch.float32)
            full_bias[row_start:row_stop] = bias
            bias_ptr = _pointer(full_bias)
        else:
            raise ValueError(f"Bias must have {rows} or {row_count} elements, got {bias.numel()}")

    output = torch.empty((batch_size, row_count), dtype=torch.float32)
    result = library.ojasx_rvq_fused_gemm(
        _pointer(encoded.packed_codes),
        _pointer(encoded.codebooks),
        _pointer(encoded.row_scales),
        bias_ptr,
        _pointer(flat_inputs),
        batch_size,
        encoded.stages,
        encoded.bits_per_stage,
        encoded.block_size,
        rows,
        columns,
        row_start,
        row_stop,
        _pointer(output),
    )
    if result != 0:
        raise RuntimeError(f"Native RVQ fused GEMM failed with status {result}")
    return output.reshape(*batch_shape, row_count)


def cuda_kernel_source_status() -> Dict[str, object]:
    """Verify presence and environment readiness of vq_kernels_cuda.cu."""
    exists = _CUDA_SOURCE.exists()
    source_sha256 = hashlib.sha256(_CUDA_SOURCE.read_bytes()).hexdigest() if exists else None
    nvcc_path = shutil.which("nvcc")
    torch_cuda = torch.cuda.is_available()
    return {
        "exists": exists,
        "source_path": str(_CUDA_SOURCE),
        "source_sha256": source_sha256,
        "nvcc_available": nvcc_path is not None,
        "nvcc_path": nvcc_path,
        "torch_cuda_available": torch_cuda,
        "ready_for_cuda_build": bool(exists and nvcc_path and torch_cuda),
    }


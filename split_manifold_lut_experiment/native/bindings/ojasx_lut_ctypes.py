"""
ojasx_lut_ctypes.py

Python ctypes wrapper for libojasx_lut.dylib / libojasx_lut.so.
Allows high-level Python code (Transformers, PyTorch, llama.cpp runner)
to invoke line-rate native C++ SIMD routines directly.

Author: OjasX Research Team & Antigravity Prime Coder
"""

import os
import ctypes
import numpy as np

# Locate shared library
LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_PATH = os.path.join(LIB_DIR, "libojasx_lut.dylib")
if not os.path.exists(LIB_PATH):
    LIB_PATH = os.path.join(LIB_DIR, "libojasx_lut.so")

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Native library not found at: {LIB_PATH}")

_lib = ctypes.CDLL(LIB_PATH)

# Function Signatures
_lib.ojasx_get_version.restype = ctypes.c_char_p
_lib.ojasx_get_backend_name.restype = ctypes.c_char_p

_lib.ojasx_create_linear.restype = ctypes.c_void_p
_lib.ojasx_create_linear.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

_lib.ojasx_set_linear_outliers.restype = ctypes.c_int
_lib.ojasx_set_linear_outliers.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

_lib.ojasx_linear_forward.restype = ctypes.c_int
_lib.ojasx_linear_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
]

_lib.ojasx_get_compressed_size.restype = ctypes.c_size_t
_lib.ojasx_get_compressed_size.argtypes = [ctypes.c_void_p]

_lib.ojasx_get_compression_ratio.restype = ctypes.c_float
_lib.ojasx_get_compression_ratio.argtypes = [ctypes.c_void_p]

_lib.ojasx_destroy_linear.argtypes = [ctypes.c_void_p]

_lib.ojasx_create_kv_cache.restype = ctypes.c_void_p
_lib.ojasx_create_kv_cache.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.ojasx_kv_cache_append.restype = ctypes.c_int
_lib.ojasx_kv_cache_append.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

_lib.ojasx_kv_cache_attention.restype = ctypes.c_int
_lib.ojasx_kv_cache_attention.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float
]

_lib.ojasx_kv_cache_get_memory_size.restype = ctypes.c_size_t
_lib.ojasx_kv_cache_get_memory_size.argtypes = [ctypes.c_void_p]

_lib.ojasx_destroy_kv_cache.argtypes = [ctypes.c_void_p]

# MoE Signatures
_lib.ojasx_create_moe.restype = ctypes.c_void_p
_lib.ojasx_create_moe.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

_lib.ojasx_moe_forward.restype = ctypes.c_int
_lib.ojasx_moe_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
]

_lib.ojasx_moe_get_compressed_size.restype = ctypes.c_size_t
_lib.ojasx_moe_get_compressed_size.argtypes = [ctypes.c_void_p]

_lib.ojasx_moe_get_compression_ratio.restype = ctypes.c_float
_lib.ojasx_moe_get_compression_ratio.argtypes = [ctypes.c_void_p]

_lib.ojasx_destroy_moe.argtypes = [ctypes.c_void_p]


def get_engine_info():
    return {
        "version": _lib.ojasx_get_version().decode("utf-8"),
        "backend": _lib.ojasx_get_backend_name().decode("utf-8")
    }


class NativeSplitManifoldLinear:
    def __init__(self, in_features: int, out_features: int, block_dim: int = 16, codebook_size: int = 256):
        self.in_features = in_features
        self.out_features = out_features
        self.block_dim = block_dim
        self.codebook_size = codebook_size
        self._handle = None

    def initialize_weights(self, codebook: np.ndarray, codes: np.ndarray, scales: np.ndarray, bias: np.ndarray = None):
        codebook_c = codebook.astype(np.float32).ctypes.data_as(ctypes.c_void_p)
        codes_c = codes.astype(np.uint8).ctypes.data_as(ctypes.c_void_p)
        scales_c = scales.astype(np.float32).ctypes.data_as(ctypes.c_void_p)
        bias_c = bias.astype(np.float32).ctypes.data_as(ctypes.c_void_p) if bias is not None else None

        self._handle = _lib.ojasx_create_linear(
            self.in_features, self.out_features, self.block_dim, self.codebook_size,
            codebook_c, codes_c, scales_c, bias_c
        )

    def set_outliers(self, row_ptrs: np.ndarray, col_indices: np.ndarray, values: np.ndarray):
        assert self._handle is not None
        nnz = len(values)
        _lib.ojasx_set_linear_outliers(
            self._handle, nnz,
            row_ptrs.astype(np.int32).ctypes.data_as(ctypes.c_void_p),
            col_indices.astype(np.int32).ctypes.data_as(ctypes.c_void_p),
            values.astype(np.float32).ctypes.data_as(ctypes.c_void_p)
        )

    def forward(self, x: np.ndarray, num_threads: int = 0) -> np.ndarray:
        assert self._handle is not None
        batch_size = x.shape[0] if x.ndim > 1 else 1
        x_flat = np.ascontiguousarray(x.reshape(batch_size, self.in_features), dtype=np.float32)
        y = np.zeros((batch_size, self.out_features), dtype=np.float32)

        ret = _lib.ojasx_linear_forward(
            self._handle,
            x_flat.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            batch_size,
            num_threads
        )
        assert ret == 0, f"Forward failed with code {ret}"
        return y.squeeze(0) if x.ndim == 1 else y

    @property
    def compressed_bytes(self) -> int:
        return _lib.ojasx_get_compressed_size(self._handle) if self._handle else 0

    @property
    def compression_ratio(self) -> float:
        return _lib.ojasx_get_compression_ratio(self._handle) if self._handle else 0.0

    def __del__(self):
        if self._handle:
            _lib.ojasx_destroy_linear(self._handle)
            self._handle = None


class NativeKVCache:
    def __init__(self, num_heads: int, head_dim: int = 128, max_seq_len: int = 4096, block_dim: int = 16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._handle = _lib.ojasx_create_kv_cache(num_heads, head_dim, max_seq_len, block_dim)

    def append(self, head_idx: int, k: np.ndarray, v: np.ndarray):
        k_c = np.ascontiguousarray(k, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        v_c = np.ascontiguousarray(v, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        ret = _lib.ojasx_kv_cache_append(self._handle, head_idx, k_c, v_c)
        assert ret == 0, f"KV append failed with code {ret}"

    def attention(self, head_idx: int, q: np.ndarray, scale_factor: float = -1.0) -> np.ndarray:
        q_c = np.ascontiguousarray(q, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        out = np.zeros(self.head_dim, dtype=np.float32)
        ret = _lib.ojasx_kv_cache_attention(self._handle, head_idx, q_c, out.ctypes.data_as(ctypes.c_void_p), scale_factor)
        assert ret == 0, f"Attention failed with code {ret}"
        return out

    @property
    def total_bytes(self) -> int:
        return _lib.ojasx_kv_cache_get_memory_size(self._handle)

    def __del__(self):
        if self._handle:
            _lib.ojasx_destroy_kv_cache(self._handle)
            self._handle = None


class NativeSplitManifoldMoE:
    def __init__(self, num_experts: int = 512, hidden_dim: int = 2560, intermediate_dim: int = 640, top_k: int = 10, block_dim: int = 16):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.top_k = top_k
        self.block_dim = block_dim
        self._handle = None

    def initialize_weights(
        self,
        codebook_in: np.ndarray,
        codebook_down: np.ndarray,
        gate_up_codes: np.ndarray,
        gate_up_scales: np.ndarray,
        down_codes: np.ndarray,
        down_scales: np.ndarray
    ):
        cb_in_c = np.ascontiguousarray(codebook_in, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        cb_down_c = np.ascontiguousarray(codebook_down, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        gu_codes_c = np.ascontiguousarray(gate_up_codes, dtype=np.uint8).ctypes.data_as(ctypes.c_void_p)
        gu_scales_c = np.ascontiguousarray(gate_up_scales, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        dn_codes_c = np.ascontiguousarray(down_codes, dtype=np.uint8).ctypes.data_as(ctypes.c_void_p)
        dn_scales_c = np.ascontiguousarray(down_scales, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)

        self._handle = _lib.ojasx_create_moe(
            self.num_experts, self.hidden_dim, self.intermediate_dim, self.top_k, self.block_dim,
            cb_in_c, cb_down_c, gu_codes_c, gu_scales_c, dn_codes_c, dn_scales_c
        )
        assert self._handle is not None, "Failed to create Native MoE engine handle"

    def forward(self, x: np.ndarray, top_k_indices: np.ndarray, top_k_weights: np.ndarray) -> np.ndarray:
        assert self._handle is not None
        x_c = np.ascontiguousarray(x, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        indices_c = np.ascontiguousarray(top_k_indices, dtype=np.int32).ctypes.data_as(ctypes.c_void_p)
        weights_c = np.ascontiguousarray(top_k_weights, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
        y_out = np.zeros(self.hidden_dim, dtype=np.float32)

        ret = _lib.ojasx_moe_forward(
            self._handle, x_c, indices_c, weights_c, y_out.ctypes.data_as(ctypes.c_void_p), len(top_k_indices)
        )
        assert ret == 0, f"MoE forward failed with code {ret}"
        return y_out

    @property
    def compressed_bytes(self) -> int:
        return _lib.ojasx_moe_get_compressed_size(self._handle) if self._handle else 0

    @property
    def compression_ratio(self) -> float:
        return _lib.ojasx_moe_get_compression_ratio(self._handle) if self._handle else 0.0

    def __del__(self):
        if self._handle:
            _lib.ojasx_destroy_moe(self._handle)
            self._handle = None


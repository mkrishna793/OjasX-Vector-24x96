"""M12: Universal TileLang & Multi-Hardware Backend Test Suite.

Verifies:
1. TileLangHardwareAutoTuner.select_tile_config across all 6 hardware families
   ("cuda", "metal"/"mps", "rocm", "sycl", "ascend", "cpu") for decode (M=1, 4)
   and prefill (M=128, 512) shapes, 24-divisible and non-24-divisible dimensions,
   and validates strict error handling for invalid targets and non-positive dimensions.
2. TileLangHardwareAutoTuner.emit_universal_kernel_source across all architectures,
   and rigorously compiles the emitted C++20 CPU SIMD kernel via g++ (-std=c++20 -O2 -shared -fPIC),
   executing ojasx_cpu_tiled_gemm_f32 via ctypes to verify exact numerical parity
   (atol=1e-5) against torch.nn.functional.linear.
3. dispatch_universal_compressed_gemm end-to-end execution parity for both
   QuantizedWeight (Adaptive RVQ) and LeechQuantizedWeight (Leech24) against
   reference dequantized PyTorch linear operations on CPU and MPS (when available).
"""

from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from ojasx_24x96.codec import QuantizationConfig, dequantize_weight, quantize_weight
from ojasx_24x96.leech_codec import (
    LeechQuantizationConfig,
    dequantize_leech_weight,
    quantize_leech_weight,
)
from ojasx_24x96.tilelang_backend import (
    HAS_TILELANG,
    HardwareTileSchedule,
    SUPPORTED_HARDWARE_TARGETS,
    TileLangHardwareAutoTuner,
    dispatch_universal_compressed_gemm,
)


# ============================================================================
# 1. TileLangHardwareAutoTuner: Tile Schedule Selection Tests
# ============================================================================


class TestTileLangHardwareAutoTunerSchedules:
    """Verifies hardware tile configuration selection across all architectures and shapes."""

    @pytest.mark.parametrize("target", ["cuda", "metal", "mps", "rocm", "sycl", "ascend", "cpu"])
    @pytest.mark.parametrize("m_val", [1, 4, 128, 512])
    @pytest.mark.parametrize("k_val", [768, 512])
    def test_select_tile_config_all_families_and_shapes(
        self, target: str, m_val: int, k_val: int
    ) -> None:
        """Verify tile configuration across all 6 hardware families, decode/prefill, and 24-divisible K."""
        n_val = 128
        schedule = TileLangHardwareAutoTuner.select_tile_config(
            M=m_val,
            N=n_val,
            K=k_val,
            device=target,
            dtype=torch.float16,
        )

        assert isinstance(schedule, HardwareTileSchedule)
        assert schedule.block_m > 0
        assert schedule.block_n > 0
        assert schedule.block_k > 0
        assert schedule.panel_size > 0
        assert schedule.pipeline_stages > 0
        assert schedule.sram_bytes > 0

        # Decode detection
        is_decode = m_val <= 16
        assert schedule.is_decode == is_decode

        # Arch normalization
        expected_arch = "metal" if target == "mps" else target
        assert schedule.target_arch == expected_arch

        # 24-divisibility impact on block_k
        if expected_arch in ("cuda", "rocm"):
            assert schedule.block_k == (24 if (k_val % 24 == 0) else 32)
            if is_decode:
                assert schedule.block_m == 16
                assert schedule.block_n == 64
                assert schedule.pipeline_stages == 2
            else:
                assert schedule.block_m == 128
                assert schedule.block_n == 128
                assert schedule.pipeline_stages == 3
        elif expected_arch == "metal":
            assert schedule.block_k == (24 if (k_val % 24 == 0) else 16)
            assert schedule.block_n == 32
            assert schedule.block_m == (8 if is_decode else 32)
            assert schedule.pipeline_stages == (1 if is_decode else 2)
        elif expected_arch in ("sycl", "ascend"):
            assert schedule.block_k == 16
            assert schedule.block_n == 64
            assert schedule.block_m == (16 if is_decode else 64)
            assert schedule.pipeline_stages == 2
        elif expected_arch == "cpu":
            assert schedule.block_k == (24 if (k_val % 24 == 0) else 16)
            assert schedule.block_m == (8 if is_decode else 32)
            assert schedule.pipeline_stages == 1

        # Dict representation check
        d = schedule.to_dict()
        assert d["target_arch"] == expected_arch
        assert d["block_m"] == schedule.block_m
        assert d["block_n"] == schedule.block_n
        assert d["block_k"] == schedule.block_k
        assert d["panel_size"] == schedule.panel_size
        assert d["pipeline_stages"] == schedule.pipeline_stages
        assert d["sram_bytes"] == schedule.sram_bytes
        assert d["is_decode"] == is_decode

    def test_normalize_target_aliases_and_devices(self) -> None:
        """Verify device string normalization for aliases, torch.device, and auto."""
        assert TileLangHardwareAutoTuner.normalize_target("cuda:0") == "cuda"
        assert TileLangHardwareAutoTuner.normalize_target("mps") == "metal"
        assert TileLangHardwareAutoTuner.normalize_target(torch.device("mps")) == "metal"
        assert TileLangHardwareAutoTuner.normalize_target("xpu") == "sycl"
        assert TileLangHardwareAutoTuner.normalize_target("hip") == "rocm"
        assert TileLangHardwareAutoTuner.normalize_target(torch.device("cpu")) == "cpu"

        auto_norm = TileLangHardwareAutoTuner.normalize_target("auto")
        assert auto_norm in ("cuda", "metal", "cpu")

    @pytest.mark.parametrize("bad_target", ["tpu", "dsp", "quantum", "unknown_device", "", "npugpu"])
    def test_invalid_hardware_target_raises_value_error(self, bad_target: str) -> None:
        """Ensure unrecognised target device names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported hardware target"):
            TileLangHardwareAutoTuner.select_tile_config(16, 64, 768, device=bad_target)

    @pytest.mark.parametrize(
        ("m", "n", "k"),
        [
            (0, 64, 768),
            (-1, 64, 768),
            (16, 0, 768),
            (16, -5, 768),
            (16, 64, 0),
            (16, 64, -24),
            (-8, -8, -8),
        ],
    )
    def test_non_positive_dimensions_raise_value_error(self, m: int, n: int, k: int) -> None:
        """Ensure zero or negative dimensions (M, N, K) raise ValueError."""
        with pytest.raises(ValueError, match="Matrix dimensions must be positive"):
            TileLangHardwareAutoTuner.select_tile_config(M=m, N=n, K=k, device="cpu")

    def test_dtype_sram_bytes_scaling(self) -> None:
        """Verify sram_bytes reflects element size (fp16/bf16 = 2 bytes, fp32 = 4 bytes)."""
        fp16_sched = TileLangHardwareAutoTuner.select_tile_config(32, 64, 768, "cpu", dtype=torch.float16)
        bf16_sched = TileLangHardwareAutoTuner.select_tile_config(32, 64, 768, "cpu", dtype=torch.bfloat16)
        fp32_sched = TileLangHardwareAutoTuner.select_tile_config(32, 64, 768, "cpu", dtype=torch.float32)

        assert fp16_sched.sram_bytes == bf16_sched.sram_bytes
        assert fp32_sched.sram_bytes == 2 * fp16_sched.sram_bytes


# ============================================================================
# 2. Universal Kernel Source Emission & Live C++20 Compilation / CDLL Tests
# ============================================================================


class TestUniversalKernelEmissionAndCompilation:
    """Verifies universal kernel code emission and executes compiled C++20 CPU SIMD kernel."""

    @pytest.mark.parametrize("target", ["metal", "cuda", "rocm", "sycl", "ascend", "cpu"])
    def test_emit_universal_kernel_source_all_targets(self, target: str) -> None:
        """Verify valid C++/CUDA/MSL source is emitted for every supported accelerator family."""
        sched = TileLangHardwareAutoTuner.select_tile_config(M=32, N=64, K=768, device=target)
        source = TileLangHardwareAutoTuner.emit_universal_kernel_source(sched)
        assert isinstance(source, str) and len(source) > 100

        if target == "metal":
            assert "#include <metal_stdlib>" in source
            assert "ojasx_metal_fused_vq_gemm" in source
            assert "struct OjasxGemmParams" in source
        elif target in ("cuda", "rocm"):
            assert "__global__" in source
            assert f"ojasx_{target}_fused_vq_gemm" in source
        elif target in ("sycl", "ascend"):
            assert f"ojasx_{target}_fused_vq_gemm" in source
        elif target == "cpu":
            assert "ojasx_cpu_tiled_gemm_f32" in source
            assert "#include <cstdint>" in source

    def test_emit_kernel_source_invalid_tile_raises(self) -> None:
        """Verify non-positive tile schedule raises ValueError during kernel emission."""
        invalid_sched = HardwareTileSchedule(
            target_arch="cpu",
            block_m=0,
            block_n=64,
            block_k=24,
            panel_size=4,
            pipeline_stages=1,
            sram_bytes=1024,
            is_decode=False,
        )
        with pytest.raises(ValueError, match="Tile dimensions must be strictly positive"):
            TileLangHardwareAutoTuner.emit_universal_kernel_source(invalid_sched)

    @pytest.mark.parametrize(
        ("m", "n", "k"),
        [
            (1, 32, 96),       # Single-row decode, 24-divisible K
            (4, 64, 768),      # Multi-row decode, 24-divisible K
            (13, 37, 128),     # Non-tile-multiple shapes (stress loop boundary conditions)
            (128, 128, 512),   # Prefill shape, non-24-divisible K
        ],
    )
    def test_compile_and_execute_cpu_cpp20_kernel_numerical_parity(
        self, tmp_path: Path, m: int, n: int, k: int
    ) -> None:
        """Strict verification: Emits C++20 kernel, compiles with g++, loads via ctypes,

        executes on random inputs, and verifies exact numerical parity against torch.nn.functional.linear.
        """
        sched = TileLangHardwareAutoTuner.select_tile_config(M=m, N=n, K=k, device="cpu")
        cpp_source = TileLangHardwareAutoTuner.emit_universal_kernel_source(sched)

        # Write C++ source to tmp_path
        src_file = tmp_path / f"ojasx_cpu_kernel_{m}_{n}_{k}.cpp"
        so_file = tmp_path / f"libojasx_cpu_{m}_{n}_{k}.so"
        src_file.write_text(cpp_source)

        # Real C++20 compilation via g++ - never mocked
        compile_cmd = [
            "g++",
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            str(src_file),
            "-o",
            str(so_file),
        ]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"g++ compilation failed:\n{res.stderr}"
        assert so_file.is_file(), "Compiled shared library not found"

        # Load DLL and resolve entrypoint
        dll = ctypes.CDLL(str(so_file))
        assert hasattr(dll, "ojasx_cpu_tiled_gemm_f32")
        kernel_fn = dll.ojasx_cpu_tiled_gemm_f32
        kernel_fn.argtypes = [
            ctypes.c_void_p,  # const float* X
            ctypes.c_void_p,  # const float* W
            ctypes.c_void_p,  # const float* Bias
            ctypes.c_void_p,  # float* Out
            ctypes.c_int64,   # M
            ctypes.c_int64,   # N
            ctypes.c_int64,   # K
        ]
        kernel_fn.restype = None

        # Generate inputs with Xavier-scaled variance for numerical stability
        torch.manual_seed(1000 + m + n + k)
        scale = 1.0 / math.sqrt(k)
        x_mat = torch.randn(m, k, dtype=torch.float32).contiguous() * scale
        w_mat = torch.randn(n, k, dtype=torch.float32).contiguous() * scale
        bias_vec = torch.randn(n, dtype=torch.float32).contiguous() * 0.05

        # 1. Test WITH Bias
        ref_out_bias = F.linear(x_mat, w_mat, bias_vec)
        cpp_out_bias = torch.zeros(m, n, dtype=torch.float32).contiguous()

        kernel_fn(
            x_mat.data_ptr(),
            w_mat.data_ptr(),
            bias_vec.data_ptr(),
            cpp_out_bias.data_ptr(),
            m,
            n,
            k,
        )
        assert torch.allclose(ref_out_bias, cpp_out_bias, atol=1e-5, rtol=1e-4), (
            f"Numerical divergence with bias for shape ({m}, {n}, {k}): "
            f"max_diff={(ref_out_bias - cpp_out_bias).abs().max().item():.2e}"
        )

        # 2. Test WITHOUT Bias (NULL pointer)
        ref_out_nobias = F.linear(x_mat, w_mat, None)
        cpp_out_nobias = torch.zeros(m, n, dtype=torch.float32).contiguous()

        kernel_fn(
            x_mat.data_ptr(),
            w_mat.data_ptr(),
            None,
            cpp_out_nobias.data_ptr(),
            m,
            n,
            k,
        )
        assert torch.allclose(ref_out_nobias, cpp_out_nobias, atol=1e-5, rtol=1e-4), (
            f"Numerical divergence without bias for shape ({m}, {n}, {k}): "
            f"max_diff={(ref_out_nobias - cpp_out_nobias).abs().max().item():.2e}"
        )


# ============================================================================
# 3. Universal Compressed GEMM Dispatch Tests (RVQ & Leech24)
# ============================================================================


class TestDispatchUniversalCompressedGEMM:
    """Verifies end-to-end dispatch parity for Adaptive RVQ and Leech24 compressed weights."""

    @pytest.mark.parametrize("prefer_subspace_lut", [True, False])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_dispatch_rvq_cpu_end_to_end(self, prefer_subspace_lut: bool, has_bias: bool) -> None:
        """Verify Adaptive RVQ compressed GEMM dispatch matches dequantized PyTorch linear on CPU."""
        torch.manual_seed(20261105)
        rows, cols = 32, 64
        weight = torch.randn(rows, cols, dtype=torch.float32)
        inputs = torch.randn(4, cols, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32) if has_bias else None

        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded, _ = quantize_weight(weight, cfg)
        assert encoded is not None

        ref_weight = dequantize_weight(encoded)
        ref_out = F.linear(inputs, ref_weight, bias)

        out, sched = dispatch_universal_compressed_gemm(
            inputs,
            encoded,
            bias=bias,
            prefer_subspace_lut=prefer_subspace_lut,
        )

        assert isinstance(sched, HardwareTileSchedule)
        assert sched.target_arch == "cpu"
        assert out.shape == ref_out.shape
        assert torch.allclose(ref_out, out, atol=1e-5, rtol=1e-4)

    def test_dispatch_rvq_multi_dim_inputs(self) -> None:
        """Verify 3D activation shapes [B, S, K] correctly map to [B, S, N]."""
        torch.manual_seed(999)
        rows, cols = 16, 48
        weight = torch.randn(rows, cols, dtype=torch.float32)
        inputs = torch.randn(2, 5, cols, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32)

        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded, _ = quantize_weight(weight, cfg)
        assert encoded is not None

        ref_weight = dequantize_weight(encoded)
        ref_out = F.linear(inputs, ref_weight, bias)

        out, sched = dispatch_universal_compressed_gemm(inputs, encoded, bias=bias)
        assert out.shape == (2, 5, rows)
        assert torch.allclose(ref_out, out, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("has_bias", [True, False])
    @pytest.mark.parametrize("cols", [48, 70])  # 48 is 24-divisible, 70 is non-24-divisible
    def test_dispatch_leech_cpu_end_to_end(self, has_bias: bool, cols: int) -> None:
        """Verify Leech24 compressed GEMM dispatch matches dequantized PyTorch linear on CPU."""
        torch.manual_seed(42 + cols)
        rows = 24
        weight = torch.randn(rows, cols, dtype=torch.float32)
        inputs = torch.randn(3, cols, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32) if has_bias else None

        lcfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded, _ = quantize_leech_weight(weight, lcfg)
        assert encoded is not None

        ref_weight = dequantize_leech_weight(encoded)
        ref_out = F.linear(inputs, ref_weight, bias)

        out, sched = dispatch_universal_compressed_gemm(inputs, encoded, bias=bias)

        assert isinstance(sched, HardwareTileSchedule)
        assert sched.target_arch == "cpu"
        assert out.shape == ref_out.shape
        assert torch.allclose(ref_out, out, atol=1e-5, rtol=1e-4)

    def test_dispatch_leech_multi_dim_inputs(self) -> None:
        """Verify 3D activation shapes [B, S, K] correctly map to [B, S, N] for Leech24."""
        torch.manual_seed(777)
        rows, cols = 16, 48
        weight = torch.randn(rows, cols, dtype=torch.float32)
        inputs = torch.randn(3, 4, cols, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32)

        lcfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded, _ = quantize_leech_weight(weight, lcfg)
        assert encoded is not None

        ref_weight = dequantize_leech_weight(encoded)
        ref_out = F.linear(inputs, ref_weight, bias)

        out, sched = dispatch_universal_compressed_gemm(inputs, encoded, bias=bias)
        assert out.shape == (3, 4, rows)
        assert torch.allclose(ref_out, out, atol=1e-5, rtol=1e-4)

    def test_dispatch_mps_when_available(self) -> None:
        """Verify compressed GEMM execution on Apple Silicon MPS device if available."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("Apple Silicon MPS not available on this platform")

        device = torch.device("mps")
        rows, cols = 16, 48

        # Test Adaptive RVQ on MPS
        torch.manual_seed(123)
        weight = torch.randn(rows, cols, dtype=torch.float32)
        x_mps = torch.randn(4, cols, dtype=torch.float32, device=device)
        bias_mps = torch.randn(rows, dtype=torch.float32, device=device)

        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded_rvq, _ = quantize_weight(weight, cfg)
        assert encoded_rvq is not None

        ref_rvq_w = dequantize_weight(encoded_rvq).to(device)
        ref_rvq_out = F.linear(x_mps, ref_rvq_w, bias_mps)

        out_rvq, sched_rvq = dispatch_universal_compressed_gemm(x_mps, encoded_rvq, bias=bias_mps)
        assert out_rvq.device.type == "mps"
        assert sched_rvq.target_arch == "metal"
        assert torch.allclose(ref_rvq_out, out_rvq, atol=1e-4, rtol=1e-3)

        # Test Leech24 on MPS
        lcfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            min_weight_elements=1,
            require_compression=False,
            max_relative_error=None,
        )
        encoded_leech, _ = quantize_leech_weight(weight, lcfg)
        assert encoded_leech is not None

        ref_leech_w = dequantize_leech_weight(encoded_leech).to(device)
        ref_leech_out = F.linear(x_mps, ref_leech_w, bias_mps)

        out_leech, sched_leech = dispatch_universal_compressed_gemm(x_mps, encoded_leech, bias=bias_mps)
        assert out_leech.device.type == "mps"
        assert sched_leech.target_arch == "metal"
        assert torch.allclose(ref_leech_out, out_leech, atol=1e-4, rtol=1e-3)

"""Unit tests for M5 fused native C++ CPU/CUDA kernels and activation-folded execution."""

import pytest
import torch
import torch.nn.functional as F

from ojasx_24x96.codec import QuantizationConfig, dequantize_weight, quantize_weight
from ojasx_24x96.leech_codec import (
    LeechQuantizationConfig,
    dequantize_leech_weight,
    quantize_leech_weight,
    rotate_activation_for_leech,
)
from ojasx_24x96.leech_native import (
    cuda_kernel_source_status,
    fused_leech_gemm_native,
    fused_rvq_gemm_native,
    native_backend_status,
)
from ojasx_24x96.module import ResidualVQLinear


def _skip_if_no_native():
    status = native_backend_status()
    if not status.get("available", False):
        pytest.skip(f"Native backend unavailable: {status.get('reason')}")


class TestLeechFusedNativeGEMM:
    """Test fused Leech24 native GEMM kernel against PyTorch reference linear layer."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("stages", [1, 2])
    @pytest.mark.parametrize("adaptive_stages", [False, True])
    @pytest.mark.parametrize("rotation_reflections", [0, 2, 4])
    def test_fused_leech_gemm_variations(
        self, batch_size: int, stages: int, adaptive_stages: bool, rotation_reflections: int
    ):
        _skip_if_no_native()
        torch.manual_seed(20261104)
        rows, columns = 16, 70  # 70 is not a multiple of 24 (padded to 72)
        weight = torch.randn(rows, columns, dtype=torch.float32)
        inputs = torch.randn(batch_size, columns, dtype=torch.float32)
        bias = torch.randn(rows, dtype=torch.float32)

        config = LeechQuantizationConfig(
            max_stages=stages,
            gain_bits=1,
            rotation_reflections=rotation_reflections,
            adaptive_stages=adaptive_stages,
            seed=42,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, config)
        assert encoded is not None

        # Reference output via PyTorch F.linear + dequantize_leech_weight
        reference_weight = dequantize_leech_weight(encoded)
        reference_out = F.linear(inputs, reference_weight, bias)

        # Native fused GEMM with pre-rotated input activation
        rotated_inputs = rotate_activation_for_leech(
            inputs,
            columns,
            encoded.rotation_reflections,
            encoded.rotation_seed,
        )
        fused_out = fused_leech_gemm_native(
            encoded,
            rotated_inputs,
            0,
            rows,
            bias=bias,
        )
        assert fused_out is not None
        assert torch.allclose(reference_out, fused_out, atol=1e-5, rtol=1e-4)

    def test_fused_leech_gemm_no_bias_and_sliced_rows(self):
        _skip_if_no_native()
        torch.manual_seed(12345)
        rows, columns = 12, 48
        weight = torch.randn(rows, columns)
        inputs = torch.randn(3, columns)

        config = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=2,
            rotation_reflections=2,
            adaptive_stages=True,
            seed=99,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, config)
        assert encoded is not None

        # Slice rows [2, 10)
        row_start, row_stop = 2, 10
        ref_full_w = dequantize_leech_weight(encoded)
        ref_slice_w = ref_full_w[row_start:row_stop]
        reference_out = F.linear(inputs, ref_slice_w, None)

        rotated_inputs = rotate_activation_for_leech(
            inputs,
            columns,
            encoded.rotation_reflections,
            encoded.rotation_seed,
        )
        fused_out = fused_leech_gemm_native(
            encoded,
            rotated_inputs,
            row_start,
            row_stop,
            bias=None,
        )
        assert fused_out is not None
        assert torch.allclose(reference_out, fused_out, atol=1e-5, rtol=1e-4)


class TestRVQFusedNativeGEMM:
    """Test fused RVQ native GEMM kernel against PyTorch reference linear layer."""

    @pytest.mark.parametrize("bits_per_stage", [1, 2, 4, 8])
    @pytest.mark.parametrize("block_size", [4, 8])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_fused_rvq_gemm_widths(
        self, bits_per_stage: int, block_size: int, batch_size: int
    ):
        _skip_if_no_native()
        torch.manual_seed(42 + bits_per_stage)
        rows, columns = 14, 30  # 30 is not a multiple of 8 (padded to 32)
        weight = torch.randn(rows, columns)
        inputs = torch.randn(batch_size, columns)
        bias = torch.randn(rows)

        config = QuantizationConfig(
            block_size=block_size,
            bits_per_stage=bits_per_stage,
            max_stages=2,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_weight(weight, config)
        assert encoded is not None

        ref_weight = dequantize_weight(encoded)
        reference_out = F.linear(inputs, ref_weight, bias)

        fused_out = fused_rvq_gemm_native(
            encoded,
            inputs,
            0,
            rows,
            bias=bias,
        )
        assert fused_out is not None
        assert torch.allclose(reference_out, fused_out, atol=1e-5, rtol=1e-4)

    def test_fused_rvq_gemm_sliced_rows_no_bias(self):
        _skip_if_no_native()
        torch.manual_seed(987)
        rows, columns = 16, 24
        weight = torch.randn(rows, columns)
        inputs = torch.randn(2, columns)

        config = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=3,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_weight(weight, config)
        assert encoded is not None

        row_start, row_stop = 3, 11
        ref_w = dequantize_weight(encoded)[row_start:row_stop]
        reference_out = F.linear(inputs, ref_w, None)

        fused_out = fused_rvq_gemm_native(
            encoded,
            inputs,
            row_start,
            row_stop,
            bias=None,
        )
        assert fused_out is not None
        assert torch.allclose(reference_out, fused_out, atol=1e-5, rtol=1e-4)


class TestResidualVQLinearFusedExecution:
    """Test ResidualVQLinear module with fused native execution enabled."""

    def test_module_fused_leech(self):
        _skip_if_no_native()
        torch.manual_seed(2026)
        rows, columns = 16, 70
        weight = torch.randn(rows, columns)
        bias = torch.randn(rows)
        inputs = torch.randn(2, 5, columns)

        config = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            adaptive_stages=True,
            seed=20261104,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert not module.fused_native_gemm
        ref_out = module(inputs)

        # Enable fused native GEMM
        module.set_fused_native_gemm(True)
        assert module.fused_native_gemm
        fused_out = module(inputs)

        # Zero dense weight caching verification
        assert module.cached_weight_bytes == 0
        assert torch.allclose(ref_out, fused_out, atol=1e-5, rtol=1e-4)

    def test_module_fused_rvq(self):
        _skip_if_no_native()
        torch.manual_seed(777)
        rows, columns = 12, 36
        weight = torch.randn(rows, columns)
        bias = torch.randn(rows)
        inputs = torch.randn(4, columns)

        config = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert not module.fused_native_gemm
        ref_out = module(inputs)

        module.set_fused_native_gemm(True)
        assert module.fused_native_gemm
        fused_out = module(inputs)

        assert module.cached_weight_bytes == 0
        assert torch.allclose(ref_out, fused_out, atol=1e-5, rtol=1e-4)


class TestCUDAKernelStatus:
    """Verify presence and validity of vq_kernels_cuda.cu source."""

    def test_cuda_kernel_source_present_and_valid(self):
        status = cuda_kernel_source_status()
        assert status["exists"] is True
        assert isinstance(status["source_sha256"], str)
        assert len(status["source_sha256"]) == 64
        assert status["source_path"].endswith("vq_kernels_cuda.cu")
        assert isinstance(status["nvcc_available"], bool)
        assert isinstance(status["torch_cuda_available"], bool)
        assert isinstance(status["ready_for_cuda_build"], bool)

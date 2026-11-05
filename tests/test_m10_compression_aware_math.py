"""Milestone M10 Verification Test Suite:
Compression-Aware Mathematical Execution & Autograd Engine.

Verifies:
1. SmoothChannelConditioner:
   - fit(weight, activations) produces finite gamma and inv_gamma with gamma * inv_gamma == 1.0.
   - Exact algebraic invariance: (condition_input(X) @ smooth_weight(W).T) matches X @ W.T to < 1e-5.
   - Outlier suppression reduces quantization relative error on synthetic channel-outlier matrices.
   - Edge cases, parameter boundary checks (alpha in [0, 1], in_features > 0), and shape mismatches.
2. subspace_lut_rvq_gemm:
   - Exact numerical parity (max_abs_diff < 1e-4) between subspace_lut_rvq_gemm(X, encoded, bias)
     and dense dequantized F.linear(X, dequantize_weight(encoded), bias) across 1D/2D/3D batch inputs,
     various block_size (4, 8, 16, 24), bits_per_stage (2, 4, 8), stages (1, 2, 3), and non-divisible
     column dimensions requiring padding.
   - ResidualVQLinear.set_subspace_lut_mode(True) dispatches through subspace_lut_rvq_gemm
     and matches set_subspace_lut_mode(False) to < 1e-4.
3. dual_folded_leech_gemm:
   - Exact numerical parity (max_abs_diff < 1e-4) between dual_folded_leech_gemm(X, leech_encoded, bias)
     and F.linear(X, dequantize_leech_weight(leech_encoded), bias) across multiple reflections (0, 1, 2, 4, 8),
     stages (1, 2, 3), gain_bits (0, 1, 2), and 1D/2D/3D shapes.
4. _CompressedLinearAutograd & Multi-Layer Forward-Backward Autograd:
   - Analytical gradient matching (< 1e-4) for inputs.grad, row_scales.grad, and bias.grad vs dense F.linear.
   - Full 2-layer nn.Sequential(ResidualVQLinear, nn.SiLU(), ResidualVQLinear) forward and backward
     propagation for both adaptive_residual_vq and leech24_shell2_residual codecs.
   - Selective requires_grad propagation and compression_aware_linear_forward wrapper verification.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx_24x96.codec import (
    QuantizationConfig,
    QuantizedWeight,
    dequantize_weight,
    quantize_weight,
)
from ojasx_24x96.leech_codec import (
    LeechQuantizationConfig,
    LeechQuantizedWeight,
    dequantize_leech_weight,
    quantize_leech_weight,
)
from ojasx_24x96.math_engine import (
    SmoothChannelConditioner,
    SmoothChannelProfile,
    _CompressedLinearAutograd,
    compression_aware_linear_forward,
    dual_folded_leech_gemm,
    subspace_lut_rvq_gemm,
)
from ojasx_24x96.module import ResidualVQLinear


# ============================================================================
# 1. SmoothChannelConditioner Verification Tests
# ============================================================================


class TestSmoothChannelConditioner:
    """Rigorous tests for SmoothChannelConditioner."""

    def test_fit_produces_finite_gamma_and_inverse(self):
        torch.manual_seed(42)
        in_features = 64
        conditioner = SmoothChannelConditioner(in_features=in_features, alpha=0.5)

        w = torch.randn(32, in_features)
        x = torch.randn(16, in_features)

        profile = conditioner.fit(w, x)

        assert isinstance(profile, SmoothChannelProfile)
        assert profile.gamma.shape == (in_features,)
        assert profile.inv_gamma.shape == (in_features,)
        assert torch.isfinite(profile.gamma).all().item()
        assert torch.isfinite(profile.inv_gamma).all().item()
        assert (profile.gamma > 0.0).all().item()
        assert (profile.inv_gamma > 0.0).all().item()

        # Exact reciprocal relationship: gamma * inv_gamma == 1.0 within float precision
        product = profile.gamma * profile.inv_gamma
        max_reciprocal_err = (product - 1.0).abs().max().item()
        assert max_reciprocal_err < 1e-6, f"Reciprocal error too high: {max_reciprocal_err}"

        # Buffers on the module must match profile
        assert torch.allclose(conditioner.gamma, profile.gamma, atol=1e-7)
        assert torch.allclose(conditioner.inv_gamma, profile.inv_gamma, atol=1e-7)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_exact_algebraic_invariance(self, alpha: float):
        """Verify (condition_input(X) @ smooth_weight(W).T) matches X @ W.T to < 1e-5."""
        torch.manual_seed(100 + int(alpha * 100))
        in_features = 96
        conditioner = SmoothChannelConditioner(in_features=in_features, alpha=alpha)

        w = torch.randn(48, in_features)
        x_2d = torch.randn(20, in_features)
        conditioner.fit(w, x_2d)

        # 2D inputs
        w_smooth = conditioner.smooth_weight(w)
        x_smooth_2d = conditioner.condition_input(x_2d)

        y_true_2d = x_2d @ w.T
        y_smooth_2d = x_smooth_2d @ w_smooth.T

        diff_2d = (y_true_2d - y_smooth_2d).abs().max().item()
        assert diff_2d < 1e-5, f"Algebraic invariance failed for 2D with alpha={alpha}: diff={diff_2d}"

        # 3D inputs [Batch1, Batch2, in_features]
        x_3d = torch.randn(3, 4, in_features)
        x_smooth_3d = conditioner.condition_input(x_3d)
        y_true_3d = x_3d @ w.T
        y_smooth_3d = x_smooth_3d @ w_smooth.T
        diff_3d = (y_true_3d - y_smooth_3d).abs().max().item()
        assert diff_3d < 1e-5, f"Algebraic invariance failed for 3D with alpha={alpha}: diff={diff_3d}"

        # 1D inputs [in_features]
        x_1d = torch.randn(in_features)
        x_smooth_1d = conditioner.condition_input(x_1d)
        y_true_1d = x_1d @ w.T
        y_smooth_1d = x_smooth_1d @ w_smooth.T
        diff_1d = (y_true_1d - y_smooth_1d).abs().max().item()
        assert diff_1d < 1e-5, f"Algebraic invariance failed for 1D with alpha={alpha}: diff={diff_1d}"

    def test_outlier_suppression_reduces_quantization_relative_error(self):
        """Construct synthetic channel outliers and verify smoothing slashes relative error."""
        torch.manual_seed(2026)
        in_features = 64
        out_features = 32

        w = torch.randn(out_features, in_features)
        x = torch.randn(128, in_features)

        # Introduce severe channel outliers: large activation and small weight in specific channels
        outlier_channels_act = [5, 17, 33]
        for ch in outlier_channels_act:
            x[:, ch] *= 75.0
            w[:, ch] *= 0.01

        # Introduce severe weight outlier in another channel
        outlier_channels_w = [11, 42]
        for ch in outlier_channels_w:
            w[:, ch] *= 60.0
            x[:, ch] *= 0.02

        # 1. Quantize baseline without smoothing
        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc_base, _ = quantize_weight(w, cfg)
        assert enc_base is not None
        w_dec_base = dequantize_weight(enc_base)

        y_true = x @ w.T
        y_base = x @ w_dec_base.T
        rel_err_base = ((y_true - y_base).norm() / y_true.norm()).item()

        # 2. Condition with SmoothChannelConditioner
        conditioner = SmoothChannelConditioner(in_features=in_features, alpha=0.5)
        conditioner.fit(w, x)

        w_smooth = conditioner.smooth_weight(w)
        enc_smooth, _ = quantize_weight(w_smooth, cfg)
        assert enc_smooth is not None
        w_dec_smooth = dequantize_weight(enc_smooth)

        x_smooth = conditioner.condition_input(x)
        y_smooth = x_smooth @ w_dec_smooth.T
        rel_err_smooth = ((y_true - y_smooth).norm() / y_true.norm()).item()

        # Verify outlier suppression reduced relative error significantly (> 2x reduction)
        assert rel_err_smooth < rel_err_base, (
            f"Expected error reduction: base={rel_err_base:.4f}, smooth={rel_err_smooth:.4f}"
        )
        assert rel_err_base / rel_err_smooth > 2.0, (
            f"Expected > 2x improvement, got ratio={rel_err_base / rel_err_smooth:.2f}"
        )

    def test_weight_only_conditioning_without_activations(self):
        """Fit with activations=None operates in weight-only channel equilibration mode."""
        torch.manual_seed(99)
        in_features = 48
        conditioner = SmoothChannelConditioner(in_features=in_features, alpha=0.5)

        w = torch.randn(32, in_features)
        w[:, 7] *= 10.0  # single channel weight spike

        profile = conditioner.fit(w, activations=None)
        assert torch.isfinite(profile.gamma).all().item()
        assert torch.isfinite(profile.inv_gamma).all().item()
        # Outlier channel should have gamma < 1.0 to suppress weight spike
        assert profile.gamma[7] < 1.0

        # Invariance still holds exactly
        x = torch.randn(10, in_features)
        y_orig = x @ w.T
        y_smooth = conditioner.condition_input(x) @ conditioner.smooth_weight(w).T
        assert (y_orig - y_smooth).abs().max().item() < 1e-5

    def test_conditioner_input_validation(self):
        """Test boundary assertions and error raising."""
        with pytest.raises(ValueError, match="in_features must be positive"):
            SmoothChannelConditioner(in_features=0)
        with pytest.raises(ValueError, match="in_features must be positive"):
            SmoothChannelConditioner(in_features=-8)

        with pytest.raises(ValueError, match="alpha must lie in"):
            SmoothChannelConditioner(in_features=16, alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must lie in"):
            SmoothChannelConditioner(in_features=16, alpha=1.1)

        cond = SmoothChannelConditioner(in_features=16)
        with pytest.raises(ValueError, match="Expected weight with last dim 16"):
            cond.fit(torch.randn(8, 20))

        with pytest.raises(ValueError, match="Expected activations with last dim 16"):
            cond.fit(torch.randn(8, 16), activations=torch.randn(4, 20))

        with pytest.raises(ValueError, match="Expected weight with last dim 16"):
            cond.smooth_weight(torch.randn(8, 20))

        with pytest.raises(ValueError, match="Expected inputs with last dim 16"):
            cond.condition_input(torch.randn(4, 20))


# ============================================================================
# 2. subspace_lut_rvq_gemm Verification Tests
# ============================================================================


class TestSubspaceLutRvqGemm:
    """Rigorous tests for subspace_lut_rvq_gemm and ResidualVQLinear LUT dispatch."""

    @pytest.mark.parametrize(
        "block_size,bits_per_stage,stages,in_features,out_features",
        [
            (4, 2, 1, 16, 32),
            (8, 4, 2, 32, 64),
            (16, 4, 3, 48, 32),
            (24, 2, 2, 48, 24),
            (8, 8, 1, 32, 16),
            # Non-divisible column dimensions (tests padding logic)
            (4, 4, 2, 15, 20),   # 15 % 4 != 0
            (8, 4, 2, 35, 20),   # 35 % 8 != 0
            (16, 2, 2, 50, 16),  # 50 % 16 != 0
            (24, 4, 1, 55, 30),  # 55 % 24 != 0
        ],
    )
    def test_subspace_lut_rvq_gemm_parity_dense(
        self,
        block_size: int,
        bits_per_stage: int,
        stages: int,
        in_features: int,
        out_features: int,
    ):
        """Verify max_abs_diff < 1e-4 between subspace_lut_rvq_gemm and dense F.linear."""
        torch.manual_seed(hash((block_size, bits_per_stage, stages, in_features)) % 100000)
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        cfg = QuantizationConfig(
            block_size=block_size,
            bits_per_stage=bits_per_stage,
            max_stages=stages,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None
        w_dense = dequantize_weight(enc)

        # 1D Input: [in_features]
        x_1d = torch.randn(in_features)
        out_lut_1d = subspace_lut_rvq_gemm(x_1d, enc, bias=b)
        out_ref_1d = F.linear(x_1d, w_dense, b)
        assert out_lut_1d.shape == (out_features,)
        diff_1d = (out_lut_1d - out_ref_1d).abs().max().item()
        assert diff_1d < 1e-4, f"1D parity failed: diff={diff_1d:.2e}"

        # 2D Input: [B, in_features]
        x_2d = torch.randn(6, in_features)
        out_lut_2d = subspace_lut_rvq_gemm(x_2d, enc, bias=b)
        out_ref_2d = F.linear(x_2d, w_dense, b)
        assert out_lut_2d.shape == (6, out_features)
        diff_2d = (out_lut_2d - out_ref_2d).abs().max().item()
        assert diff_2d < 1e-4, f"2D parity failed: diff={diff_2d:.2e}"

        # 3D Input: [B1, B2, in_features]
        x_3d = torch.randn(2, 4, in_features)
        out_lut_3d = subspace_lut_rvq_gemm(x_3d, enc, bias=b)
        out_ref_3d = F.linear(x_3d, w_dense, b)
        assert out_lut_3d.shape == (2, 4, out_features)
        diff_3d = (out_lut_3d - out_ref_3d).abs().max().item()
        assert diff_3d < 1e-4, f"3D parity failed: diff={diff_3d:.2e}"

        # Bias=None test
        out_lut_nobias = subspace_lut_rvq_gemm(x_2d, enc, bias=None)
        out_ref_nobias = F.linear(x_2d, w_dense, None)
        diff_nobias = (out_lut_nobias - out_ref_nobias).abs().max().item()
        assert diff_nobias < 1e-4, f"No-bias parity failed: diff={diff_nobias:.2e}"

    @pytest.mark.parametrize("row_block_size", [1, 7, 16, 64, 256])
    def test_subspace_lut_row_chunking_invariance(self, row_block_size: int):
        """Verify row_block_size chunking does not alter numerical results."""
        torch.manual_seed(777)
        in_features, out_features = 48, 35
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None
        w_dense = dequantize_weight(enc)

        x = torch.randn(5, in_features)
        out_lut = subspace_lut_rvq_gemm(x, enc, bias=b, row_block_size=row_block_size)
        out_ref = F.linear(x, w_dense, b)
        diff = (out_lut - out_ref).abs().max().item()
        assert diff < 1e-4, f"Row block size {row_block_size} failed: diff={diff:.2e}"

    def test_residual_vq_linear_subspace_lut_mode_dispatch(self):
        """Verify ResidualVQLinear.set_subspace_lut_mode toggles dispatch and matches standard."""
        torch.manual_seed(123)
        in_features, out_features = 64, 32
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None

        layer = ResidualVQLinear.from_encoded(
            enc, bias=b, source_dtype=torch.float32, device=torch.device("cpu")
        )

        x = torch.randn(4, in_features)

        # Standard dequantization mode
        layer.set_subspace_lut_mode(False)
        out_standard = layer(x)

        # LUT mode
        layer.set_subspace_lut_mode(True)
        out_lut = layer(x)

        diff = (out_standard - out_lut).abs().max().item()
        assert diff < 1e-4, f"Mismatch between LUT mode and standard mode: diff={diff:.2e}"

    def test_subspace_lut_feature_dim_mismatch(self):
        """Verify dimension mismatch raises ValueError."""
        w = torch.randn(16, 32)
        cfg = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=1, require_compression=False, max_relative_error=None)
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None

        with pytest.raises(ValueError, match="Input feature mismatch"):
            subspace_lut_rvq_gemm(torch.randn(4, 24), enc)


# ============================================================================
# 3. dual_folded_leech_gemm Verification Tests
# ============================================================================


class TestDualFoldedLeechGemm:
    """Rigorous tests for dual_folded_leech_gemm."""

    @pytest.mark.parametrize("reflections", [0, 1, 2, 4, 8])
    @pytest.mark.parametrize("stages", [1, 2, 3])
    @pytest.mark.parametrize("gain_bits", [0, 1, 2])
    def test_dual_folded_leech_gemm_parity_dense(
        self,
        reflections: int,
        stages: int,
        gain_bits: int,
    ):
        """Verify max_abs_diff < 1e-4 between dual_folded_leech_gemm and dense F.linear."""
        torch.manual_seed(hash((reflections, stages, gain_bits)) % 100000)
        in_features = 48  # 2 * 24D
        out_features = 24

        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        cfg = LeechQuantizationConfig(
            max_stages=stages,
            gain_bits=gain_bits,
            rotation_reflections=reflections,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_leech_weight(w, cfg)
        assert enc is not None
        w_dense = dequantize_leech_weight(enc)

        # 1D Input: [in_features]
        x_1d = torch.randn(in_features)
        out_leech_1d = dual_folded_leech_gemm(x_1d, enc, bias=b)
        out_ref_1d = F.linear(x_1d, w_dense, b)
        assert out_leech_1d.shape == (out_features,)
        diff_1d = (out_leech_1d - out_ref_1d).abs().max().item()
        assert diff_1d < 1e-4, f"Leech 1D parity failed: diff={diff_1d:.2e}"

        # 2D Input: [B, in_features]
        x_2d = torch.randn(5, in_features)
        out_leech_2d = dual_folded_leech_gemm(x_2d, enc, bias=b)
        out_ref_2d = F.linear(x_2d, w_dense, b)
        assert out_leech_2d.shape == (5, out_features)
        diff_2d = (out_leech_2d - out_ref_2d).abs().max().item()
        assert diff_2d < 1e-4, f"Leech 2D parity failed: diff={diff_2d:.2e}"

        # 3D Input: [B1, B2, in_features]
        x_3d = torch.randn(2, 3, in_features)
        out_leech_3d = dual_folded_leech_gemm(x_3d, enc, bias=b)
        out_ref_3d = F.linear(x_3d, w_dense, b)
        assert out_leech_3d.shape == (2, 3, out_features)
        diff_3d = (out_leech_3d - out_ref_3d).abs().max().item()
        assert diff_3d < 1e-4, f"Leech 3D parity failed: diff={diff_3d:.2e}"

        # Bias=None
        out_leech_nobias = dual_folded_leech_gemm(x_2d, enc, bias=None)
        out_ref_nobias = F.linear(x_2d, w_dense, None)
        diff_nobias = (out_leech_nobias - out_ref_nobias).abs().max().item()
        assert diff_nobias < 1e-4, f"Leech no-bias parity failed: diff={diff_nobias:.2e}"

    @pytest.mark.parametrize("row_block_size", [1, 7, 13, 64, 128])
    def test_dual_folded_leech_row_chunking_invariance(self, row_block_size: int):
        """Verify row_block_size chunking does not alter numerical output for Leech GEMM."""
        torch.manual_seed(888)
        in_features, out_features = 48, 30
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)

        cfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_leech_weight(w, cfg)
        assert enc is not None
        w_dense = dequantize_leech_weight(enc)

        x = torch.randn(4, in_features)
        out_leech = dual_folded_leech_gemm(x, enc, bias=b, row_block_size=row_block_size)
        out_ref = F.linear(x, w_dense, b)
        diff = (out_leech - out_ref).abs().max().item()
        assert diff < 1e-4, f"Row block size {row_block_size} failed: diff={diff:.2e}"

    def test_dual_folded_leech_dim_mismatch(self):
        """Verify dimension mismatch raises ValueError."""
        w = torch.randn(24, 24)
        cfg = LeechQuantizationConfig(max_stages=1, gain_bits=0, require_compression=False, max_relative_error=None)
        enc, _ = quantize_leech_weight(w, cfg)
        assert enc is not None

        with pytest.raises(ValueError, match="Input feature mismatch"):
            dual_folded_leech_gemm(torch.randn(4, 30), enc)


# ============================================================================
# 4. _CompressedLinearAutograd & Multi-Layer Forward-Backward Tests
# ============================================================================


class TestCompressedLinearAutograd:
    """Rigorous tests for forward-backward autograd across compressed linear layers."""

    @pytest.mark.parametrize("use_subspace_lut", [False, True])
    def test_single_layer_analytical_gradients_rvq(self, use_subspace_lut: bool):
        """Verify inputs.grad, row_scales.grad, bias.grad match dense analytical gradients within < 1e-4."""
        torch.manual_seed(42)
        in_features, out_features = 32, 16
        batch_size = 4

        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)
        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None

        row_scales_init = enc.row_scales.clone()
        w_dense_init = dequantize_weight(enc)
        w_unit = w_dense_init / row_scales_init.unsqueeze(1)

        # Dense analytical reference
        x_ref = torch.randn(batch_size, in_features, requires_grad=True)
        s_ref = row_scales_init.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)

        w_dense = s_ref.unsqueeze(1) * w_unit
        y_ref = F.linear(x_ref, w_dense, b_ref)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)

        # Compressed Autograd execution
        x_test = x_ref.detach().clone().requires_grad_(True)
        s_test = row_scales_init.clone().requires_grad_(True)
        b_test = b.clone().requires_grad_(True)

        y_test = _CompressedLinearAutograd.apply(
            x_test,
            s_test,
            b_test,
            enc,
            8,
            use_subspace_lut,
        )
        y_test.backward(grad_out)

        # Assert forward match
        fwd_diff = (y_ref - y_test).abs().max().item()
        assert fwd_diff < 1e-4, f"Forward diff too high ({use_subspace_lut=}): {fwd_diff:.2e}"

        # Assert gradient match
        dx_diff = (x_ref.grad - x_test.grad).abs().max().item()
        ds_diff = (s_ref.grad - s_test.grad).abs().max().item()
        db_diff = (b_ref.grad - b_test.grad).abs().max().item()

        assert dx_diff < 1e-4, f"dx_diff too high ({use_subspace_lut=}): {dx_diff:.2e}"
        assert ds_diff < 1e-4, f"ds_diff too high ({use_subspace_lut=}): {ds_diff:.2e}"
        assert db_diff < 1e-4, f"db_diff too high ({use_subspace_lut=}): {db_diff:.2e}"

    def test_single_layer_analytical_gradients_leech(self):
        """Verify inputs.grad, row_scales.grad, bias.grad match dense gradients for Leech codec."""
        torch.manual_seed(99)
        in_features, out_features = 48, 24
        batch_size = 4

        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)
        cfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=0,
            rotation_reflections=2,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_leech_weight(w, cfg)
        assert enc is not None

        row_scales_init = enc.row_scales.clone()
        w_dense_init = dequantize_leech_weight(enc)
        w_unit = w_dense_init / row_scales_init.unsqueeze(1)

        # Dense reference
        x_ref = torch.randn(batch_size, in_features, requires_grad=True)
        s_ref = row_scales_init.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)

        w_dense = s_ref.unsqueeze(1) * w_unit
        y_ref = F.linear(x_ref, w_dense, b_ref)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)

        # Leech autograd
        x_test = x_ref.detach().clone().requires_grad_(True)
        s_test = row_scales_init.clone().requires_grad_(True)
        b_test = b.clone().requires_grad_(True)

        y_test = _CompressedLinearAutograd.apply(
            x_test,
            s_test,
            b_test,
            enc,
            12,
            False,
        )
        y_test.backward(grad_out)

        # Assert forward match
        fwd_diff = (y_ref - y_test).abs().max().item()
        assert fwd_diff < 1e-4, f"Leech forward diff too high: {fwd_diff:.2e}"

        # Assert gradient match
        dx_diff = (x_ref.grad - x_test.grad).abs().max().item()
        ds_diff = (s_ref.grad - s_test.grad).abs().max().item()
        db_diff = (b_ref.grad - b_test.grad).abs().max().item()

        assert dx_diff < 1e-4, f"Leech dx_diff too high: {dx_diff:.2e}"
        assert ds_diff < 1e-4, f"Leech ds_diff too high: {ds_diff:.2e}"
        assert db_diff < 1e-4, f"Leech db_diff too high: {db_diff:.2e}"

    def test_selective_requires_grad_and_bias_none(self):
        """Verify autograd correctly handles selective subsets of requires_grad and bias=None."""
        torch.manual_seed(55)
        w = torch.randn(16, 24)
        cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=1,
            require_compression=False,
            max_relative_error=None,
            min_weight_elements=1,
        )
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None

        # 1. Only inputs requires_grad, bias is None
        x = torch.randn(3, 24, requires_grad=True)
        s = enc.row_scales.detach().clone()
        y = _CompressedLinearAutograd.apply(x, s, None, enc, 8, False)
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all().item()
        assert s.grad is None

        # 2. Only row_scales requires_grad
        x = torch.randn(3, 24, requires_grad=False)
        s = enc.row_scales.detach().clone().requires_grad_(True)
        b = torch.randn(16, requires_grad=False)
        y = _CompressedLinearAutograd.apply(x, s, b, enc, 8, False)
        y.sum().backward()
        assert x.grad is None
        assert s.grad is not None and torch.isfinite(s.grad).all().item()
        assert b.grad is None

        # 3. Only bias requires_grad
        s = enc.row_scales.detach().clone()
        b = torch.randn(16, requires_grad=True)
        y = _CompressedLinearAutograd.apply(x, s, b, enc, 8, False)
        y.sum().backward()
        assert x.grad is None
        assert s.grad is None
        assert b.grad is not None and torch.isfinite(b.grad).all().item()

    @pytest.mark.parametrize("use_subspace_lut", [False, True])
    def test_multi_layer_sequential_autograd_rvq(self, use_subspace_lut: bool):
        """Verify full forward-backward through 2-layer nn.Sequential(ResidualVQLinear, SiLU, ResidualVQLinear) for RVQ."""
        torch.manual_seed(101)
        d_in, d_hidden, d_out = 32, 48, 16
        batch_size = 4

        w1 = torch.randn(d_hidden, d_in)
        b1 = torch.randn(d_hidden)
        cfg1 = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, require_compression=False, max_relative_error=None, min_weight_elements=1)
        enc1, _ = quantize_weight(w1, cfg1)
        assert enc1 is not None

        w2 = torch.randn(d_out, d_hidden)
        b2 = torch.randn(d_out)
        cfg2 = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, require_compression=False, max_relative_error=None, min_weight_elements=1)
        enc2, _ = quantize_weight(w2, cfg2)
        assert enc2 is not None

        # ResidualVQLinear layers
        l1 = ResidualVQLinear.from_encoded(enc1, bias=b1, source_dtype=torch.float32, device=torch.device("cpu"))
        l2 = ResidualVQLinear.from_encoded(enc2, bias=b2, source_dtype=torch.float32, device=torch.device("cpu"))
        l1.row_scales.requires_grad_(True)
        l1.bias.requires_grad_(True)
        l2.row_scales.requires_grad_(True)
        l2.bias.requires_grad_(True)

        if use_subspace_lut:
            l1.set_subspace_lut_mode(True)
            l2.set_subspace_lut_mode(True)

        net_rvq = nn.Sequential(l1, nn.SiLU(), l2)

        # Dense reference network
        s1_ref = enc1.row_scales.clone().requires_grad_(True)
        b1_ref = b1.clone().requires_grad_(True)
        w1_unit = dequantize_weight(enc1) / enc1.row_scales.unsqueeze(1)

        s2_ref = enc2.row_scales.clone().requires_grad_(True)
        b2_ref = b2.clone().requires_grad_(True)
        w2_unit = dequantize_weight(enc2) / enc2.row_scales.unsqueeze(1)

        x_ref = torch.randn(batch_size, d_in, requires_grad=True)
        x_rvq = x_ref.detach().clone().requires_grad_(True)

        # Forward reference
        h1_ref = F.linear(x_ref, s1_ref.unsqueeze(1) * w1_unit, b1_ref)
        a1_ref = F.silu(h1_ref)
        y_ref = F.linear(a1_ref, s2_ref.unsqueeze(1) * w2_unit, b2_ref)

        # Forward RVQ network
        y_rvq = net_rvq(x_rvq)

        fwd_diff = (y_ref - y_rvq).abs().max().item()
        assert fwd_diff < 1e-4, f"Multi-layer RVQ forward diff too high: {fwd_diff:.2e}"

        # Scalar loss backward
        loss_ref = y_ref.sum()
        loss_ref.backward()

        loss_rvq = y_rvq.sum()
        loss_rvq.backward()

        # All gradients must exist and be finite
        assert x_rvq.grad is not None and torch.isfinite(x_rvq.grad).all().item()
        assert l1.row_scales.grad is not None and torch.isfinite(l1.row_scales.grad).all().item()
        assert l1.bias.grad is not None and torch.isfinite(l1.bias.grad).all().item()
        assert l2.row_scales.grad is not None and torch.isfinite(l2.row_scales.grad).all().item()
        assert l2.bias.grad is not None and torch.isfinite(l2.bias.grad).all().item()

        # Gradient parity against dense reference
        assert (x_ref.grad - x_rvq.grad).abs().max().item() < 1e-4
        assert (s1_ref.grad - l1.row_scales.grad).abs().max().item() < 1e-4
        assert (b1_ref.grad - l1.bias.grad).abs().max().item() < 1e-4
        assert (s2_ref.grad - l2.row_scales.grad).abs().max().item() < 1e-4
        assert (b2_ref.grad - l2.bias.grad).abs().max().item() < 1e-4

    def test_multi_layer_sequential_autograd_leech(self):
        """Verify full forward-backward through 2-layer nn.Sequential(ResidualVQLinear, SiLU, ResidualVQLinear) for Leech."""
        torch.manual_seed(202)
        d_in, d_hidden, d_out = 48, 72, 24
        batch_size = 4

        w1 = torch.randn(d_hidden, d_in)
        b1 = torch.randn(d_hidden)
        cfg1 = LeechQuantizationConfig(max_stages=2, gain_bits=0, rotation_reflections=2, require_compression=False, max_relative_error=None, min_weight_elements=1)
        enc1, _ = quantize_leech_weight(w1, cfg1)
        assert enc1 is not None

        w2 = torch.randn(d_out, d_hidden)
        b2 = torch.randn(d_out)
        cfg2 = LeechQuantizationConfig(max_stages=2, gain_bits=0, rotation_reflections=2, require_compression=False, max_relative_error=None, min_weight_elements=1)
        enc2, _ = quantize_leech_weight(w2, cfg2)
        assert enc2 is not None

        # ResidualVQLinear layers
        l1 = ResidualVQLinear.from_encoded(enc1, bias=b1, source_dtype=torch.float32, device=torch.device("cpu"))
        l2 = ResidualVQLinear.from_encoded(enc2, bias=b2, source_dtype=torch.float32, device=torch.device("cpu"))
        l1.row_scales.requires_grad_(True)
        l1.bias.requires_grad_(True)
        l2.row_scales.requires_grad_(True)
        l2.bias.requires_grad_(True)

        net_leech = nn.Sequential(l1, nn.SiLU(), l2)

        # Dense reference
        s1_ref = enc1.row_scales.clone().requires_grad_(True)
        b1_ref = b1.clone().requires_grad_(True)
        w1_unit = dequantize_leech_weight(enc1) / enc1.row_scales.unsqueeze(1)

        s2_ref = enc2.row_scales.clone().requires_grad_(True)
        b2_ref = b2.clone().requires_grad_(True)
        w2_unit = dequantize_leech_weight(enc2) / enc2.row_scales.unsqueeze(1)

        x_ref = torch.randn(batch_size, d_in, requires_grad=True)
        x_leech = x_ref.detach().clone().requires_grad_(True)

        # Forward reference
        h1_ref = F.linear(x_ref, s1_ref.unsqueeze(1) * w1_unit, b1_ref)
        a1_ref = F.silu(h1_ref)
        y_ref = F.linear(a1_ref, s2_ref.unsqueeze(1) * w2_unit, b2_ref)

        # Forward Leech network
        y_leech = net_leech(x_leech)

        fwd_diff = (y_ref - y_leech).abs().max().item()
        assert fwd_diff < 1e-4, f"Multi-layer Leech forward diff too high: {fwd_diff:.2e}"

        # Scalar loss backward
        loss_ref = y_ref.sum()
        loss_ref.backward()

        loss_leech = y_leech.sum()
        loss_leech.backward()

        # All gradients must exist and be finite
        assert x_leech.grad is not None and torch.isfinite(x_leech.grad).all().item()
        assert l1.row_scales.grad is not None and torch.isfinite(l1.row_scales.grad).all().item()
        assert l1.bias.grad is not None and torch.isfinite(l1.bias.grad).all().item()
        assert l2.row_scales.grad is not None and torch.isfinite(l2.row_scales.grad).all().item()
        assert l2.bias.grad is not None and torch.isfinite(l2.bias.grad).all().item()

        # Gradient parity against dense reference
        assert (x_ref.grad - x_leech.grad).abs().max().item() < 1e-4
        assert (s1_ref.grad - l1.row_scales.grad).abs().max().item() < 1e-4
        assert (b1_ref.grad - l1.bias.grad).abs().max().item() < 1e-4
        assert (s2_ref.grad - l2.row_scales.grad).abs().max().item() < 1e-4
        assert (b2_ref.grad - l2.bias.grad).abs().max().item() < 1e-4

    def test_compression_aware_linear_forward_wrapper(self):
        """Verify compression_aware_linear_forward wrapper matches module forward in eval and autograd modes."""
        torch.manual_seed(303)
        in_features, out_features = 32, 16
        w = torch.randn(out_features, in_features)
        b = torch.randn(out_features)
        cfg = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, require_compression=False, max_relative_error=None, min_weight_elements=1)
        enc, _ = quantize_weight(w, cfg)
        assert enc is not None

        layer = ResidualVQLinear.from_encoded(enc, bias=b, source_dtype=torch.float32, device=torch.device("cpu"))

        # Eval mode (no grad)
        with torch.no_grad():
            x_eval = torch.randn(3, in_features)
            y1 = compression_aware_linear_forward(x_eval, layer, use_subspace_lut=True)
            y2 = layer(x_eval)
            assert (y1 - y2).abs().max().item() < 1e-4

        # Autograd mode
        x_train = torch.randn(3, in_features, requires_grad=True)
        y_train = compression_aware_linear_forward(x_train, layer, use_subspace_lut=True)
        loss = y_train.sum()
        loss.backward()
        assert x_train.grad is not None and torch.isfinite(x_train.grad).all().item()

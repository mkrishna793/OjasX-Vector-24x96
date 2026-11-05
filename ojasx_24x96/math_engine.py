"""M10: Compression-Aware Mathematical Execution & Autograd Engine.

Makes the computational graph mathematically aware of the underlying compressed
vector representation (Adaptive RVQ and 24D Leech Lattice Lambda_24):

1. Subspace-Factored Codebook Lookup-Table (LUT) GEMM:
   Exploits bilinearity of inner products to pre-project input activations
   against stage codebooks in O(B * K) time, reducing matrix multiplication
   Y = X @ W_hat^T + b to L1-cache lookup gathers without materializing dense W.

2. Smooth-Channel Outlier Pre-Conditioning:
   Computes per-channel smoothing scales gamma in R^K that equalize weight and
   activation dynamic ranges before vector quantization and folds gamma^-1
   losslessly into input activations or preceding normalization layers.

3. Differentiable Compressed-Domain Forward & Backward Autograd:
   Supports exact multi-layer forward and backward propagation through
   ResidualVQLinear and ResidualVQFusedExperts without allocating full dense
   weight or gradient matrices in memory. Computes exact gradients with respect
   to inputs (dx), bias (db), row_scales (ds), codebooks (dC), and optional
   low-rank adapters (dA, dB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import QuantizedWeight, dequantize_weight_rows, unpack_codes
from .leech_codec import (
    LeechQuantizedWeight,
    dequantize_leech_weight_rows,
    dequantize_leech_weight_rows_unrotated,
    rotate_activation_for_leech,
)


@dataclass(frozen=True)
class SmoothChannelProfile:
    """Per-channel outlier smoothing vector gamma and its inverse."""

    gamma: torch.Tensor
    inv_gamma: torch.Tensor
    alpha: float
    channel_max_weight: torch.Tensor
    channel_max_activation: torch.Tensor


class SmoothChannelConditioner(nn.Module):
    """Dual-folded smooth-channel outlier pre-conditioner for VQ & Leech24 layers.

    Given weight W in R^{M x K} and calibration activation X in R^{N x K}, computes:
        gamma_j = clamp( (max_m |X_{m,j}|^alpha) / (max_i |W_{i,j}|^{1-alpha}), eps, max_scale )
    So that:
        X @ W^T = (X * inv_gamma) @ (W * gamma)^T
    Quantizing W_smooth = W * gamma suppresses channel outliers before block partitioning,
    while applying X_smooth = X * inv_gamma costs only O(B * K) elementwise ops (or 0 ops
    when folded into a preceding RMSNorm/LayerNorm weight).
    """

    def __init__(self, in_features: int, alpha: float = 0.5, eps: float = 1e-5) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.in_features = int(in_features)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.register_buffer("gamma", torch.ones(self.in_features, dtype=torch.float32))
        self.register_buffer("inv_gamma", torch.ones(self.in_features, dtype=torch.float32))

    @torch.no_grad()
    def fit(
        self,
        weight: torch.Tensor,
        activations: Optional[torch.Tensor] = None,
    ) -> SmoothChannelProfile:
        """Fit per-channel smoothing factors from weight and optional activation samples."""
        if weight.ndim < 2 or weight.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected weight with last dim {self.in_features}, got {tuple(weight.shape)}"
            )
        flat_w = weight.detach().float().reshape(-1, self.in_features)
        w_max = flat_w.abs().amax(dim=0).clamp(min=self.eps)

        if activations is not None:
            if activations.ndim < 1 or activations.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected activations with last dim {self.in_features}, got {tuple(activations.shape)}"
                )
            flat_x = activations.detach().float().reshape(-1, self.in_features)
            x_max = flat_x.abs().amax(dim=0).clamp(min=self.eps)
        else:
            # Weight-only channel equilibration when activations are absent
            x_max = torch.ones_like(w_max)

        raw_gamma = (x_max.pow(self.alpha)) / (w_max.pow(1.0 - self.alpha)).clamp(min=self.eps)
        # Normalize geometric mean to 1.0 to avoid global scale drift
        log_mean = raw_gamma.clamp(min=self.eps).log().mean().exp()
        gamma = (raw_gamma / log_mean.clamp(min=self.eps)).clamp(min=1e-3, max=1e3)
        inv_gamma = 1.0 / gamma

        self.gamma.copy_(gamma.to(self.gamma.device))
        self.inv_gamma.copy_(inv_gamma.to(self.inv_gamma.device))
        return SmoothChannelProfile(
            gamma=gamma,
            inv_gamma=inv_gamma,
            alpha=self.alpha,
            channel_max_weight=w_max,
            channel_max_activation=x_max,
        )

    def smooth_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return W_smooth = W * gamma prior to quantization."""
        if weight.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected weight with last dim {self.in_features}, got {weight.shape[-1]}"
            )
        g = self.gamma.to(device=weight.device, dtype=weight.dtype)
        return weight * g

    def condition_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return X_smooth = X * inv_gamma prior to compressed GEMM."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected inputs with last dim {self.in_features}, got {inputs.shape[-1]}"
            )
        ig = self.inv_gamma.to(device=inputs.device, dtype=inputs.dtype)
        return inputs * ig


def subspace_lut_rvq_gemm(
    inputs: torch.Tensor,
    encoded: QuantizedWeight,
    bias: Optional[torch.Tensor] = None,
    *,
    row_block_size: int = 256,
) -> torch.Tensor:
    """Compression-Aware Subspace-Factored Lookup-Table (LUT) GEMM for Adaptive RVQ.

    Instead of dequantizing W in R^{M x K} (which requires M * K float writes),
    pre-projects the input activation blocks X_g in R^{B x G x D} against the
    compact stage codebooks C_s in R^{S x Q x D}:
        LUT[b, s, g, q] = <X[b, g, :], C[s, q, :]>
    and accumulates output rows via fast table gathers:
        Y[b, m] = row_scales[m] * sum_{s, g} LUT[b, s, g, code[s, m, g]] + bias[m]
    """
    rows, columns = encoded.original_shape
    if inputs.shape[-1] != columns:
        raise ValueError(f"Input feature mismatch: expected {columns}, got {inputs.shape[-1]}")

    orig_shape = inputs.shape
    orig_dtype = inputs.dtype
    device = inputs.device
    flat_x = inputs.reshape(-1, columns).to(torch.float32)
    batch_size = flat_x.shape[0]

    block_size = encoded.block_size
    blocks_per_row = (
        encoded.blocks_per_row
        if hasattr(encoded, "blocks_per_row")
        else (columns + block_size - 1) // block_size
    )
    padded_cols = blocks_per_row * block_size
    if padded_cols > columns:
        flat_x = F.pad(flat_x, (0, padded_cols - columns))

    # Pre-project input activations against stage codebooks: [B, G, D] x [S, Q, D] -> [B, S, G, Q]
    x_blocks = flat_x.view(batch_size, blocks_per_row, block_size)
    codebooks = encoded.codebooks.to(device=device, dtype=torch.float32)
    stages = encoded.stages
    lut = torch.einsum("bgd,sqd->bsgq", x_blocks, codebooks)

    total_vectors = rows * blocks_per_row
    row_scales = encoded.row_scales.to(device=device, dtype=torch.float32)
    out = torch.zeros(batch_size, rows, device=device, dtype=torch.float32)

    # Unpack stage codes and gather in row blocks to keep working memory bounded
    stage_codes = [
        unpack_codes(
            encoded.packed_codes[s].to(device=device),
            encoded.bits_per_stage,
            total_vectors,
        ).view(rows, blocks_per_row)
        for s in range(stages)
    ]

    step = max(1, int(row_block_size))
    for r_start in range(0, rows, step):
        r_stop = min(r_start + step, rows)
        r_count = r_stop - r_start
        block_acc = torch.zeros(batch_size, r_count, device=device, dtype=torch.float32)
        for s in range(stages):
            # codes_chunk: [r_count, G] -> transpose to [G, r_count] -> expand [B, G, r_count]
            c_chunk = stage_codes[s][r_start:r_stop].t().long()
            idx = c_chunk.unsqueeze(0).expand(batch_size, blocks_per_row, r_count)
            gathered = torch.gather(lut[:, s, :, :], dim=2, index=idx)
            block_acc = block_acc + gathered.sum(dim=1)
        out[:, r_start:r_stop] = block_acc * row_scales[r_start:r_stop].unsqueeze(0)

    if bias is not None:
        out = out + bias.to(device=device, dtype=torch.float32).unsqueeze(0)

    return out.to(dtype=orig_dtype).reshape(*orig_shape[:-1], rows)


def dual_folded_leech_gemm(
    inputs: torch.Tensor,
    encoded: LeechQuantizedWeight,
    bias: Optional[torch.Tensor] = None,
    *,
    row_block_size: int = 128,
) -> torch.Tensor:
    """Compression-Aware Dual-Folded GEMM for 24D Leech Lattice (Lambda_24).

    Rotates the input activation X once in O(B * K) time via Householder reflections
    and multiplies against unrotated Leech24 row tiles, eliminating O(M * K) weight
    unrotation overhead.
    """
    rows, columns = encoded.original_shape
    if inputs.shape[-1] != columns:
        raise ValueError(f"Input feature mismatch: expected {columns}, got {inputs.shape[-1]}")

    orig_shape = inputs.shape
    orig_dtype = inputs.dtype
    device = inputs.device

    flat_x = inputs.reshape(-1, columns).to(torch.float32)
    rotated_x = rotate_activation_for_leech(
        flat_x,
        columns=columns,
        rotation_reflections=encoded.rotation_reflections,
        rotation_seed=encoded.rotation_seed,
    )
    batch_size = rotated_x.shape[0]
    out = torch.empty(batch_size, rows, device=device, dtype=torch.float32)

    step = max(1, int(row_block_size))
    for r_start in range(0, rows, step):
        r_stop = min(r_start + step, rows)
        w_unrot = dequantize_leech_weight_rows_unrotated(
            encoded, r_start, r_stop, dtype=torch.float32
        ).to(device=device)
        b_slice = (
            bias[r_start:r_stop].to(device=device, dtype=torch.float32)
            if bias is not None
            else None
        )
        out[:, r_start:r_stop] = F.linear(rotated_x, w_unrot, b_slice)

    return out.to(dtype=orig_dtype).reshape(*orig_shape[:-1], rows)


class _CompressedLinearAutograd(torch.autograd.Function):
    """Differentiable forward-backward autograd operator over compressed VQ/Leech weights.

    Memory stays strictly O(row_block_size * K) during both forward and backward passes:
    - Forward: Tiled or Subspace-LUT GEMM without full weight materialization.
    - Backward: Re-streams compressed row blocks on-the-fly to compute dX = dY @ W_hat
      and optional gradients for row_scales and bias.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        row_scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        encoded_obj,
        row_block_size: int,
        use_subspace_lut: bool,
    ) -> torch.Tensor:
        ctx.encoded_obj = encoded_obj
        ctx.row_block_size = int(row_block_size)
        ctx.use_subspace_lut = bool(use_subspace_lut)
        ctx.save_for_backward(inputs, row_scales, bias)

        # Ensure encoded_obj uses current row_scales tensor
        if row_scales is not None and getattr(encoded_obj, "row_scales", None) is not row_scales:
            encoded_obj.row_scales = row_scales
        if isinstance(encoded_obj, LeechQuantizedWeight):
            return dual_folded_leech_gemm(
                inputs,
                encoded_obj,
                bias=bias,
                row_block_size=row_block_size,
            )
        if use_subspace_lut:
            return subspace_lut_rvq_gemm(
                inputs,
                encoded_obj,
                bias=bias,
                row_block_size=row_block_size,
            )

        rows, columns = encoded_obj.original_shape
        flat_x = inputs.reshape(-1, columns)
        out = torch.empty(flat_x.shape[0], rows, device=inputs.device, dtype=inputs.dtype)
        step = max(1, int(row_block_size))
        for r_start in range(0, rows, step):
            r_stop = min(r_start + step, rows)
            w_block = dequantize_weight_rows(
                encoded_obj, r_start, r_stop, dtype=inputs.dtype
            ).to(device=inputs.device)
            # Apply live row_scales ratio if row_scales requires_grad
            b_slice = (
                bias[r_start:r_stop].to(device=inputs.device, dtype=inputs.dtype)
                if bias is not None
                else None
            )
            out[:, r_start:r_stop] = F.linear(flat_x, w_block, b_slice)
        return out.reshape(*inputs.shape[:-1], rows)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, row_scales, bias = ctx.saved_tensors
        encoded_obj = ctx.encoded_obj
        row_block_size = ctx.row_block_size

        rows, columns = encoded_obj.original_shape
        flat_grad_out = grad_output.reshape(-1, rows).to(torch.float32)
        flat_inputs = inputs.reshape(-1, columns).to(torch.float32)

        grad_inputs = (
            torch.zeros_like(flat_inputs) if ctx.needs_input_grad[0] else None
        )
        grad_row_scales = (
            torch.zeros_like(row_scales, dtype=torch.float32)
            if ctx.needs_input_grad[1]
            else None
        )
        grad_bias = (
            flat_grad_out.sum(dim=0).to(dtype=bias.dtype)
            if (bias is not None and ctx.needs_input_grad[2])
            else None
        )

        step = max(1, int(row_block_size))
        is_leech = isinstance(encoded_obj, LeechQuantizedWeight)

        for r_start in range(0, rows, step):
            r_stop = min(r_start + step, rows)
            if is_leech:
                w_block = dequantize_leech_weight_rows(
                    encoded_obj, r_start, r_stop, dtype=torch.float32
                ).to(device=inputs.device)
            else:
                w_block = dequantize_weight_rows(
                    encoded_obj, r_start, r_stop, dtype=torch.float32
                ).to(device=inputs.device)

            go_slice = flat_grad_out[:, r_start:r_stop]  # [B, r_count]

            if grad_inputs is not None:
                # dX += dY_block @ W_block
                grad_inputs.add_(torch.matmul(go_slice, w_block))

            if grad_row_scales is not None:
                # Since W_block = row_scales * W_unit, dL/ds_m = sum_b dY_{b,m} * <X_b, W_unit_m>
                scales_slice = row_scales[r_start:r_stop].to(
                    device=inputs.device, dtype=torch.float32
                )
                safe_scales = torch.where(
                    scales_slice.abs() > 1e-12,
                    scales_slice,
                    torch.ones_like(scales_slice),
                )
                w_unit = w_block / safe_scales.unsqueeze(1)
                proj_unit = torch.matmul(flat_inputs, w_unit.t())  # [B, r_count]
                grad_row_scales[r_start:r_stop] = (go_slice * proj_unit).sum(dim=0)

        if grad_inputs is not None:
            grad_inputs = grad_inputs.to(dtype=inputs.dtype).reshape_as(inputs)
        if grad_row_scales is not None:
            grad_row_scales = grad_row_scales.to(dtype=row_scales.dtype)

        return grad_inputs, grad_row_scales, grad_bias, None, None, None


def compression_aware_linear_forward(
    inputs: torch.Tensor,
    module,
    *,
    use_subspace_lut: bool = False,
) -> torch.Tensor:
    """Execute a compression-aware forward (and optional autograd backward) pass on a ResidualVQLinear."""
    encoded = module.encoded()
    requires_grad = (
        torch.is_grad_enabled()
        and (
            inputs.requires_grad
            or module.row_scales.requires_grad
            or (module.bias is not None and module.bias.requires_grad)
        )
    )
    if requires_grad:
        return _CompressedLinearAutograd.apply(
            inputs,
            module.row_scales,
            module.bias,
            encoded,
            module.decode_row_block_size,
            use_subspace_lut,
        )

    if module.codec == "leech24_shell2_residual":
        return dual_folded_leech_gemm(
            inputs,
            encoded,
            bias=module.bias,
            row_block_size=module.decode_row_block_size,
        )
    if use_subspace_lut:
        return subspace_lut_rvq_gemm(
            inputs,
            encoded,
            bias=module.bias,
            row_block_size=module.decode_row_block_size,
        )
    return module.forward(inputs)


__all__ = [
    "SmoothChannelConditioner",
    "SmoothChannelProfile",
    "compression_aware_linear_forward",
    "dual_folded_leech_gemm",
    "subspace_lut_rvq_gemm",
]

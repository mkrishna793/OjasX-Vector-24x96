"""
split_manifold_engine.py

Split-Manifold Look-Up Table (LUT) Execution Engine.
Realizes zero-unpack compressed-domain linear algebra for LLM inference.

Mathematical Guarantee:
  W = W_manifold + W_outlier
  LUT[g, k] = <x_g, C[k]>  (computed once per layer into SRAM/L2 cache)
  y_manifold[m] = sum_g LUT[g, codes[m, g]]  (zero floating-point weight fetches)
  y[m] = y_manifold[m] + sparse_gemv(W_outlier, x)

Author: OjasX Research Team & Antigravity
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class SplitManifoldLUTLinear(nn.Module):
    """
    Linear layer executing directly in the compressed domain without unpacking weights.
    Decomposes weights into low-variance manifold (quantized via vector LUT) and
    high-magnitude outliers (exact floating point sparse matrix).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_dim: int = 16,
        codebook_size: int = 256,
        outlier_tau: float = 3.5,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert in_features % block_dim == 0, f"in_features ({in_features}) must be divisible by block_dim ({block_dim})"
        assert codebook_size <= 256, "codebook_size must fit in uint8 (<= 256)"

        self.in_features = in_features
        self.out_features = out_features
        self.block_dim = block_dim
        self.num_groups = in_features // block_dim
        self.codebook_size = codebook_size
        self.outlier_tau = outlier_tau
        self.dtype = dtype

        # Codebook: [codebook_size, block_dim]
        self.register_buffer("codebook", torch.zeros((codebook_size, block_dim), dtype=dtype))

        # Compressed manifold codes: [out_features, num_groups] uint8 (1 byte per block)
        self.register_buffer("codes", torch.zeros((out_features, self.num_groups), dtype=torch.uint8))

        # Per-channel scale: [out_features]
        self.register_buffer("scales", torch.ones(out_features, dtype=dtype))

        # Sparse outlier matrix (COO indices and values)
        self.register_buffer("outlier_indices", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("outlier_values", torch.empty((0,), dtype=dtype))
        self.num_outliers = 0

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def lut_sram_bytes(self) -> int:
        """Returns the exact byte size of the pre-projected LUT table in SRAM."""
        # num_groups * codebook_size * sizeof(dtype)
        elem_size = 4 if self.dtype in (torch.float32, torch.int32) else 2
        return self.num_groups * self.codebook_size * elem_size

    @property
    def compression_ratio(self) -> float:
        """Computes true compression ratio compared to dense FP16/BF16 weights."""
        dense_bytes = self.out_features * self.in_features * 2  # BF16/FP16 baseline
        # Compressed: uint8 codes + codebook + scales + sparse outliers
        code_bytes = self.out_features * self.num_groups * 1
        codebook_bytes = self.codebook_size * self.block_dim * 2
        scale_bytes = self.out_features * 2
        outlier_bytes = self.num_outliers * (8 + 2)  # int64 index + fp16 value
        compressed_bytes = code_bytes + codebook_bytes + scale_bytes + outlier_bytes
        return dense_bytes / max(1, compressed_bytes)

    @classmethod
    def from_dense(
        cls,
        dense_weight: torch.Tensor,
        dense_bias: Optional[torch.Tensor] = None,
        block_dim: int = 16,
        codebook_size: int = 256,
        outlier_tau: float = 3.5,
    ) -> "SplitManifoldLUTLinear":
        """
        Calibrates and fits the Split-Manifold LUT Linear layer from a dense weight tensor.
        """
        out_features, in_features = dense_weight.shape
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            block_dim=block_dim,
            codebook_size=codebook_size,
            outlier_tau=outlier_tau,
            bias=dense_bias is not None,
            dtype=dense_weight.dtype,
        )

        with torch.no_grad():
            W = dense_weight.clone().float()

            # 1. Outlier Separation: |W| >= tau * std(W)
            std = W.std().item()
            mean = W.mean().item()
            threshold = outlier_tau * std
            outlier_mask = (W - mean).abs() >= threshold

            outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).t()  # [2, num_outliers]

            # 2. Extract Residual Manifold
            W_manifold = W.clone()
            W_manifold[outlier_mask] = 0.0

            # 3. Vector Quantization of Manifold Subspaces
            # Partition into [out_features, num_groups, block_dim]
            num_groups = in_features // block_dim
            W_blocks = W_manifold.view(out_features, num_groups, block_dim)

            # Per-row scale normalization
            row_scales = W_blocks.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            W_blocks_norm = W_blocks / row_scales
            layer.scales = row_scales.view(out_features).to(dense_weight.dtype)

            # Flatten all normalized blocks: [out_features * num_groups, block_dim]
            all_vectors = W_blocks_norm.reshape(-1, block_dim)

            # Initialize codebook via uniform quantiles or k-means
            # Fast deterministic codebook initialization
            indices_sample = torch.linspace(0, all_vectors.shape[0] - 1, codebook_size, dtype=torch.long)
            centroids = all_vectors[indices_sample].clone()

            # 5 iterations of fast vector k-means
            for _ in range(5):
                # Distances: ||v - c||^2 = ||v||^2 + ||c||^2 - 2 v c^T
                dists = torch.cdist(all_vectors, centroids)
                cluster_ids = torch.argmin(dists, dim=1)

                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(codebook_size, device=centroids.device)
                new_centroids.index_add_(0, cluster_ids, all_vectors)
                counts.index_add_(0, cluster_ids, torch.ones_like(cluster_ids, dtype=torch.float32))

                mask_valid = counts > 0
                centroids[mask_valid] = new_centroids[mask_valid] / counts[mask_valid].unsqueeze(1)

            layer.codebook = centroids.to(dense_weight.dtype)

            # Assign best centroid to every block
            final_dists = torch.cdist(all_vectors, centroids)
            best_codes = torch.argmin(final_dists, dim=1).view(out_features, num_groups)
            layer.codes = best_codes.to(torch.uint8)

            # 4. Exact Residual Outlier Computation
            # Reconstruct quantized manifold: W_hat_manifold
            unpacked_blocks = centroids[best_codes.long()] # [M, G, D]
            W_hat_manifold = (unpacked_blocks * row_scales).reshape(out_features, in_features)

            # Residual outliers: W_outlier = (W - W_hat_manifold) * outlier_mask
            # Guaranteed to cancel any centroid spillover at outlier coordinates!
            outlier_values = (W - W_hat_manifold)[outlier_mask]

            layer.outlier_indices = outlier_indices
            layer.outlier_values = outlier_values.to(dense_weight.dtype)
            layer.num_outliers = outlier_values.numel()

            if dense_bias is not None:
                layer.bias.data.copy_(dense_bias.data)

        return layer

    def build_lut(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-projects activation blocks against all centroids once per forward pass.
        x: [..., in_features]
        Returns:
          LUT: [..., num_groups, codebook_size]
        """
        # Reshape x into [..., num_groups, block_dim]
        *batch_dims, in_feat = x.shape
        x_blocks = x.view(*batch_dims, self.num_groups, self.block_dim)

        # Bilinear dot product: LUT[g, k] = <x_blocks[g], codebook[k]>
        # x_blocks: [..., G, D], codebook: [K_cb, D]
        # einsum: "... g d, k d -> ... g k"
        lut = torch.matmul(x_blocks, self.codebook.t())  # [..., G, K_cb]
        return lut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure Zero-Unpack Forward Pass.
        Weights are NEVER decompressed or unpacked into dense floating-point memory!
        """
        orig_shape = x.shape
        *batch_dims, in_feat = orig_shape
        x_flat = x.view(-1, in_feat)  # [B, K]
        batch_size = x_flat.shape[0]

        # 1. Pre-project activations against codebook -> SRAM Look-Up Table
        # LUT shape: [B, num_groups, codebook_size]
        lut = self.build_lut(x_flat)

        # 2. Zero-Unpack Manifold Execution via Integer Index Gather
        # self.codes: [out_features, num_groups] (uint8)
        # lut: [B, num_groups, codebook_size]
        # We gather: gathered[b, m, g] = lut[b, g, codes[m, g]]
        codes_expanded = self.codes.unsqueeze(0).expand(batch_size, -1, -1).long()  # [B, M, G]
        lut_expanded = lut.unsqueeze(1).expand(-1, self.out_features, -1, -1)      # [B, M, G, K_cb]

        # Gather along centroid dimension: [B, M, G]
        gathered = torch.gather(lut_expanded, dim=3, index=codes_expanded.unsqueeze(3)).squeeze(3)

        # Accumulate over groups: [B, M]
        y_manifold = gathered.sum(dim=2) * self.scales.unsqueeze(0)

        # 3. Add Sparse Outlier Component
        if self.num_outliers > 0:
            # Reconstruct sparse matrix multiplication efficiently: y_outlier = x * W_outlier^T
            # outlier_indices: [2, num_outliers] where row 0 is m (out_feat), row 1 is k (in_feat)
            m_idx = self.outlier_indices[0]
            k_idx = self.outlier_indices[1]

            # x_flat[:, k_idx] has shape [B, num_outliers]
            # Multiply by outlier values and scatter-add to output row m_idx
            outlier_prod = x_flat[:, k_idx] * self.outlier_values.unsqueeze(0)  # [B, num_outliers]

            y_outlier = torch.zeros_like(y_manifold)
            y_outlier.index_add_(1, m_idx, outlier_prod)
            y = y_manifold + y_outlier
        else:
            y = y_manifold

        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)

        # Reshape to original batch dimensions
        return y.view(*batch_dims, self.out_features)

    def dequantize_weights(self) -> torch.Tensor:
        """
        Diagnostic ONLY: unpacks weights into dense matrix to verify exactness.
        Not used during forward pass.
        """
        with torch.no_grad():
            # codes: [out_features, num_groups]
            # codebook: [codebook_size, block_dim]
            unpacked_blocks = self.codebook[self.codes.long()]  # [M, G, D]
            W_manifold = (unpacked_blocks * self.scales.view(-1, 1, 1)).reshape(self.out_features, self.in_features)

            W_outlier = torch.zeros_like(W_manifold)
            if self.num_outliers > 0:
                W_outlier[self.outlier_indices[0], self.outlier_indices[1]] = self.outlier_values

            return W_manifold + W_outlier

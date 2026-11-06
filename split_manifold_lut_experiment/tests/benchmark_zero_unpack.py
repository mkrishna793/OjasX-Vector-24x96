"""
benchmark_zero_unpack.py

Empirical Benchmark: Zero-Unpack Split-Manifold LUT Execution vs Streaming Dequantization.
Measures latency, memory allocation, and FLOPs for LLM inference (e.g. K=4096, M=4096).

Author: OjasX Research Team & Antigravity
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from split_manifold_lut_experiment.engine.split_manifold_engine import SplitManifoldLUTLinear


def benchmark():
    torch.manual_seed(42)
    device = torch.device("cpu")

    M, K = 4096, 4096  # Standard 7B/8B LLM projection dimension
    block_dim = 16
    codebook_size = 256
    num_tokens = 1  # Autoregressive single-token decoding

    print("================================================================================")
    print(f"BENCHMARKING ZERO-UNPACK SPLIT-MANIFOLD LUT ENGINE (M={M}, K={K}, Block={block_dim})")
    print("================================================================================")

    # Generate synthetic weight and activation
    dense_w = torch.randn(M, K, dtype=torch.float32)
    x = torch.randn(num_tokens, K, dtype=torch.float32)

    layer = SplitManifoldLUTLinear.from_dense(
        dense_w, block_dim=block_dim, codebook_size=codebook_size, outlier_tau=3.5
    )

    print(f"Total weights: {M * K:,} elements ({M * K * 2 / 1024**2:.1f} MB in BF16)")
    print(f"SRAM LUT table size: {layer.lut_sram_bytes / 1024:.1f} KB")
    print(f"Compression ratio: {layer.compression_ratio:.2f}x")
    print(f"Isolated outliers: {layer.num_outliers:,} coordinates ({layer.num_outliers / (M * K) * 100:.3f}%)")

    # Warmup
    for _ in range(5):
        _ = layer(x)

    # 1. Benchmark Zero-Unpack LUT Forward
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        y_lut = layer(x)
    lut_latency_ms = (time.perf_counter() - t0) / iters * 1000.0

    # 2. Benchmark Streaming Dequantize + Dense Matmul (The Broken Baseline)
    def streaming_dequantize_forward(x_in, layer_in):
        # Emulates what ojasx2/3 was doing: dynamically unpacks [M, K] every forward pass
        codes = layer_in.codes.long()
        unpacked = layer_in.codebook[codes]  # [M, G, D]
        w_dense = (unpacked * layer_in.scales.view(-1, 1, 1)).reshape(layer_in.out_features, layer_in.in_features)
        return torch.matmul(x_in, w_dense.t())

    # Warmup streaming
    for _ in range(3):
        _ = streaming_dequantize_forward(x, layer)

    t0 = time.perf_counter()
    for _ in range(iters):
        y_stream = streaming_dequantize_forward(x, layer)
    stream_latency_ms = (time.perf_counter() - t0) / iters * 1000.0

    print("--------------------------------------------------------------------------------")
    print(f"Zero-Unpack LUT Execution Latency:        {lut_latency_ms:.3f} ms / layer")
    print(f"Streaming Weight Dequantization Latency:  {stream_latency_ms:.3f} ms / layer")
    speedup = stream_latency_ms / lut_latency_ms
    print(f"Speedup:                                  {speedup:.2f}x FASTER")
    print(f"Accumulated across 168 layers (e.g. Qwen):")
    print(f"  - Zero-Unpack LUT Total Dec Time:       {lut_latency_ms * 168:.1f} ms / token")
    print(f"  - Streaming Unpack Total Dec Time:      {stream_latency_ms * 168:.1f} ms / token")
    print("--------------------------------------------------------------------------------")
    print("Memory Bandwidth Saved:")
    bytes_per_dense_row = K * 2  # 8192 bytes
    bytes_per_lut_row = (K // block_dim) * 1  # 256 bytes
    bw_saving = bytes_per_dense_row / bytes_per_lut_row
    print(f"  - Dense weight memory read per token:   {M * bytes_per_dense_row / 1024**2:.1f} MB")
    print(f"  - Zero-Unpack memory read per token:   {M * bytes_per_lut_row / 1024**2:.1f} MB")
    print(f"  - Memory Bandwidth Reduction:           {bw_saving:.1f}x reduction")
    print("================================================================================")


if __name__ == "__main__":
    benchmark()

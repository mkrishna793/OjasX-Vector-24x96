"""
test_moe_ctypes.py

Empirically verifies that Python ctypes can initialize and run the 512-Expert
Split-Manifold MoE engine directly in native ARM NEON / AVX2 C++.
"""

import numpy as np
from split_manifold_lut_experiment.native.bindings.ojasx_lut_ctypes import NativeSplitManifoldMoE, get_engine_info

def test_native_moe():
    info = get_engine_info()
    print("OjasX Engine Info:", info)
    assert "Apple_Silicon_ARM_NEON" in info["backend"] or "AVX" in info["backend"] or "CPU" in info["backend"]

    num_experts = 512
    hidden_dim = 256
    intermediate_dim = 64
    top_k = 10
    block_dim = 16
    num_groups = hidden_dim // block_dim
    inter_groups = intermediate_dim // block_dim

    moe = NativeSplitManifoldMoE(num_experts, hidden_dim, intermediate_dim, top_k, block_dim)

    # Synthetic codebooks & weights
    cb_in = np.random.randn(256, block_dim).astype(np.float32) * 0.05
    cb_down = np.random.randn(256, block_dim).astype(np.float32) * 0.05
    gu_codes = np.random.randint(0, 256, size=(num_experts, 2 * intermediate_dim, num_groups), dtype=np.uint8)
    gu_scales = np.ones((num_experts, 2 * intermediate_dim), dtype=np.float32)
    dn_codes = np.random.randint(0, 256, size=(num_experts, hidden_dim, inter_groups), dtype=np.uint8)
    dn_scales = np.ones((num_experts, hidden_dim), dtype=np.float32)

    moe.initialize_weights(cb_in, cb_down, gu_codes, gu_scales, dn_codes, dn_scales)

    print(f"Native 512-Expert MoE Initialized.")
    print(f"Compressed RAM Footprint: {moe.compressed_bytes / (1024*1024):.2f} MB")
    print(f"Compression Ratio:        {moe.compression_ratio:.2f}x")
    assert moe.compression_ratio > 20.0

    # Test top-10 routed forward
    x = np.random.randn(hidden_dim).astype(np.float32)
    top_k_indices = np.array([3, 17, 89, 142, 201, 310, 399, 450, 488, 510], dtype=np.int32)
    top_k_weights = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)

    y_out = moe.forward(x, top_k_indices, top_k_weights)

    print(f"Forward completed successfully! Output norm: {np.linalg.norm(y_out):.4f}")
    assert y_out.shape == (hidden_dim,)
    assert not np.isnan(y_out).any()
    assert np.linalg.norm(y_out) > 0.0

    print("ALL NATIVE MOE CTYPES TESTS PASSED!")

if __name__ == "__main__":
    test_native_moe()

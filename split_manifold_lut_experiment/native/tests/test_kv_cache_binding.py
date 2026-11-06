"""
test_kv_cache_binding.py

Verifies Python integration with Native C++ KV-Cache Engine.
Tests token appending, memory reduction, and attention context generation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings")))
from ojasx_lut_ctypes import NativeKVCache

def test_kv():
    print("Testing Native Zero-Unpack KV-Cache Engine...")
    num_heads = 8
    head_dim = 128
    max_seq_len = 4096

    cache = NativeKVCache(num_heads=num_heads, head_dim=head_dim, max_seq_len=max_seq_len, block_dim=16)

    # Append 500 tokens
    for t in range(500):
        k = np.random.randn(head_dim).astype(np.float32)
        v = np.random.randn(head_dim).astype(np.float32)
        for h in range(num_heads):
            cache.append(h, k, v)

    # Query attention
    q = np.random.randn(head_dim).astype(np.float32)
    ctx = cache.attention(head_idx=0, q=q)

    print(f"Appended 500 tokens across {num_heads} heads.")
    print(f"Total KV Cache in RAM: {cache.total_bytes / 1024:.1f} KB")
    print(f"Uncompressed Baseline: {(2 * 500 * num_heads * head_dim * 4) / 1024:.1f} KB")
    print(f"Compression Ratio:    {((2 * 500 * num_heads * head_dim * 4)) / max(1, cache.total_bytes):.1f}x smaller")
    print(f"Attention Context vector shape: {ctx.shape}, sample values: {ctx[:4]}")
    assert not np.isnan(ctx[0])
    print("KV-Cache Python Verification: PASSED!")

if __name__ == "__main__":
    test_kv()

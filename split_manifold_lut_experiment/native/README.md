# OjasX Native Split-Manifold LUT SIMD Engine & Portability Suite

High-performance, standalone C++20 engine implementing **Zero-Unpack Look-Up Table (LUT)** matrix-vector multiplication, **512-Expert High-Density MoE Gather**, and **8-Bit Quantized KV-Cache**.

- **Vector SIMD Backends**: Apple Silicon ARM NEON (`__ARM_NEON`), Linux x86_64 AVX2 (`__AVX2__`).
- **Dynamic Shared Library**: `libojasx_lut.dylib` (macOS) / `libojasx_lut.so` (Linux).
- **GGML / llama.cpp Adapter**: Built-in support for `GGML_TYPE_OJASX_LUT` (100) and `GGML_TYPE_OJASX_MOE` (101).
- **Python Bindings**: Direct zero-overhead ctypes bridge (`ojasx_lut_ctypes.py`).

---

## Directory Structure

```text
split_manifold_lut_experiment/native/
├── CMakeLists.txt                 # Modern CMake build definition
├── Makefile                       # One-command build (`make`, `make test`, `make benchmark`)
├── LLAMA_CPP_INTEGRATION.md       # Integration guide for llama.cpp / GGML
├── libojasx_lut.dylib             # Compiled universal C-ABI dynamic library
├── ojasx_run                      # Standalone native C++ CLI benchmark runner
├── test_native_lut                # Comprehensive C++ verification test suite
├── include/
│   ├── ojasx_lut.h                # Core SplitManifoldLinear and SplitManifoldMoE headers
│   ├── ojasx_lut_c_api.h          # Universal C-ABI export headers
│   ├── ojasx_lut_kv_cache.h       # Zero-unpack 8-bit quantized KV-cache header
│   └── ojasx_ggml_adapter.h       # llama.cpp / GGML custom tensor operator adapter
├── src/
│   ├── ojasx_lut_cpu.cpp          # ARM NEON / AVX2 vectorized LUT pre-projection & gather
│   ├── ojasx_lut_kv_cache.cpp     # Quantized KV-cache append & attention score kernels
│   ├── ojasx_lut_c_api.cpp        # C-ABI export implementations
│   ├── ojasx_ggml_adapter.cpp     # GGML graph compute forward dispatchers
│   └── ojasx_run.cpp              # Standalone CLI binary source
├── bindings/
│   └── ojasx_lut_ctypes.py        # Python ctypes wrappers for all C-ABI functions
├── patches/
│   └── ggml-ojasx.patch           # Git patch for upstream llama.cpp / GGML
└── tests/
    ├── test_native_lut.cpp        # 7-test verification suite (Linear, MoE, KV, GGML, C-ABI)
    ├── test_moe_ctypes.py         # Python ctypes test for 512-Expert MoE
    └── test_kv_cache_binding.py   # Python ctypes test for KV-cache
```

---

## Quickstart

### 1. Build Everything
```bash
cd split_manifold_lut_experiment/native
make -j4
```

### 2. Run the Test Suite
```bash
make test
```
All 7 tests verify bit-exact correctness against uncompressed float ground truths with 0 errors.

### 3. Run Empirical Microbenchmarks
```bash
./ojasx_run --benchmark
```

### 4. Run Python ctypes Tests
```bash
PYTHONPATH=../../ python3 tests/test_moe_ctypes.py
PYTHONPATH=../../ python3 tests/test_kv_cache_binding.py
```

---

## Empirical Benchmark Results (Apple Silicon M-Series)

| Workload | Dense Baseline | OjasX Compressed | Memory Slashed | Latency Per Token | Effective Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Layer (2560 -> 2560)** | 25.00 MB | **0.42 MB** | **60.09x** | **237.45 μs** | 110.40 GB/s equiv. |
| **MLP / Gate Layer (2560 -> 6912)** | 67.50 MB | **1.10 MB** | **61.55x** | **575.79 μs** | 122.93 GB/s equiv. |
| **512-Expert MoE (Top-10 Active)** | 9,600 MB (9.6 GB) | **157.53 MB** | **60.94x** | **1,779.96 μs** | Inactive experts: 0 bytes |
| **KV-Cache (2048 tokens, 8 heads)** | 16.00 MB | **0.67 MB** | **23.81x** | **18.66 μs** append | 6.73 μs logits query |

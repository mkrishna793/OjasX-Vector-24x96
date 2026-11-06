# OjasX: Split-Manifold Look-Up Table (LUT) Execution & Extreme Vector Quantization

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](split_manifold_lut_experiment/native/)
[![Hardware Support](https://img.shields.io/badge/SIMD-ARM_NEON_%2F_AVX2-green.svg)](split_manifold_lut_experiment/native/)
[![Formal Verification](https://img.shields.io/badge/Proofs-Lean4_%2F_Rocq_%2F_Z3-purple.svg)](split_manifold_lut_experiment/lean/)

**OjasX** is an ultra-high-efficiency inference engine and vector quantization architecture designed to run massive large language models (including 120B+ parameter high-density MoEs) on consumer workstations and mobile edge chips.

Unlike standard quantization formats (AWQ, GPTQ, GGUF Q4_K) that continuously decompress quantized weights back into floating-point numbers on every token, OjasX introduces **Split-Manifold LUT Execution**: weights **never unpack into RAM**, eliminating memory bandwidth bottlenecks and achieving up to **$61.5\text{x}$ memory bandwidth reduction**.

---

## Key Breakthroughs & Core Architecture

```
                 Input Activation Vector x [hidden_dim]
                                   |
                                   v
             +-------------------------------------------+
             |   Pre-Project Activation into L1 LUT      |
             |        LUT = x . Codebook^T  (~16 KB)     |
             +---------------------+---------------------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
     +-------------------------------+   +-------------------------------+
     | Dense Linear / MLP Layers     |   | 512-Expert MoE Layer          |
     | - Pure 8-bit index gather     |   | - Top-K active gather from L1 |
     | - Zero weight decompression   |   | - Inactive 502 experts untouched|
     | - Exact CSR outlier bypass    |   | - 0 bytes inactive bus traffic|
     +---------------+---------------+   +---------------+---------------+
                     |                                   |
                     +-------------+-------------+-------+
                                   |
                                   v
             +-------------------------------------------+
             |   Zero-Unpack 8-Bit Quantized KV-Cache    |
             |   - 23.8x to 42.6x KV cache reduction     |
             |   - Direct Q . K^T attention over codes   |
             +-------------------------------------------+
```

### 1. Zero-Unpack SRAM Look-Up Table (LUT) Gather
- Pre-projects the activation vector $x$ once against 256 codebook centroids into an L1 cache Look-Up Table ($\text{LUT} \in \mathbb{R}^{G \times 256}$, size $16\text{ KB} - 160\text{ KB}$).
- Computes output neurons via pure 8-bit integer index lookup and accumulation:
  $$\hat{y}_m = \sum_{g=1}^{G} \text{LUT}[g, \text{code}[m, g]]$$
- Eliminates millions of floating-point multiplications per token and prevents memory bus saturation.

### 2. Heavy-Tail CSR Sparse Outlier Bypass
- Standard vector quantization fails on attention projection layers because $0.38\%$ of weights are extreme outliers ($15\sigma$ to $36.6\sigma$). Clamping them destroys softmax dynamic range and causes repetition loops.
- OjasX extracts these outliers into Compressed Sparse Row (CSR) format, achieving **exact dynamic range retention** and **zero repetition loops**.

### 3. High-Density 512-Expert MoE Engine
- In high-density MoE architectures (e.g. Qwen 3.8 / 120B with 512 experts per layer), all 512 experts receive the identical input activation vector $x$.
- Pre-projecting $x$ once into the shared L1 LUT allows the 10 active experts to gather outputs simultaneously. The remaining **502 inactive experts consume 0 bytes of memory bus bandwidth**.

### 4. Zero-Unpack 8-Bit Quantized KV-Cache
- Stores historical Key and Value tokens directly in 8-bit quantized manifold codes.
- Slashes KV-cache memory consumption by **$23.8\text{x}$ to $42.6\text{x}$**, enabling context windows beyond $10\text{M}+$ tokens in limited RAM.

### 5. Dual-Tier Hybrid Architecture (Universal GGML / llama.cpp Adapter)
- Exposes `GGML_TYPE_OJASX_LUT` (type 100) and `GGML_TYPE_OJASX_MOE` (type 101).
- Dispatches GGML tensor graph operations directly into native ARM NEON / AVX2 kernels while retaining `llama.cpp`'s tokenizers, prompt management, and CLI tools.

---

## Empirical Benchmarks (Apple Silicon M-Series ARM NEON)

Measured with the native C++ CLI runner (`./ojasx_run --benchmark`):

| Workload | Dense Baseline | OjasX Compressed | Memory Slashed | Latency Per Token | Effective Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Layer ($2560 \to 2560$)** | 25.00 MB | **0.42 MB** | **60.09x** | **236.82 μs** | 110.69 GB/s equiv. (1.84 GB/s actual) |
| **MLP / Gate Layer ($2560 \to 6912$)** | 67.50 MB | **1.10 MB** | **61.55x** | **575.14 μs** | 123.06 GB/s equiv. (2.00 GB/s actual) |
| **512-Expert MoE (Top-10 Active)** | 9,600 MB (9.6 GB) | **157.53 MB** | **60.94x** | **1,783.42 μs** | Inactive 502 experts: 0 bytes bus traffic |
| **KV-Cache (2048 tokens, 8 heads)** | 16.00 MB | **0.67 MB** | **23.81x** | **18.62 μs** append | 6.82 μs logits query |

---

## Mathematical Proofs & Formal Verification

The Split-Manifold bilinear pre-projection theorem and error bounds are mathematically proved and machine-checked:

1. **Lean 4 Formal Proof**: [`split_manifold_lut_experiment/lean/SplitManifoldLUT.lean`](split_manifold_lut_experiment/lean/SplitManifoldLUT.lean)  
   Proves algebraic identity $\hat{y}_m = x \cdot \hat{W}_m = \sum_g \text{LUT}[g, \text{code}[m,g]]$.
2. **Rocq / Coq Constructive Proof**: [`split_manifold_lut_experiment/rocq/SplitManifoldLUT.v`](split_manifold_lut_experiment/rocq/SplitManifoldLUT.v)  
   Proves exact equivalence and preservation of dynamic range under sparse outlier splitting.
3. **Z3 SMT Invariant Verification**: [`split_manifold_lut_experiment/z3/verify_split_manifold_z3.py`](split_manifold_lut_experiment/z3/verify_split_manifold_z3.py)  
   Exhaustively verifies non-overflow bounds, index alignment, and cacheline boundaries.

---

## Directory Layout

```text
OjasX-24X96/
├── split_manifold_lut_experiment/ # Split-Manifold LUT Autonomous Engine
│   ├── native/                    # High-performance C++20 SIMD engine & C-ABI
│   │   ├── include/               # Engine headers (ojasx_lut.h, ojasx_lut_c_api.h, ojasx_ggml_adapter.h)
│   │   ├── src/                   # Vectorized implementations (ARM NEON, AVX2, CUDA, Metal)
│   │   ├── bindings/              # Python ctypes bindings (ojasx_lut_ctypes.py)
│   │   ├── patches/               # llama.cpp / GGML integration patch
│   │   ├── tests/                 # 7 native C++ test suites (100% passing)
│   │   ├── Makefile               # One-command build (`make`, `make test`, `make benchmark`)
│   │   ├── CMakeLists.txt         # Modern CMake build system
│   │   ├── LLAMA_CPP_INTEGRATION.md # llama.cpp step-by-step integration guide
│   │   └── README.md              # Native engine documentation
│   ├── engine/                    # Python Split-Manifold reference engine
│   ├── tools/                     # Streaming quantizers & Colab validation harnesses
│   ├── lean/                      # Machine-checked Lean 4 mathematical proof
│   ├── rocq/                      # Coq / Rocq constructive verification
│   └── z3/                        # Z3 SMT solver formal boundary checks
├── ojasx_24x96/                   # Core Python package & vector quantization engine
├── benchmarks/                    # Comprehensive benchmark scripts and JSON traces
├── reports/                       # Technical milestone reports & A100 verification logs
├── tests/                         # Pytest test suite for modules M0-M14
├── pyproject.toml                 # Package configuration
└── LICENSE                        # Apache 2.0 License
```

---

## Quickstart

### 1. Compile & Run the Native C++ Engine
```bash
cd split_manifold_lut_experiment/native
make -j4
make test          # Runs 7 native C++ test suites (100% pass)
make benchmark     # Measures latency, throughput, and memory bandwidth
```

### 2. Standalone CLI Inference Runner
```bash
# Display SIMD backend and engine info
./split_manifold_lut_experiment/native/ojasx_run --info

# Microbenchmark 512-Expert MoE Layer
./split_manifold_lut_experiment/native/ojasx_run --moe 512 2560 10

# Microbenchmark Zero-Unpack KV-Cache
./split_manifold_lut_experiment/native/ojasx_run --kv 2048
```

### 3. Python ctypes Integration
```bash
PYTHONPATH=. python3 split_manifold_lut_experiment/native/tests/test_moe_ctypes.py
PYTHONPATH=. python3 split_manifold_lut_experiment/native/tests/test_kv_cache_binding.py
```

### 4. llama.cpp Integration
Follow the step-by-step guide in [`split_manifold_lut_experiment/native/LLAMA_CPP_INTEGRATION.md`](split_manifold_lut_experiment/native/LLAMA_CPP_INTEGRATION.md):
```bash
cd /path/to/llama.cpp
git apply /path/to/OjasX-24X96/split_manifold_lut_experiment/native/patches/ggml-ojasx.patch
cmake -B build -DGGML_OJASX=ON -DCMAKE_EXE_LINKER_FLAGS="-lojasx_lut"
cmake --build build -j
```

---

## Technical Reports

- [Native Dual-Tier Hybrid Engine Report](reports/native_dual_tier_hybrid_engine_report.md)
- [Split-Manifold LUT Mathematical Engine](reports/split_manifold_lut_math_engine.md)
- [Qwen 3.8-Flash-Next 120B Evaluation Report](reports/qwen38_flash_next_a100_evaluation_report.md)
- [Qwen 3.8-27B A100 Full Evaluation Report](reports/qwen38_27b_a100_full_evaluation_report.md)

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

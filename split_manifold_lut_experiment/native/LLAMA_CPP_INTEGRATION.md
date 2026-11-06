# Integrating OjasX Split-Manifold LUT into `llama.cpp`

This guide explains how to compile and link `llama.cpp` against the **OjasX Native Engine (`libojasx_lut`)** to enable zero-unpack LUT inference.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   llama.cpp High-Level API   |
                      | (Tokenizers, Samplers, CLI) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    GGML Execution Graph     |
                      +--------------+--------------+
                                     |
                [Is Tensor Type GGML_TYPE_OJASX_LUT?]
                               /            \
                             YES             NO
                             /                \
                            v                  v
              +--------------------------+  +--------------------------+
              |   ojasx_ggml_adapter     |  | Standard GGML FP16 / Q4  |
              | Zero-Unpack L1 LUT Gather|  | Unpack + Float MACs      |
              +--------------------------+  +--------------------------+
                            |
                            v
              +--------------------------+
              |    libojasx_lut.dylib    |
              |  ARM NEON / AVX2 Engine  |
              +--------------------------+
```

---

## 3-Step Integration Guide

### Step 1: Apply the GGML Extension Patch
Navigate to your `llama.cpp` clone and apply the provided patch:
```bash
cd llama.cpp
git apply /path/to/OjasX/split_manifold_lut_experiment/native/patches/ggml-ojasx.patch
```

### Step 2: Include Headers and Link `libojasx_lut`
When invoking CMake for `llama.cpp`, pass the include and library paths:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OJASX=ON \
    -DCMAKE_CXX_FLAGS="-I/path/to/OjasX/split_manifold_lut_experiment/native/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/path/to/OjasX/split_manifold_lut_experiment/native -lojasx_lut"
cmake --build build -j
```

### Step 3: Run Inference
```bash
./build/bin/main -m qwen38_ojasx_compressed.gguf -p "Explain quantum entanglement:" -n 128
```

---

## Benefits over Standard Q4_K / Q8_0
1. **Zero Weight Decompression**: Weights remain compressed as uint8 codes in host RAM. No decompression into float buffers on every token.
2. **MoE Zero-Arithmetic Inactive Expert Bypass**: In 512-expert architectures (e.g. Qwen 3.8 / 120B), only the active top-10 experts read codes from the shared $8\text{ KB}$ L1 cache activation table. The 502 inactive experts consume **0 memory bus bandwidth**.
3. **Zero-Unpack 8-Bit KV-Cache**: KV cache history is stored in 8-bit codes, reducing memory consumption by $23.8\text{x}$ to $42.6\text{x}$.

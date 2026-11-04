# Native CPU Leech24 streaming decoder

The optional native backend addresses the measured bottleneck in the original
Leech24 memory profile without changing the codec, planner, checkpoint, or dense
fallback. It is a locally built scalar C++ CPU decoder with an exact PyTorch
reference path. Linux CPU is the executed platform; CUDA and other operating
systems are not claimed as qualified.

## Why this was the next step

A cProfile run of three pinned-TinyGPT memory forwards took 0.337 seconds. Row
slicing and reconstruction accounted for approximately 0.279 seconds, including
0.134 seconds in shell-index decoding, with additional bit unpacking, presence
counting, temporary tensors, and repeated encoded-view validation. Dense GEMM was
not the main Leech-specific bottleneck.

The first implementation goal was therefore row decoding, not a new shell,
entropy coder, or shell-codebook cache. A full 196,560-by-24 float32 codebook would
have added roughly 18.9 MB and worked against the RAM objective.

## Implemented design

`native/leech24_cpu.cpp` and `leech_native.py` provide:

- independent C++ Golay-word, octad, coordinate-pair, and shell-point decoding;
- direct little-endian 18-bit index and 0/1/2-bit gain unpacking;
- adaptive presence scanning and compact active-index mapping;
- residual-stage accumulation without a materialized shell codebook;
- bounded padded row-block output, never a retained full decoded model;
- a small C ABI built from source with the host C++ compiler;
- no Python development headers or PyTorch C++ extension build;
- `python`, `native`, and `auto` backend selections;
- automatic fallback in `auto` mode and loud failure in explicit `native` mode;
- `OJASX_LEECH_NATIVE_DISABLE=1` as an explicit opt-out.

The native code fuses index/shell/gain/residual reconstruction. It deliberately
leaves the final Householder operations to the same PyTorch functions used by the
reference decoder. This preserves bitwise rounding: native C++ arithmetic initially
differed by a few float32 ULPs, which became visible after BF16 conversion. Using
the reference Householder operations removed those differences. PyTorch also
continues to perform the final dtype conversion and GEMM; a fused native GEMM is
not claimed.

The encoded view is now constructed once per forward rather than once per row
chunk. It is only a lightweight view of existing payloads. Every row decode stays
bounded, and no dense weight is cached across forwards in memory mode.

## Correctness evidence

Permanent tests cover:

1. every one of the **196,561** native indices against the independent PyTorch
   construction, with exact integer equality;
2. unaligned and partial row ranges for 0/1/2 gain bits, adaptive stages, and
   Householder reflections, with bitwise float32 equality;
3. Python/native streaming and full-cache module paths;
4. disabled-backend auto fallback and explicit-native fail-closed behavior;
5. existing planner, manifest, save/meta-reload, and learned-codec regressions.

The real-artifact qualification additionally requires:

- identical Python/native memory-prefill SHA-256 hashes in fresh processes;
- identical Python/native speed-prefill SHA-256 hashes;
- identical generated token IDs in every quantized worker;
- zero decoded-cache bytes in native memory mode;
- unchanged persistent tensor inventories.

The checked run passes all of these checks. The pinned TinyGPT generated
`[111, 110, 103, 32]` in every quantized configuration.

## Measured runtime result

All rows below use the same BF16 artifact with SHA-256
`1a3e9d5e7210cd44cb5dbea266ddb488b8d5034658b3c0161fe9bd94dfcc3aa9`.
Each worker is a fresh CPython process with one CPU thread. Decode generates four
tokens while recomputing context without KV caching, so these are not throughput
measurements.

| Worker | Warm prefill median | Decode median/token | Sampled decode peak RSS |
| --- | ---: | ---: | ---: |
| Dense BF16 | 0.015580 s | 0.018844 s | 727,838,720 B |
| Leech Python memory | 0.096814 s | 0.103795 s | 730,882,048 B |
| Leech native memory | 0.023041 s | 0.027315 s | 730,390,528 B |
| Leech Python speed cache | 0.015150 s | 0.018497 s | 730,505,216 B |
| Leech native speed cache | 0.015970 s | 0.019562 s | 729,169,920 B |

Derived results:

- Native memory decode is **3.800x faster** than Python Leech memory decode.
- Native memory decode time is **1.450x dense**, improved from **5.508x dense**
  for Python Leech in the same worker set.
- Native speed-cache decode is **1.038x dense** in this tiny sample; this is
  near timing parity within the probe, not a demonstrated speedup.
- Native memory retains **0 decoded-cache bytes**. Speed mode retains
  **184,320 bytes**, exactly as before.
- Native memory initializes **0 bytes of Python Golay/Leech tensor tables**.
  Python memory initializes **207,136 bytes**. The C++ table object is
  **21,628 bytes**.
- Native sampled decode RSS is **491,520 bytes lower** than Python Leech in this
  run, but remains **2,551,808 bytes higher than dense**. RSS sampling and a tiny
  model make small differences noisy; neither value is a general RAM law.

Persistent state remains **1,861,927 bytes** in every quantized worker versus
**2,013,952 dense bytes**. Native changes execution, not compression. The shared
library, C++ runtime, Python objects, allocator retention, activations, and GEMM
workspace are not included in the table-byte counts.

## API

New Leech modules default to `auto`: use native CPU decoding when it builds and
loads successfully, otherwise use the reference implementation.

```python
from ojasx_vector_quantization_runner import (
    native_leech_backend_status,
    set_leech_decode_backend,
)

print(native_leech_backend_status())
set_leech_decode_backend(model, "native")  # Require native; fail if unavailable.
set_leech_decode_backend(model, "python")  # Force the reference implementation.
set_leech_decode_backend(model, "auto")    # Native with fallback.
```

Direct row/full decoders also accept `backend="python"`, `"native"`, or `"auto"`.
Changing a module backend clears its decoded-weight cache to avoid reusing a
cache produced under a different execution contract.

## Reproduce

First create the fixed real artifact as described in `LEECH24.md`, then run:

```bash
python -m ojasx_vector_quantization_runner.leech24_native_qualify \
  --artifact /tmp/leech24-qualification/run/encoded \
  --evidence ojasx_vector_quantization_runner/evidence/leech24-native-runtime.json
```

The dense artifact must be the `source-bf16` sibling produced by
`leech24_qualify`. Run permanent tests with:

```bash
python -m pytest -q \
  tests/test_ojasx_vector_quantization_runner.py \
  tests/test_ojasx_vector_quantization_generalized.py \
  tests/test_ojasx_vector_quantization_leech24.py
```

Native tests skip only the two compiler-dependent checks when no backend can be
built. The disabled-backend fallback test still executes. Machine-readable
results are in `evidence/leech24-native-runtime.json`.

## Remaining boundary and next step

This is a substantial Leech-runtime improvement, not a dense-RAM victory or a
production-kernel claim. The remaining TinyGPT gap includes runtime/allocator
overhead that cannot be solved by a 152 KB persistent-state saving. A fused GEMM,
SIMD, multithreading, CUDA, or larger-model benefit is unmeasured.

The next qualification should use a larger local BF16 checkpoint where weight
residency dominates process overhead, with learned/Leech/dense candidates in one
planner and explicit quality, RSS, prefill, and decode targets. Native must retain
the same exact-output and fail-closed contracts there before broader deployment
claims are appropriate.

# OJASX Vector Quantization Runner

A **standalone, inference-only** adaptive multi-stage residual vector-quantization
project for PyTorch and local Hugging Face generative language models.

This directory is deliberately isolated from `ojasx`, `ojasx2`, the E8 codecs, and
the M0–M4 training/scheduling foundation. An executable test rejects any import of
those packages from this project. Nothing in the OjasX core is modified.

## Status and honest scope

The current implementation is a **hardened research runner**, not a claim of
production readiness or state-of-the-art model compression. It provides:

- deterministic learned vector codebooks with 1/2/4/8-bit code packing;
- genuine 24D Leech minimum-shell VQ with exact Golay construction/indexing;
- mixed learned/Leech/dense candidates in the same calibrated global planner;
- arbitrary learned-VQ vector dimensions from 2 through 64, including 24D blocks;
- multiple residual stages with a zero-codeword non-expansion guard;
- per-row RMS scaling and exact padding/truncation;
- per-tensor relative-error measurement and error-budget admission;
- automatic original-layer fallback for failed budgets or ineffective compression;
- inference-only `ResidualVQLinear` and HF `ResidualVQConv1D` modules;
- row-streaming/active-expert memory execution and explicit decoded-weight caching;
- optional fused native CPU Leech row decoding with exact fallback and output tests;
- model-independent tensor IR, role policies, and fail-closed execution adapters;
- multi-candidate Pareto planning under persistent-state/cache/decode constraints;
- bounded local-output calibration and independent end-to-end quality gates;
- a pinned larger-BF16 checkpoint qualification harness with fresh-process RSS
  gates (harness only; no larger-checkpoint result is currently claimed);
- exact `nn.Linear` replacement plus allowlisted Conv1D/fused-expert adapters;
- fail-closed dense retention for unknown `nn.Linear` subclasses;
- independent expert-projection admission and dense fallback;
- shared/tied-weight preservation by default;
- local HF conversion, sharded safetensors persistence, and direct meta-model loading;
- auditable `vq_plan.json` persistence with post-apply byte-prediction checks;
- task-aware causal/multimodal model reconstruction for reviewed composite configs;
- duplicate-shard, missing-tensor, shape, schema, and unexpected-tensor checks;
- tokenizer, generation-config, and chat-template preservation;
- local chat, per-text next-token evaluation, and forward benchmarking;
- independent Z3 and Rocq codec proofs with permanent counterexamples.

It does **not** currently implement a general infinite-Leech closest-vector solver,
TCQ, NVMe residual streaming, contextual sparsity, cross-layer folding, Unsloth
integration, or an LM Studio/llama.cpp tensor type. Those require separate
research and runtime kernels.

## Codec

For each row \(r\), the encoder computes an RMS scale \(s_r\). Zero rows retain an
exact zero scale. The row is divided into vectors \(x\) of `block_size` values.
Each stage \(j\) learns a codebook \(C_j\) and stores the nearest-centroid index:

\[
a_j = \operatorname{argmin}_k \|R_j - C_j[k]\|_2^2,
\qquad R_{j+1} = R_j - C_j[a_j],
\qquad R_0 = x.
\]

Every codebook contains a zero vector. Therefore choosing that centroid can leave
the residual unchanged, and nearest-centroid selection cannot increase its squared
norm. This is a non-expansion guarantee, **not** a guarantee of low error.

The reconstruction is:

\[
\hat{x} = \sum_{j=1}^{S} C_j[a_j],
\qquad \hat{W}_r = s_r\hat{x},
\]

followed by removal of container padding. The encoder measures the actual stored
reconstruction against the source matrix. It admits the shortest residual prefix
that satisfies `max_relative_error`, or retains the original layer if no prefix
qualifies. A candidate that is not smaller is also rejected by default.

### Payload bits versus real storage

The nominal code-payload rate is:

```text
selected_stages * bits_per_stage / block_size
```

That number excludes codebooks, scales, bias, metadata, embeddings, unquantized
layers, KV cache, activations, and runtime overhead. Reports therefore also include
`measured_bits_per_weight` and actual encoded bytes for each quantized weight.

The reproducible synthetic low-entropy matrix in `evidence/codec-smoke.json` uses a
0.5-bit payload but measures **2.0 total bits/weight**. It compresses
**16,384 bytes to 1,024 bytes** with relative reconstruction error approximately
**9.80e-8**. This specially structured matrix is not evidence that arbitrary LLM
weights reach the same result.

## Leech24 finite-shell codec

`leech24.py` and `leech_codec.py` implement a separate algebraic 24D codec. They
construct and verify the extended Golay code, decode all 196,561 origin/minimum-
shell indices bijectively, pack each direction in 18 bits, and compete every
shape--gain codeword against the origin for finite-code residual non-expansion.
Shared seeded Householder reflections, zero/one/two-bit radial gains, adaptive
presence bitmaps, and row scales are optional codec/planner choices. This is not
a nearest-point solver for the infinite Leech lattice.

Every index, gain, bitmap, scale, and gain-table byte is counted. Bounded index
ranges and matrix rows can be decoded without retaining a full dense weight. The
same tensor IR, activation calibration, global planner, quality gates, manifests,
and meta-reload path accept mixed learned-VQ, Leech, and dense candidates.

See [`LEECH24.md`](LEECH24.md) for the construction, exact storage formula,
matched-actual-byte synthetic learned-VQ comparison, five-step implementation
map, and real trained-checkpoint reproduction. The real checkpoint probe reduced
persistent state by **7.5486%** and passed its scoped gates, but did **not** reduce
sampled process RSS on the roughly 2 MB model; streaming was slower and speed
mode used a full decoded cache. No universal RAM, accuracy, or latency win is
claimed.

[`LEECH24_NATIVE.md`](LEECH24_NATIVE.md) adds the measured native CPU follow-up.
Its exact-output memory decoder is **3.800x faster than Python Leech** on the
pinned artifact and improves decode from **5.508x to 1.450x dense time**. Native
memory still has zero decoded cache and slightly lower RSS than Python Leech,
but remains above dense RSS on this tiny model. The backend defaults to `auto`,
with explicit Python/native modes and a compiler-less fallback.

## Memory and computation

`ResidualVQLinear` stores packed codes, codebooks, scales, and optional bias. Two
runtime profiles make the memory/latency tradeoff explicit:

- `memory` (the default) retains no decoded dense matrix. Ordinary projections
  decode bounded row chunks; fused expert banks decode only the active expert.
  The row bound is configurable and persisted with generalized plans;
- `speed` caches full decoded matrices after their first use and reuses them on
  later forwards. Buffer updates invalidate version-tracked caches. This can
  reduce repeated decode work, but state plus a same-dtype full cache is larger
  than dense by exactly the encoded payload bytes.

`runtime_storage_report` separately reports VQ bytes, decoded-cache bytes,
unique persistent state-tensor bytes, and their sum. Exact tied aliases are
counted once. These are tensor inventories, **not** RSS or allocator measurements.
Streaming reduces dense-weight workspace, but codebook gathers, activations,
temporary float32 chunks, KV cache, and runtime overhead remain separate.
Arithmetic computation does not automatically decrease, and no universal speedup
or whole-process peak-RAM reduction is claimed.

The generic PyTorch path can move to devices supported by the installed PyTorch.
Only CPU execution has been tested in this workspace. CUDA, ROCm, MPS, Metal,
llama.cpp, and dedicated GPU lookup/gather kernels are not claimed as qualified.

## Generalized tensor IR and planning

The generalized API separates discovery, candidate generation, calibration,
planning, application, and validation. It requires an explicit memory budget and
at least one local or end-to-end quality budget. Calibration is mandatory unless
weight-only planning is explicitly enabled. Uncalled layers and unsupported or
pre-quantized source contracts fail dense by default.

The planner uses actual encoded bytes and reports whether its candidate search was
exact or beam-truncated. Applying a plan verifies the predicted persistent state
against the real tensor inventory. Local calibration errors guide selection but
are not model-accuracy bounds; independent logits/KL/top-1/NLL gates return an
explicit passed, failed, or unvalidated status.

See [`GENERALIZED_VQ.md`](GENERALIZED_VQ.md) for the encoded-byte and RAM formulas,
per-role policies, streaming/speed-mode identity, and API example,
[`LEECH24.md`](LEECH24.md) for the algebraic codec and real-checkpoint results,
and [`LARGER_BF16.md`](LARGER_BF16.md) for the pinned larger-checkpoint
qualification contract. The larger harness distinguishes persistent bytes,
sampled phase RSS, process high-water marks, and weight-dominated load
allocations; it has not yet produced a larger-model pass.

The reproducible tiny generalized smoke selects one of two linear layers and
reduces unique persistent state from **16,768 to 11,264 bytes**. On its eight
held-out random output positions it records relative logits error **0.033535**,
mean KL approximately **3.40e-5**, and top-1 agreement **0.875**, passing its
explicit illustrative gates. These are tiny synthetic contract measurements, not
real-model accuracy or RAM certification.

## HF architecture compatibility

The ordinary replacement pass handles exact `nn.Linear` instances and HF
`Conv1D`. Unknown `nn.Linear` subclasses remain dense: their custom forward or
reshape semantics cannot safely be replaced by `F.linear`. A decoded `.weight`
compatibility property supports HF code that directly reads weight dtype, shape,
or orientation without retaining a dense matrix in memory mode.

Separate allowlisted adapters cover verified ordinary and clamped fused-expert
contracts. They do not infer compatibility from similar attribute names in custom
remote code.

With Transformers 5.22.1 (also the latest PyPI release when checked), tiny random
CPU models have quantized save/meta-load/output-equality tests for:

- Qwen2, Qwen3, Qwen3-Next, Llama, Gemma3, Mistral, and GPT-2;
- Qwen3.5 text/MoE, Qwen2-MoE, Qwen3-MoE, and Mixtral;
- Gemma4 text/MoE, GLM-MoE-DSA, and DeepSeek-V4;
- composite Qwen3.5, Gemma4, and GLM5-Next text-input paths.

These include architecture classes used by Qwen3.8, Gemma4, GLM-5.3/Flash, and
DeepSeek-V4. They do **not** constitute execution of those official pretrained
checkpoints. Composite tests use `input_ids` only; image/audio/video inputs and
processor-based chat are not yet qualified. DeepSeek-V4.1 is not registered in the
installed latest Transformers and is not claimed as supported. Native FP8/FP4
checkpoint requantization is also unverified.

The official model IDs, exact config contracts, executed scopes, unsupported
boundaries, and the roadmap for generalized VQ are recorded in
[`CURRENT_MODELS.md`](CURRENT_MODELS.md).

By default, `lm_head` remains dense. Embeddings, normalization layers, unsupported
routing/custom modules, and unsupported tensor types remain unchanged. Shared or
tied weights are preserved by default. Include/exclude globs can protect sensitive
modules. Universal architecture support and perfect accuracy are not claimed.

Conversion loads the source model once. Quantized loading creates the architecture
under `accelerate.init_empty_weights`, using causal or multimodal auto-loading as
recorded in the manifest. It reconstructs VQ modules and assigns safetensors shards
without first loading dense source weights. On
Transformers 5 the checkpoint stores the exact runtime tensor layout with
`save_original_format=False`; use this runner's loader, not dense `from_pretrained`.

## Local conversion and chat

Install PyTorch plus the HF dependencies in an environment, then run commands from
the repository root:

```bash
python -m pip install torch transformers accelerate safetensors tokenizers
```

Convert a **local** HF instruct-model directory:

```bash
python -m ojasx_vector_quantization_runner convert /path/to/source /path/to/vq-output \
  --block-size 8 --bits-per-stage 4 --max-stages 4 \
  --max-relative-error 0.20
```

The output directory must be empty, preventing stale shards from mixing with a new
model. Use `--exclude` repeatedly for additional module globs. A failed conversion
is preferable to silently producing a larger or excessively inaccurate model.

Inspect the actual layer measurements:

```bash
python -m ojasx_vector_quantization_runner inspect /path/to/vq-output
```

Compare original and quantized models on explicit texts:

```bash
python -m ojasx_vector_quantization_runner evaluate /path/to/source /path/to/vq-output \
  --text "A held-out evaluation prompt." \
  --text "Another prompt in the language you want to test."
```

This reports next-token NLL, perplexity, top-1 agreement, and reference-to-candidate
KL for those texts only. Use representative held-out benchmarks and human chat
evaluation before choosing a deployment configuration.

Run local chat:

```bash
python -m ojasx_vector_quantization_runner chat /path/to/vq-output \
  --device cpu --prompt "Answer in English: what is vector quantization?"
```

The runner refuses to guess a missing chat template. It uses the saved tokenizer's
template and generation metadata, which avoids a common source of broken Qwen chat
behavior. It cannot repair a tokenizer or model that was already incorrect. Add
`--runtime-profile speed` to trade decoded-cache RAM for potentially lower repeated
decode work. In memory mode, `--decode-row-block-size` controls the streaming row
chunk and defaults to the saved generalized-plan value (or 64).

Benchmark both profiles on an explicit prompt:

```bash
python -m ojasx_vector_quantization_runner benchmark /path/to/vq-output \
  --device cpu --text "A benchmark prompt." --warmup 2 --iterations 5
```

The report checks numerical output agreement across profiles using dtype-specific
or explicit tolerances and records median/mean/range latency plus tensor/cache
bytes. Chunked and full GEMM can differ slightly in floating-point rounding. It
measures forward passes only, not
generation throughput, service latency, or allocator high-water marks.

`evaluate` now includes aggregate and per-text NLL, perplexity, top-1 agreement,
and reference-to-candidate KL. Per-text results help expose a regression on one
language or prompt that an aggregate could hide; they still do not prove natural
language quality or that generated answers use the requested language.

## Python API

```python
from ojasx_vector_quantization_runner import QuantizationConfig, quantize_model
from ojasx_vector_quantization_runner.hf import (
    VQChatRunner,
    save_quantized_pretrained,
)

config = QuantizationConfig(
    block_size=8,
    bits_per_stage=4,
    max_stages=4,
    max_relative_error=0.20,
)
model, report = quantize_model(model, config)
save_quantized_pretrained(model, "/tmp/vq-model", tokenizer=tokenizer, report=report)

runner = VQChatRunner.from_pretrained("/tmp/vq-model", device="cpu")
answer = runner.respond([{"role": "user", "content": "Hello"}])
```

This is inference-only. Codes and codebooks are buffers, not trainable quantization
parameters. Quantization-aware training, Unsloth training integration, and backward
through a codebook are outside this project's current contract.

## Formal foundation

The independent artifacts are in `formal/`:

- exact integer 1/2/4-bit packing and byte bounds;
- residual reconstruction and exact zero-residual reconstruction;
- nonzero row-scale roundtrip;
- zero-anchor non-expansion;
- padding bounds, error-budget admission, and compression-ratio guards;
- counterexamples for out-of-range codes, unanchored residual growth, the lack of
  an accuracy implication from compression, and payload-only bit-rate omissions.

Current evidence:

- **9/9 Z3 proofs** and **4/4 intended counterexamples**;
- **7/7 Rocq declarations** checked with jsCoq and closed under the global context;
- a deliberately false Rocq theorem rejected.

These are exact Boolean, integer, canonical-rational, or Z3-real learned-codec
laws. They do not prove the Golay/Leech construction; Leech has separate
exhaustive index/membership and brute-force-search tests. The formal artifacts
are not IEEE-kernel proofs, statistical rate-distortion theorems, model-quality
proofs, HF architecture-coverage proofs, or hardware-performance proofs.

## Reproduce

```bash
python -m pytest -q tests/test_ojasx_vector_quantization_runner.py \
  tests/test_ojasx_vector_quantization_generalized.py \
  tests/test_ojasx_vector_quantization_leech24.py \
  tests/test_ojasx_vector_quantization_larger_bf16.py
python -m ojasx_vector_quantization_runner.smoke
python -m ojasx_vector_quantization_runner.runtime_smoke
python -m ojasx_vector_quantization_runner.generalized_smoke
python -m ojasx_vector_quantization_runner.leech24_smoke
python ojasx_vector_quantization_runner/formal/verify_z3.py
# Real-checkpoint download and qualification: see LEECH24.md.
# Then benchmark its native CPU decoder:
python -m ojasx_vector_quantization_runner.leech24_native_qualify \
  --artifact /tmp/leech24-qualification/run/encoded \
  --evidence ojasx_vector_quantization_runner/evidence/leech24-native-runtime.json

# Rocq/jsCoq route:
npm install --prefix .tools/jscoq jscoq@0.17.1
python ojasx_vector_quantization_runner/formal/verify_rocq.py
```

Machine-readable results are in `evidence/codec-smoke.json`,
`evidence/runtime-smoke.json`, `evidence/generalized-smoke.json`,
`evidence/leech24-smoke.json`, `evidence/leech24-real-qualification.json`,
`evidence/leech24-native-runtime.json`, `formal/z3-results.json`, and
`formal/rocq-results.json`.

The checked-in runtime smoke uses one tiny random Qwen3 on one CPU input. In its
recorded environment, median forward latency was **2.676 ms** in streaming-memory
mode and **0.866 ms** in speed mode; encoded persistent state was **92,032 bytes**, while
state plus the speed cache was **239,488 bytes** versus **213,888 dense-state
bytes**. This illustrates the measured tradeoff, not a general speedup or memory
guarantee. The same file records substantial random-model logit error so
compression and top-1 agreement on seven tokens cannot be mistaken for accuracy
certification.

Current workspace evidence additionally includes **78 focused runner tests**
(including **19 generalized-planning tests**, **16 Leech tests**, and **6
larger-BF16 harness tests**), a complete repository run of **1388 passed and 72
explicitly skipped tests**, and all **7 static/symbolic stages** passing across
**359 Python files**. The raw logs for this final harness run are preserved as
`evidence/larger-bf16-focused-pytest.txt`,
`evidence/larger-bf16-full-pytest.txt`, and
`evidence/larger-bf16-static-verification.txt`; the earlier native logs remain
unchanged. The separate real-checkpoint evidence is scoped exactly as described
in `LEECH24.md`. Test counts are not large-model quality, hardware-speed, or
production-architecture certification. No larger trained checkpoint has been
acquired or qualified. Rocq results remain historical because jsCoq was not
reinstalled for this final run.

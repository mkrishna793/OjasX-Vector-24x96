# Current-model compatibility and generalization plan

Checked on **2026-11-04** against Hugging Face model cards/configurations and PyPI.
The installed and latest available PyPI Transformers version in this workspace is
**5.22.1**. No newer release was available to install.

## Official checkpoints versus executed runner tests

The runner tests construct tiny random models locally. They exercise architecture,
quantization, persistence, and forward contracts without downloading massive
weights. **An architecture-class roundtrip is not a real-checkpoint quality result.**

A separate Leech24 experiment does use the real trained
`HeathLoganCampbell/one-million-params` mixed-corpus TinyGPT checkpoint. It is not
one of the official HF checkpoints below and does not upgrade their evidence
level. On a pinned substitute byte corpus, two selected Leech tensors passed the
configured quality gates, saved 7.5486% of unique BF16 persistent state, and
reloaded from meta tensors without dense-source materialization. Sampled process
RSS did not decrease on this roughly 2 MB model. See `LEECH24.md` and
`evidence/leech24-real-qualification.json` for the exact revision, corpus,
selection, gates, RSS, and timing limits.

| Official HF checkpoint | Official config contract | Actually executed in the isolated runner | Remaining boundary |
| --- | --- | --- | --- |
| [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3_5`, `Qwen3_5ForConditionalGeneration`; native vision encoder and hybrid Gated DeltaNet/full attention | Tiny composite `Qwen3_5ForConditionalGeneration`: quantized save/meta-load/text-input forward with exact pre-save output equality | No 27B weights, processor/chat, image/video, multilingual quality, RAM, or generation benchmark |
| [`Qwen/Qwen3.8-2.4T-A95B`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | `qwen3_5_moe_text`, `Qwen3_5MoeForCausalLM`; hybrid attention and 512 routed experts in the official config | Tiny `Qwen3_5MoeForCausalLM` with quantized ordinary and fused-expert projections | No real checkpoint, 512-expert scale, FP8 variant, quality, or throughput test |
| [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | `gemma4`, `Gemma4ForConditionalGeneration`; composite text/vision config | Tiny composite Gemma4 text-input roundtrip; separate tiny Gemma4 text and MoE roundtrips | No official weights, processor, image/audio/video, chat, or quality test |
| [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) | `glm_moe_dsa`, `GlmMoeDsaForCausalLM`; sparse attention, MoE, FP8 source metadata | Tiny `GlmMoeDsaForCausalLM` with direct `.weight` compatibility and fused-expert roundtrip | Native FP8 loading/requantization and the 753B checkpoint are unverified |
| [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | `glm5_next`, `Glm5NextForConditionalGeneration`; hybrid linear/indexed attention and clamped expert gating | Tiny composite GLM5-Next with both tested attention types and clamped fused-expert roundtrip | No official weights, native FP8 path, multimodal inputs, quality, or speed test |
| [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | `deepseek_v4`, `DeepseekV4ForCausalLM`; grouped projections, compressed attention, clamped MoE, FP8/FP4 metadata | Tiny `DeepseekV4ForCausalLM`; unsupported grouped-linear subclass remains dense, other admitted tensors roundtrip | Native FP8/FP4, grouped-linear quantization, real weights, quality, and speed are unverified |
| [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek_v41`, `DeepseekV41ForCausalLM`; additional composite/text/vision and DSpark/engram fields | **Not executed**: Transformers 5.22.1 does not register `deepseek_v41` in this environment | Requires a registered HF implementation or a separately reviewed adapter; V4 must not be substituted silently |

Qwen3-Next also has a tiny quantized text/MoE roundtrip. Qwen3.8 is built on the
Qwen3.5 architectural foundation according to its
[official repository](https://github.com/QwenLM/Qwen3.8), but that statement does
not make every Qwen3.5/3.8 size, modality, or quantized artifact interchangeable.

## Safeguards implemented for generalization

- **Task-aware loading:** causal text configs use `AutoModelForCausalLM`; composite
  vision configs use `AutoModelForMultimodalLM`. The selected task is saved in the
  manifest and checked when loading.
- **Fail-closed tensor replacement:** exact `nn.Linear` instances are supported.
  Unknown `nn.Linear` subclasses remain dense because their forward/layout contract
  may differ. DeepSeek-V4's grouped linear is a concrete regression case.
- **Explicit HF adapters:** HF `Conv1D` and allowlisted fused-expert classes have
  separate contracts. Ordinary and clamped expert gating are distinguished in the
  module and manifest; a gate-contract mismatch is rejected.
- **Direct-weight compatibility:** HF code that reads `module.weight` receives a
  decoded tensor with the original source dtype and source orientation. Memory
  profile still does not retain that tensor.
- **Dense fallback and tied-weight safety:** rejected projections remain dense;
  shared/tied weights are preserved by default.
- **Persistence checks:** schema, shapes, source dtype, task, duplicate/missing/
  unexpected shard tensors, and exact pre-save versus reloaded outputs are tested.
- **Generalized planning:** tensor IR, per-role protection, multi-candidate local
  calibration, hard byte constraints, and independent end-to-end quality gates are
  implemented in `GENERALIZED_VQ.md`.
- **Low-residency execution:** memory mode streams row chunks and active-expert
  matrices instead of materializing a complete dense weight on every forward.

## What “robust and generalized VQ” still requires

There is no quantization setting that is *perfect* for every model. Lossy VQ
discards information. Exact packing, non-expanding residual selection, and dense
fallback can be guaranteed; universal model accuracy cannot. A credible system
should instead optimize and validate the following closed loop.

### 1. Checkpoint admission before quantization

Record the exact model ID, immutable revision, config hash, architecture, model
task, Transformers version, source dtype, existing `quantization_config`, and
processor/tokenizer hashes. Reject unknown tasks and unsupported native FP4/FP8
layouts explicitly. Do not silently requantize an already-quantized checkpoint as
if it were the original BF16 model.

### 2. Tensor-role policies, not only module names

Classify attention, ordinary FFN, shared/routed experts, routers, embeddings,
output heads, vision/audio encoders, projectors, indexers, recurrent state
parameters, norms, MTP/DSpark heads, and unknown custom tensors. Each role needs
an explicit quantize/fallback policy and coverage report. Unknown roles should
remain dense by default.

### 3. Output-aware mixed-precision admission

Per-weight Frobenius error is not enough. Use representative calibration data to
measure each candidate's effect on hidden outputs and logits. Allocate residual
stages/code widths/block sizes under a **global storage and model-quality budget**.
Sensitive routers, indexers, projectors, embeddings, or output heads may need more
stages or dense retention. Every candidate must remain reversible to dense fallback.

### 4. Real quality gates

Evaluate reference and candidate on held-out multilingual text, task benchmarks,
long context, greedy chat continuations, and the actual instruction template. For
multimodal models, include image/audio/video suites and processor-level chat
roundtrips. Report per-sample/per-language regressions, not only one aggregate
perplexity. A conversion should fail its declared gate rather than ship silently.

### 5. Runtime and memory qualification

Benchmark prefill and autoregressive decode separately on each target backend.
Measure end-to-end tokens/s, time to first token, peak RSS/VRAM, decoded-cache
bytes, temporary decode workspace, and KV cache. Persistent weight compression
alone does not prove lower peak RAM or faster inference.

### 6. Reproducible qualification matrix

Start with the smallest official BF16 checkpoint for each exact architecture and
record its revision. Then test larger variants, MoE scale, multilingual chat,
multimodal inputs, native pre-quantized artifacts, and optimized backends. Keep
architecture-class smoke tests, real-checkpoint tests, and performance claims as
separate evidence levels.

## Next implementation priorities

1. Processor-aware conversion/chat for Qwen3.8, Gemma4, and GLM5-Next, followed by
   actual image/video input roundtrips.
2. Explicit BF16-source admission and native FP8/FP4 preservation/dequantization
   policies with executed checkpoints.
3. Extend the implemented local calibration/Pareto planner with cross-layer
   interaction search and quality-guided refinement after end-to-end failures.
4. Expand the scoped real TinyGPT evaluation to official small-checkpoint
   multilingual and task suites; then larger-model prefill/decode and actual
   peak-memory measurements on qualified hardware.
5. DeepSeek-V4.1 support only after its exact implementation is available and its
   additional tensors have reviewed contracts.
6. Optimized VQ decode/gather kernels and separate Unsloth/LM Studio export work.

# Generalized VQ: tensor IR, planning, calibration, gates, and low-RAM execution

This document describes the implemented generalized workflow. It does not claim
universal accuracy, universal architecture support, or measured production RAM
savings on a large checkpoint.

## 1. Five connected improvements

### 1. Model-independent tensor IR

`ir.py` defines `TensorSpec`, `QuantizationCandidate`, and `QuantizationPlan`.
Every represented tensor has a name, matrix/logical shape, source dtype and bytes,
role, layout, decode group, shared-weight group, source-quantization state, and an
explicit protection reason when it cannot be quantized.

`adapters.py` discovers exact `nn.Linear`, HF `Conv1D`, allowlisted fused expert
banks, embeddings, and unknown `nn.Linear` subclasses. Layout safety and role
policy are separate decisions. A familiar attribute name never makes an unknown
custom module compatible.

Default role policy:

| Role/layout | Default generalized policy |
| --- | --- |
| Attention, feed-forward, routed/shared experts | Candidates allowed; calibration required by default |
| Recurrent/hybrid projections recognized by the adapter | Candidates allowed; calibration required by default |
| Router and shared-expert gate | Dense protection |
| Embedding and output head | Dense protection |
| Vision/audio/multimodal projector | Dense protection until explicitly reviewed and calibrated |
| Unknown `nn.Linear` subclass | Dense, unsupported contract |
| Shared/tied weight | Dense protection |
| Native FP8/FP4/other pre-quantized source | Dense unless requantization is explicitly opted in |
| Uncalled or uncalibrated tensor | Dense under the default planner |

### 2. Multiple candidates and a global planner

`planner.py` generates several residual-VQ candidates per tensor plus a dense
fallback. Learned candidates use actual packed codes, codebooks, and scales;
Leech candidates use packed 18-bit shell indices, radial gains, presence bitmaps,
and row scales. The two codec families can compete in one candidate list. Nominal
payload bits per weight are not used as storage evidence. See `LEECH24.md` for the
24D construction and its exact byte formula.

Each fused expert bank forms one decode group. Its two projections can be dense,
VQ, or mixed. The multiple-choice Pareto search minimizes additive squared local
error among complete plans satisfying all hard memory and per-tensor constraints.
It uses maximum savings as a secondary objective. If the deterministic frontier
beam truncates the search, the plan says `candidate_optimality: bounded`; otherwise
it says `exact_over_generated_candidates`. Neither label means globally optimal
model accuracy.

### 3. Bounded activation calibration

`calibration.py` runs supplied inputs once with forward hooks and stores a bounded
number of activation rows. Candidate outputs are compared with the original local
module outputs. Uncalled modules remain uncalibrated and fail dense by default.
Fused projections are measured independently while the other projection remains
dense.

For candidate output \(\hat Y\) and reference output \(Y\), aggregated over all
captured invocations, the local error is:

\[
e_t = \sqrt{\frac{\sum (\hat Y-Y)^2}{\sum Y^2 + 10^{-30}}}.
\]

The sums are accumulated before taking the ratio; this is not an average of
per-batch ratios. The planner minimizes \(\sum_t e_t^2\) subject to its hard
constraints. This additive value is a **local heuristic**, not a bound on final
NLL, KL, task accuracy, or interacting cross-layer errors.

### 4. Independent end-to-end quality gates

`gates.py` compares a separate reference model and the candidate on supplied
validation inputs. Supported gates are:

- global relative logits L2 error;
- maximum absolute logits error;
- reference-to-candidate KL;
- top-1 agreement;
- next-token NLL delta when causal labels are available.

Nonfinite candidate outputs fail. Requesting an NLL gate without supervised labels
also fails. At least one threshold is required. A `GeneralizedQuantizationResult`
is explicitly `passed`, `failed`, or `unvalidated`; `require_accepted()` refuses an
unvalidated or failed result.

Passing these gates is evidence only for the supplied inputs, position sampling,
and thresholds. Held-out multilingual, task, long-context, chat, and modality-
specific evaluations remain separate requirements.

### 5. Low-residency execution and auditable persistence

Memory mode now uses **streaming decode**:

- ordinary learned-VQ and Leech Linear/Conv1D weights decode in bounded output-row
  chunks;
- packed learned codes and 18-bit Leech indices are unpacked only for the required,
  possibly unaligned bit range, with adaptive presence restoring inactive vectors;
- fused MoE banks decode only the active expert's matrices inside the expert loop;
- no full dense weight is retained between forwards.

Leech rows default to an `auto` native-CPU/reference backend; explicit `python`,
`native`, and disabled-native fallback modes are tested. Native preserves bounded
rows and exact reference outputs but does not change the persistent plan or prove
lower whole-process RSS. See `LEECH24_NATIVE.md`.

Speed mode decodes and caches complete matrices. It is not the low-RAM profile.
Inference tensors without mutation-version tracking bypass the speed cache rather
than risking stale weights.

A generalized plan can be saved as `vq_plan.json` with the checkpoint. Loading uses
its saved streaming row-block value unless explicitly overridden. `inspect` prints
the plan, including per-role coverage, selected candidates, calibration coverage,
memory predictions, and any executed end-to-end gates.

## 2. Storage and RAM mathematics

### Exact encoded persistent bytes

For a matrix with \(r\) rows, \(c\) columns, block size \(d\), code width \(k\),
\(S\) residual stages, and \(K=2^k\) centroids:

\[
V = r\left\lceil\frac{c}{d}\right\rceil
\]

is the number of vectors. The implementation's exact encoded weight bytes are:

\[
b_{\mathrm{VQ}} =
S\left\lceil\frac{Vk}{8}\right\rceil + 4SKd + 4r.
\]

The terms are packed codes, float32 codebooks, and float32 row scales. Bias and all
unquantized state are additional. This formula is directly tested for every
supported code width.

The measured rate is \(8b_{\mathrm{VQ}}/(rc)\), not merely \(Sk/d\). Small tensors
can fail compression because codebooks dominate; the planner then retains dense
or chooses another candidate.

Leech uses the same actual-byte rule but a different payload: 18-bit direction
indices, packed radial gain codes, optional presence bits, small float32 gain
tables, and float32 row scales. Its complete formula and runtime-table accounting
are in `LEECH24.md`. Shared decoder tables are reported separately as runtime
overhead rather than silently treated as free model state.

### Exact persistent-state prediction

Let \(B_0\) be the dense model's unique persistent state-tensor bytes, counting
exact tied aliases once. For selected encoded weights \(Q\):

\[
B(p) = B_0 - \sum_{t\in Q}
\left(b_{t,\mathrm{original}}-b_{t,\mathrm{encoded}}\right).
\]

This includes biases, embeddings, normalization parameters, and other dense state
through \(B_0\). `apply_quantization_plan` measures the resulting state and raises
if it differs from this prediction.

These are tensor-inventory bytes, not tokenizer/metadata files, RSS, allocator
high-water marks, activations, or KV cache.

### Streaming temporary decode output

With row-block bound \(R\), an ordinary projection's decoded output chunk is at most:

\[
D_t = \min(r,R)\,c\,s_a,
\]

where \(s_a\) is the activation/cache dtype size used by the estimate. For a fused
bank, the estimate covers one active expert's selected projections rather than all
experts. Padded float32 intermediates, codebook gathers, inputs, outputs, and other
workspace remain additional.

Therefore streaming reduces dense-weight workspace; it does not by itself prove a
whole-process peak-RAM bound.

### Why speed mode increases state-plus-cache bytes

Let \(C=\sum_{t\in Q} b_{t,\mathrm{original}}\) when the decoded cache uses the same
dtype and shape as the original weights. Then:

\[
B(p)+C
= B_0 + \sum_{t\in Q} b_{t,\mathrm{encoded}}
> B_0.
\]

Thus a full same-dtype decoded cache makes total state-plus-cache **larger than
dense** by exactly the encoded payloads. Use `max_state_plus_cache_bytes` when
planning speed mode. Use memory/streaming mode when weight residency is the goal.

## 3. API example

```python
import copy
import torch
from ojasx_vector_quantization_runner import (
    PlanningConstraints,
    QualityGateThresholds,
    QuantizationConfig,
    default_candidate_configs,
    default_leech_candidate_configs,
    quantize_model_generalized,
)
from ojasx_vector_quantization_runner.hf import save_quantized_pretrained

reference = copy.deepcopy(model).eval()  # Required only for direct end-to-end gates.
calibration_inputs = [
    {"input_ids": torch.tensor([[1, 2, 3, 4]])},
    {"input_ids": torch.tensor([[5, 6, 7, 8]])},
]
validation_inputs = [
    {"input_ids": torch.tensor([[9, 10, 11, 12]])},
]

search = QuantizationConfig(
    max_relative_error=None,
    max_codebook_samples=1024,
    kmeans_iterations=8,
    require_compression=True,
)
constraints = PlanningConstraints(
    target_state_ratio=0.60,             # Persistent tensor-state target.
    max_tensor_estimated_error=0.10,     # Example only; tune for the real task.
    runtime_profile="memory",
    decode_row_block_size=32,
)
result = quantize_model_generalized(
    model,
    constraints=constraints,
    candidate_configs=(
        default_candidate_configs(search) + default_leech_candidate_configs()
    ),
    calibration_inputs=calibration_inputs,
    reference_model=reference,
    quality_thresholds=QualityGateThresholds(
        max_relative_logits_l2_error=0.05,
        max_mean_nll_delta=0.02,
        min_top1_agreement=0.95,
    ),
    validation_inputs=validation_inputs,
)
result.require_accepted()  # Raises for failed or missing end-to-end validation.
save_quantized_pretrained(
    result.model,
    "/tmp/generalized-vq",
    report=result.report,
    plan=result.plan,
)
```

The numeric thresholds above are illustrations, not recommended universal values.
Direct validation keeps a reference and candidate model resident. For large-model
conversion, use a reference-output/held-out evaluation strategy that fits the
available conversion machine, then benchmark the loaded candidate independently.

Weight-only planning requires the explicit `allow_weight_only=True` option. It is a
weaker fallback, not an equivalent to activation calibration.

## 4. Remaining work required for production claims

1. Expand the scoped real TinyGPT Leech qualification in `LEECH24.md` to revised
   official checkpoints and held-out multilingual/task suites.
2. Add cross-layer interaction/search and quality-guided refinement when an
   end-to-end gate fails.
3. Measure actual peak RSS/VRAM, prefill, decode, KV cache, and allocator workspace
   on larger models and each deployment backend; the tiny real probe did not show
   a whole-process RSS saving.
4. Add processor-aware multimodal calibration and validation.
5. Implement reviewed native FP8/FP4 source policies and optimized streaming VQ
   kernels. The generic PyTorch streaming path favors memory, not guaranteed speed.

Tiny synthetic tests establish the implemented contracts only. The scoped real
TinyGPT result adds one trained-checkpoint gate/runtime measurement, but neither
evidence level proves that every real model can meet a chosen RAM target while
preserving task accuracy.

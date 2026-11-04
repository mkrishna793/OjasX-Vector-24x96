# Leech24 residual vector quantization: implementation and qualification

This runner now includes a genuine 24-dimensional Leech-lattice **minimum-shell**
vector codec alongside learned residual VQ and dense fallback. It is an
inference-only research implementation. The five requested stages below are
implemented and executed, but the measurements do not establish universal
accuracy, compression, speed, or process-RAM superiority.

The implementation is independently written in PyTorch. Its mathematical
references are Conway--Sloane's Golay/Leech construction, van der Ouderaa et al.,
[arXiv:2603.11021](https://arxiv.org/abs/2603.11021), and the MIT/Apache LLVQ
reference at `ddecad3196f4f7fc872e03ca5c0499deb12c47e8`. No novelty or general
Leech closest-vector-problem solver is claimed.

## 1. Correct lattice prototype and finite-code non-expansion

### Construction and searched set

`leech24.py` constructs the extended binary `[24,12,8]` Golay code and verifies
all 4,096 words. Its weight distribution is:

```text
1 + 759*x^8 + 2576*x^12 + 759*x^16 + x^24
```

In the integer embedding `sqrt(8) * Lambda_24`, coordinates have uniform parity.
For an even point, `(x/2) mod 2` must be Golay and `sum(x) == 0 mod 8`. For an odd
point, `((x-1)/2) mod 2` must be Golay and `sum(x) == 4 mod 8`. The lattice point
is the integer vector divided by `sqrt(8)`.

The quantizer searches the origin and the 196,560 minimum-shell points, not the
infinite lattice. The integer shell has squared norm 32; `decode_unit_vectors`
divides it by `sqrt(32)` to obtain unit directional codewords. Its exact classes
are:

| Index class | Count | Construction |
| --- | ---: | --- |
| Origin | 1 | Index zero |
| Even octad | 97,152 | 759 Golay octads with 128 even-sign patterns |
| Two-axis | 1,104 | 276 coordinate pairs with four `+/-4` sign choices |
| Odd Golay | 98,304 | 4,096 Golay words with 24 distinguished coordinates |

All **196,561** indices are decoded in the permanent test and checked for lattice
membership, uniqueness, and shell membership. Algebraic nearest-shell search is
compared against explicit brute-force dot products. The smoke records a maximum
absolute dot discrepancy of **3.55e-15** on its float64 probe.

### Orthogonal transform and residual feedback

Rows are RMS-normalized, padded to 24D vectors, and transformed by a shared,
seeded product of Householder reflections. The inverse transform is applied after
residual reconstruction. This is an orthogonal transform; an odd number of
reflections need not be a determinant-one rotation.

For unit shell direction `u`, radial gain `g`, and residual `r`, the squared-error
improvement is:

```text
2*g*<r,u> - g^2
```

The encoder maximizes this expression over the exact shell directions and the
small shared gain table. The origin explicitly competes with every shape--gain
codeword. A non-improving codeword is replaced by the origin, so a stage cannot
increase the exact finite-code squared residual. Later adaptive stages can also
leave a vector inactive.

This argument is specific to the implemented finite code and origin competition.
It is **not** an assertion that a modulo, range-limited, or finite-shell codec
inherits every property of an unbounded nearest-lattice decoder. No such untested
variant is used here.

## 2. Compact, range-decodable indices and exact byte accounting

There are 196,561 representable points including the origin, so each direction
uses an 18-bit index. `pack_indices18` and `unpack_indices18_range` implement exact
packing and bounded range decoding. Packing the complete index domain takes
**442,263 bytes**, including final-byte padding, and roundtrips exactly.

For `r` matrix rows, `c` columns, and `V = r*ceil(c/24)` vectors, let `A_j` be the
active count in stage `j`, `g` the gain-code width (`0`, `1`, or `2`), and `S` the
stored stage count. Encoded weight bytes are exactly:

```text
sum_j ceil(18*A_j/8) + ceil(g*A_j/8)
+ (adaptive ? (S-1)*ceil(V/8) : 0)
+ 4*S*2^g
+ 4*r
```

The terms are direction indices, gain codes, later-stage presence bitmaps,
float32 per-stage gain tables, and float32 row scales. Bias, unquantized tensors,
file containers, manifests, and runtime tables are additional. The planner uses
these measured encoded bytes, not `18/24` or another nominal payload rate.

The CPU decoder's shared integer tables measure **207,136 bytes** in the synthetic
smoke. Encoder-only floating-point search tables are initialized separately and
are not needed by a fresh inference decoder. Shared tables and algorithmically
generated Householder vectors are runtime overhead, not free model state.

## 3. Rotation/adaptive ablations and learned-VQ comparison

`leech24_smoke.py` uses the same seeded `32 x 96` float32 matrix for every row and
counts actual encoded tensors for both codec families. For each Leech candidate,
it selects the lowest-error generated learned-VQ candidate at or below the same
actual-byte budget and separately reports exact-byte matches. It is a finite
candidate comparison, not a globally optimal rate--distortion frontier.

Representative unrotated adaptive rows are:

| Codec/configuration | Actual bytes | Measured bits/weight | Relative error |
| --- | ---: | ---: | ---: |
| Leech, 0 gain bits, 3 stages | 1,024 | 2.667 | 0.592819 |
| Learned VQ, 8D/2-bit, 4 stages | 1,024 | 2.667 | 0.635489 |
| Leech, 1 gain bit, 3 stages | 1,096 | 2.854 | 0.252070 |
| Best generated learned VQ within 1,096 bytes | 1,024 | 2.667 | 0.635489 |
| Leech, 2 gain bits, 3 stages | 1,168 | 3.042 | 0.249128 |
| Best generated learned VQ within 1,168 bytes | 1,024 | 2.667 | 0.635489 |

At exactly 1,168 bytes, the generated learned 24D/1-bit/five-stage candidate has
relative error 0.918056. These numbers describe one synthetic matrix only.

The ablations prevent overclaiming: two Householder reflections slightly worsen
every paired result on this matrix, despite identical storage. For the one-bit
unrotated case, adaptive and non-adaptive residuals have identical error, while
adaptive presence metadata increases storage from **1,064 to 1,096 bytes** because
all later-stage vectors remain active. Rotation and adaptivity are planner
options, not assumed improvements.

## 4. Tensor IR, mixed candidates, global planning, and persistence

`LeechQuantizationConfig` and `LeechQuantizedWeight` are integrated through:

- `ir.py`: typed learned/Leech/dense candidates and codec-specific configuration;
- `planner.py`: mixed candidate generation, measured-byte Pareto planning, and
  dense fallback;
- `calibration.py`: the same bounded local-output calibration for either codec;
- `generalized.py`: calibration, global application, and independent quality gates;
- `module.py`: mixed-codec buffers, exact manifests, bounded streaming, speed
  caches, and codec-aware constructors during meta reload;
- `hf.py`: existing reviewed architecture and sharded-persistence contracts.

A permanent regression puts learned residual VQ and Leech candidates for the same
tensors into one global planner and checks dense alternatives and post-apply byte
prediction. Existing role protections remain unchanged: routers, tied weights,
unknown custom linears, unsupported layouts, and other protected tensors remain
dense by default. Leech does not bypass architecture admission or quality gates.

Use a mixed search explicitly:

```python
from ojasx_vector_quantization_runner import (
    QuantizationConfig,
    default_candidate_configs,
    default_leech_candidate_configs,
)

configs = (
    default_candidate_configs(QuantizationConfig(max_relative_error=None))
    + default_leech_candidate_configs()
)
# Pass configs as candidate_configs=configs to quantize_model_generalized.
```

The real qualification below deliberately used `default_leech_candidate_configs`
plus dense fallback to isolate Leech behavior; the mixed-planner path is covered
by the separate regression.

## 5. Streaming and real trained-checkpoint qualification

### Protocol

`leech24_qualify.py` uses the real trained
`HeathLoganCampbell/one-million-params` release `v0.1.0` checkpoint
`tiny-gpt-1m-mixed.pt`, architecture commit
`170de2c5760cfdea3f375f6e5aaf2a8c45933850`, step 19,999. The source SHA-256 is
pinned as `05b9bc603f9a826867dadc67302021d3dd77248af6ef42b3682844759d974228`.
It has 1,006,976 unique parameters and is explicitly cast from FP32 to BF16.

Validation uses three pinned byte windows from Tiny Shakespeare at
`karpathy/char-rnn@6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`, with SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
This is a substitute corpus, **not** the checkpoint's original mixed-corpus
validation split. Only `blocks.0.*` was eligible, with target persistent-state
ratio 0.95. All other tensors remained dense.

The planner selected three-stage, one-gain-bit, two-reflection adaptive Leech
candidates for:

- `blocks.0.attn.qkv`: 98,304 to 18,543 weight bytes;
- `blocks.0.mlp.w2`: 86,016 to 13,752 weight bytes.

### Quality and persistent state

| Measurement | Result |
| --- | ---: |
| Supervised validation positions | 381 |
| Relative logits L2 error | 0.0667559510 |
| Maximum absolute logits error | 3.34375 |
| Mean reference-to-candidate KL | 0.0245918452 |
| Top-1 agreement | 0.9186351706 |
| Mean NLL delta | +0.0272461498 |
| Dense BF16 unique persistent state | 2,013,952 bytes |
| Reloaded quantized persistent state | 1,861,927 bytes |
| Persistent saving | 152,025 bytes (7.5486%) |
| Encoded artifact files, including metadata | 1,916,144 bytes |

All configured gates passed. Predicted and reloaded persistent bytes agree
exactly. The encoded model was reconstructed on BF16 meta tensors and loaded
without a dense source checkpoint. This is a small real-checkpoint gate result,
not broad pretrained-quality certification.

### Actual fresh-process runtime results

In this original Python-backend qualification, each runtime worker was a fresh
CPython process. Sampled phase RSS and `ru_maxrss` are recorded in the evidence.
Four greedy tokens recomputed context without KV caching; all three workers
generated `[111, 110, 103, 32]`.

| Measurement | Dense | Leech memory | Leech speed cache |
| --- | ---: | ---: | ---: |
| Persistent state after run | 2,013,952 B | 1,861,927 B | 1,861,927 B |
| Retained decoded-cache bytes | 0 B | 0 B | 184,320 B |
| State plus decoded cache | 2,013,952 B | 1,861,927 B | 2,046,247 B |
| Sampled prefill peak RSS | 726,138,880 B | 728,653,824 B | 727,138,304 B |
| Sampled decode peak RSS | 728,723,456 B | 731,774,976 B | 731,140,096 B |
| Median seconds/recomputed token | 0.018039 | 0.092994 | 0.018299 |
| Decode time relative to dense | 1.000x | 5.155x | 1.014x |

**The tiny-model process-RAM objective is not demonstrated here.** Quantized
memory mode's sampled decode RSS is 3,051,520 bytes higher than dense, because
PyTorch/library and decode overhead dominate a roughly 2 MB model. Memory mode
does avoid retaining any full decoded model weight and reduces persistent state,
but its generic PyTorch row decoder is slower. Speed mode nearly restores this
tiny decode timing by retaining full decoded matrices; state plus cache is
32,295 bytes larger than dense, exactly the encoded payload size. These four-token
timings are not throughput claims. Larger-model RAM and optimized-kernel tests
remain necessary.

### Native CPU follow-up

The subsequent `LEECH24_NATIVE.md` implementation keeps the same artifact and
codec but replaces the row-reconstruction bottleneck with an optional native CPU
backend. On its final matched worker set, native memory decode is **3.800x** the
Python Leech speed and **1.450x dense time**, with bitwise Python/native prefill
hashes and zero decoded cache. It initializes no Python lattice tables and uses a
21,628-byte C++ table object. Sampled RSS remains above dense on this tiny model,
so the original RAM limitation still stands. See
`evidence/leech24-native-runtime.json` for the separate current measurements.

## Reproduce

From the repository root, with the runner's Python/HF dependencies installed:

```bash
python -m pytest -q \
  tests/test_ojasx_vector_quantization_runner.py \
  tests/test_ojasx_vector_quantization_generalized.py \
  tests/test_ojasx_vector_quantization_leech24.py
python -m ojasx_vector_quantization_runner.leech24_smoke
```

Download the pinned real inputs through the GitHub contents API and verify them:

```bash
WORK=/tmp/leech24-qualification
mkdir -p "$WORK"
gh api 'repos/HeathLoganCampbell/one-million-params/contents/models/tiny-gpt-1m-mixed.pt?ref=170de2c5760cfdea3f375f6e5aaf2a8c45933850' \
  -H 'Accept: application/vnd.github.raw+json' > "$WORK/tiny-gpt-1m-mixed.pt"
gh api 'repos/karpathy/char-rnn/contents/data/tinyshakespeare/input.txt?ref=6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e' \
  -H 'Accept: application/vnd.github.raw+json' > "$WORK/tinyshakespeare-input.txt"
sha256sum "$WORK/tiny-gpt-1m-mixed.pt" "$WORK/tinyshakespeare-input.txt"
```

Then run conversion, gates, save/meta-reload, and all three fresh-process workers:

```bash
python -m ojasx_vector_quantization_runner.leech24_qualify qualify \
  --source "$WORK/tiny-gpt-1m-mixed.pt" \
  --data "$WORK/tinyshakespeare-input.txt" \
  --workspace "$WORK/run" \
  --evidence ojasx_vector_quantization_runner/evidence/leech24-real-qualification.json \
  --include 'blocks.0.*' --target-state-ratio 0.95
```

Machine-readable evidence is in `evidence/leech24-smoke.json` and
`evidence/leech24-real-qualification.json`. The generated model and downloaded
checkpoint remain scratch artifacts and are not committed to Git.

# Larger trained-BF16 checkpoint qualification

`larger_bf16_qualify.py` is a checkpoint-driven harness for local, stock
Hugging Face causal language models. It connects calibrated RVQ/Leech24
candidate generation, global memory planning, independent quality gates,
encoded save/meta-reload, and fresh-process runtime measurements.

**No larger-checkpoint pass, compression ratio, accuracy result, or dense-RAM
reduction is currently claimed.** The checked-in tests use a tiny synthetic
fixture only to test the harness. A real result must name the source repository,
immutable revision, license, training declaration, exact source-file hashes, and
resulting evidence file.

## Source and data contract

The source directory must contain a usable config, tokenizer, and one or more
safetensors weight shards. The parent command audits tensor dtypes without
loading the full dense state through safetensors metadata. Floating-point tensor
bytes must meet `--min-source-bf16-fraction` (default **0.95**). The loaded dense
reference and conversion source are explicitly BF16; FP8/FP4 or arbitrary
quantized checkpoints are not silently promoted into a BF16 qualification.

A separate provenance JSON file declares:

- `repository`, immutable `revision`, `license`, and a concrete `training`
  description;
- `trained_checkpoint: true` and `bf16_checkpoint: true`;
- unique parameter count, counting exact tied/storage aliases once;
- every regular source file's exact byte size and SHA-256;
- optionally `model_type` and a local Transformers `tokenizer_class` override.

The harness recomputes hashes, sizes, model type, unique parameter count, and
weight-shard identities. It rejects unclear licenses such as `other`,
`unknown`, or `to-be-decided`; relative/absolute path escapes; undeclared
weight files; missing files; and identity mismatches. These checks audit the
supplied declaration; they do not independently prove that the training
narrative is true. Official upstream hashes should be recorded when available.

Calibration and validation text must be different files with independently
pinned SHA-256 values. Token windows are deterministic, spread across each
file, and never synthesized by repeating a short file. Calibration is used for
candidate selection. Validation is reserved for dense-reference and quantized
quality gates. This is inference-only conversion, not fine-tuning.

Raw checkpoint formats such as the llama2.c `.bin` files are not directly
accepted by this generic HF command. They require a separate, tested converter
that preserves the original payload hash and documents the exact tensor mapping.
A converter or an FP32-to-BF16 cast is not evidence of training, quality, or RAM
savings by itself.

## Running a qualification

From the repository root, with local files already acquired:

```bash
python -m ojasx_vector_quantization_runner.larger_bf16_qualify qualify \
  --source /path/to/pinned-hf-model \
  --provenance /path/to/provenance.json \
  --calibration-data /path/to/calibration.txt \
  --calibration-sha256 CALIBRATION_SHA256 \
  --validation-data /path/to/validation.txt \
  --validation-sha256 VALIDATION_SHA256 \
  --workspace /path/to/run \
  --evidence /path/to/evidence.json \
  --codec mixed \
  --target-state-ratio 0.90 \
  --max-relative-logits 0.10 \
  --max-kl 0.05 \
  --min-top1 0.90 \
  --max-nll-delta 0.05 \
  --max-dense-reference-nll 6.0 \
  --runtime-repeats 3 \
  --min-rss-savings-bytes 1048576 \
  --min-rss-savings-fraction 0.01
```

Thresholds are experiment-specific acceptance criteria, not claimed safe
universal defaults. Use `--help` for candidate, calibration, activation, and
runtime controls. The command is local-only and does not download model files.
Custom remote-code architectures need a reviewed adapter/source implementation;
`trust_remote_code` is not enabled by this harness.

## What a full pass requires

Unless its corresponding explicit smoke-test opt-out is supplied, acceptance
requires all of the following:

1. At least **25,000,000 unique parameters**, at least **16,000,000 bytes** of
   quantizable weight bytes, and at least **1,000,000 bytes** of measured
   persistent-state savings.
2. Successful calibrated conversion and an explicit BF16 dense reference, with
   dense fallback wherever a candidate is inadmissible.
3. Dense-reference NLL screening and independent validation gates for relative
   logits error, mean KL, top-1 agreement, and NLL delta.
4. Persistent encoded save, meta-only encoded reload without dense-source
   materialization, exact predicted/reloaded byte equality, and repeated
   quality gates after reload.
5. Alternating fresh dense/quantized subprocesses. Quantized workers receive
   only the encoded artifact and runtime token IDs, never the dense source path.
6. Deterministic prefill logits and generated tokens within each mode, exact
   persistent-byte accounting, and zero decoded-cache bytes in memory mode.
7. Strictly positive repeated median sampled-RSS savings in **load**, **prefill**,
   and **decode without a KV cache**, meeting both configured absolute and
   fractional minima. A zero saving cannot pass even if a caller configures a
   zero threshold.
8. Dense unique persistent weight bytes exceeding the positive non-weight
   allocation increment attributed to model loading.

The evidence separately reports whether dense weights exceed the **entire**
pre-model import baseline. That stronger whole-process ratio is descriptive,
not hidden inside the incremental-load gate. `--require-nonoverlapping-rss`
can additionally require every dense repetition's phase RSS to exceed every
quantized repetition's phase RSS; it is not enabled by default.

The runtime monitor combines sampled `/proc/self/status` RSS with process
`ru_maxrss` high-water marks and reports them separately. These are measurements
of the executed environment, not universal allocator, GPU, throughput, or
latency guarantees. Decode deliberately recomputes without a KV cache and does
not measure normal cached generation. Speedup is not an acceptance criterion.

## Status meanings

- `passed`: every declared quality, persistence, runtime, and RAM gate passed.
- `failed`: the evidence records one or more failed gates; no RAM claim is made.
- `passed_without_full_ram_qualification`: explicit small-test opt-outs disabled
  the scale and/or RSS acceptance requirements. This is an integration result,
  not a full larger-model RAM qualification.

`--no-require-rss-reduction` and `--no-require-weight-dominated-load` are the
CLI's explicit opt-outs (the Python arguments are
`require_rss_reduction=False` and `require_weight_dominated_load=False`). They
exist for synthetic integration tests and must not be used to relabel a result
as a full RAM pass.

## Focused harness verification

```bash
python -m pytest -q tests/test_ojasx_vector_quantization_larger_bf16.py
```

These tests cover provenance/hash rejection, BF16 dtype auditing, deterministic
token windows, runtime aggregation and zero-savings rejection, worker
reproducibility, complete tiny save/meta-reload execution, and independent
prefill-hash verification. They do not substitute for a trained larger-model
run.

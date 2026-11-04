"""Larger-BF16 qualification harness tests; no larger-model result is claimed."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ojasx_vector_quantization_runner.hf import load_quantized_pretrained
from ojasx_vector_quantization_runner.larger_bf16_qualify import (
    _sha256,
    _tensor_sha256,
    _token_windows,
    _unique_parameter_count,
    qualify,
    summarize_runtime_workers,
    validate_source_provenance,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_fixture(root: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    source = root / "source"
    source.mkdir(parents=True)
    torch.manual_seed(123)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=48,
        intermediate_size=72,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    unique_parameters = _unique_parameter_count(model)
    model.save_pretrained(source, safe_serialization=True)
    words = [
        "<unk>", "<bos>", "<eos>", "hello", "world", "tiny", "stories",
        "are", "useful", "for", "testing", "quality", "and", "memory",
        "once", "upon", "a", "time", "there", "was", "small", "child",
        "who", "liked", "books",
    ]
    vocabulary = {word: index for index, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(source)
    # Keep the local generic tokenizer explicit in this synthetic fixture.
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    tokenizer_config["tokenizer_class"] = "TokenizersBackend"
    _write_json(source / "tokenizer_config.json", tokenizer_config)
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in source.iterdir()
        if path.is_file()
    }
    declaration = {
        "schema_version": 1,
        "repository": "synthetic/test-only",
        "revision": "fixture-v1",
        "license": "MIT",
        "training": "synthetic schema fixture, not a trained-model or quality claim",
        "trained_checkpoint": True,
        "bf16_checkpoint": True,
        "unique_parameters": unique_parameters,
        "model_type": "qwen2",
        "tokenizer_class": "TokenizersBackend",
        "files": files,
    }
    provenance = root / "provenance.json"
    _write_json(provenance, declaration)
    calibration = root / "calibration.txt"
    calibration.write_text(
        "hello world tiny stories are useful for testing quality and memory "
        "once upon a time there was a small child who liked books"
    )
    validation = root / "validation.txt"
    validation.write_text(
        "once upon a time a small child liked tiny books and useful stories "
        "for quality testing"
    )
    return source, provenance, calibration, validation


def test_source_provenance_hashes_license_and_bf16_audit(tmp_path):
    source, provenance, _, _ = _source_fixture(tmp_path)
    report = validate_source_provenance(source, provenance)
    assert report["dtype_audit"]["bf16_fraction_of_floating_tensor_bytes"] == 1.0
    assert report["verified_files"]["model.safetensors"]["bytes"] > 0
    declaration = json.loads(provenance.read_text())
    missing_count = copy.deepcopy(declaration)
    del missing_count["unique_parameters"]
    missing_count_path = tmp_path / "missing-count.json"
    _write_json(missing_count_path, missing_count)
    with pytest.raises(ValueError, match="unique_parameters"):
        validate_source_provenance(source, missing_count_path)
    unclear = copy.deepcopy(declaration)
    unclear["license"] = "to-be-decided"
    unclear_path = tmp_path / "unclear.json"
    _write_json(unclear_path, unclear)
    with pytest.raises(ValueError, match="license"):
        validate_source_provenance(source, unclear_path)
    (source / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source_provenance(source, provenance)


def test_nested_weight_shards_and_all_source_files_are_hash_pinned(tmp_path):
    from safetensors import safe_open

    source, provenance, _, _ = _source_fixture(tmp_path)
    shard = source / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    nested = source / "weights" / shard.name
    nested.parent.mkdir()
    shard.replace(nested)
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {"total_size": nested.stat().st_size},
            "weight_map": {key: "weights/model.safetensors" for key in keys},
        },
    )
    declaration = json.loads(provenance.read_text())
    declaration["files"] = {
        path.relative_to(source).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source.rglob("*")
        if path.is_file()
    }
    _write_json(provenance, declaration)
    report = validate_source_provenance(source, provenance)
    assert report["dtype_audit"]["weight_files"] == ["weights/model.safetensors"]
    (source / "weights" / "unpinned.txt").write_text("not declared")
    with pytest.raises(ValueError, match="Every regular source file"):
        validate_source_provenance(source, provenance)


def test_token_windows_are_deterministic_bounded_and_spread():
    class Tokenizer:
        def __call__(self, text, add_special_tokens):
            assert add_special_tokens
            return {"input_ids": list(range(100))}

    path = Path(__file__)
    first = _token_windows(Tokenizer(), path, sequence_length=10, batches=3)
    second = _token_windows(Tokenizer(), path, sequence_length=10, batches=3)
    assert len(first) == 3
    assert all(batch["input_ids"].shape == (1, 10) for batch in first)
    assert all(torch.equal(left["input_ids"], right["input_ids"]) for left, right in zip(first, second))
    assert first[0]["input_ids"][0, 0].item() == 0
    assert first[-1]["input_ids"][0, -1].item() == 99
    with pytest.raises(ValueError, match="required"):
        _token_windows(Tokenizer(), path, sequence_length=101, batches=1)


def _runtime_worker(mode: str, repetition: int, *, reduced: bool = True):
    dense = mode == "dense"
    pre_model = 700
    base = (1000 if dense else 900) + repetition * 5
    if dense or not reduced:
        base = 1000 + repetition * 5
    persistent = 300 if dense else 200
    return {
        "mode": mode,
        "profile": "memory",
        "artifact": "/dense-artifact" if dense else "/quantized-artifact",
        "runtime_input": "/shared-runtime-input",
        "pre_model": {"current_rss_bytes": pre_model, "process_peak_rss_bytes": pre_model},
        "load": {
            "sampled_phase_peak_rss_bytes": base,
            "process_peak_rss_bytes": base,
            "seconds": 1.0,
        },
        "prefill": {
            "sampled_phase_peak_rss_bytes": base + 100,
            "process_peak_rss_bytes": base + 100,
            "median_seconds": 0.2 if dense else 0.3,
            "logits_sha256": "dense-hash" if dense else "quantized-hash",
        },
        "decode_without_kv_cache": {
            "sampled_phase_peak_rss_bytes": base + 200,
            "process_peak_rss_bytes": base + 200,
            "median_seconds_per_recomputed_token": 0.3 if dense else 0.4,
            "generated_token_ids": [4, 5],
        },
        "persistent_storage_after_run": {
            "model_unique_state_tensor_bytes": persistent,
            "decoded_cache_bytes": 0,
        },
        "source_dense_checkpoint_passed_to_worker": dense,
    }


def _summary(*, reduced=True, require_nonoverlapping=True, require_weight=True):
    dense = [_runtime_worker("dense", index) for index in range(3)]
    quantized = [_runtime_worker("quantized", index, reduced=reduced) for index in range(3)]
    return summarize_runtime_workers(
        dense,
        quantized,
        dense_persistent_bytes=300,
        quantized_persistent_bytes=200,
        require_rss_reduction=True,
        require_weight_dominated_load=require_weight,
        min_rss_savings_bytes=0,
        min_rss_savings_fraction=0.0,
        require_nonoverlapping_rss=require_nonoverlapping,
    )


def test_runtime_summary_separates_rss_weight_dominance_and_latency():
    report = _summary()
    assert report["acceptance"]["passed"]
    assert report["acceptance"]["measured_median_rss_reduction_in_all_phases"]
    assert report["rss"]["load"]["median_savings_bytes"] == 100
    assert report["rss"]["load"]["ranges_nonoverlapping"]
    assert report["persistent_savings_bytes"] == 100
    residency = report["weight_residency"]
    assert residency["dominates_incremental_nonweight_load_allocation"]
    assert not residency["dominates_entire_pre_model_process_baseline"]
    assert report["latency"]["quantized_decode_median_of_worker_medians_seconds"] == 0.4
    assert report["ram_claim"].startswith("Qualified median")
    relaxed = _summary(require_weight=False)
    assert relaxed["acceptance"]["passed"]
    assert relaxed["ram_claim"] == "No qualified dense-RAM reduction claim"


def test_runtime_summary_rejects_zero_reduction_and_inconsistent_workers():
    report = _summary(reduced=False, require_nonoverlapping=False)
    assert not report["rss"]["load"]["gate_passed"]
    assert not report["acceptance"]["passed"]
    assert report["ram_claim"] == "No qualified dense-RAM reduction claim"
    dense = [_runtime_worker("dense", index) for index in range(2)]
    quantized = [_runtime_worker("quantized", index) for index in range(2)]
    quantized[1]["prefill"]["logits_sha256"] = "different"
    inconsistent = summarize_runtime_workers(
        dense,
        quantized,
        dense_persistent_bytes=300,
        quantized_persistent_bytes=200,
        require_rss_reduction=False,
        require_weight_dominated_load=False,
        min_rss_savings_bytes=1,
        min_rss_savings_fraction=0.0,
        require_nonoverlapping_rss=False,
    )
    assert not inconsistent["acceptance"]["reproducibility_and_accounting_passed"]
    assert not inconsistent["acceptance"]["passed"]


def _qualification_args(root: Path, source: Path, provenance: Path, calibration: Path, validation: Path):
    return SimpleNamespace(
        source=str(source),
        provenance=str(provenance),
        calibration_data=str(calibration),
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        validation_data=str(validation),
        validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
        workspace=str(root / "run"),
        evidence=str(root / "evidence.json"),
        codec="rvq",
        include="*",
        target_state_ratio=0.95,
        max_tensor_estimated_error=10.0,
        decode_row_block_size=7,
        max_frontier_states=64,
        max_relative_logits=2.0,
        max_absolute_logits=None,
        max_kl=10.0,
        min_top1=0.0,
        max_nll_delta=10.0,
        max_dense_reference_nll=20.0,
        min_unique_parameters=1,
        min_quantizable_weight_bytes=1,
        min_persistent_savings_bytes=1,
        min_source_bf16_fraction=0.95,
        sequence_length=8,
        calibration_batches=1,
        validation_batches=2,
        max_activation_rows=4,
        max_invocations_per_module=1,
        max_positions_per_batch=16,
        max_shard_size="512MB",
        leech_backend="python",
        num_threads=1,
        runtime_repeats=1,
        prefill_warmup=0,
        prefill_iterations=1,
        decode_tokens=1,
        require_rss_reduction=False,
        require_weight_dominated_load=False,
        min_rss_savings_bytes=1,
        min_rss_savings_fraction=0.0,
        require_nonoverlapping_rss=False,
    )


def test_tiny_fixture_full_qualification_and_independent_prefill_hash(tmp_path):
    source, provenance, calibration, validation = _source_fixture(tmp_path)
    args = _qualification_args(tmp_path, source, provenance, calibration, validation)
    assert qualify(args) == 0
    evidence = json.loads(Path(args.evidence).read_text())
    assert evidence["status"] == "passed_without_full_ram_qualification"
    assert evidence["dense_ram_reduction_claim"] == "No qualified dense-RAM reduction claim"
    assert evidence["quality_after_meta_reload"]["passed"]
    artifact = tmp_path / "run" / "encoded"
    assert evidence["artifact"]["actual_reloaded_persistent_bytes"] == (
        evidence["generalized"]["plan"]["memory"]["predicted_model_state_bytes"]
    )
    quantized_worker = evidence["runtime_workers"]["quantized_memory"][0]
    assert not quantized_worker["source_dense_checkpoint_passed_to_worker"]
    assert quantized_worker["persistent_storage_after_run"]["decoded_cache_bytes"] == 0
    runtime_input = json.loads((tmp_path / "run" / "runtime-input.json").read_text())
    inputs = {key: torch.tensor(value) for key, value in runtime_input.items()}
    model = load_quantized_pretrained(artifact, load_tokenizer=False)
    with torch.inference_mode():
        expected = _tensor_sha256(model(**inputs, use_cache=False).logits)
    assert quantized_worker["prefill"]["logits_sha256"] == expected

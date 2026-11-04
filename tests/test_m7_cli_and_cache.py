"""Milestone M7: Thin Production CLI & Offline Cache Manager Tests.

Validates:
1. OjasXOfflineCache registration, resolution, eviction, and strict zero-download enforcement.
2. CLI subcommands:
   - doctor (readiness audit, compiler check, cache path)
   - prepare (streaming bounded-RAM conversion)
   - inspect (OQPack, legacy VQ, and checkpoint structural profile)
   - verify (cryptographic integrity & non-zero tamper detection)
   - chat (KV-cached generation with token IDs or prompt)
   - bench (prefill latency, cached vs uncached decode speedup, throughput)
   - serve (--dry-run configuration and readiness validation)
3. Delegation of legacy commands (convert, evaluate, benchmark) to runner.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ojasx_24x96.cli import OjasXOfflineCache, build_m7_parser, main, run_cli
from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.module import quantize_model
from ojasx_24x96.oqpack import save_oqpack


@pytest.fixture
def tiny_llama_fixture(tmp_path: Path):
    """Create a miniature Llama causal LM, quantize it, and save as canonical OQPack v1."""
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_cache=True,
    )
    model = LlamaForCausalLM(cfg).eval()

    # Source checkpoint
    src_dir = tmp_path / "source_llama"
    model.save_pretrained(src_dir, safe_serialization=True)

    # Quantize and save OQPack
    q_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=1,
        min_weight_elements=1,
        require_compression=False,
    )
    q_model, _ = quantize_model(model, q_cfg)
    oqpack_dir = tmp_path / "tiny_oqpack"
    save_oqpack(q_model, oqpack_dir, model_id="tiny-llama-m7")
    cfg.save_pretrained(oqpack_dir)

    return src_dir, oqpack_dir


def test_offline_cache_register_resolve_and_evict(tmp_path: Path, tiny_llama_fixture):
    """Verify OjasXOfflineCache registers, resolves, lists, and evicts packages strictly offline."""
    _, oqpack_dir = tiny_llama_fixture
    cache_dir = tmp_path / "test_cache"
    cache = OjasXOfflineCache(cache_dir=cache_dir)

    assert cache.list_packages() == []

    # Register package
    entry = cache.register_package(oqpack_dir, alias="tiny-alias")
    assert entry["model_id"] == "tiny-llama-m7"
    assert entry["alias"] == "tiny-alias"
    assert entry["format_version"] == "oqpack.v1"
    assert entry["size_bytes"] > 0
    assert len(entry["sha256"]) == 64
    assert entry["shards"] >= 1

    # List packages
    packages = cache.list_packages()
    assert len(packages) == 1
    assert packages[0]["model_id"] == "tiny-llama-m7"

    # Resolve by alias
    resolved_alias = cache.resolve("tiny-alias")
    assert resolved_alias == oqpack_dir.resolve()

    # Resolve by model_id
    resolved_id = cache.resolve("tiny-llama-m7")
    assert resolved_id == oqpack_dir.resolve()

    # Resolve by direct local path
    resolved_path = cache.resolve(str(oqpack_dir))
    assert resolved_path == oqpack_dir.resolve()

    # Evict by alias
    evicted = cache.evict("tiny-alias")
    assert evicted is True
    assert cache.list_packages() == []

    # Evict non-existent
    assert cache.evict("non-existent") is False


def test_strict_offline_mode_enforced_no_silent_downloads(tmp_path: Path):
    """Enforce that resolving an uncached remote model ID strictly fails offline without network calls."""
    cache_dir = tmp_path / "test_cache"
    cache = OjasXOfflineCache(cache_dir=cache_dir)

    # Disallow socket creation during resolution
    def _fail_on_network(*args, **kwargs):
        raise AssertionError("Network socket opened during offline resolution!")

    with patch.object(socket, "socket", side_effect=_fail_on_network):
        with pytest.raises(
            FileNotFoundError,
            match="Offline mode enforced: meta-llama/Llama-3-8b not found in local cache or filesystem; silent downloads are disabled",
        ):
            cache.resolve("meta-llama/Llama-3-8b")


def test_cli_doctor(capsys):
    """Verify doctor subcommand inspects environment, compiler, and kernel readiness."""
    code = run_cli(["doctor", "--json"])
    assert code == 0

    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["silent_downloads_disabled"] is True
    assert report["dense_fallback_disabled"] is True
    assert "python_version" in report
    assert "torch_version" in report
    assert "native_cpu_backend" in report
    assert "cuda_kernel_source" in report
    assert "offline_cache_dir" in report


def test_cli_prepare_and_inspect_and_verify(tmp_path: Path, tiny_llama_fixture, capsys):
    """Verify prepare converts checkpoint, inspect profiles it, and verify checks cryptographic validity."""
    src_dir, _ = tiny_llama_fixture
    out_dir = tmp_path / "prepared_checkpoint"

    # 1. prepare
    prep_code = run_cli(
        [
            "prepare",
            str(src_dir),
            str(out_dir),
            "--codec",
            "adaptive_residual_vq",
            "--bits-per-stage",
            "4",
            "--max-stages",
            "2",
        ]
    )
    assert prep_code == 0
    prep_out = json.loads(capsys.readouterr().out)
    assert prep_out["total_tensors"] > 0
    assert prep_out["quantized_tensors"] > 0
    assert (out_dir / "conversion_state.json").exists()

    # 2. inspect on streamed checkpoint
    insp_code = run_cli(["inspect", str(out_dir)])
    assert insp_code == 0
    insp_out = json.loads(capsys.readouterr().out)
    assert "format_version" in insp_out
    assert "shards" in insp_out
    assert "compression_ratio" in insp_out
    assert "architecture_profile" in insp_out
    assert "memory_admission" in insp_out

    # 3. verify on streamed checkpoint
    ver_code = run_cli(["verify", str(out_dir)])
    assert ver_code == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["valid"] is True
    assert ver_out["checksums_verified"] is True

    # 4. verify with --no-checksums
    ver_fast_code = run_cli(["verify", str(out_dir), "--no-checksums"])
    assert ver_fast_code == 0
    ver_fast_out = json.loads(capsys.readouterr().out)
    assert ver_fast_out["valid"] is True
    assert ver_fast_out["checksums_verified"] is False


def test_cli_verify_tamper_detection(tmp_path: Path, tiny_llama_fixture, capsys):
    """Verify that tampering with an OQPack safetensors shard triggers non-zero exit code and error report."""
    _, oqpack_dir = tiny_llama_fixture

    # Initially valid
    code = run_cli(["verify", str(oqpack_dir)])
    assert code == 0
    res = json.loads(capsys.readouterr().out)
    assert res["valid"] is True

    # Tamper with shard file
    shard_file = next(oqpack_dir.glob("shard-*.safetensors"))
    shard_bytes = bytearray(shard_file.read_bytes())
    shard_bytes[-1] ^= 0xFF  # Corrupt single byte
    shard_file.write_bytes(shard_bytes)

    # Verification must fail with non-zero exit code
    tamper_code = run_cli(["verify", str(oqpack_dir)])
    assert tamper_code != 0
    tamper_res = json.loads(capsys.readouterr().out)
    assert tamper_res["valid"] is False
    assert "error" in tamper_res


def test_cli_inspect_oqpack(tiny_llama_fixture, capsys):
    """Verify inspect on canonical OQPack v1 container outputs all required M7 fields."""
    _, oqpack_dir = tiny_llama_fixture
    code = run_cli(["inspect", str(oqpack_dir)])
    assert code == 0

    report = json.loads(capsys.readouterr().out)
    assert report["format_version"] == "oqpack.v1"
    assert report["model_id"] == "tiny-llama-m7"
    assert report["shards"] >= 1
    assert "compression_ratio" in report
    assert "architecture_profile" in report
    assert "memory_admission" in report


def test_cli_chat_with_kv_cache(tmp_path: Path, tiny_llama_fixture, capsys):
    """Verify chat subcommand runs OjasXInferenceEngine with genuine KV caching."""
    _, oqpack_dir = tiny_llama_fixture
    cache_dir = tmp_path / "chat_cache"

    with patch.dict(os.environ, {"OJASX_CACHE_DIR": str(cache_dir)}):
        cache = OjasXOfflineCache()
        cache.register_package(oqpack_dir, alias="chat-llama")

        code = run_cli(
            [
                "chat",
                "chat-llama",
                "--prompt",
                "1, 2, 3",
                "--max-new-tokens",
                "4",
                "--temperature",
                "0.0",
            ]
        )
        assert code == 0

        out = json.loads(capsys.readouterr().out)
        assert out["used_kv_cache"] is True
        assert out["prompt_token_ids"] == [1, 2, 3]
        assert len(out["generated_token_ids"]) == 4
        assert out["kv_cache_bytes"] > 0
        assert out["decode_steps"] == 4
        assert "admission_report" in out


def test_cli_bench(tiny_llama_fixture, capsys):
    """Verify bench subcommand measures prefill latency, decode speedup, and native throughput."""
    _, oqpack_dir = tiny_llama_fixture
    code = run_cli(
        [
            "bench",
            str(oqpack_dir),
            "--seq-len",
            "8",
            "--max-new-tokens",
            "4",
            "--backend",
            "auto",
        ]
    )
    assert code == 0

    bench_res = json.loads(capsys.readouterr().out)
    assert "prefill_latency_ms" in bench_res
    assert "cached_decode_latency_ms" in bench_res
    assert "uncached_decode_latency_ms" in bench_res
    assert "cached_vs_uncached_speedup" in bench_res
    assert "kv_cache_bytes" in bench_res
    assert "native_fused_kernel_throughput_tok_s" in bench_res
    assert bench_res["seq_len"] == 8
    assert bench_res["max_new_tokens"] == 4


def test_cli_serve_dry_run(tiny_llama_fixture, capsys):
    """Verify serve --dry-run validates configuration without starting background process."""
    _, oqpack_dir = tiny_llama_fixture
    code = run_cli(
        [
            "serve",
            str(oqpack_dir),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--memory-budget-mb",
            "512",
            "--dry-run",
        ]
    )
    assert code == 0

    serve_res = json.loads(capsys.readouterr().out)
    assert serve_res["status"] == "ready"
    assert serve_res["dry_run"] is True
    assert serve_res["host"] == "0.0.0.0"
    assert serve_res["port"] == 9000
    assert serve_res["memory_budget_mb"] == 512


def test_cli_delegates_legacy_subcommands():
    """Verify legacy subcommands (convert, evaluate, benchmark) delegate to runner.main."""
    with patch("ojasx_24x96.runner.main", return_value=0) as mock_runner_main:
        code = run_cli(["convert", "dummy_source", "dummy_output"])
        assert code == 0
        mock_runner_main.assert_called_once_with(["convert", "dummy_source", "dummy_output"])

    with patch("ojasx_24x96.runner.main", return_value=0) as mock_runner_main:
        code = run_cli(["evaluate", "source", "quantized", "--text", "hello"])
        assert code == 0
        mock_runner_main.assert_called_once_with(["evaluate", "source", "quantized", "--text", "hello"])

    with patch("ojasx_24x96.runner.main", return_value=0) as mock_runner_main:
        code = main(["benchmark", "model", "--text", "hello"])
        assert code == 0
        mock_runner_main.assert_called_once_with(["benchmark", "model", "--text", "hello"])


def test_build_m7_parser_structure():
    """Verify parser supports all M7 and legacy subcommands with required arguments."""
    parser = build_m7_parser()
    actions = {subaction.dest for subaction in parser._actions if hasattr(subaction, "dest")}
    assert "command" in actions

    # Test parser help doesn't crash
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0

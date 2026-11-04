"""End-to-End M0–M8 Full Vertical Slice Qualification & Conformance Test Suite.

Asserts 100% compliance across all OjasX-24X96 milestones:
- M0: Contract & 7-Bucket Memory Admission
- M1: Canonical OQPack Container & Zero-Copy mmap
- M2: Streaming Checkpoint Ingestion & Crash Resumption
- M3: Quality Certificate Verification
- M4: KV-Cached Runtime & Multi-Turn Prefix Reuse
- M5: Fused Native C++ Leech24 & RVQ GEMM Kernels
- M6: Stable C ABI & llama.cpp GGML Custom Op Adapter
- M7: Production CLI & Strict Offline Cache
- M8: OpenAI HTTP Server & Sandboxed Tool Agent Runtime
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import urllib.request

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from benchmarks.run_m0_m8_vertical_slice import run_full_vertical_slice
from ojasx_24x96.backend_abi import (
    LlamaCppGGMLAdapter,
    abi_header_status,
    run_backend_conformance_suite,
)
from ojasx_24x96.cli import OjasXOfflineCache, run_cli
from ojasx_24x96.codec import QuantizationConfig, quantize_weight
from ojasx_24x96.contract import (
    evaluate_memory_admission,
    introspect_architecture,
    max_admissible_context_length,
)
from ojasx_24x96.engine import OjasXInferenceEngine, SamplingParams
from ojasx_24x96.leech_codec import LeechQuantizationConfig, quantize_leech_weight
from ojasx_24x96.module import ResidualVQLinear
from ojasx_24x96.oqpack import (
    load_quantized_model_from_oqpack,
    save_oqpack,
    verify_oqpack_integrity,
)
from ojasx_24x96.server import OjasXOpenAIServer
from ojasx_24x96.structured_tools import (
    OjasXToolUsingAgent,
    SandboxedToolRunner,
    ToolCall,
    ToolSpec,
)


class TestFullM0M8VerticalSlice:
    """Test suite executing the end-to-end M0–M8 vertical slice pipeline."""

    def test_pipeline_end_to_end(self):
        """Run the complete M0-M8 vertical slice qualification pipeline."""
        report = run_full_vertical_slice()

        # Overall Pipeline Summary Assertions
        summary = report["summary"]
        assert summary["status"] == "PASS"
        assert summary["all_passed"] is True
        expected_milestones = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
        assert summary["milestones_passed"] == expected_milestones
        assert summary["elapsed_seconds"] > 0

        # M0 (Contract & Introspection)
        m0 = report["m0_contract"]
        assert m0["status"] == "PASS"
        assert m0["qwen_is_hybrid"] is True
        assert "linear_recurrent" in m0["qwen_attention_kinds"]
        assert "full_attention" in m0["qwen_attention_kinds"]
        assert m0["llama_is_hybrid"] is False
        assert m0["admission_4k_total_bytes"] > 0
        assert m0["max_context_under_64mb"] > 0

        # M2 (Streaming Ingestion & Resumption)
        m2 = report["m2_streaming_ingest"]
        assert m2["status"] == "PASS"
        assert m2["total_tensors"] >= 20
        assert m2["quantized_tensors"] > 0
        assert m2["resumed_tensors"] > 0
        assert m2["compression_ratio"] > 1.0

        # M3 (Quality Certificate)
        m3 = report["m3_certificate"]
        assert m3["status"] == "PASS"
        assert m3["certificate_passed"] is True
        assert m3["mean_kl_divergence"] < 0.1
        assert m3["top1_agreement"] > 0.5
        assert m3["top5_agreement"] > 0.8

        # M1 (OQPack Container)
        m1 = report["m1_oqpack"]
        assert m1["status"] == "PASS"
        assert m1["format_version"] == "oqpack.v1"
        assert m1["shards"] >= 1
        assert m1["exact_logit_match"] is True

        # M4 (KV-Cached Runtime)
        m4 = report["m4_kv_runtime"]
        assert m4["status"] == "PASS"
        assert m4["matches_exactly"] is True
        assert m4["max_logit_abs_diff"] < 1e-5
        assert m4["turn2_prefill_tokens"] == 2
        assert m4["turn3_prefill_tokens"] == 2
        assert m4["kv_bytes_after_reset"] == 0

        # M5 (Native Fused Kernels)
        m5 = report["m5_fused_kernels"]
        assert m5["status"] == "PASS"
        assert m5["fused_modules_in_model"] >= 1
        assert m5["leech24_cached_bytes"] == 0
        assert m5["rvq_cached_bytes"] == 0
        assert m5["leech24_fused_abs_diff"] < 1e-4
        assert m5["rvq_fused_abs_diff"] < 1e-4
        assert m5["full_model_fused_max_diff"] < 1e-4

        # M6 (Backend C ABI & llama.cpp GGML Adapter)
        m6 = report["m6_backend_abi"]
        assert m6["status"] == "PASS"
        assert m6["abi_version"] == 1
        assert m6["conformance_passed"] is True
        assert m6["conformance_checks"]["abi_header_and_symbols"] is True
        assert m6["conformance_checks"]["leech24_matmul_conformance"] is True
        assert m6["conformance_checks"]["rvq_matmul_conformance"] is True
        assert m6["conformance_checks"]["zero_dense_weight_cache"] is True
        assert m6["sidecar_format"] == "ggml_ojasx_adapter.v1"
        assert m6["sidecar_quantized_tensors"] >= 1
        assert m6["ggml_matmul_zero_dense_cache"] is True

        # M7 (CLI & Offline Cache)
        m7 = report["m7_cli_cache"]
        assert m7["status"] == "PASS"
        assert m7["package_registered"] == "vslice_qwen"
        assert m7["package_total_bytes"] > 0
        assert m7["cli_verify_exit_code"] == 0
        assert m7["cli_doctor_exit_code"] == 0

        # M8 (OpenAI Server & Sandboxed Tool Agent)
        m8 = report["m8_server_and_agent"]
        assert m8["status"] == "PASS"
        assert m8["sse_chunks_received"] > 0
        assert m8["sse_completed_cleanly"] is True
        assert m8["sandboxed_tool_success"] is True
        assert m8["sandboxed_tool_output"] == 42
        assert m8["sandboxed_pid_isolated"] is True
        assert m8["agent_prefix_reused"] is True


class TestMilestoneIsolation:
    """Targeted granular tests for each milestone to ensure fault isolation."""

    def test_m0_hybrid_introspection(self):
        """Verify Qwen3.8 hybrid DeltaNet/full-attention introspection & admission."""
        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=12,
            linear_value_head_dim=12,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
        )
        model = Qwen3_5ForCausalLM(cfg).eval()
        profile = introspect_architecture(model)

        assert profile.is_hybrid_attention is True
        assert profile.layer_kv_specs[0].attention_kind == "linear_recurrent"
        assert profile.layer_kv_specs[1].attention_kind == "full_attention"

        admission = evaluate_memory_admission(
            profile,
            encoded_weight_bytes=10000,
            dense_weight_bytes=5000,
            context_length=2048,
            budget_bytes=24 * 1024**3,
        )
        assert admission.admitted is True
        assert admission.total_required_bytes < 24 * 1024**3

    def test_m5_leech_and_rvq_zero_weight_cache(self):
        """Verify both Leech24 and RVQ layers execute with 0 dense weight cache."""
        # 1. Leech24 layer
        w_leech = torch.randn(12, 48)
        enc_leech, _ = quantize_leech_weight(
            w_leech,
            LeechQuantizationConfig(max_stages=2, gain_bits=2, min_weight_elements=1, max_relative_error=None),
        )
        mod_leech = ResidualVQLinear.from_encoded(
            enc_leech, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )
        inp_leech = torch.randn(2, 48)
        ref_l = mod_leech(inp_leech)
        mod_leech.set_fused_native_gemm(True)
        fused_l = mod_leech(inp_leech)
        assert mod_leech.cached_weight_bytes == 0
        assert torch.allclose(ref_l, fused_l, atol=1e-4)

        # 2. RVQ layer
        w_rvq = torch.randn(12, 32)
        enc_rvq, _ = quantize_weight(
            w_rvq,
            QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, min_weight_elements=1, max_relative_error=None),
        )
        mod_rvq = ResidualVQLinear.from_encoded(
            enc_rvq, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )
        inp_rvq = torch.randn(2, 32)
        ref_r = mod_rvq(inp_rvq)
        mod_rvq.set_fused_native_gemm(True)
        fused_r = mod_rvq(inp_rvq)
        assert mod_rvq.cached_weight_bytes == 0
        assert torch.allclose(ref_r, fused_r, atol=1e-4)

    def test_m6_c_abi_and_conformance(self):
        """Verify C ABI descriptors, conformance suite, and GGML adapter dispatch."""
        hdr = abi_header_status()
        assert hdr["verified"] is True
        suite = run_backend_conformance_suite()
        assert suite["passed"] is True

        w = torch.randn(8, 24)
        enc, _ = quantize_leech_weight(w, LeechQuantizationConfig(max_stages=1, min_weight_elements=1, max_relative_error=None))
        layer = ResidualVQLinear.from_encoded(
            enc, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )
        inp = torch.randn(1, 24)
        out = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(layer, inp, prefer_native=True)
        assert layer.cached_weight_bytes == 0
        assert out.shape == (1, 8)

    def test_m7_offline_cache_strict_enforcement(self):
        """Verify offline cache resolution and strict error on missing models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OjasXOfflineCache(cache_dir=Path(tmpdir))
            with pytest.raises(FileNotFoundError, match="Offline mode enforced"):
                cache.resolve("non_existent_model_id")

    def test_m8_sandboxed_tool_agent_loop(self):
        """Verify tool execution sandbox isolation and KV cache reuse across turns."""
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        model = LlamaForCausalLM(cfg).eval()
        engine = OjasXInferenceEngine(model, max_context_length=128)

        tool = ToolSpec(
            name="adder",
            description="Adds two numbers",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
            python_callable_source="def adder(x, y):\n    return x + y\n",
        )
        runner = SandboxedToolRunner(tools=[tool])
        agent = OjasXToolUsingAgent(engine, runner)

        result = agent.run_agent_turn(
            [1, 2, 3],
            tool_call_decider=lambda _: ToolCall(call_id="call_add", name="adder", arguments={"x": 19, "y": 23}),
            params=SamplingParams(max_new_tokens=4),
        )

        assert result["tool_result"].success is True
        assert result["tool_result"].output == 42
        assert result["tool_result"].sandbox_pid != os.getpid()
        assert result["prefix_reused"] is True

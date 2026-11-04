"""End-to-End M0–M8 Full Vertical Slice Qualification & Conformance Pipeline.

Executes the complete production qualification sequence for OjasX-24X96 across
all milestones M0 through M8:
- M0: Hardware/Architecture Introspection & 7-Bucket Memory Admission
- M1: Canonical OQPack Container (oqpack.v1) Multi-Shard Save & mmap Reload
- M2: Bounded-RAM Streaming Checkpoint Ingestion & Resumption
- M3: Quality Certificate (Perplexity Delta, Forward KL, Top-1/Top-5 Agreement)
- M4: KV-Cached Reference Runtime (use_cache Parity & 3-Turn Prefix Reuse)
- M5: Fused Native C++ Kernels (Leech24 & RVQ with 0 Bytes Dense Weight Cache)
- M6: Stable C ABI & llama.cpp GGML Custom Op Adapter Sidecar Export
- M7: Production CLI (oqx) & Offline Cache Manager
- M8: OpenAI-Compatible HTTP Server & Sandboxed Tool-Using Agent Runtime
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from ojasx_24x96.backend_abi import (
    LlamaCppGGMLAdapter,
    abi_header_status,
    run_backend_conformance_suite,
)
from ojasx_24x96.certificates import (
    QualityCertificateThresholds,
    generate_quality_certificate,
)
from ojasx_24x96.cli import OjasXOfflineCache, run_cli
from ojasx_24x96.codec import QuantizationConfig, quantize_weight
from ojasx_24x96.contract import (
    evaluate_memory_admission,
    introspect_architecture,
    max_admissible_context_length,
)
from ojasx_24x96.engine import (
    OjasXInferenceEngine,
    SamplingParams,
)
from ojasx_24x96.generalized import quantize_model_generalized
from ojasx_24x96.ingest import stream_convert_checkpoint
from ojasx_24x96.leech_codec import LeechQuantizationConfig, quantize_leech_weight
from ojasx_24x96.module import ResidualVQLinear
from ojasx_24x96.oqpack import (
    load_quantized_model_from_oqpack,
    save_oqpack,
    verify_oqpack_integrity,
)
from ojasx_24x96.planner import PlanningConstraints
from ojasx_24x96.server import OjasXOpenAIServer
from ojasx_24x96.structured_tools import (
    OjasXToolUsingAgent,
    SandboxedToolRunner,
    ToolCall,
    ToolSpec,
)


def run_full_vertical_slice() -> Dict[str, Any]:
    """Execute the end-to-end M0–M8 production qualification pipeline."""
    t_start = time.perf_counter()
    results: Dict[str, Any] = {
        "pipeline_name": "OjasX-24X96 M0-M8 Full Vertical Slice Qualification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # =========================================================================
    # Milestone M0: Generalized Architecture Introspection & Memory Admission
    # =========================================================================
    torch.manual_seed(20261104)

    # 1. Qwen3.8 / Qwen3_5 (Hybrid Gated-DeltaNet Linear + Full Attention)
    qwen_cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=24,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=24,
        linear_value_head_dim=24,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
    )
    qwen_model = Qwen3_5ForCausalLM(qwen_cfg).eval()
    qwen_profile = introspect_architecture(qwen_model)

    # 2. Llama Architecture Profile
    llama_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=True,
    )
    llama_model = LlamaForCausalLM(llama_cfg).eval()
    llama_profile = introspect_architecture(llama_model)

    # 7-Bucket Total Memory Admission calculation
    qwen_admission = evaluate_memory_admission(
        qwen_profile,
        encoded_weight_bytes=52_000,
        dense_weight_bytes=26_000,
        context_length=4096,
        batch_size=1,
        kv_dtype=torch.float16,
        budget_bytes=24 * 1024**3,
    )
    max_ctx_qwen = max_admissible_context_length(
        qwen_profile,
        budget_bytes=64 * 1024**2,
        encoded_weight_bytes=52_000,
        dense_weight_bytes=26_000,
        batch_size=1,
        kv_dtype=torch.float16,
    )

    assert qwen_profile.is_hybrid_attention is True, "Qwen3.8 must be recognized as hybrid attention"
    assert llama_profile.is_hybrid_attention is False, "Llama must be recognized as standard attention"
    assert qwen_admission.admitted is True, "Admission under 24GB budget must succeed"
    assert max_ctx_qwen > 0, "Max admissible context length must be positive"

    results["m0_contract"] = {
        "status": "PASS",
        "qwen_model_type": qwen_profile.model_type,
        "qwen_is_hybrid": qwen_profile.is_hybrid_attention,
        "qwen_attention_kinds": [s.attention_kind for s in qwen_profile.layer_kv_specs],
        "llama_is_hybrid": llama_profile.is_hybrid_attention,
        "admission_4k_total_bytes": qwen_admission.total_required_bytes,
        "admission_4k_kv_cache_bytes": qwen_admission.kv_cache_bytes,
        "max_context_under_64mb": max_ctx_qwen,
    }

    # =========================================================================
    # Milestone M2: Bounded-RAM Streaming Safetensors Ingestion & Resumption
    # =========================================================================
    with tempfile.TemporaryDirectory() as tmp_m2:
        ckpt_dir = Path(tmp_m2) / "raw_ckpt"
        stream_out_dir = Path(tmp_m2) / "streamed_out"
        ckpt_dir.mkdir(parents=True)

        sd = {k: v.contiguous() for k, v in llama_model.state_dict().items()}
        keys = list(sd.keys())
        mid = len(keys) // 2
        shard1 = {k: sd[k] for k in keys[:mid]}
        shard2 = {k: sd[k] for k in keys[mid:]}
        save_file(shard1, str(ckpt_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(ckpt_dir / "model-00002-of-00002.safetensors"))
        w_map = {
            k: ("model-00001-of-00002.safetensors" if i < mid else "model-00002-of-00002.safetensors")
            for i, k in enumerate(keys)
        }
        (ckpt_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": w_map})
        )

        # 1. Crash simulation
        crashed = False
        try:
            stream_convert_checkpoint(
                ckpt_dir,
                stream_out_dir,
                codec="leech24_shell2_residual",
                leech_config=LeechQuantizationConfig(
                    max_stages=2,
                    gain_bits=2,
                    rotation_reflections=2,
                    min_weight_elements=64,
                    max_relative_error=None,
                ),
                fail_after_tensors=3,
            )
        except RuntimeError:
            crashed = True
        assert crashed, "Crash simulation must fail after designated tensor count"

        # 2. Resumption simulation
        m2_report = stream_convert_checkpoint(
            ckpt_dir,
            stream_out_dir,
            codec="leech24_shell2_residual",
            leech_config=LeechQuantizationConfig(
                max_stages=2,
                gain_bits=2,
                rotation_reflections=2,
                min_weight_elements=64,
                max_relative_error=None,
            ),
            resume=True,
        )
        assert m2_report.resumed_tensors > 0, "Resumed tensors count must be positive"
        assert m2_report.quantized_tensors > 0, "Quantized tensors count must be positive"

        results["m2_streaming_ingest"] = {
            "status": "PASS",
            "total_tensors": m2_report.total_tensors,
            "quantized_tensors": m2_report.quantized_tensors,
            "resumed_tensors": m2_report.resumed_tensors,
            "compression_ratio": round(m2_report.compression_ratio, 3),
            "peak_single_tensor_bytes": m2_report.peak_single_tensor_bytes,
        }

    # =========================================================================
    # Model Quantization & Setup for M1, M3, M4, M5, M6, M7, M8
    # =========================================================================
    eval_batches = [torch.randint(0, 128, (2, 16), dtype=torch.long)]
    long_batches = [torch.randint(0, 128, (1, 48), dtype=torch.long)]

    working_qwen = copy.deepcopy(qwen_model)
    q_result = quantize_model_generalized(
        working_qwen,
        candidate_configs=[
            QuantizationConfig(
                block_size=8,
                bits_per_stage=4,
                max_stages=2,
                kmeans_iterations=6,
                max_relative_error=None,
                min_weight_elements=64,
            ),
            LeechQuantizationConfig(
                max_stages=2,
                gain_bits=2,
                rotation_reflections=2,
                max_relative_error=None,
                min_weight_elements=64,
            ),
        ],
        calibration_inputs=eval_batches,
        constraints=PlanningConstraints(
            target_state_ratio=0.999,
            max_tensor_weight_error=1.0,
            runtime_profile="memory",
            architecture_profile=qwen_profile,
            max_total_runtime_bytes=24 * 1024**3,
            context_length=4096,
        ),
    )
    q_model = q_result.model

    # =========================================================================
    # Milestone M3: Quality Certificate Generation & Gate Verification
    # =========================================================================
    cert = generate_quality_certificate(
        qwen_model,
        q_model,
        eval_batches,
        long_context_batches=long_batches,
        plan=q_result.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=50.0,
            max_kl_divergence=2.0,
            min_top1_agreement=0.10,
            min_top5_agreement=0.25,
            max_long_context_kl_drift=2.0,
        ),
    )
    assert cert.passed is True, "Quality certificate must pass all declared thresholds"

    results["m3_certificate"] = {
        "status": "PASS",
        "certificate_passed": cert.passed,
        "mean_kl_divergence": round(cert.mean_kl_divergence, 5),
        "top1_agreement": round(cert.top1_agreement, 4),
        "top5_agreement": round(cert.top5_agreement, 4),
        "long_context_kl_divergence": (
            round(cert.long_context_kl_divergence, 5)
            if cert.long_context_kl_divergence is not None
            else None
        ),
    }

    # =========================================================================
    # Milestone M1: OQPack Container Packaging & Zero-Copy mmap Load
    # =========================================================================
    with tempfile.TemporaryDirectory() as tmp_m1:
        oq_dir = Path(tmp_m1) / "oqpack_model"
        manifest = save_oqpack(
            q_model,
            oq_dir,
            plan=q_result.plan,
            architecture_profile=qwen_profile,
            memory_admission=qwen_admission,
            quality_certificate=cert.to_dict(),
            model_id="qwen3_8_vertical_slice",
            max_shard_bytes=64 * 1024,  # Force multi-shard splitting
        )
        verified_manifest = verify_oqpack_integrity(oq_dir, verify_checksums=True)
        assert len(verified_manifest.shards) >= 1, "Must contain at least 1 verified shard"

        skeleton = copy.deepcopy(qwen_model)
        reloaded_model, _ = load_quantized_model_from_oqpack(
            skeleton, oq_dir, verify_checksums=True
        )

        with torch.inference_mode():
            out_q = q_model(eval_batches[0])
            out_r = reloaded_model(eval_batches[0])
            logits_q = out_q.logits if hasattr(out_q, "logits") else out_q
            logits_r = out_r.logits if hasattr(out_r, "logits") else out_r
            oqpack_exact_match = bool(torch.equal(logits_q, logits_r))
        assert oqpack_exact_match is True, "Reloaded OQPack must match in-memory quantized model bitwise"

        # Keep oq_dir available for M7 by copying to a persistent temp location
        persisted_oq_dir = Path(tempfile.mkdtemp(prefix="oqpack_vslice_"))
        for item in oq_dir.iterdir():
            if item.is_file():
                persisted_oq_dir.joinpath(item.name).write_bytes(item.read_bytes())

    results["m1_oqpack"] = {
        "status": "PASS",
        "format_version": verified_manifest.format_version,
        "shards": len(verified_manifest.shards),
        "exact_logit_match": oqpack_exact_match,
        "verified_tensors": len(verified_manifest.tensors),
    }

    # =========================================================================
    # Milestone M4: Genuine KV-Cached Runtime & Multi-Turn Prefix Reuse
    # =========================================================================
    engine = OjasXInferenceEngine(
        q_model,
        max_context_length=512,
        memory_budget_bytes=2 * 1024**3,
        kv_dtype=torch.float32,
    )
    parity = engine.verify_cached_vs_uncached_parity(
        [10, 20, 30, 40], max_new_tokens=4, atol=1e-4
    )
    assert parity.matches_exactly is True, "Cached vs uncached token IDs must match exactly"
    assert parity.max_logit_abs_diff < 1e-4, "Max logit abs diff must remain below 1e-4"

    # 3-turn multi-turn conversational prefix reuse
    t1 = engine.chat_turn([11, 22], SamplingParams(max_new_tokens=3, temperature=0.0))
    t2 = engine.chat_turn([33, 44], SamplingParams(max_new_tokens=3, temperature=0.0))
    t3 = engine.chat_turn([55, 66], SamplingParams(max_new_tokens=3, temperature=0.0))
    assert t2.prefill_tokens == 2, "Turn 2 must reuse prefix (only 2 prompt tokens prefilled)"
    assert t3.prefill_tokens == 2, "Turn 3 must reuse prefix (only 2 prompt tokens prefilled)"

    # Reset conversational state
    engine.reset_chat()
    kv_allocated_after_reset = engine.kv_manager.stats().allocated_bytes
    assert kv_allocated_after_reset == 0, "reset_chat() must release all KV-cache memory back to 0 bytes"

    results["m4_kv_runtime"] = {
        "status": "PASS",
        "matches_exactly": parity.matches_exactly,
        "max_logit_abs_diff": float(parity.max_logit_abs_diff),
        "turn1_prefill_tokens": t1.prefill_tokens,
        "turn2_prefill_tokens": t2.prefill_tokens,
        "turn3_prefill_tokens": t3.prefill_tokens,
        "turn3_kv_bytes": t3.kv_cache_bytes,
        "kv_bytes_after_reset": kv_allocated_after_reset,
    }

    # =========================================================================
    # Milestone M5: Fused Native Kernels (Leech24 & RVQ with 0 Cache Bytes)
    # =========================================================================
    # 1. Dedicated Leech24 layer verification
    leech_cfg = LeechQuantizationConfig(
        max_stages=2, gain_bits=2, rotation_reflections=2, min_weight_elements=1, max_relative_error=None
    )
    w_leech = torch.randn(16, 72)
    b_leech = torch.randn(16)
    enc_leech, _ = quantize_leech_weight(w_leech, leech_cfg)
    leech_module = ResidualVQLinear.from_encoded(
        enc_leech, bias=b_leech, source_dtype=torch.float32, device=torch.device("cpu")
    )
    inp_leech = torch.randn(2, 4, 72)
    ref_out_leech = leech_module(inp_leech)
    leech_module.set_fused_native_gemm(True)
    fused_out_leech = leech_module(inp_leech)
    assert leech_module.cached_weight_bytes == 0, "Fused Leech24 layer must not cache dense weights"
    diff_leech = float((ref_out_leech - fused_out_leech).abs().max().item())
    assert diff_leech < 1e-4, f"Fused Leech diff too high: {diff_leech}"

    # 2. Dedicated Adaptive RVQ layer verification
    rvq_cfg = QuantizationConfig(
        block_size=8, bits_per_stage=4, max_stages=2, min_weight_elements=1, max_relative_error=None
    )
    w_rvq = torch.randn(16, 32)
    b_rvq = torch.randn(16)
    enc_rvq, _ = quantize_weight(w_rvq, rvq_cfg)
    rvq_module = ResidualVQLinear.from_encoded(
        enc_rvq, bias=b_rvq, source_dtype=torch.float32, device=torch.device("cpu")
    )
    inp_rvq = torch.randn(2, 4, 32)
    ref_out_rvq = rvq_module(inp_rvq)
    rvq_module.set_fused_native_gemm(True)
    fused_out_rvq = rvq_module(inp_rvq)
    assert rvq_module.cached_weight_bytes == 0, "Fused RVQ layer must not cache dense weights"
    diff_rvq = float((ref_out_rvq - fused_out_rvq).abs().max().item())
    assert diff_rvq < 1e-4, f"Fused RVQ diff too high: {diff_rvq}"

    # 3. Full model fused execution
    fused_count = 0
    for mod in q_model.modules():
        if isinstance(mod, ResidualVQLinear):
            mod.set_fused_native_gemm(True)
            fused_count += 1
    with torch.inference_mode():
        out_fused = q_model(eval_batches[0])
        logits_fused = out_fused.logits if hasattr(out_fused, "logits") else out_fused
        max_fused_diff = float((logits_q - logits_fused).abs().max().item())
    assert max_fused_diff < 1e-4, "Full model fused output must match reference within 1e-4"

    results["m5_fused_kernels"] = {
        "status": "PASS",
        "fused_modules_in_model": fused_count,
        "leech24_fused_abs_diff": diff_leech,
        "leech24_cached_bytes": leech_module.cached_weight_bytes,
        "rvq_fused_abs_diff": diff_rvq,
        "rvq_cached_bytes": rvq_module.cached_weight_bytes,
        "full_model_fused_max_diff": max_fused_diff,
    }

    # =========================================================================
    # Milestone M6: Backend C ABI & llama.cpp Adapter Sidecar Export
    # =========================================================================
    abi_hdr = abi_header_status()
    assert abi_hdr["verified"] is True, "C ABI header verification failed"

    m6_conformance = run_backend_conformance_suite()
    assert m6_conformance["passed"] is True, "M6 C ABI conformance suite must pass 100%"

    with tempfile.TemporaryDirectory() as tmp_m6:
        sidecar_path = Path(tmp_m6) / "ggml_ojasx_adapter.json"
        adapter_manifest = LlamaCppGGMLAdapter.export_ggml_adapter_manifest(q_model, sidecar_path)
        assert sidecar_path.exists(), "Sidecar manifest must be created on disk"

        # Dispatch custom ggml matmul on one of the quantized layers
        q_layer = next(m for m in q_model.modules() if isinstance(m, ResidualVQLinear))
        test_in = torch.randn(2, q_layer.in_features)
        ggml_out = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(q_layer, test_in, prefer_native=True)
        assert q_layer.cached_weight_bytes == 0, "GGML custom matmul must allocate 0 dense weight bytes"

    results["m6_backend_abi"] = {
        "status": "PASS",
        "abi_version": abi_hdr["abi_version"],
        "conformance_passed": m6_conformance["passed"],
        "conformance_checks": {k: v["passed"] for k, v in m6_conformance["checks"].items()},
        "sidecar_format": adapter_manifest["format"],
        "sidecar_quantized_tensors": len(adapter_manifest["quantized_tensors"]),
        "ggml_matmul_zero_dense_cache": bool(q_layer.cached_weight_bytes == 0),
    }

    # =========================================================================
    # Milestone M7: Thin CLI (oqx) & Offline Cache Manager
    # =========================================================================
    with tempfile.TemporaryDirectory() as tmp_cache:
        cache = OjasXOfflineCache(cache_dir=Path(tmp_cache))
        cache_entry = cache.register_package(persisted_oq_dir, alias="vslice_qwen")
        resolved_path = cache.resolve("vslice_qwen")
        assert resolved_path.resolve() == persisted_oq_dir.resolve(), "Offline cache resolution failed"

        # CLI execution: verify subcommand
        ret_verify = run_cli(["verify", str(persisted_oq_dir)])
        assert ret_verify == 0, "CLI verify subcommand must return code 0"

        # CLI execution: doctor subcommand
        ret_doc = run_cli(["doctor", "--json"])
        assert ret_doc == 0, "CLI doctor subcommand must return code 0"

    results["m7_cli_cache"] = {
        "status": "PASS",
        "package_registered": cache_entry.get("alias", "vslice_qwen"),
        "package_total_bytes": cache_entry.get("size_bytes", 0),
        "cli_verify_exit_code": ret_verify,
        "cli_doctor_exit_code": ret_doc,
    }

    # =========================================================================
    # Milestone M8: OpenAI Server & Sandboxed Tool Agent Runtime
    # =========================================================================
    # 1. Ephemeral OpenAI HTTP Server
    server = OjasXOpenAIServer(engine, host="127.0.0.1", port=0)
    srv_host, srv_port = server.start_background()
    base_url = f"http://{srv_host}:{srv_port}"

    # SSE streaming chat completion with response_format={"type": "json_object"}
    req_payload = json.dumps({
        "model": "qwen3_8_vertical_slice",
        "messages": [{"role": "user", "content": "Format structured output as JSON"}],
        "max_tokens": 4,
        "stream": True,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=req_payload,
        headers={"Content-Type": "application/json"},
    )
    sse_lines: List[str] = []
    with urllib.request.urlopen(req) as resp:
        while True:
            raw_line = resp.readline().decode("utf-8")
            if not raw_line:
                break
            stripped = raw_line.strip()
            sse_lines.append(stripped)
            if stripped == "data: [DONE]":
                break

    server.shutdown()
    assert any("assistant" in l for l in sse_lines), "SSE stream must include assistant role"
    assert sse_lines[-1] == "data: [DONE]", "SSE stream must conclude with data: [DONE]"

    # 2. Sandboxed Tool-Using Agent with Subprocess Isolation & KV-Cache Reuse
    calc_tool = ToolSpec(
        name="multiplier",
        description="Multiplies two numbers",
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        python_callable_source="def multiplier(a, b):\n    return a * b\n",
    )
    tool_runner = SandboxedToolRunner(tools=[calc_tool])
    agent = OjasXToolUsingAgent(engine, tool_runner)

    agent_turn = agent.run_agent_turn(
        [10, 20, 30],
        tool_call_decider=lambda _: ToolCall(
            call_id="call_vslice_1", name="multiplier", arguments={"a": 6, "b": 7}
        ),
        params=SamplingParams(max_new_tokens=4),
    )

    assert agent_turn["tool_result"].success is True, "Sandboxed tool execution must succeed"
    assert agent_turn["tool_result"].output == 42, "Multiplier output must equal 42"
    assert agent_turn["tool_result"].sandbox_pid != os.getpid(), "Sandbox must execute in isolated subprocess"
    assert agent_turn["prefix_reused"] is True, "Agent turn 2 must reuse KV cache from turn 1"

    results["m8_server_and_agent"] = {
        "status": "PASS",
        "server_base_url": base_url,
        "sse_chunks_received": len([l for l in sse_lines if l.startswith("data: ") and l != "data: [DONE]"]),
        "sse_completed_cleanly": bool(sse_lines[-1] == "data: [DONE]"),
        "sandboxed_tool_success": agent_turn["tool_result"].success,
        "sandboxed_tool_output": agent_turn["tool_result"].output,
        "sandboxed_pid_isolated": bool(agent_turn["tool_result"].sandbox_pid != os.getpid()),
        "agent_prefix_reused": agent_turn["prefix_reused"],
    }

    # Clean up persisted OQPack dir
    for f in persisted_oq_dir.iterdir():
        f.unlink()
    persisted_oq_dir.rmdir()

    # =========================================================================
    # Final Qualification Summary
    # =========================================================================
    elapsed = time.perf_counter() - t_start
    results["summary"] = {
        "status": "PASS",
        "milestones_passed": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
        "all_passed": True,
        "elapsed_seconds": round(elapsed, 2),
    }

    return results


if __name__ == "__main__":
    report = run_full_vertical_slice()
    out_file = Path(__file__).resolve().parent / "m0_m8_vertical_slice_results.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"M0-M8 Vertical Slice Qualification Complete in {report['summary']['elapsed_seconds']}s")
    print(f"Results saved to {out_file}")
    print(json.dumps(report, indent=2))

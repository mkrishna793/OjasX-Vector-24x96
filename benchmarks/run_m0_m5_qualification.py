"""End-to-end M0-M5 benchmark and qualification harness for OjasX-24X96.

Evaluates modern open-source architectures:
1. Qwen3.8 / Qwen3_5 (Hybrid Gated-DeltaNet + Full Attention)
2. Gemma 4 (Alternating Sliding-Window + Global GQA)
3. GLM-5.3 / GlmMoeDsa (Multi-head Latent Attention + Routed & Shared MoE)
4. DeepSeek-V4 (Multi-head Latent Attention + Dense/MoE)
5. Custom PyTorch Model (with user-defined custom Linear projection)

Validates across M0-M5:
- M0: Generalized architecture introspection & 7-bucket memory admission
- M1: Canonical OQPack (oqpack.v1) sharded save, SHA-256 verification & zero-copy mmap load
- M2: Bounded-RAM streaming safetensors checkpoint ingestion & crash-resume
- M3: Layer sensitivity profiling, KV-aware Pareto planning & quality certificates (PPL, KL, Top-1/Top-5)
- M4: Genuine KV-cached runtime (use_cache=True vs use_cache=False parity) & multi-turn chat prefix reuse
- M5: Fused native C++ Leech24 & RVQ GEMM kernels with zero dense weight cache
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from ojasx_24x96 import (
    LeechQuantizationConfig,
    OjasXInferenceEngine,
    PlanningConstraints,
    QualityCertificateThresholds,
    QuantizationConfig,
    SamplingParams,
    cuda_kernel_source_status,
    evaluate_memory_admission,
    generate_quality_certificate,
    introspect_architecture,
    load_quantized_model_from_oqpack,
    max_admissible_context_length,
    native_leech_backend_status,
    quantize_model,
    quantize_model_generalized,
    save_oqpack,
    stream_convert_checkpoint,
    verify_oqpack_integrity,
)


class CustomGatedProjection(nn.Module):
    """User-defined custom linear projection layer tested via duck-typed M0 adapter."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.04)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class CustomSequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 96) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj1 = CustomGatedProjection(hidden_size, hidden_size * 2)
        self.act = nn.SiLU()
        self.proj2 = CustomGatedProjection(hidden_size * 2, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = self.embed(input_ids)
        h = h + self.proj2(self.act(self.proj1(h)))
        return self.head(h)


def build_modern_models() -> Dict[str, nn.Module]:
    torch.manual_seed(20261104)
    models: Dict[str, nn.Module] = {}

    # 1. Qwen3.8 / Qwen3_5 (Hybrid Gated-DeltaNet + Full Attention)
    qwen_cfg = Qwen3_5TextConfig(
        vocab_size=256,
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
    models["Qwen3.8-Hybrid (Qwen3_5)"] = Qwen3_5ForCausalLM(qwen_cfg).eval()

    # 2. Gemma 4 (Alternating Sliding-Window + Global GQA)
    gemma_cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        global_num_key_value_heads=2,
        head_dim=24,
        global_head_dim=24,
        sliding_window=64,
    )
    models["Gemma-4-Hybrid"] = Gemma4ForCausalLM(gemma_cfg).eval()

    # 3. GLM-5.3 / GlmMoeDsa (MLA + Routed/Shared MoE)
    glm_cfg = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        moe_intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=24,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=8,
    )
    models["GLM-5.3-MoE-MLA"] = GlmMoeDsaForCausalLM(glm_cfg).eval()

    # 4. DeepSeek-V4 (MLA + Routed/Shared MoE Architecture)
    ds_cfg = DeepseekV4Config(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        moe_intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=24,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=8,
    )
    models["DeepSeek-V4-MLA"] = DeepseekV4ForCausalLM(ds_cfg).eval()

    # 5. Custom PyTorch Module with CustomGatedProjection
    models["Custom-Gated-PyTorch"] = CustomSequenceClassifier(vocab_size=256, hidden_size=96).eval()

    return models


def run_benchmark() -> Dict[str, Any]:
    models = build_modern_models()
    eval_batches = [
        torch.randint(0, 256, (2, 16), dtype=torch.long),
        torch.randint(0, 256, (2, 16), dtype=torch.long),
    ]
    long_batches = [
        torch.randint(0, 256, (1, 48), dtype=torch.long),
    ]

    results: Dict[str, Any] = {
        "native_cpu_status": native_leech_backend_status(),
        "cuda_kernel_status": cuda_kernel_source_status(),
        "models": {},
    }

    for name, base_model in models.items():
        t0 = time.perf_counter()
        arch_profile = introspect_architecture(base_model)

        # Quantize using Generalized Pareto VQ (both RVQ + Leech24 candidates)
        candidate_configs = [
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
        ]
        working_model = copy.deepcopy(base_model)
        q_result = quantize_model_generalized(
            working_model,
            candidate_configs=candidate_configs,
            calibration_inputs=eval_batches[:1],
            constraints=PlanningConstraints(
                target_state_ratio=0.999,
                max_tensor_weight_error=1.0,
                runtime_profile="memory",
                architecture_profile=arch_profile,
                max_total_runtime_bytes=24 * 1024**3,
                context_length=4096,
            ),
        )
        q_model = q_result.model

        # M0: 7-Bucket Total Memory Admission at 4096 context
        encoded_bytes = sum(l.storage_bytes for l in q_result.report.layers if l.selected)
        dense_bytes = sum(l.original_bytes for l in q_result.report.layers if not l.selected)
        admission = evaluate_memory_admission(
            arch_profile,
            encoded_weight_bytes=encoded_bytes,
            dense_weight_bytes=dense_bytes,
            context_length=4096,
            batch_size=1,
            kv_dtype=torch.float16,
            budget_bytes=24 * 1024**3,  # 24 GB laptop budget
        )
        max_ctx_64mb = max_admissible_context_length(
            arch_profile,
            budget_bytes=64 * 1024**2,
            encoded_weight_bytes=encoded_bytes,
            dense_weight_bytes=dense_bytes,
            batch_size=1,
            kv_dtype=torch.float16,
        )

        # M3: Quality Certificate
        cert = generate_quality_certificate(
            base_model,
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

        # M1: OQPack Save, Integrity Check & Zero-Copy Load
        with tempfile.TemporaryDirectory() as tmpdir:
            oq_dir = Path(tmpdir) / "oqpack_artifact"
            manifest = save_oqpack(
                q_model,
                oq_dir,
                plan=q_result.plan,
                architecture_profile=arch_profile,
                memory_admission=admission,
                quality_certificate=cert.to_dict(),
                model_id=name,
                max_shard_bytes=64 * 1024,  # Force multi-shard splitting
            )
            verified_manifest = verify_oqpack_integrity(oq_dir, verify_checksums=True)
            skeleton = copy.deepcopy(base_model)
            reloaded_model, _ = load_quantized_model_from_oqpack(
                skeleton, oq_dir, verify_checksums=True
            )
            with torch.inference_mode():
                out_q = q_model(eval_batches[0])
                out_r = reloaded_model(eval_batches[0])
                logits_q = out_q.logits if hasattr(out_q, "logits") else out_q
                logits_r = out_r.logits if hasattr(out_r, "logits") else out_r
                oqpack_exact_match = bool(torch.equal(logits_q, logits_r))

        # M4: KV-Cache Parity & Multi-Turn Chat (for CausalLMs that support use_cache)
        kv_parity_info: Dict[str, Any] = {"supported": False}
        if name in {"Qwen3.8-Hybrid (Qwen3_5)", "Gemma-4-Hybrid"}:
            engine = OjasXInferenceEngine(
                q_model,
                max_context_length=512,
                memory_budget_bytes=2 * 1024**3,
                kv_dtype=torch.float32,
            )
            parity = engine.verify_cached_vs_uncached_parity(
                [10, 20, 30, 40], max_new_tokens=4, atol=1e-4
            )
            t1 = engine.chat_turn([11, 22], SamplingParams(max_new_tokens=3, temperature=0.0))
            t2 = engine.chat_turn([33, 44], SamplingParams(max_new_tokens=3, temperature=0.0))
            kv_bytes_turn2 = t2.kv_cache_bytes
            engine.reset_chat()
            kv_bytes_after_reset = engine.kv_manager.stats().allocated_bytes
            kv_parity_info = {
                "supported": True,
                "matches_exactly": parity.matches_exactly,
                "max_logit_abs_diff": parity.max_logit_abs_diff,
                "turn1_prefill_tokens": t1.prefill_tokens,
                "turn2_prefill_tokens": t2.prefill_tokens,
                "turn2_kv_bytes": kv_bytes_turn2,
                "kv_bytes_after_reset": kv_bytes_after_reset,
            }

        # M5: Fused Native C++ Kernel Execution Check
        fused_modules = 0
        for mod in q_model.modules():
            if hasattr(mod, "set_fused_native_gemm"):
                mod.set_fused_native_gemm(True)
                fused_modules += 1
        with torch.inference_mode():
            out_fused = q_model(eval_batches[0])
            logits_fused = out_fused.logits if hasattr(out_fused, "logits") else out_fused
            max_fused_diff = float((logits_q - logits_fused).abs().max().item())

        elapsed = time.perf_counter() - t0
        results["models"][name] = {
            "model_type": arch_profile.model_type,
            "architecture_class": arch_profile.architecture_class,
            "attention_types": sorted({spec.attention_kind for spec in arch_profile.layer_kv_specs}),
            "selected_layers": q_result.report.selected_layers,
            "total_eligible_layers": len(q_result.report.layers),
            "weight_compression_ratio": round(
                q_result.report.selected_original_bytes / max(1, sum(l.storage_bytes for l in q_result.report.layers if l.selected)),
                3,
            ),
            "original_weight_bytes": q_result.report.original_bytes,
            "compressed_weight_bytes": q_result.report.stored_bytes,
            "m0_admission_4k_total_bytes": admission.total_required_bytes,
            "m0_kv_cache_4k_bytes": admission.kv_cache_bytes,
            "m0_max_ctx_under_64mb": max_ctx_64mb,
            "m1_oqpack_shards": len(verified_manifest.shards),
            "m1_oqpack_exact_match": oqpack_exact_match,
            "m3_certificate_passed": cert.passed,
            "m3_mean_kl": round(cert.mean_kl_divergence, 5),
            "m3_top1_agreement": round(cert.top1_agreement, 4),
            "m3_top5_agreement": round(cert.top5_agreement, 4),
            "m4_kv_cache_runtime": kv_parity_info,
            "m5_fused_native_modules": fused_modules,
            "m5_max_fused_abs_diff": max_fused_diff,
            "elapsed_sec": round(elapsed, 2),
        }

    # M2: Streaming Ingestion & Crash-Resume Verification on LlamaForCausalLM
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "source_ckpt"
        out_dir = Path(tmpdir) / "streamed_out"
        ckpt_dir.mkdir(parents=True)
        llama_cfg = LlamaConfig(
            vocab_size=256,
            hidden_size=96,
            intermediate_size=192,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        llama = LlamaForCausalLM(llama_cfg).eval()
        sd = {k: v.contiguous() for k, v in llama.state_dict().items()}
        keys = list(sd.keys())
        mid = len(keys) // 2
        save_file({k: sd[k] for k in keys[:mid]}, str(ckpt_dir / "model-00001-of-00002.safetensors"))
        save_file({k: sd[k] for k in keys[mid:]}, str(ckpt_dir / "model-00002-of-00002.safetensors"))
        weight_map = {
            k: ("model-00001-of-00002.safetensors" if i < mid else "model-00002-of-00002.safetensors")
            for i, k in enumerate(keys)
        }
        (ckpt_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
        try:
            stream_convert_checkpoint(
                ckpt_dir,
                out_dir,
                codec="leech24_shell2_residual",
                leech_config=LeechQuantizationConfig(
                    max_stages=2, gain_bits=2, rotation_reflections=2, max_relative_error=None, min_weight_elements=64
                ),
                fail_after_tensors=3,
            )
        except RuntimeError:
            pass
        m2_report = stream_convert_checkpoint(
            ckpt_dir,
            out_dir,
            codec="leech24_shell2_residual",
            leech_config=LeechQuantizationConfig(
                max_stages=2, gain_bits=2, rotation_reflections=2, max_relative_error=None, min_weight_elements=64
            ),
            resume=True,
        )
        results["m2_streaming_ingest"] = {
            "total_tensors": m2_report.total_tensors,
            "quantized_tensors": m2_report.quantized_tensors,
            "resumed_tensors": m2_report.resumed_tensors,
            "compression_ratio": round(m2_report.compression_ratio, 3),
            "peak_single_tensor_ratio": round(
                m2_report.peak_single_tensor_bytes / max(1, m2_report.original_bytes), 4
            ),
        }

    return results


if __name__ == "__main__":
    summary = run_benchmark()
    out_path = Path(__file__).resolve().parent / "m0_m5_qualification_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

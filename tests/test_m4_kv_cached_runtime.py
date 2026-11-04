"""Milestone M4: Genuine KV-Cached Reference Runtime & Multi-Turn Chat Engine Tests."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)
from transformers.cache_utils import DynamicCache

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.contract import AdmissionError
from ojasx_24x96.engine import (
    ChatTurn,
    GenerationResult,
    KVCacheParityReport,
    OjasXInferenceEngine,
    SamplingParams,
    apply_execution_profile,
)
from ojasx_24x96.generalized import quantize_model_generalized
from ojasx_24x96.kv_cache import KVCacheStats, OjasXKVCacheManager
from ojasx_24x96.leech_codec import LeechQuantizationConfig
from ojasx_24x96.module import quantize_model
from ojasx_24x96.planner import PlanningConstraints


def _create_tiny_model(architecture: str) -> nn.Module:
    """Create a fast, miniature test model conforming to modern architectural profiles."""
    if architecture == "llama":
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        return LlamaForCausalLM(cfg).eval()

    if architecture == "gemma4":
        cfg = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            max_position_embeddings=64,
            sliding_window=32,
            layer_types=["sliding_attention", "full_attention"],
            tie_word_embeddings=True,
            use_cache=True,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=64,
        )
        cfg.allow_global_per_layer_attribute_access = True
        return Gemma4ForCausalLM(cfg).eval()

    if architecture == "qwen3_5":
        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            tie_word_embeddings=True,
            use_cache=True,
        )
        return Qwen3_5ForCausalLM(cfg).eval()

    raise ValueError(f"Unknown test architecture: {architecture}")


def _quantize_with_codec(model: nn.Module, codec: str) -> nn.Module:
    """Apply adaptive_residual_vq or leech24_shell2_residual vector quantization."""
    if codec == "adaptive_residual_vq":
        config = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            max_relative_error=None,
            require_compression=False,
            min_weight_elements=1,
        )
        quantized, report = quantize_model(model, config)
        assert report.selected_layers > 0, "At least one layer must be ARVQ quantized"
        return quantized

    if codec == "leech24_shell2_residual":
        leech_config = LeechQuantizationConfig(
            max_stages=2,
            max_relative_error=None,
            gain_bits=1,
            rotation_reflections=2,
            adaptive_stages=True,
            max_gain_samples=128,
            gain_grid_size=8,
            chunk_size=16,
            min_weight_elements=1,
            require_compression=False,
            seed=20261104,
        )
        result = quantize_model_generalized(
            model,
            constraints=PlanningConstraints(
                target_state_ratio=0.95,
                max_tensor_estimated_error=10.0,
                decode_row_block_size=7,
            ),
            calibration_inputs=[{"input_ids": torch.tensor([[1, 2, 3, 4]])}],
            candidate_configs=[leech_config],
            include=("*self_attn*", "*mlp*"),
        )
        assert result.report.selected_layers > 0, "At least one layer must be Leech quantized"
        return model

    raise ValueError(f"Unknown codec: {codec}")


# ---------------------------------------------------------------------------
# Test 1: KV Cache Manager Lifecycle, Accounting, Cropping, and Bounds
# ---------------------------------------------------------------------------


def test_kv_cache_manager_lifecycle_and_accounting():
    mgr = OjasXKVCacheManager(max_context_length=64, default_dtype=torch.float16)

    # Initial zero state
    stats = mgr.stats()
    assert stats.allocated_bytes == 0
    assert stats.sequence_length == 0
    assert stats.max_context_length == 64
    assert stats.num_cached_layers == 0
    assert stats.dtype == "float16"
    assert isinstance(stats.to_dict(), dict)

    # Mock DynamicCache with key and value tensors
    cache = DynamicCache()

    class MockLayer:
        def __init__(self, seq_len: int = 8):
            self.keys = torch.zeros(1, 2, seq_len, 16, dtype=torch.float16)
            self.values = torch.zeros(1, 2, seq_len, 16, dtype=torch.float16)

    cache.layers.append(MockLayer(8))
    cache.layers.append(MockLayer(8))

    mgr.update(cache)
    stats_updated = mgr.stats()
    assert stats_updated.sequence_length == 8
    assert stats_updated.num_cached_layers == 2
    # 2 layers * 2 tensors * (1 * 2 * 8 * 16 * 2 bytes) = 2048 bytes
    assert stats_updated.allocated_bytes == 2048
    assert stats_updated.dtype == "float16"

    # Cropping
    mgr.crop(5)
    stats_cropped = mgr.stats()
    assert stats_cropped.sequence_length == 5
    assert cache.layers[0].keys.shape[-2] == 5
    assert cache.layers[1].values.shape[-2] == 5
    # 4 tensors * (1 * 2 * 5 * 16 * 2 bytes) = 1280 bytes
    assert stats_cropped.allocated_bytes == 1280

    # Context admission bound enforcement
    with pytest.raises(ValueError, match="admission bound"):
        mgr.update(cache, sequence_length=100)

    # Reset
    mgr.reset()
    assert mgr.stats().allocated_bytes == 0
    assert mgr.stats().sequence_length == 0
    assert mgr.past_key_values is None


# ---------------------------------------------------------------------------
# Test 2: Cached vs Uncached Parity Across Architectures & Codecs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["llama", "gemma4", "qwen3_5"])
@pytest.mark.parametrize("codec", ["adaptive_residual_vq", "leech24_shell2_residual"])
def test_cached_vs_uncached_parity_all_architectures(architecture: str, codec: str):
    raw_model = _create_tiny_model(architecture)
    quantized_model = _quantize_with_codec(raw_model, codec)

    engine = OjasXInferenceEngine(
        quantized_model,
        max_context_length=64,
        execution_profile="memory",
    )

    prompt = [1, 2, 3, 4]
    report = engine.verify_cached_vs_uncached_parity(prompt, max_new_tokens=4, atol=1e-4)

    assert isinstance(report, KVCacheParityReport)
    assert report.matches_exactly is True
    assert report.cached_kv_bytes > 0
    assert report.max_logit_abs_diff <= 1e-4
    assert report.cached_token_ids == report.uncached_token_ids
    assert len(report.cached_token_ids) == 4


# ---------------------------------------------------------------------------
# Test 3: Multi-Turn Chat Reuses Cached Prefix Tokens Across 3 Turns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["llama", "qwen3_5"])
def test_multi_turn_chat_reuses_cached_prefix(architecture: str):
    raw_model = _create_tiny_model(architecture)
    quantized_model = _quantize_with_codec(raw_model, "adaptive_residual_vq")

    engine = OjasXInferenceEngine(quantized_model, max_context_length=128)
    params = SamplingParams(max_new_tokens=4, temperature=0.0)

    # Turn 1: Fresh prompt of 2 tokens
    res1 = engine.chat_turn([10, 20], params)
    assert res1.prefill_tokens == 2
    assert len(res1.generated_token_ids) == 4
    bytes_turn1 = res1.kv_cache_bytes
    assert bytes_turn1 > 0

    # Turn 2: User sends 2 new tokens -> Only delta (2 tokens) should be prefilled!
    res2 = engine.chat_turn([30, 40], params)
    assert res2.prefill_tokens == 2, f"Expected 2 delta tokens prefilled, got {res2.prefill_tokens}"
    assert len(res2.generated_token_ids) == 4
    bytes_turn2 = res2.kv_cache_bytes
    assert bytes_turn2 > bytes_turn1

    # Turn 3: User sends 2 new tokens -> Only delta (2 tokens) should be prefilled!
    res3 = engine.chat_turn([50, 60], params)
    assert res3.prefill_tokens == 2, f"Expected 2 delta tokens prefilled, got {res3.prefill_tokens}"
    assert len(res3.generated_token_ids) == 4
    bytes_turn3 = res3.kv_cache_bytes
    assert bytes_turn3 > bytes_turn2

    # Check conversation turn history: 3 user turns + 3 assistant turns = 6
    assert len(engine.chat_history) == 6
    assert engine.chat_history[0].role == "user"
    assert engine.chat_history[1].role == "assistant"
    assert engine.chat_history[2].role == "user"
    assert engine.chat_history[3].role == "assistant"
    assert engine.chat_history[4].role == "user"
    assert engine.chat_history[5].role == "assistant"

    # Reset chat frees all KV cache back to 0 bytes
    engine.reset_chat()
    assert engine.kv_manager.stats().allocated_bytes == 0
    assert len(engine.chat_history) == 0
    assert len(engine.conversation_tokens) == 0


# ---------------------------------------------------------------------------
# Test 4: Stream Generate Yields Identical Tokens to Generate
# ---------------------------------------------------------------------------


def test_stream_generate_yields_identical_tokens():
    raw_model = _create_tiny_model("llama")
    quantized_model = _quantize_with_codec(raw_model, "adaptive_residual_vq")
    engine = OjasXInferenceEngine(quantized_model, max_context_length=64)

    prompt = [2, 4, 6]

    # Greedy comparison
    greedy_params = SamplingParams(max_new_tokens=6, temperature=0.0)
    gen_greedy = engine.generate(prompt, greedy_params, use_cache=True, reset_cache=True)
    stream_greedy = list(engine.stream_generate(prompt, greedy_params, use_cache=True, reset_cache=True))
    assert gen_greedy.generated_token_ids == stream_greedy

    # Stochastic comparison with explicit seed
    sample_params = SamplingParams(max_new_tokens=6, temperature=0.8, top_k=5, seed=2026)
    gen_sample = engine.generate(prompt, sample_params, use_cache=True, reset_cache=True)
    stream_sample = list(engine.stream_generate(prompt, sample_params, use_cache=True, reset_cache=True))
    assert gen_sample.generated_token_ids == stream_sample


# ---------------------------------------------------------------------------
# Test 5: Early Stopping with EOS and Stop Sequences
# ---------------------------------------------------------------------------


def test_early_stopping_eos_and_stop_sequences():
    raw_model = _create_tiny_model("llama")
    quantized_model = _quantize_with_codec(raw_model, "adaptive_residual_vq")
    engine = OjasXInferenceEngine(quantized_model, max_context_length=64)

    prompt = [1, 2, 3]
    baseline = engine.generate(prompt, SamplingParams(max_new_tokens=10, temperature=0.0))
    assert len(baseline.generated_token_ids) == 10
    assert baseline.stop_reason == "max_new_tokens"

    # Stop via EOS token
    target_eos = baseline.generated_token_ids[2]
    first_eos_idx = baseline.generated_token_ids.index(target_eos)
    res_eos = engine.generate(
        prompt,
        SamplingParams(max_new_tokens=10, temperature=0.0, eos_token_ids=(target_eos,)),
    )
    assert res_eos.stop_reason == "eos"
    assert res_eos.generated_token_ids == baseline.generated_token_ids[: first_eos_idx + 1]

    # Stop via multi-token stop sequence
    target_stop_seq = baseline.generated_token_ids[3:5]
    # Find first occurrence of target_stop_seq in baseline
    first_seq_end = next(
        idx + len(target_stop_seq)
        for idx in range(len(baseline.generated_token_ids) - len(target_stop_seq) + 1)
        if baseline.generated_token_ids[idx : idx + len(target_stop_seq)] == target_stop_seq
    )
    res_seq = engine.generate(
        prompt,
        SamplingParams(max_new_tokens=10, temperature=0.0, stop_sequences=(target_stop_seq,)),
    )
    assert res_seq.stop_reason == "stop_sequence"
    assert res_seq.generated_token_ids == baseline.generated_token_ids[:first_seq_end]


# ---------------------------------------------------------------------------
# Test 6: Memory Admission Budget Enforces AdmissionError
# ---------------------------------------------------------------------------


def test_memory_admission_error_on_exceeded_budget():
    model = _create_tiny_model("llama")

    # A budget of 1000 bytes is far smaller than model weights (~300 KB) + activations + KV cache
    with pytest.raises(AdmissionError) as exc_info:
        _ = OjasXInferenceEngine(
            model,
            max_context_length=512,
            memory_budget_bytes=1000,
        )

    error_message = str(exc_info.value)
    assert "Memory admission rejected" in error_message
    assert "total_required_bytes" in error_message
    assert "budget_bytes" in error_message
    assert exc_info.value.report.admitted is False

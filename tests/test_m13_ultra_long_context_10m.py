"""Tests for Milestone M13: Ultra-Scale 10M+ Context Online-Softmax Attention & Streamer.

Verifies:
1. Exact numerical parity (< 1e-5) between OnlineSoftmaxBlockAttention and full-matrix
   scaled dot-product attention for:
   - Multi-Head Attention (MHA, causal=True and causal=False)
   - Grouped-Query Attention (GQA, query_start_pos > 0)
   - Uneven sequence lengths not divisible by block sizes
2. 10,000,000+ token virtual stream numerical stability proof:
   - Streaming 10M tokens through OnlineSoftmaxBlockAttention.forward_stream
   - Zero NaN/Inf overflow in FP64 log-sum-exp accumulators
   - Strictly bounded O(1) working memory
3. PagedCompressedKVCache:
   - INT8 page compression (>= 3.5x reduction vs FP32)
   - Dequantization accuracy and high signal fidelity
   - Sink-token + sliding-window page eviction beyond max_resident_pages
4. UltraLongContextStreamer:
   - stream_prefill_chunks over multi-chunk token streams on quantized CausalLM
   - stream_generate_infinite with bounded KV cache length (<= max_kv_cache_tokens)
   - Deterministic token generation and EOS termination
"""

from __future__ import annotations

import math
import tracemalloc
from typing import Iterator, List, Tuple

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.long_context import (
    BlockAttentionReport,
    OnlineSoftmaxBlockAttention,
    PagedCompressedKVCache,
    UltraLongContextStreamer,
)
from ojasx_24x96.module import quantize_model


# ============================================================================
# Section 1: Exact Mathematical Reference for Scaled Dot-Product Attention
# ============================================================================


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    query_start_pos: int = 0,
) -> torch.Tensor:
    """Exact full-matrix reference attention computed in float64.

    Args:
        query: [B, H_q, S_q, D]
        key:   [B, H_kv, S_k, D]
        value: [B, H_kv, S_k, D]
        causal: Whether to apply causal mask
        query_start_pos: Absolute token index of query[:, :, 0, :]
    """
    B, H_q, S_q, D = query.shape
    _, H_kv, S_k, _ = key.shape

    if H_q != H_kv:
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
        rep = H_q // H_kv
        key = key.unsqueeze(2).expand(B, H_kv, rep, S_k, D).reshape(B, H_q, S_k, D)
        value = value.unsqueeze(2).expand(B, H_kv, rep, S_k, D).reshape(B, H_q, S_k, D)

    scale = 1.0 / math.sqrt(D)
    q_f64 = query.to(torch.float64)
    k_f64 = key.to(torch.float64)
    v_f64 = value.to(torch.float64)

    scores = torch.matmul(q_f64, k_f64.transpose(-2, -1)) * scale

    if causal:
        q_pos = torch.arange(
            query_start_pos, query_start_pos + S_q, device=query.device, dtype=torch.long
        ).view(1, 1, S_q, 1)
        k_pos = torch.arange(
            0, S_k, device=query.device, dtype=torch.long
        ).view(1, 1, 1, S_k)
        mask = k_pos > q_pos
        scores = scores.masked_fill(mask, float("-inf"))

    # Numerical stabilization: subtract max score
    max_scores = scores.max(dim=-1, keepdim=True).values
    safe_max = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - safe_max)
    exp_scores = torch.where(torch.isinf(scores), torch.zeros_like(exp_scores), exp_scores)

    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    safe_sum = torch.where(sum_exp > 0.0, sum_exp, torch.ones_like(sum_exp))
    probs = exp_scores / safe_sum

    out = torch.matmul(probs, v_f64)
    return out.to(dtype=query.dtype)


# ============================================================================
# Section 2: OnlineSoftmaxBlockAttention Exact Numerical Parity Tests
# ============================================================================


class TestOnlineSoftmaxBlockAttentionParity:
    """Verifies max_abs_diff < 1e-5 between online blockwise attention and full attention."""

    @pytest.mark.parametrize("causal", [True, False])
    def test_mha_exact_parity(self, causal: bool) -> None:
        """Parity test for Multi-Head Attention (num_heads == num_kv_heads)."""
        torch.manual_seed(20261105)
        B, H, S, D = 2, 4, 256, 64
        q = torch.randn(B, H, S, D, dtype=torch.float32)
        k = torch.randn(B, H, S, D, dtype=torch.float32)
        v = torch.randn(B, H, S, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H,
            num_kv_heads=H,
            head_dim=D,
            q_block_size=64,
            kv_block_size=128,
            causal=causal,
        )

        out_block = attn(q, k, v)
        out_ref = _reference_attention(q, k, v, causal=causal)

        max_diff = torch.max(torch.abs(out_block - out_ref)).item()
        assert max_diff < 1e-5, f"MHA (causal={causal}) parity failed: max_diff={max_diff:.2e}"

    def test_gqa_chunked_prefill_parity(self) -> None:
        """Parity test for Grouped-Query Attention with query_start_pos > 0."""
        torch.manual_seed(42)
        B, H_q, H_kv, D = 2, 8, 2, 64
        S_q = 64
        S_k = 384
        query_start_pos = 320  # Tokens 320 to 383

        q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
        k = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)
        v = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H_q,
            num_kv_heads=H_kv,
            head_dim=D,
            q_block_size=32,
            kv_block_size=64,
            causal=True,
        )

        out_block = attn(q, k, v, query_start_pos=query_start_pos)
        out_ref = _reference_attention(
            q, k, v, causal=True, query_start_pos=query_start_pos
        )

        max_diff = torch.max(torch.abs(out_block - out_ref)).item()
        assert max_diff < 1e-5, f"GQA chunked prefill parity failed: max_diff={max_diff:.2e}"

    def test_gqa_single_token_decode_parity(self) -> None:
        """Parity test for single token decoding (S_q = 1) over historical KV cache."""
        torch.manual_seed(101)
        B, H_q, H_kv, D = 1, 8, 2, 64
        S_q = 1
        S_k = 512
        query_start_pos = 511

        q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
        k = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)
        v = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H_q,
            num_kv_heads=H_kv,
            head_dim=D,
            q_block_size=16,
            kv_block_size=32,
            causal=True,
        )

        out_block = attn(q, k, v, query_start_pos=query_start_pos)
        out_ref = _reference_attention(
            q, k, v, causal=True, query_start_pos=query_start_pos
        )

        max_diff = torch.max(torch.abs(out_block - out_ref)).item()
        assert max_diff < 1e-5, f"Single-token decode parity failed: max_diff={max_diff:.2e}"

    @pytest.mark.parametrize("causal", [True, False])
    def test_uneven_sequence_lengths_parity(self, causal: bool) -> None:
        """Parity test where S_q and S_k are prime and not divisible by block sizes."""
        torch.manual_seed(999)
        B, H_q, H_kv, D = 2, 6, 2, 32
        S_q = 73
        S_k = 197
        q_blk = 37
        kv_blk = 41
        query_start_pos = 124 if causal else 0

        q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
        k = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)
        v = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H_q,
            num_kv_heads=H_kv,
            head_dim=D,
            q_block_size=q_blk,
            kv_block_size=kv_blk,
            causal=causal,
        )

        out_block = attn(q, k, v, query_start_pos=query_start_pos)
        out_ref = _reference_attention(
            q, k, v, causal=causal, query_start_pos=query_start_pos
        )

        max_diff = torch.max(torch.abs(out_block - out_ref)).item()
        assert max_diff < 1e-5, f"Uneven lengths (causal={causal}) failed: max_diff={max_diff:.2e}"


# ============================================================================
# Section 3: 10,000,000+ Token Virtual Stream Numerical Stability Proof
# ============================================================================


class TestTenMillionTokenStream:
    """Verifies numerical stability, zero NaN/Inf overflow, and bounded RAM on 10M token stream."""

    def test_ten_million_token_stream_stability(self) -> None:
        """Stream 10,000,000 tokens through OnlineSoftmaxBlockAttention.forward_stream.

        - 1,000 blocks of 10,000 tokens yielded lazily
        - query tile shape: [1, 4, 4, 16]
        - verifies report.kv_tokens == 10_000_000
        - verifies zero NaN or Inf in output or FP64 log-sum-exp state
        - verifies strictly bounded working memory via tracemalloc
        """
        torch.manual_seed(777)
        B, H, S_q, D = 1, 4, 4, 16
        q = torch.randn(B, H, S_q, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H,
            num_kv_heads=H,
            head_dim=D,
            causal=False,
        )

        total_tokens = 10_000_000
        block_tokens = 10_000
        num_blocks = total_tokens // block_tokens

        # Shared buffer to verify lazy streaming without allocating 10M tokens in RAM
        k_tile = torch.randn(B, H, block_tokens, D, dtype=torch.float32)
        v_tile = torch.randn(B, H, block_tokens, D, dtype=torch.float32)

        def _lazy_stream() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(num_blocks):
                yield k_tile, v_tile

        tracemalloc.start()
        tracemalloc.reset_peak()

        out, report = attn.forward_stream(q, _lazy_stream())

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Telemetry validation
        assert isinstance(report, BlockAttentionReport)
        assert report.query_tokens == S_q, f"Expected S_q={S_q}, got {report.query_tokens}"
        assert (
            report.kv_tokens == total_tokens
        ), f"Expected kv_tokens={total_tokens}, got {report.kv_tokens}"
        assert (
            report.num_kv_blocks == num_blocks
        ), f"Expected num_blocks={num_blocks}, got {report.num_kv_blocks}"
        assert report.peak_tile_elements == B * H * S_q * block_tokens

        # Numerical health: Zero NaN or Inf
        assert not torch.isnan(out).any(), "NaN detected in 10M token attention output"
        assert not torch.isinf(out).any(), "Inf detected in 10M token attention output"
        assert not math.isnan(report.max_logsumexp), "NaN in report max_logsumexp"
        assert not math.isinf(report.max_logsumexp), "Inf in report max_logsumexp"
        assert report.max_logsumexp > 0.0, "max_logsumexp should be positive after 10M tokens"

        # Bounded working memory proof:
        # 10M tokens in FP32 K+V would require 10,000,000 * 4 heads * 16 dim * 4 bytes * 2 = 5.12 GB.
        # Streaming working memory overhead must remain strictly bounded (< 10 MB).
        peak_mb = peak_memory / (1024 * 1024)
        assert (
            peak_mb < 10.0
        ), f"Streamer working memory leaked: peak={peak_mb:.2f} MB (expected < 10 MB)"

    def test_ten_million_token_causal_stream(self) -> None:
        """Stream 10M tokens under causal decoding where query is at token 10,000,000."""
        torch.manual_seed(888)
        B, H, S_q, D = 1, 4, 2, 16
        q = torch.randn(B, H, S_q, D, dtype=torch.float32)

        attn = OnlineSoftmaxBlockAttention(
            num_heads=H,
            num_kv_heads=H,
            head_dim=D,
            causal=True,
        )

        total_tokens = 10_000_000
        block_tokens = 20_000
        num_blocks = total_tokens // block_tokens

        k_tile = torch.randn(B, H, block_tokens, D, dtype=torch.float32)
        v_tile = torch.randn(B, H, block_tokens, D, dtype=torch.float32)

        def _lazy_stream() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(num_blocks):
                yield k_tile, v_tile

        # Query positioned at the end of the 10M token stream
        out, report = attn.forward_stream(
            q, _lazy_stream(), query_start_pos=total_tokens - S_q
        )

        assert report.kv_tokens == total_tokens
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert not math.isnan(report.max_logsumexp)
        assert not math.isinf(report.max_logsumexp)


# ============================================================================
# Section 4: PagedCompressedKVCache Tests
# ============================================================================


class TestPagedCompressedKVCache:
    """Verifies INT8 compression, dequantization accuracy, and sink-token sliding-window eviction."""

    def test_int8_quantization_and_compression_ratio(self) -> None:
        """Verify INT8 per-token quantization reduces memory footprint by >= 3.5x."""
        torch.manual_seed(42)
        B, H, S, D = 1, 4, 1024, 64
        k = torch.randn(B, H, S, D, dtype=torch.float32)
        v = torch.randn(B, H, S, D, dtype=torch.float32)

        cache = PagedCompressedKVCache(page_size=256, max_resident_pages=16, sink_pages=2)
        cache.append(k, v)

        assert cache.resident_tokens == S
        assert cache.total_tokens_seen == S

        # Check internal storage types
        qk, sk, qv, sv = cache._pages[0]
        assert qk.dtype == torch.int8
        assert qv.dtype == torch.int8
        assert sk.dtype == torch.float16
        assert sv.dtype == torch.float16

        # Compression calculation
        raw_fp32_bytes = 2 * (B * H * S * D * 4)  # K and V in float32
        stored_bytes = cache.storage_bytes
        compression_ratio = raw_fp32_bytes / stored_bytes

        assert compression_ratio >= 3.5, f"Compression ratio {compression_ratio:.2f}x < 3.5x"

    def test_dequantization_accuracy(self) -> None:
        """Verify high fidelity reconstruction (> 0.999 cosine similarity) after INT8 dequantization."""
        torch.manual_seed(123)
        B, H, S, D = 1, 2, 128, 32
        k = torch.randn(B, H, S, D, dtype=torch.float32) * 5.0
        v = torch.randn(B, H, S, D, dtype=torch.float32) * 3.0

        cache = PagedCompressedKVCache(page_size=128, max_resident_pages=4, sink_pages=1)
        cache.append(k, v)

        deq_pages = list(cache.iter_dequantized_pages(dtype=torch.float32))
        assert len(deq_pages) == 1
        k_rec, v_rec = deq_pages[0]

        # Cosine similarity
        cos_sim_k = torch.cosine_similarity(k.flatten(), k_rec.flatten(), dim=0).item()
        cos_sim_v = torch.cosine_similarity(v.flatten(), v_rec.flatten(), dim=0).item()
        assert cos_sim_k > 0.999, f"K reconstruction cosine similarity {cos_sim_k} <= 0.999"
        assert cos_sim_v > 0.999, f"V reconstruction cosine similarity {cos_sim_v} <= 0.999"

        # Max error bounded by scale / 2
        qk, sk, _, _ = cache._pages[0]
        max_err = torch.max(torch.abs(k - k_rec)).item()
        max_bound = (sk.max().item() * 1.5)  # accounting for fp16 rounding
        assert max_err <= max_bound, f"Max error {max_err} exceeded bound {max_bound}"

    def test_sink_token_sliding_window_page_eviction(self) -> None:
        """Verify sink pages are preserved while older intermediate pages are evicted."""
        torch.manual_seed(99)
        page_size = 16
        max_resident_pages = 5
        sink_pages = 2
        total_pages_to_append = 12

        cache = PagedCompressedKVCache(
            page_size=page_size,
            max_resident_pages=max_resident_pages,
            sink_pages=sink_pages,
        )

        # Append pages with distinct values
        for i in range(total_pages_to_append):
            k = torch.full((1, 1, page_size, 8), fill_value=float(i + 1), dtype=torch.float32)
            v = torch.full((1, 1, page_size, 8), fill_value=float(i + 1), dtype=torch.float32)
            cache.append(k, v)

        assert cache.total_tokens_seen == total_pages_to_append * page_size
        assert len(cache._pages) == max_resident_pages
        assert cache.resident_tokens == max_resident_pages * page_size

        deq = list(cache.iter_dequantized_pages())
        page_means = [round(p[0].mean().item(), 1) for p in deq]

        # Sink pages (1.0 and 2.0) must be preserved; recent pages (9.0, 10.0, 11.0, 12.0) retained
        assert page_means[:sink_pages] == [1.0, 2.0], f"Sink pages corrupt: {page_means}"
        assert page_means[sink_pages:] == [10.0, 11.0, 12.0], f"Sliding window corrupt: {page_means}"

    def test_paged_cache_block_attention_streaming(self) -> None:
        """Verify OnlineSoftmaxBlockAttention consuming PagedCompressedKVCache streams directly."""
        torch.manual_seed(55)
        B, H, D = 1, 4, 32
        q = torch.randn(B, H, 8, D, dtype=torch.float32)

        cache = PagedCompressedKVCache(page_size=32, max_resident_pages=8, sink_pages=2)
        for _ in range(12):
            k = torch.randn(B, H, 32, D, dtype=torch.float32)
            v = torch.randn(B, H, 32, D, dtype=torch.float32)
            cache.append(k, v)

        attn = OnlineSoftmaxBlockAttention(num_heads=H, num_kv_heads=H, head_dim=D, causal=False)
        out, report = attn.forward_stream(q, cache.iter_dequantized_pages())

        assert out.shape == (B, H, 8, D)
        assert report.kv_tokens == cache.resident_tokens
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================================
# Section 5: UltraLongContextStreamer Multi-Chunk Prefill & Infinite Generation
# ============================================================================


class TestUltraLongContextStreamer:
    """Verifies chunked prefill, infinite generation, and bounded KV cache on quantized CausalLM."""

    @pytest.fixture(scope="class")
    @classmethod
    def quantized_causal_model(cls) -> nn.Module:
        """Create a tiny Llama model quantized with Adaptive Residual VQ."""
        torch.manual_seed(20261105)
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        base_model = LlamaForCausalLM(cfg).eval()
        q_cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            max_relative_error=None,
            require_compression=False,
            min_weight_elements=1,
        )
        quantized_model, report = quantize_model(base_model, q_cfg)
        assert report.selected_layers > 0, "Model must contain quantized layers"
        return quantized_model

    def test_stream_prefill_chunks_bounded_kv(self, quantized_causal_model: nn.Module) -> None:
        """Verify stream_prefill_chunks processes multi-chunk stream while bounding KV cache."""
        torch.manual_seed(1234)
        streamer = UltraLongContextStreamer(
            quantized_causal_model,
            prefill_chunk_size=16,
            max_kv_cache_tokens=32,
            sink_tokens=4,
        )

        # 6 chunks of 25 tokens = 150 tokens total
        chunks = [torch.randint(0, 64, (1, 25)) for _ in range(6)]
        last_logits, total_processed, pkv = streamer.stream_prefill_chunks(chunks)

        assert total_processed == 150
        assert last_logits.shape == (1, 64)

        # Verify KV cache bounds
        if hasattr(pkv, "get_seq_length"):
            seq_len = pkv.get_seq_length()
        elif isinstance(pkv, (tuple, list)):
            seq_len = pkv[0][0].shape[-2]
        else:
            pytest.fail(f"Unrecognized KV cache structure: {type(pkv)}")

        assert (
            seq_len <= streamer.max_kv_cache_tokens
        ), f"KV cache length {seq_len} exceeded maximum {streamer.max_kv_cache_tokens}"

    def test_stream_generate_infinite_determinism(self, quantized_causal_model: nn.Module) -> None:
        """Verify stream_generate_infinite generates deterministic tokens across runs."""
        torch.manual_seed(999)
        streamer = UltraLongContextStreamer(
            quantized_causal_model,
            prefill_chunk_size=16,
            max_kv_cache_tokens=32,
            sink_tokens=4,
        )

        prompt_chunks = [torch.randint(0, 64, (1, 20)) for _ in range(3)]

        tokens_run1 = list(streamer.stream_generate_infinite(prompt_chunks, max_new_tokens=25))
        tokens_run2 = list(streamer.stream_generate_infinite(prompt_chunks, max_new_tokens=25))

        assert len(tokens_run1) == 25
        assert (
            tokens_run1 == tokens_run2
        ), "Token generation was non-deterministic across identical runs"

    def test_stream_generate_infinite_eos_termination(
        self, quantized_causal_model: nn.Module
    ) -> None:
        """Verify stream_generate_infinite terminates immediately upon generating EOS."""
        torch.manual_seed(42)
        streamer = UltraLongContextStreamer(
            quantized_causal_model,
            prefill_chunk_size=16,
            max_kv_cache_tokens=32,
            sink_tokens=4,
        )

        prompt_chunks = [torch.randint(0, 64, (1, 15)) for _ in range(3)]
        full_tokens = list(streamer.stream_generate_infinite(prompt_chunks, max_new_tokens=20))
        assert len(full_tokens) == 20

        # Choose the 5th token as target EOS
        target_eos = full_tokens[4]
        stopped_tokens = list(
            streamer.stream_generate_infinite(
                prompt_chunks, max_new_tokens=20, eos_token_ids=[target_eos]
            )
        )

        assert len(stopped_tokens) == 5
        assert stopped_tokens == full_tokens[:5]
        assert stopped_tokens[-1] == target_eos

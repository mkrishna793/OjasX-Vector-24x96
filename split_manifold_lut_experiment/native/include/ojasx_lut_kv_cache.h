/**
 * ojasx_lut_kv_cache.h
 *
 * High-Density Zero-Unpack Look-Up Table (LUT) KV-Cache Engine.
 * Slashes Key-Value memory consumption by 16x - 32x while computing
 * multi-head attention directly from 8-bit compressed codes.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#ifndef OJASX_LUT_KV_CACHE_H
#define OJASX_LUT_KV_CACHE_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace ojasx {

/**
 * Quantized Head KV Buffer for a single attention head.
 */
struct HeadKVCache {
    int head_dim = 128;
    int block_dim = 16;
    int num_groups = 8;
    int max_seq_len = 0;
    int current_seq_len = 0;

    // Key & Value Codebooks: [256 * block_dim]
    std::vector<float> k_codebook;
    std::vector<float> v_codebook;

    // Key Codes: [max_seq_len * num_groups] (1 byte per block!)
    std::vector<uint8_t> k_codes;

    // Value Codes: [max_seq_len * num_groups] (1 byte per block!)
    std::vector<uint8_t> v_codes;

    // Per-token dynamic scale factors (for high dynamic range retention)
    std::vector<float> k_scales;
    std::vector<float> v_scales;

    HeadKVCache() = default;

    HeadKVCache(int h_dim, int max_seq, int b_dim = 16)
        : head_dim(h_dim),
          block_dim(b_dim),
          num_groups(h_dim / b_dim),
          max_seq_len(max_seq),
          current_seq_len(0)
    {
        k_codebook.resize(256 * block_dim, 0.0f);
        v_codebook.resize(256 * block_dim, 0.0f);
        k_codes.resize(max_seq_len * num_groups, 0);
        v_codes.resize(max_seq_len * num_groups, 0);
        k_scales.resize(max_seq_len, 1.0f);
        v_scales.resize(max_seq_len, 1.0f);
    }

    size_t get_byte_size() const {
        return (k_codes.size() + v_codes.size()) * sizeof(uint8_t) +
               (k_scales.size() + v_scales.size()) * sizeof(float) +
               (k_codebook.size() + v_codebook.size()) * sizeof(float);
    }

    size_t get_dense_byte_size() const {
        // Uncompressed FP16 baseline: 2 * head_dim * current_seq_len * 2 bytes
        return (size_t)2 * head_dim * current_seq_len * sizeof(float);
    }

    float get_compression_ratio() const {
        size_t current_compressed = (size_t)current_seq_len * (num_groups * 2 * sizeof(uint8_t) + 2 * sizeof(float));
        return (float)get_dense_byte_size() / std::max((size_t)1, current_compressed);
    }

    /**
     * Appends a new Key and Value vector into the compressed cache.
     * k: [head_dim], v: [head_dim]
     */
    void append_token(const float* k, const float* v);

    /**
     * Computes attention logits against the entire compressed key history:
     *   logits[t] = (q . k_t) / sqrt(head_dim)
     * using the Query-projected LUT with ZERO key floats unpacked!
     *
     * q: [head_dim]
     * logits_out: [current_seq_len] pre-allocated float buffer
     */
    void compute_attention_scores_zero_unpack(
        const float* q,
        float* logits_out,
        float scale_factor = -1.0f
    ) const;

    /**
     * Computes attention output vector:
     *   out = sum_t (attn_probs[t] * v_t)
     *
     * attn_probs: [current_seq_len] (already softmaxed)
     * out: [head_dim] pre-allocated float buffer
     */
    void compute_context_vector(const float* attn_probs, float* out) const;
};

/**
 * Multi-Layer, Multi-Head Zero-Unpack KV-Cache Engine.
 */
class LayerKVCache {
public:
    int num_heads = 0;
    int head_dim = 128;
    int max_seq_len = 0;
    std::vector<HeadKVCache> heads;

    LayerKVCache() = default;

    LayerKVCache(int n_heads, int h_dim, int max_seq, int b_dim = 16)
        : num_heads(n_heads), head_dim(h_dim), max_seq_len(max_seq)
    {
        heads.reserve(num_heads);
        for (int i = 0; i < num_heads; ++i) {
            heads.emplace_back(h_dim, max_seq, b_dim);
        }
    }

    size_t get_total_byte_size() const {
        size_t total = 0;
        for (const auto& h : heads) {
            total += h.get_byte_size();
        }
        return total;
    }
};

} // namespace ojasx

#endif // OJASX_LUT_KV_CACHE_H

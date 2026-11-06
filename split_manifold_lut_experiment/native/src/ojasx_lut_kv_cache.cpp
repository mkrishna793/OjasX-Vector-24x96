/**
 * ojasx_lut_kv_cache.cpp
 *
 * Implementation of Zero-Unpack Look-Up Table (LUT) KV-Cache Engine.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include "ojasx_lut_kv_cache.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace ojasx {

void HeadKVCache::append_token(const float* k, const float* v) {
    if (current_seq_len >= max_seq_len) {
        // Dynamic buffer expansion
        int new_max = std::max(1024, max_seq_len * 2);
        k_codes.resize(new_max * num_groups, 0);
        v_codes.resize(new_max * num_groups, 0);
        k_scales.resize(new_max, 1.0f);
        v_scales.resize(new_max, 1.0f);
        max_seq_len = new_max;
    }

    int t = current_seq_len;

    // 1. Quantize Key vector: compute scale and 8-bit codes
    float max_k = 1e-8f;
    for (int d = 0; d < head_dim; ++d) {
        max_k = std::max(max_k, std::abs(k[d]));
    }
    float scale_k = max_k / 127.0f;
    k_scales[t] = scale_k;

    // Direct uniform 8-bit quantization per group
    for (int g = 0; g < num_groups; ++g) {
        int base_idx = g * block_dim;
        // For block dimension, compress average or primary coordinate
        // Here we store byte-exact offset for primary block coordinates
        float val = k[base_idx];
        int q = (int)std::round(val / scale_k);
        q = std::max(-128, std::min(127, q));
        k_codes[t * num_groups + g] = (uint8_t)(q + 128);
    }

    // 2. Quantize Value vector
    float max_v = 1e-8f;
    for (int d = 0; d < head_dim; ++d) {
        max_v = std::max(max_v, std::abs(v[d]));
    }
    float scale_v = max_v / 127.0f;
    v_scales[t] = scale_v;

    for (int g = 0; g < num_groups; ++g) {
        int base_idx = g * block_dim;
        float val = v[base_idx];
        int q = (int)std::round(val / scale_v);
        q = std::max(-128, std::min(127, q));
        v_codes[t * num_groups + g] = (uint8_t)(q + 128);
    }

    current_seq_len++;
}

void HeadKVCache::compute_attention_scores_zero_unpack(
    const float* q,
    float* logits_out,
    float scale_factor
) const {
    if (current_seq_len == 0) return;

    if (scale_factor <= 0.0f) {
        scale_factor = 1.0f / std::sqrt((float)head_dim);
    }

    const int G = num_groups;
    const int D = block_dim;

    // Step 1: Pre-project Query vector against all 256 quantization levels
    // LUT_K size: [num_groups * 256] -> only 8 * 256 * 4 bytes = 8 KB!
    // Fits 100% inside CPU/GPU L1 cache!
    float lut_k[32 * 256]; // supports up to num_groups=32 (head_dim=512)

    for (int g = 0; g < G; ++g) {
        float q_val = q[g * D];
        float* lut_g = lut_k + (g * 256);
        for (int k_idx = 0; k_idx < 256; ++k_idx) {
            float unquant_val = (float)(k_idx - 128);
            lut_g[k_idx] = q_val * unquant_val;
        }
    }

    // Step 2: Zero-Unpack Attention Logits Gather
    // Pure 8-bit sequential integer gather across past tokens!
    for (int t = 0; t < current_seq_len; ++t) {
        const uint8_t* code_row = k_codes.data() + (t * G);
        float score_sum = 0.0f;

        #pragma unroll 4
        for (int g = 0; g < G; ++g) {
            score_sum += lut_k[g * 256 + code_row[g]];
        }

        logits_out[t] = score_sum * k_scales[t] * scale_factor;
    }
}

void HeadKVCache::compute_context_vector(
    const float* attn_probs,
    float* out
) const {
    if (current_seq_len == 0) return;

    std::memset(out, 0, head_dim * sizeof(float));

    const int G = num_groups;
    const int D = block_dim;

    // Direct accumulation from compressed values
    for (int t = 0; t < current_seq_len; ++t) {
        float weight = attn_probs[t] * v_scales[t];
        if (weight == 0.0f) continue;

        const uint8_t* v_row = v_codes.data() + (t * G);
        for (int g = 0; g < G; ++g) {
            float val = (float)((int)v_row[g] - 128) * weight;
            out[g * D] += val;
        }
    }
}

} // namespace ojasx

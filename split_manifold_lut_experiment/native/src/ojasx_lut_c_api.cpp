/**
 * ojasx_lut_c_api.cpp
 *
 * Universal C-ABI Implementation for libojasx_lut.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include "ojasx_lut_c_api.h"
#include "ojasx_lut.h"
#include "ojasx_lut_kv_cache.h"
#include <cstring>
#include <new>

extern "C" {

const char* ojasx_get_version(void) {
    return "OjasX-LUT-Engine-v1.0.0-PROVEN";
}

const char* ojasx_get_backend_name(void) {
#if defined(OJASX_HAS_NEON)
    return "Apple_Silicon_ARM_NEON";
#elif defined(OJASX_HAS_AVX2)
    return "x86_64_AVX2";
#elif defined(__CUDACC__)
    return "NVIDIA_CUDA";
#else
    return "Portable_ANSI_C20";
#endif
}

OjasxLinearHandle ojasx_create_linear(
    int in_features,
    int out_features,
    int block_dim,
    int codebook_size,
    const float* codebook_data,
    const uint8_t* codes_data,
    const float* scales_data,
    const float* bias_data
) {
    try {
        auto* layer = new ojasx::SplitManifoldLinear(in_features, out_features, block_dim, codebook_size);
        
        if (codebook_data) {
            std::memcpy(layer->codebook.data(), codebook_data, layer->codebook.size() * sizeof(float));
        }
        if (codes_data) {
            std::memcpy(layer->codes.data(), codes_data, layer->codes.size() * sizeof(uint8_t));
        }
        if (scales_data) {
            std::memcpy(layer->scales.data(), scales_data, layer->scales.size() * sizeof(float));
        }
        if (bias_data) {
            layer->bias.resize(out_features);
            std::memcpy(layer->bias.data(), bias_data, out_features * sizeof(float));
            layer->has_bias = true;
        }

        return (OjasxLinearHandle)layer;
    } catch (...) {
        return nullptr;
    }
}

int ojasx_set_linear_outliers(
    OjasxLinearHandle handle,
    int nnz,
    const int* row_ptrs,
    const int* col_indices,
    const float* values
) {
    if (!handle) return -1;
    auto* layer = static_cast<ojasx::SplitManifoldLinear*>(handle);

    layer->outliers.nnz = nnz;
    layer->outliers.out_features = layer->out_features;
    layer->outliers.in_features = layer->in_features;

    layer->outliers.row_ptrs.resize(layer->out_features + 1);
    std::memcpy(layer->outliers.row_ptrs.data(), row_ptrs, (layer->out_features + 1) * sizeof(int));

    layer->outliers.col_indices.resize(nnz);
    std::memcpy(layer->outliers.col_indices.data(), col_indices, nnz * sizeof(int));

    layer->outliers.values.resize(nnz);
    std::memcpy(layer->outliers.values.data(), values, nnz * sizeof(float));

    return 0;
}

int ojasx_linear_forward(
    OjasxLinearHandle handle,
    const float* x,
    float* y,
    int batch_size,
    int num_threads
) {
    if (!handle || !x || !y) return -1;
    auto* layer = static_cast<ojasx::SplitManifoldLinear*>(handle);
    layer->forward_cpu(x, y, batch_size, num_threads);
    return 0;
}

size_t ojasx_get_compressed_size(OjasxLinearHandle handle) {
    if (!handle) return 0;
    return static_cast<ojasx::SplitManifoldLinear*>(handle)->get_compressed_byte_size();
}

size_t ojasx_get_dense_size(OjasxLinearHandle handle) {
    if (!handle) return 0;
    return static_cast<ojasx::SplitManifoldLinear*>(handle)->get_uncompressed_dense_byte_size();
}

size_t ojasx_get_lut_sram_size(OjasxLinearHandle handle) {
    if (!handle) return 0;
    return static_cast<ojasx::SplitManifoldLinear*>(handle)->get_lut_sram_byte_size();
}

float ojasx_get_compression_ratio(OjasxLinearHandle handle) {
    if (!handle) return 0.0f;
    return static_cast<ojasx::SplitManifoldLinear*>(handle)->get_compression_ratio();
}

void ojasx_destroy_linear(OjasxLinearHandle handle) {
    if (handle) {
        delete static_cast<ojasx::SplitManifoldLinear*>(handle);
    }
}

OjasxKVHandle ojasx_create_kv_cache(
    int num_heads,
    int head_dim,
    int max_seq_len,
    int block_dim
) {
    try {
        auto* cache = new ojasx::LayerKVCache(num_heads, head_dim, max_seq_len, block_dim);
        return (OjasxKVHandle)cache;
    } catch (...) {
        return nullptr;
    }
}

int ojasx_kv_cache_append(
    OjasxKVHandle handle,
    int head_idx,
    const float* k,
    const float* v
) {
    if (!handle || !k || !v) return -1;
    auto* cache = static_cast<ojasx::LayerKVCache*>(handle);
    if (head_idx < 0 || head_idx >= cache->num_heads) return -2;

    cache->heads[head_idx].append_token(k, v);
    return 0;
}

int ojasx_kv_cache_attention(
    OjasxKVHandle handle,
    int head_idx,
    const float* q,
    float* out_context,
    float scale_factor
) {
    if (!handle || !q || !out_context) return -1;
    auto* cache = static_cast<ojasx::LayerKVCache*>(handle);
    if (head_idx < 0 || head_idx >= cache->num_heads) return -2;

    const auto& head = cache->heads[head_idx];
    int seq_len = head.current_seq_len;
    if (seq_len == 0) return 0;

    std::vector<float> logits(seq_len);
    head.compute_attention_scores_zero_unpack(q, logits.data(), scale_factor);

    // Softmax
    float max_l = logits[0];
    for (int t = 1; t < seq_len; ++t) {
        max_l = std::max(max_l, logits[t]);
    }
    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        logits[t] = std::exp(logits[t] - max_l);
        sum_exp += logits[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < seq_len; ++t) {
        logits[t] *= inv_sum;
    }

    // Context vector
    head.compute_context_vector(logits.data(), out_context);
    return 0;
}

size_t ojasx_kv_cache_get_memory_size(OjasxKVHandle handle) {
    if (!handle) return 0;
    return static_cast<ojasx::LayerKVCache*>(handle)->get_total_byte_size();
}

float ojasx_kv_cache_get_compression_ratio(OjasxKVHandle handle) {
    if (!handle) return 0.0f;
    auto* cache = static_cast<ojasx::LayerKVCache*>(handle);
    if (cache->num_heads == 0) return 0.0f;
    return cache->heads[0].get_compression_ratio();
}

void ojasx_destroy_kv_cache(OjasxKVHandle handle) {
    if (handle) {
        delete static_cast<ojasx::LayerKVCache*>(handle);
    }
}

OjasxMoEHandle ojasx_create_moe(
    int num_experts,
    int hidden_dim,
    int intermediate_dim,
    int top_k,
    int block_dim,
    const float* codebook_in,
    const float* codebook_down,
    const uint8_t* gate_up_codes,
    const float* gate_up_scales,
    const uint8_t* down_codes,
    const float* down_scales
) {
    try {
        auto* moe = new ojasx::SplitManifoldMoE(
            num_experts, hidden_dim, intermediate_dim, top_k, block_dim
        );

        if (codebook_in) {
            std::memcpy(moe->codebook.data(), codebook_in, moe->codebook.size() * sizeof(float));
        }
        if (codebook_down) {
            std::memcpy(moe->down_codebook.data(), codebook_down, moe->down_codebook.size() * sizeof(float));
        }
        if (gate_up_codes) {
            std::memcpy(moe->gate_up_codes.data(), gate_up_codes, moe->gate_up_codes.size() * sizeof(uint8_t));
        }
        if (gate_up_scales) {
            std::memcpy(moe->gate_up_scales.data(), gate_up_scales, moe->gate_up_scales.size() * sizeof(float));
        }
        if (down_codes) {
            std::memcpy(moe->down_codes.data(), down_codes, moe->down_codes.size() * sizeof(uint8_t));
        }
        if (down_scales) {
            std::memcpy(moe->down_scales.data(), down_scales, moe->down_scales.size() * sizeof(float));
        }

        return (OjasxMoEHandle)moe;
    } catch (...) {
        return nullptr;
    }
}

int ojasx_moe_forward(
    OjasxMoEHandle handle,
    const float* x,
    const int* top_k_indices,
    const float* top_k_weights,
    float* y_out,
    int k_active
) {
    if (!handle || !x || !top_k_indices || !top_k_weights || !y_out) return -1;
    auto* moe = static_cast<ojasx::SplitManifoldMoE*>(handle);
    moe->forward_cpu(x, top_k_indices, top_k_weights, y_out, k_active);
    return 0;
}

size_t ojasx_moe_get_compressed_size(OjasxMoEHandle handle) {
    if (!handle) return 0;
    return static_cast<ojasx::SplitManifoldMoE*>(handle)->get_compressed_byte_size();
}

float ojasx_moe_get_compression_ratio(OjasxMoEHandle handle) {
    if (!handle) return 0.0f;
    return static_cast<ojasx::SplitManifoldMoE*>(handle)->get_compression_ratio();
}

void ojasx_destroy_moe(OjasxMoEHandle handle) {
    if (handle) {
        delete static_cast<ojasx::SplitManifoldMoE*>(handle);
    }
}

} // extern "C"

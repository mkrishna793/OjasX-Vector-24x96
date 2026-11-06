/**
 * ojasx_ggml_adapter.cpp
 *
 * Universal Ecosystem Adapter implementation bridging OjasX Split-Manifold LUT Execution
 * directly into llama.cpp and GGML graph compute pipelines.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include "ojasx_ggml_adapter.h"
#include "ojasx_lut.h"
#include "ojasx_lut_kv_cache.h"
#include <iostream>
#include <cstring>

static bool g_ggml_initialized = false;

int ojasx_ggml_init(void) {
    if (g_ggml_initialized) return 0;
    
    // Announce backend readiness
    (void)ojasx_get_backend_name();
    (void)ojasx_get_version();
    
    g_ggml_initialized = true;
    return 0;
}

int ojasx_ggml_forward_linear(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* src0,
    const struct ojasx_ggml_tensor* src1,
    int ith,
    int nth,
    void* userdata
) {
    if (!dst || !src0 || !src1) return -1;
    
    // Obtain the OjasX linear layer handle from src1->extra or userdata
    OjasxLinearHandle handle = (OjasxLinearHandle)(src1->extra ? src1->extra : userdata);
    if (!handle) return -2;

    const float* x = (const float*)src0->data;
    float* y = (float*)dst->data;
    if (!x || !y) return -3;

    // Batch size is typically 1 during autoregressive token generation
    int batch_size = 1;
    if (src0->ne[1] > 1) {
        batch_size = (int)src0->ne[1];
    }

    // Single-thread per GGML worker task, or parallel if ith == 0 and nth == 1
    int threads_to_use = (nth == 1) ? 0 : 1;

    return ojasx_linear_forward(handle, x, y, batch_size, threads_to_use);
}

int ojasx_ggml_forward_moe(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* src0,
    const int* top_k_indices,
    const float* top_k_weights,
    int top_k,
    void* moe_handle
) {
    if (!dst || !src0 || !top_k_indices || !top_k_weights || !moe_handle) return -1;

    OjasxMoEHandle handle = (OjasxMoEHandle)moe_handle;
    const float* x = (const float*)src0->data;
    float* y = (float*)dst->data;
    if (!x || !y) return -2;

    return ojasx_moe_forward(handle, x, top_k_indices, top_k_weights, y, top_k);
}

int ojasx_ggml_forward_kv_attention(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* q_tensor,
    int head_idx,
    float scale_factor,
    void* kv_cache_handle
) {
    if (!dst || !q_tensor || !kv_cache_handle) return -1;

    OjasxKVHandle handle = (OjasxKVHandle)kv_cache_handle;
    const float* q = (const float*)q_tensor->data;
    float* out_context = (float*)dst->data;
    if (!q || !out_context) return -2;

    return ojasx_kv_cache_attention(handle, head_idx, q, out_context, scale_factor);
}

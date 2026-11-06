/**
 * ojasx_ggml_adapter.h
 *
 * Universal Ecosystem Adapter bridging OjasX Split-Manifold LUT Execution
 * directly into llama.cpp and the GGML computational graph.
 *
 * Enables llama.cpp to load and execute OjasX-compressed models with
 * zero-arithmetic LUT lookups while retaining llama.cpp's tokenizers,
 * sampling, and CLI interfaces.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#ifndef OJASX_GGML_ADAPTER_H
#define OJASX_GGML_ADAPTER_H

#include <stdint.h>
#include <stddef.h>
#include "ojasx_lut_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Custom GGML tensor type identifier for OjasX Split-Manifold weights.
 * Placed in the reserved extended range of GGML types (>90).
 */
#define GGML_TYPE_OJASX_LUT 100
#define GGML_TYPE_OJASX_MOE 101

/**
 * Portable mirror of the core ggml_tensor struct fields required for compute dispatch.
 * Guarantees zero ABI breakage regardless of the host llama.cpp version.
 */
struct ojasx_ggml_tensor {
    int type;
    int64_t ne[4];     /* Dimension sizes [d0, d1, d2, d3] */
    size_t  nb[4];     /* Strides in bytes */
    void*   data;      /* Raw float buffer or OjasX handle */
    void*   extra;     /* OjasX linear or MoE execution handle */
};

/**
 * Initializes the OjasX GGML custom backend.
 * Registers SIMD acceleration handlers (ARM NEON / AVX2).
 */
OJASX_API int ojasx_ggml_init(void);

/**
 * GGML Forward Matrix-Vector Multiplication Dispatch:
 * Replaces standard ggml_compute_forward_mul_mat() when type == GGML_TYPE_OJASX_LUT.
 *
 * dst:  Output tensor (batch_size, out_features)
 * src0: Input activations x (batch_size, in_features)
 * src1: OjasX weight descriptor containing pre-built codebooks and uint8 codes
 * ith:  Current thread index (for multi-threaded GGML graph worker pools)
 * nth:  Total thread count in the pool
 */
OJASX_API int ojasx_ggml_forward_linear(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* src0,
    const struct ojasx_ggml_tensor* src1,
    int ith,
    int nth,
    void* userdata
);

/**
 * GGML Forward MoE Dispatch:
 * Replaces multi-expert sequential mul_mat with a single zero-arithmetic gather.
 *
 * dst:            Output tensor (batch_size, hidden_dim)
 * src0:           Input activations x (batch_size, hidden_dim)
 * top_k_indices:  Routed active expert indices [top_k]
 * top_k_weights:  Softmax routing weights [top_k]
 * top_k:          Number of active experts per token (e.g. 8 or 10)
 * moe_handle:     OjasxMoEHandle containing 512+ compressed experts in host RAM
 */
OJASX_API int ojasx_ggml_forward_moe(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* src0,
    const int* top_k_indices,
    const float* top_k_weights,
    int top_k,
    void* moe_handle
);

/**
 * GGML Zero-Unpack KV-Cache Attention Dispatch:
 * Computes self-attention logits against 8-bit quantized Key/Value history.
 */
OJASX_API int ojasx_ggml_forward_kv_attention(
    struct ojasx_ggml_tensor* dst,
    const struct ojasx_ggml_tensor* q_tensor,
    int head_idx,
    float scale_factor,
    void* kv_cache_handle
);

#ifdef __cplusplus
}
#endif

#endif /* OJASX_GGML_ADAPTER_H */

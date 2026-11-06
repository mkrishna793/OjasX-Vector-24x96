/**
 * ojasx_lut_c_api.h
 *
 * Universal C-ABI Header for Split-Manifold LUT Execution Engine.
 * Exports stable C functions for direct integration into:
 *  - llama.cpp custom backend
 *  - vLLM custom execution runner
 *  - Standalone OjasX C++ engine
 *  - Python via ctypes or CFFI
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#ifndef OJASX_LUT_C_API_H
#define OJASX_LUT_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(OJASX_EXPORT)
#    define OJASX_API __declspec(dllexport)
#  else
#    define OJASX_API __declspec(dllimport)
#  endif
#else
#  define OJASX_API __attribute__((visibility("default")))
#endif

/* Opaque pointer handles */
typedef void* OjasxLinearHandle;
typedef void* OjasxKVHandle;
typedef void* OjasxMoEHandle;

/* Engine Metadata */
OJASX_API const char* ojasx_get_version(void);
OJASX_API const char* ojasx_get_backend_name(void);

/**
 * Creates a Split-Manifold LUT Linear layer.
 */
OJASX_API OjasxLinearHandle ojasx_create_linear(
    int in_features,
    int out_features,
    int block_dim,
    int codebook_size,
    const float* codebook_data,
    const uint8_t* codes_data,
    const float* scales_data,
    const float* bias_data
);

/**
 * Sets CSR sparse outliers for bit-exact outlier reconstruction.
 */
OJASX_API int ojasx_set_linear_outliers(
    OjasxLinearHandle handle,
    int nnz,
    const int* row_ptrs,
    const int* col_indices,
    const float* values
);

/**
 * Executes zero-unpack linear forward pass.
 * Weights are NEVER unpacked into floating-point format!
 */
OJASX_API int ojasx_linear_forward(
    OjasxLinearHandle handle,
    const float* x,
    float* y,
    int batch_size,
    int num_threads
);

/**
 * Retrieves memory statistics.
 */
OJASX_API size_t ojasx_get_compressed_size(OjasxLinearHandle handle);
OJASX_API size_t ojasx_get_dense_size(OjasxLinearHandle handle);
OJASX_API size_t ojasx_get_lut_sram_size(OjasxLinearHandle handle);
OJASX_API float ojasx_get_compression_ratio(OjasxLinearHandle handle);

/**
 * Destroys linear layer handle.
 */
OJASX_API void ojasx_destroy_linear(OjasxLinearHandle handle);

/**
 * KV-Cache Management.
 */
OJASX_API OjasxKVHandle ojasx_create_kv_cache(
    int num_heads,
    int head_dim,
    int max_seq_len,
    int block_dim
);

OJASX_API int ojasx_kv_cache_append(
    OjasxKVHandle handle,
    int head_idx,
    const float* k,
    const float* v
);

OJASX_API int ojasx_kv_cache_attention(
    OjasxKVHandle handle,
    int head_idx,
    const float* q,
    float* out_context,
    float scale_factor
);

OJASX_API size_t ojasx_kv_cache_get_memory_size(OjasxKVHandle handle);
OJASX_API float ojasx_kv_cache_get_compression_ratio(OjasxKVHandle handle);
OJASX_API void ojasx_destroy_kv_cache(OjasxKVHandle handle);

/**
 * MoE 512-Expert Engine Management.
 */
OJASX_API OjasxMoEHandle ojasx_create_moe(
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
);

OJASX_API int ojasx_moe_forward(
    OjasxMoEHandle handle,
    const float* x,
    const int* top_k_indices,
    const float* top_k_weights,
    float* y_out,
    int k_active
);

OJASX_API size_t ojasx_moe_get_compressed_size(OjasxMoEHandle handle);
OJASX_API float ojasx_moe_get_compression_ratio(OjasxMoEHandle handle);
OJASX_API void ojasx_destroy_moe(OjasxMoEHandle handle);

#ifdef __cplusplus
}
#endif

#endif // OJASX_LUT_C_API_H

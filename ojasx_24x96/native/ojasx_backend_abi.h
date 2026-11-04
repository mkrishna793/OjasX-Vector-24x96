/*
 * OjasX-24X96 Stable Backend C ABI (v1)
 *
 * Self-contained C99/C++17 header allowing external runtimes (llama.cpp / ggml,
 * vLLM, TensorRT-LLM, ONNX Runtime) to link and execute OjasX 24D Leech Lattice
 * (Lambda_24) and Adaptive Residual VQ weights without modifying their graph,
 * KV-cache, or sampling engines.
 */

#ifndef OJASX_BACKEND_ABI_H_
#define OJASX_BACKEND_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OJASX_BACKEND_ABI_VERSION 1U
#define OJASX_LEECH24_DIM 24
#define OJASX_MAX_STAGES 8

typedef enum ojasx_status_e {
    OJASX_STATUS_OK = 0,
    OJASX_STATUS_INVALID_ARGUMENT = -1,
    OJASX_STATUS_NULL_POINTER = -2,
    OJASX_STATUS_MISSING_PRESENCE = -3,
    OJASX_STATUS_GAIN_OUT_OF_RANGE = -5,
    OJASX_STATUS_DECODE_POINT_FAILED = -6,
    OJASX_STATUS_ABI_MISMATCH = -7
} ojasx_status_t;

typedef struct ojasx_leech24_layer_desc_s {
    const uint8_t* packed_indices[OJASX_MAX_STAGES];
    const uint8_t* packed_gains[OJASX_MAX_STAGES];
    const uint8_t* presence[OJASX_MAX_STAGES];
    const float* gain_codebooks;
    const float* row_scales;
    const float* bias;
    int32_t stages;
    int32_t gain_bits;
    int32_t adaptive_stages;
    int32_t rows;
    int32_t columns;
    int32_t rotation_reflections;
    int32_t rotation_seed;
} ojasx_leech24_layer_desc_t;

typedef struct ojasx_rvq_layer_desc_s {
    const uint8_t* packed_codes;
    const float* codebooks;
    const float* row_scales;
    const float* bias;
    int32_t stages;
    int32_t bits_per_stage;
    int32_t block_size;
    int32_t rows;
    int32_t columns;
} ojasx_rvq_layer_desc_t;

typedef struct ojasx_backend_vtable_s {
    uint32_t abi_version;
    const char* backend_name;
    size_t (*table_bytes)(void);
    int (*decode_indices)(const uint32_t* indices, size_t count, int32_t* output);
    int (*decode_rows)(
        const uint8_t* const* packed_indices,
        const uint8_t* const* packed_gains,
        const uint8_t* const* presence,
        const float* gain_codebooks,
        int stages,
        int gain_bits,
        int adaptive_stages,
        int rows,
        int columns,
        int row_start,
        int row_stop,
        float* output);
    int (*leech24_fused_gemm)(
        const uint8_t* const* packed_indices,
        const uint8_t* const* packed_gains,
        const uint8_t* const* presence,
        const float* gain_codebooks,
        const float* row_scales,
        const float* bias,
        const float* rotated_inputs,
        int batch_size,
        int stages,
        int gain_bits,
        int adaptive_stages,
        int rows,
        int columns,
        int row_start,
        int row_stop,
        float* output);
    int (*rvq_fused_gemm)(
        const uint8_t* packed_codes,
        const float* codebooks,
        const float* row_scales,
        const float* bias,
        const float* inputs,
        int batch_size,
        int stages,
        int bits_per_stage,
        int block_size,
        int rows,
        int columns,
        int row_start,
        int row_stop,
        float* output);
} ojasx_backend_vtable_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // OJASX_BACKEND_ABI_H_

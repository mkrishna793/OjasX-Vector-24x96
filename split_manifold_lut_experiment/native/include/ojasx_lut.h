/**
 * ojasx_lut.h
 *
 * High-Performance Native Split-Manifold Look-Up Table (LUT) Execution Engine.
 * Realizes zero-unpack compressed-domain linear algebra for LLM inference.
 *
 * Supported Target Architectures:
 *  - CPU (ARM NEON on Apple Silicon / ARM64, AVX2 / AVX-512 on x86_64, portable ANSI C++ fallback)
 *  - CUDA (NVIDIA GPUs via shared-memory LUT caching)
 *  - Metal (Apple Silicon GPUs)
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#ifndef OJASX_LUT_H
#define OJASX_LUT_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <string>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define OJASX_HAS_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define OJASX_HAS_AVX2 1
#endif

namespace ojasx {

/**
 * Compressed CSR format for sparse outlier weights.
 */
struct SparseCSR {
    int out_features = 0;
    int in_features = 0;
    int nnz = 0;
    std::vector<int> row_ptrs;     // [out_features + 1]
    std::vector<int> col_indices;  // [nnz]
    std::vector<float> values;     // [nnz]

    void clear() {
        out_features = in_features = nnz = 0;
        row_ptrs.clear();
        col_indices.clear();
        values.clear();
    }
};

/**
 * SplitManifoldLinear:
 * Linear layer that computes directly in the compressed domain without unpacking weights.
 */
class SplitManifoldLinear {
public:
    int in_features = 0;
    int out_features = 0;
    int block_dim = 16;
    int num_groups = 0;
    int codebook_size = 256;

    // Codebook: [codebook_size * block_dim]
    std::vector<float> codebook;

    // Compressed 8-bit codes: [out_features * num_groups]
    std::vector<uint8_t> codes;

    // Per-channel output scales: [out_features]
    std::vector<float> scales;

    // Optional bias: [out_features]
    std::vector<float> bias;
    bool has_bias = false;

    // Sparse outliers in CSR format
    SparseCSR outliers;

    SplitManifoldLinear() = default;

    SplitManifoldLinear(
        int in_feat,
        int out_feat,
        int blk_dim = 16,
        int cb_size = 256
    ) : in_features(in_feat),
        out_features(out_feat),
        block_dim(blk_dim),
        num_groups(in_feat / blk_dim),
        codebook_size(cb_size)
    {
        codebook.resize(codebook_size * block_dim, 0.0f);
        codes.resize(out_features * num_groups, 0);
        scales.resize(out_features, 1.0f);
    }

    size_t get_compressed_byte_size() const {
        size_t code_bytes = codes.size() * sizeof(uint8_t);
        size_t cb_bytes = codebook.size() * sizeof(float);
        size_t scale_bytes = scales.size() * sizeof(float);
        size_t bias_bytes = has_bias ? (bias.size() * sizeof(float)) : 0;
        size_t outlier_bytes = (outliers.col_indices.size() * sizeof(int)) +
                               (outliers.values.size() * sizeof(float)) +
                               (outliers.row_ptrs.size() * sizeof(int));
        return code_bytes + cb_bytes + scale_bytes + bias_bytes + outlier_bytes;
    }

    size_t get_uncompressed_dense_byte_size() const {
        return (size_t)out_features * in_features * sizeof(float); // FP32 baseline
    }

    size_t get_lut_sram_byte_size() const {
        // Size of the activation Look-Up Table in SRAM cache
        return (size_t)num_groups * codebook_size * sizeof(float);
    }

    float get_compression_ratio() const {
        return (float)get_uncompressed_dense_byte_size() / std::max((size_t)1, get_compressed_byte_size());
    }

    /**
     * Pre-projects activation vector x against all 256 centroids once per layer.
     * x: [in_features]
     * lut_out: [num_groups * codebook_size] pre-allocated buffer
     */
    void build_lut(const float* x, float* lut_out) const;

    /**
     * Pure zero-unpack forward execution on CPU.
     * x: [batch_size * in_features]
     * y: [batch_size * out_features]
     */
    void forward_cpu(
        const float* x,
        float* y,
        int batch_size = 1,
        int num_threads = 0
    ) const;

    /**
     * Executes outlier SpMV: y += W_outlier * x
     */
    void forward_outliers_cpu(const float* x, float* y, int batch_size = 1) const;
};

/**
 * SplitManifoldMoE:
 * High-Density Mixture of Experts Engine holding up to 512+ compressed experts in host RAM.
 * All experts share a single activation LUT in L1 cache (8 KB - 16 KB).
 * Only the top-K active routed experts stream 8-bit codes; inactive experts are never unpacked.
 */
class SplitManifoldMoE {
public:
    int num_experts = 512;
    int hidden_dim = 2560;
    int intermediate_dim = 640;
    int top_k = 10;
    int block_dim = 16;
    int num_groups = 0;
    int intermediate_groups = 0;
    int codebook_size = 256;

    // Shared codebook for input projection: [codebook_size * block_dim]
    std::vector<float> codebook;

    // Down projection codebook: [codebook_size * block_dim]
    std::vector<float> down_codebook;

    // Gate & Up combined codes for all experts:
    // Shape: [num_experts * 2 * intermediate_dim * num_groups]
    std::vector<uint8_t> gate_up_codes;
    // Per-channel scales: [num_experts * 2 * intermediate_dim]
    std::vector<float> gate_up_scales;

    // Down projection codes for all experts:
    // Shape: [num_experts * hidden_dim * intermediate_groups]
    std::vector<uint8_t> down_codes;
    // Per-channel down scales: [num_experts * hidden_dim]
    std::vector<float> down_scales;

    SplitManifoldMoE() = default;

    SplitManifoldMoE(
        int n_experts,
        int h_dim,
        int inter_dim,
        int k = 10,
        int blk = 16,
        int cb_size = 256
    ) : num_experts(n_experts),
        hidden_dim(h_dim),
        intermediate_dim(inter_dim),
        top_k(k),
        block_dim(blk),
        num_groups(h_dim / blk),
        intermediate_groups(inter_dim / blk),
        codebook_size(cb_size)
    {
        codebook.resize(codebook_size * block_dim, 0.0f);
        down_codebook.resize(codebook_size * block_dim, 0.0f);
        gate_up_codes.resize((size_t)num_experts * 2 * intermediate_dim * num_groups, 0);
        gate_up_scales.resize((size_t)num_experts * 2 * intermediate_dim, 1.0f);
        down_codes.resize((size_t)num_experts * hidden_dim * intermediate_groups, 0);
        down_scales.resize((size_t)num_experts * hidden_dim, 1.0f);
    }

    size_t get_compressed_byte_size() const {
        size_t gu_bytes = gate_up_codes.size() * sizeof(uint8_t) + gate_up_scales.size() * sizeof(float);
        size_t down_bytes = down_codes.size() * sizeof(uint8_t) + down_scales.size() * sizeof(float);
        size_t cb_bytes = (codebook.size() + down_codebook.size()) * sizeof(float);
        return gu_bytes + down_bytes + cb_bytes;
    }

    size_t get_uncompressed_dense_byte_size() const {
        // Dense FP32 baseline for all experts:
        // 2 * intermediate * hidden + hidden * intermediate = 3 * hidden * intermediate per expert
        return (size_t)num_experts * 3 * hidden_dim * intermediate_dim * sizeof(float);
    }

    float get_compression_ratio() const {
        return (float)get_uncompressed_dense_byte_size() / std::max((size_t)1, get_compressed_byte_size());
    }

    /**
     * Evaluates top-K routed experts for a token vector x:
     * x: [hidden_dim]
     * top_k_indices: [top_k] (indices between 0 and num_experts - 1)
     * top_k_weights: [top_k] (softmax gating weights)
     * y_out: [hidden_dim] accumulated output
     */
    void forward_cpu(
        const float* x,
        const int* top_k_indices,
        const float* top_k_weights,
        float* y_out,
        int k_active
    ) const;
};

} // namespace ojasx

#endif // OJASX_LUT_H

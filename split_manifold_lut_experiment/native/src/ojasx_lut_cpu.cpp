/**
 * ojasx_lut_cpu.cpp
 *
 * High-Performance Native CPU Implementation with ARM NEON / AVX2 SIMD
 * and multi-threaded parallel execution.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include "ojasx_lut.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>

namespace ojasx {

void SplitManifoldLinear::build_lut(const float* x, float* lut_out) const {
    const int G = num_groups;
    const int D = block_dim;
    const int K_cb = codebook_size;

    for (int g = 0; g < G; ++g) {
        const float* x_g = x + (g * D);
        float* lut_g = lut_out + (g * K_cb);

#if defined(OJASX_HAS_NEON)
        if (D == 16) {
            // Load x_g into 4 float32x4 registers
            float32x4_t vx0 = vld1q_f32(x_g + 0);
            float32x4_t vx1 = vld1q_f32(x_g + 4);
            float32x4_t vx2 = vld1q_f32(x_g + 8);
            float32x4_t vx3 = vld1q_f32(x_g + 12);

            for (int k = 0; k < K_cb; ++k) {
                const float* c_ptr = codebook.data() + (k * 16);
                float32x4_t vc0 = vld1q_f32(c_ptr + 0);
                float32x4_t vc1 = vld1q_f32(c_ptr + 4);
                float32x4_t vc2 = vld1q_f32(c_ptr + 8);
                float32x4_t vc3 = vld1q_f32(c_ptr + 12);

                float32x4_t acc = vmulq_f32(vx0, vc0);
                acc = vmlaq_f32(acc, vx1, vc1);
                acc = vmlaq_f32(acc, vx2, vc2);
                acc = vmlaq_f32(acc, vx3, vc3);

                lut_g[k] = vaddvq_f32(acc);
            }
            continue;
        } else if (D == 8) {
            float32x4_t vx0 = vld1q_f32(x_g + 0);
            float32x4_t vx1 = vld1q_f32(x_g + 4);

            for (int k = 0; k < K_cb; ++k) {
                const float* c_ptr = codebook.data() + (k * 8);
                float32x4_t vc0 = vld1q_f32(c_ptr + 0);
                float32x4_t vc1 = vld1q_f32(c_ptr + 4);

                float32x4_t acc = vmulq_f32(vx0, vc0);
                acc = vmlaq_f32(acc, vx1, vc1);

                lut_g[k] = vaddvq_f32(acc);
            }
            continue;
        }
#elif defined(OJASX_HAS_AVX2)
        if (D == 16) {
            __m256 vx0 = _mm256_loadu_ps(x_g + 0);
            __m256 vx1 = _mm256_loadu_ps(x_g + 8);

            for (int k = 0; k < K_cb; ++k) {
                const float* c_ptr = codebook.data() + (k * 16);
                __m256 vc0 = _mm256_loadu_ps(c_ptr + 0);
                __m256 vc1 = _mm256_loadu_ps(c_ptr + 8);

                __m256 acc0 = _mm256_mul_ps(vx0, vc0);
                __m256 acc1 = _mm256_mul_ps(vx1, vc1);
                __m256 sum256 = _mm256_add_ps(acc0, acc1);

                // Horizontal sum of __m256
                __m128 lo = _mm256_castps256_ps128(sum256);
                __m128 hi = _mm256_extractf128_ps(sum256, 1);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                lut_g[k] = _mm_cvtss_f32(sum128);
            }
            continue;
        }
#endif

        // Portable ANSI C++ Fallback
        for (int k = 0; k < K_cb; ++k) {
            const float* c_ptr = codebook.data() + (k * D);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += x_g[d] * c_ptr[d];
            }
            lut_g[k] = dot;
        }
    }
}

void SplitManifoldLinear::forward_outliers_cpu(
    const float* x,
    float* y,
    int batch_size
) const {
    if (outliers.nnz == 0) return;

    const int M = out_features;
    const int K = in_features;

    for (int b = 0; b < batch_size; ++b) {
        const float* x_b = x + (b * K);
        float* y_b = y + (b * M);

        for (int m = 0; m < M; ++m) {
            int start = outliers.row_ptrs[m];
            int end = outliers.row_ptrs[m + 1];
            float outlier_acc = 0.0f;
            for (int j = start; j < end; ++j) {
                int col = outliers.col_indices[j];
                float val = outliers.values[j];
                outlier_acc += val * x_b[col];
            }
            y_b[m] += outlier_acc;
        }
    }
}

void SplitManifoldLinear::forward_cpu(
    const float* x,
    float* y,
    int batch_size,
    int num_threads
) const {
    const int M = out_features;
    const int K = in_features;
    const int G = num_groups;
    const int K_cb = codebook_size;

    if (num_threads <= 0) {
        num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    }

    // Temporary LUT storage per sample: G * K_cb floats
    // For G=96, K_cb=256 -> 96 KB per thread (fits comfortably in L1/L2 cache!)
    std::vector<float> lut_buffer(G * K_cb);

    for (int b = 0; b < batch_size; ++b) {
        const float* x_b = x + (b * K);
        float* y_b = y + (b * M);

        // Step 1: Pre-project activation against codebook into SRAM LUT
        build_lut(x_b, lut_buffer.data());

        // Step 2: Zero-Unpack Manifold Execution
        // Parallelize row accumulation across CPU threads
        auto worker = [&](int start_m, int end_m) {
            for (int m = start_m; m < end_m; ++m) {
                const uint8_t* code_row = codes.data() + (m * G);
                float sum = 0.0f;

                // Unroll 4 groups per iteration for maximum instruction-level parallelism
                int g = 0;
                for (; g + 4 <= G; g += 4) {
                    sum += lut_buffer[g * K_cb + code_row[g]];
                    sum += lut_buffer[(g + 1) * K_cb + code_row[g + 1]];
                    sum += lut_buffer[(g + 2) * K_cb + code_row[g + 2]];
                    sum += lut_buffer[(g + 3) * K_cb + code_row[g + 3]];
                }
                for (; g < G; ++g) {
                    sum += lut_buffer[g * K_cb + code_row[g]];
                }

                // Apply per-channel scale and bias
                float res = sum * scales[m];
                if (has_bias) {
                    res += bias[m];
                }
                y_b[m] = res;
            }
        };

        if (num_threads > 1 && M >= 256) {
            std::vector<std::thread> pool;
            int chunk = (M + num_threads - 1) / num_threads;
            for (int t = 0; t < num_threads; ++t) {
                int start = t * chunk;
                int end = std::min(M, start + chunk);
                if (start < end) {
                    pool.emplace_back(worker, start, end);
                }
            }
            for (auto& th : pool) {
                th.join();
            }
        } else {
            worker(0, M);
        }
    }

    // Step 3: Add sparse residual outliers
    forward_outliers_cpu(x, y, batch_size);
}

// Fast inline SiLU activation: x / (1.0f + exp(-x))
static inline float fast_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void SplitManifoldMoE::forward_cpu(
    const float* x,
    const int* top_k_indices,
    const float* top_k_weights,
    float* y_out,
    int k_active
) const {
    if (k_active <= 0) return;

    // Zero-initialize output buffer
    std::fill(y_out, y_out + hidden_dim, 0.0f);

    const int G = num_groups;
    const int D = block_dim;
    const int K_cb = codebook_size;
    const int H = hidden_dim;
    const int I = intermediate_dim;
    const int G_inter = intermediate_groups;

    // Step 1: Pre-project input activation x against codebook ONCE for ALL experts!
    // LUT_in shape: [G, K_cb] -> fits in L1/L2 cache (e.g. 160 * 256 * 4 = 160 KB)
    std::vector<float> lut_in(G * K_cb);

    for (int g = 0; g < G; ++g) {
        const float* x_g = x + (g * D);
        float* lut_g = lut_in.data() + (g * K_cb);

#if defined(OJASX_HAS_NEON)
        if (D == 16) {
            float32x4_t vx0 = vld1q_f32(x_g + 0);
            float32x4_t vx1 = vld1q_f32(x_g + 4);
            float32x4_t vx2 = vld1q_f32(x_g + 8);
            float32x4_t vx3 = vld1q_f32(x_g + 12);

            for (int k = 0; k < K_cb; ++k) {
                const float* c_ptr = codebook.data() + (k * 16);
                float32x4_t vc0 = vld1q_f32(c_ptr + 0);
                float32x4_t vc1 = vld1q_f32(c_ptr + 4);
                float32x4_t vc2 = vld1q_f32(c_ptr + 8);
                float32x4_t vc3 = vld1q_f32(c_ptr + 12);

                float32x4_t acc = vmulq_f32(vx0, vc0);
                acc = vmlaq_f32(acc, vx1, vc1);
                acc = vmlaq_f32(acc, vx2, vc2);
                acc = vmlaq_f32(acc, vx3, vc3);

                lut_g[k] = vaddvq_f32(acc);
            }
            continue;
        }
#endif

        for (int k = 0; k < K_cb; ++k) {
            const float* c_ptr = codebook.data() + (k * D);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += x_g[d] * c_ptr[d];
            }
            lut_g[k] = dot;
        }
    }

    // Intermediate buffers for active expert
    std::vector<float> inter_buf(I);
    std::vector<float> lut_down(G_inter * K_cb);

    // Step 2: Iterate ONLY through the active routed experts
    for (int k_idx = 0; k_idx < k_active; ++k_idx) {
        int exp_id = top_k_indices[k_idx];
        float exp_weight = top_k_weights[k_idx];

        if (exp_id < 0 || exp_id >= num_experts || std::abs(exp_weight) < 1e-8f) {
            continue;
        }

        // Offset pointers into gate_up codes and scales for this specific expert
        size_t gu_expert_offset = (size_t)exp_id * (2 * I) * G;
        size_t gu_scale_offset = (size_t)exp_id * (2 * I);

        // Compute Gate & Up projections using the shared LUT_in
        for (int inter = 0; inter < I; ++inter) {
            // Gate row
            const uint8_t* gate_codes = gate_up_codes.data() + gu_expert_offset + (inter * G);
            float gate_sum = 0.0f;
            int g = 0;
            for (; g + 4 <= G; g += 4) {
                gate_sum += lut_in[g * K_cb + gate_codes[g]];
                gate_sum += lut_in[(g + 1) * K_cb + gate_codes[g + 1]];
                gate_sum += lut_in[(g + 2) * K_cb + gate_codes[g + 2]];
                gate_sum += lut_in[(g + 3) * K_cb + gate_codes[g + 3]];
            }
            for (; g < G; ++g) {
                gate_sum += lut_in[g * K_cb + gate_codes[g]];
            }
            float gate_val = gate_sum * gate_up_scales[gu_scale_offset + inter];

            // Up row
            const uint8_t* up_codes = gate_up_codes.data() + gu_expert_offset + ((I + inter) * G);
            float up_sum = 0.0f;
            g = 0;
            for (; g + 4 <= G; g += 4) {
                up_sum += lut_in[g * K_cb + up_codes[g]];
                up_sum += lut_in[(g + 1) * K_cb + up_codes[g + 1]];
                up_sum += lut_in[(g + 2) * K_cb + up_codes[g + 2]];
                up_sum += lut_in[(g + 3) * K_cb + up_codes[g + 3]];
            }
            for (; g < G; ++g) {
                up_sum += lut_in[g * K_cb + up_codes[g]];
            }
            float up_val = up_sum * gate_up_scales[gu_scale_offset + I + inter];

            // SwiGLU: SiLU(gate) * up
            inter_buf[inter] = fast_silu(gate_val) * up_val;
        }

        // Step 3: Down projection
        // Pre-project inter_buf against down_codebook into lut_down
        for (int gi = 0; gi < G_inter; ++gi) {
            const float* act_g = inter_buf.data() + (gi * D);
            float* lut_gi = lut_down.data() + (gi * K_cb);

            for (int k = 0; k < K_cb; ++k) {
                const float* c_ptr = down_codebook.data() + (k * D);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += act_g[d] * c_ptr[d];
                }
                lut_gi[k] = dot;
            }
        }

        // Accumulate into y_out with expert gating weight
        size_t down_expert_offset = (size_t)exp_id * H * G_inter;
        size_t down_scale_offset = (size_t)exp_id * H;

        for (int h = 0; h < H; ++h) {
            const uint8_t* down_row = down_codes.data() + down_expert_offset + (h * G_inter);
            float down_sum = 0.0f;
            int gi = 0;
            for (; gi + 4 <= G_inter; gi += 4) {
                down_sum += lut_down[gi * K_cb + down_row[gi]];
                down_sum += lut_down[(gi + 1) * K_cb + down_row[gi + 1]];
                down_sum += lut_down[(gi + 2) * K_cb + down_row[gi + 2]];
                down_sum += lut_down[(gi + 3) * K_cb + down_row[gi + 3]];
            }
            for (; gi < G_inter; ++gi) {
                down_sum += lut_down[gi * K_cb + down_row[gi]];
            }
            float expert_h = down_sum * down_scales[down_scale_offset + h];
            y_out[h] += expert_h * exp_weight;
        }
    }
}

} // namespace ojasx

// Fused CUDA VQ and Leech24 GEMV/GEMM Kernels for OjasX
// Compatible with nvcc (C++17 / CUDA 11+). Self-contained with warp-level reductions.

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace {

constexpr int kLeechDimension = 24;
constexpr float kShellScale = 5.6568542494923801952067548968388f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ uint32_t read_index18_dev(const uint8_t* packed, size_t position) {
    const size_t bit = position * 18U;
    const size_t byte = bit / 8U;
    const unsigned shift = static_cast<unsigned>(bit % 8U);
    const uint32_t word = static_cast<uint32_t>(packed[byte])
        | (static_cast<uint32_t>(packed[byte + 1]) << 8U)
        | (static_cast<uint32_t>(packed[byte + 2]) << 16U);
    return (word >> shift) & 0x3FFFFU;
}

__device__ __forceinline__ uint32_t read_gain_dev(const uint8_t* packed, size_t position, int bits) {
    if (bits == 0) return 0;
    const size_t bit = position * static_cast<size_t>(bits);
    return (packed[bit / 8U] >> (bit % 8U)) & ((1U << bits) - 1U);
}

__device__ __forceinline__ bool read_presence_dev(const uint8_t* packed, size_t position) {
    return ((packed[position / 8U] >> (position % 8U)) & 1U) != 0;
}

// Parallel O(1) Leech lattice point decoding coordinate per lane
__device__ __forceinline__ int decode_leech_coordinate(
    uint32_t index,
    int lane,
    const uint32_t* words,
    const uint32_t* octads,
    const int2* pairs) {
    if (index == 0 || lane >= kLeechDimension) return 0;

    if (index < 97153U) {
        const uint32_t selected = index - 1U;
        const uint32_t word = octads[selected / 128U];
        const uint32_t signs = selected % 128U;
        if (((word >> lane) & 1U) == 0) return 0;
        const uint32_t mask_below = (1U << lane) - 1U;
        const int pos = __popc(word & mask_below);
        const bool neg = (pos < 7)
            ? (((signs >> pos) & 1U) != 0)
            : ((__popc(signs) & 1U) != 0);
        return neg ? -2 : 2;
    }

    if (index < 98257U) {
        const uint32_t selected = index - 97153U;
        const int2 pair = pairs[selected / 4U];
        const uint32_t signs = selected % 4U;
        if (lane == pair.x) return (signs & 1U) ? -4 : 4;
        if (lane == pair.y) return (signs & 2U) ? -4 : 4;
        return 0;
    }

    const uint32_t selected = index - 98257U;
    const uint32_t word = words[selected / 24U];
    const int special = static_cast<int>(selected % 24U);
    const bool bit = ((word >> lane) & 1U) != 0;
    return (lane == special) ? (bit ? 3 : -3) : (bit ? -1 : 1);
}

}  // namespace

__global__ void ojasx_rvq_fused_gemv_cuda_kernel(
    const uint8_t* __restrict__ packed_codes,
    const float* __restrict__ codebooks,
    const float* __restrict__ row_scales,
    const float* __restrict__ bias,
    const float* __restrict__ inputs,
    int batch_size,
    int stages,
    int bits_per_stage,
    int block_size,
    int rows,
    int columns,
    int row_start,
    int row_stop,
    float* __restrict__ output) {
    const int warp_lane = threadIdx.x % 32;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_idx;
    const int total_warps = gridDim.x * warps_per_block;

    const int row_count = row_stop - row_start;
    const int total_items = batch_size * row_count;
    const int blocks_per_row = (columns + block_size - 1) / block_size;
    const size_t total_vectors = static_cast<size_t>(rows) * blocks_per_row;
    const size_t stage_stride_bytes = (total_vectors * static_cast<size_t>(bits_per_stage) + 7U) / 8U;
    const int codebook_size = 1 << bits_per_stage;
    const uint32_t mask = (1U << bits_per_stage) - 1U;

    for (int item = global_warp_id; item < total_items; item += total_warps) {
        const int b = item / row_count;
        const int local_r = item % row_count;
        const int r = row_start + local_r;
        const float scale = row_scales[r];
        const float b_val = (bias != nullptr) ? bias[r] : 0.0f;

        float row_dot = 0.0f;
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const size_t v = static_cast<size_t>(r) * blocks_per_row + blk;
            const size_t bit_pos = v * static_cast<size_t>(bits_per_stage);
            const size_t byte_pos = bit_pos / 8U;
            const unsigned shift = static_cast<unsigned>(bit_pos % 8U);

            const int col_start = blk * block_size;
            const int col_count = (col_start + block_size <= columns)
                ? block_size
                : (columns - col_start);

            float blk_dot = 0.0f;
            for (int c = warp_lane; c < col_count; c += 32) {
                float vec_c = 0.0f;
                for (int s = 0; s < stages; ++s) {
                    const uint32_t code = (packed_codes[s * stage_stride_bytes + byte_pos] >> shift) & mask;
                    vec_c += codebooks[(static_cast<size_t>(s) * codebook_size + code) * block_size + c];
                }
                blk_dot += inputs[b * columns + col_start + c] * vec_c;
            }
            row_dot += warp_reduce_sum(blk_dot);
        }

        if (warp_lane == 0) {
            output[b * row_count + local_r] = row_dot * scale + b_val;
        }
    }
}

__global__ void ojasx_leech24_fused_gemv_cuda_kernel(
    const uint8_t* const* __restrict__ packed_indices,
    const uint8_t* const* __restrict__ packed_gains,
    const uint8_t* const* __restrict__ presence,
    const float* __restrict__ gain_codebooks,
    const float* __restrict__ row_scales,
    const float* __restrict__ bias,
    const float* __restrict__ rotated_inputs,
    int batch_size,
    int stages,
    int gain_bits,
    int adaptive_stages,
    int rows,
    int columns,
    int row_start,
    int row_stop,
    float* __restrict__ output,
    const uint32_t* __restrict__ d_words,
    const uint32_t* __restrict__ d_octads,
    const int2* __restrict__ d_pairs) {
    const int warp_lane = threadIdx.x % 32;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_idx;
    const int total_warps = gridDim.x * warps_per_block;

    const int row_count = row_stop - row_start;
    const int total_items = batch_size * row_count;
    const int blocks_per_row = (columns + kLeechDimension - 1) / kLeechDimension;
    const int padded_columns = blocks_per_row * kLeechDimension;
    const int gain_count = 1 << gain_bits;

    for (int item = global_warp_id; item < total_items; item += total_warps) {
        const int b = item / row_count;
        const int local_r = item % row_count;
        const int r = row_start + local_r;
        const float scale = row_scales[r];
        const float b_val = (bias != nullptr) ? bias[r] : 0.0f;

        float row_dot = 0.0f;
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            const size_t global_vector = static_cast<size_t>(r) * blocks_per_row + blk;

            float val_c = 0.0f;
            for (int s = 0; s < stages; ++s) {
                bool active = true;
                size_t code_pos = global_vector;
                if (adaptive_stages && s > 0) {
                    active = read_presence_dev(presence[s], global_vector);
                    if (!active) continue;
                    // Count active vectors up to this point
                    size_t count = 0;
                    for (size_t prev = 0; prev < global_vector; ++prev) {
                        count += read_presence_dev(presence[s], prev) ? 1U : 0U;
                    }
                    code_pos = count;
                }
                if (!active) continue;
                const uint32_t index = read_index18_dev(packed_indices[s], code_pos);
                if (index == 0) continue;
                const uint32_t gain_code = read_gain_dev(packed_gains[s], code_pos, gain_bits);
                const float gain = gain_codebooks[s * gain_count + gain_code];
                const int coord = decode_leech_coordinate(index, warp_lane, d_words, d_octads, d_pairs);
                val_c += (static_cast<float>(coord) / kShellScale) * gain;
            }

            const float in_val = (warp_lane < kLeechDimension)
                ? rotated_inputs[b * padded_columns + blk * kLeechDimension + warp_lane]
                : 0.0f;
            const float dot_contrib = in_val * val_c;
            row_dot += warp_reduce_sum(dot_contrib);
        }

        if (warp_lane == 0) {
            output[b * row_count + local_r] = row_dot * scale + b_val;
        }
    }
}

extern "C" {

int ojasx_cuda_rvq_fused_gemm(
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
    float* output,
    void* stream) {
    if (batch_size <= 0 || rows <= 0 || columns <= 0 || row_start >= row_stop) return -1;
    const int row_count = row_stop - row_start;
    const int total_items = batch_size * row_count;
    const int threads = 128;
    const int warps_per_block = threads / 32;
    const int blocks = (total_items + warps_per_block - 1) / warps_per_block;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    ojasx_rvq_fused_gemv_cuda_kernel<<<blocks, threads, 0, s>>>(
        packed_codes, codebooks, row_scales, bias, inputs,
        batch_size, stages, bits_per_stage, block_size, rows, columns,
        row_start, row_stop, output);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int ojasx_cuda_leech24_fused_gemm(
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
    float* output,
    const uint32_t* d_words,
    const uint32_t* d_octads,
    const void* d_pairs,
    void* stream) {
    if (batch_size <= 0 || rows <= 0 || columns <= 0 || row_start >= row_stop) return -1;
    const int row_count = row_stop - row_start;
    const int total_items = batch_size * row_count;
    const int threads = 128;
    const int warps_per_block = threads / 32;
    const int blocks = (total_items + warps_per_block - 1) / warps_per_block;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    ojasx_leech24_fused_gemv_cuda_kernel<<<blocks, threads, 0, s>>>(
        packed_indices, packed_gains, presence, gain_codebooks,
        row_scales, bias, rotated_inputs,
        batch_size, stages, gain_bits, adaptive_stages, rows, columns,
        row_start, row_stop, output,
        d_words, d_octads, static_cast<const int2*>(d_pairs));
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // extern "C"

#else

// Host/CPU stubs when not compiling with nvcc
extern "C" {

int ojasx_cuda_rvq_fused_gemm(
    const uint8_t*, const float*, const float*, const float*,
    const float*, int, int, int, int, int, int, int, int, float*, void*) {
    return -999;  // CUDA compilation required
}

int ojasx_cuda_leech24_fused_gemm(
    const uint8_t* const*, const uint8_t* const*, const uint8_t* const*,
    const float*, const float*, const float*, const float*,
    int, int, int, int, int, int, int, int, float*,
    const uint32_t*, const uint32_t*, const void*, void*) {
    return -999;  // CUDA compilation required
}

}  // extern "C"

#endif

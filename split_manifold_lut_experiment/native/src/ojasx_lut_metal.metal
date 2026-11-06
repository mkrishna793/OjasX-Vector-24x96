/**
 * ojasx_lut_metal.metal
 *
 * Metal Compute Kernel for Split-Manifold LUT Execution on Apple Silicon GPUs.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include <metal_stdlib>
using namespace metal;

// Kernel 1: Build LUT
kernel void build_lut_metal(
    device const float* x [[buffer(0)]],
    device const float* codebook [[buffer(1)]],
    device float* lut [[buffer(2)]],
    constant int& num_groups [[buffer(3)]],
    constant int& block_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int g = gid.y;
    int k = gid.x;

    if (g >= num_groups || k >= 256) return;

    device const float* x_g = x + (g * block_dim);
    device const float* c_k = codebook + (k * block_dim);

    float dot = 0.0f;
    for (int d = 0; d < 16; ++d) {
        if (d < block_dim) {
            dot += x_g[d] * c_k[d];
        }
    }

    lut[g * 256 + k] = dot;
}

// Kernel 2: Zero-Unpack Manifold GEMV
kernel void split_manifold_gemv_metal(
    device const float* lut [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& out_features [[buffer(5)]],
    constant int& num_groups [[buffer(6)]],
    constant int& has_bias [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    int m = gid;
    if (m >= out_features) return;

    device const uchar* code_row = codes + (m * num_groups);
    float sum = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
        uchar code = code_row[g];
        sum += lut[g * 256 + code];
    }

    float res = sum * scales[m];
    if (has_bias != 0) {
        res += bias[m];
    }
    y[m] = res;
}

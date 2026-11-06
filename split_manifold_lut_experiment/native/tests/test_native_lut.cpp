/**
 * test_native_lut.cpp
 *
 * Comprehensive Native C++ Test Suite for Split-Manifold LUT Engine.
 *
 * Author: OjasX Research Team & Antigravity Prime Coder
 */

#include "ojasx_lut.h"
#include "ojasx_lut_kv_cache.h"
#include "ojasx_lut_c_api.h"
#include "ojasx_ggml_adapter.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

void test_metadata() {
    std::cout << "[Test 1] Metadata & Target Backend..." << std::endl;
    const char* ver = ojasx_get_version();
    const char* backend = ojasx_get_backend_name();
    std::cout << "  Version: " << ver << std::endl;
    std::cout << "  Backend: " << backend << std::endl;
    assert(ver != nullptr);
    assert(backend != nullptr);
    std::cout << "  PASSED!" << std::endl;
}

void test_linear_correctness() {
    std::cout << "\n[Test 2] Split-Manifold LUT Linear Layer Numerical Test..." << std::endl;
    const int in_f = 256;
    const int out_f = 128;
    const int blk_dim = 16;
    const int cb_size = 256;
    const int G = in_f / blk_dim;

    ojasx::SplitManifoldLinear layer(in_f, out_f, blk_dim, cb_size);

    // Initialize codebook with distinct orthogonal centroids
    for (int k = 0; k < cb_size; ++k) {
        for (int d = 0; d < blk_dim; ++d) {
            layer.codebook[k * blk_dim + d] = (float)((k % 17) - 8) * 0.1f + (float)d * 0.01f;
        }
    }

    // Assign codes deterministically
    for (int m = 0; m < out_f; ++m) {
        layer.scales[m] = 1.0f + 0.01f * (float)m;
        for (int g = 0; g < G; ++g) {
            layer.codes[m * G + g] = (uint8_t)((m * 3 + g * 7) % 256);
        }
    }

    // Input vector
    std::vector<float> x(in_f);
    for (int i = 0; i < in_f; ++i) {
        x[i] = std::sin((float)i * 0.2f);
    }

    // Compute via native forward
    std::vector<float> y_native(out_f, 0.0f);
    layer.forward_cpu(x.data(), y_native.data(), 1, 1);

    // Compute ground truth by unpacking codes manually
    std::vector<float> y_ref(out_f, 0.0f);
    for (int m = 0; m < out_f; ++m) {
        float row_sum = 0.0f;
        for (int g = 0; g < G; ++g) {
            uint8_t code = layer.codes[m * G + g];
            const float* c = layer.codebook.data() + (code * blk_dim);
            float dot = 0.0f;
            for (int d = 0; d < blk_dim; ++d) {
                dot += x[g * blk_dim + d] * c[d];
            }
            row_sum += dot;
        }
        y_ref[m] = row_sum * layer.scales[m];
    }

    // Check error
    float max_err = 0.0f;
    for (int m = 0; m < out_f; ++m) {
        float err = std::abs(y_native[m] - y_ref[m]);
        max_err = std::max(max_err, err);
    }

    std::cout << "  Max Discrepancy vs Manual Unpack: " << max_err << std::endl;
    assert(max_err < 1e-4f);
    std::cout << "  Compression Ratio: " << layer.get_compression_ratio() << "x" << std::endl;
    std::cout << "  SRAM LUT Cache Size: " << layer.get_lut_sram_byte_size() << " bytes" << std::endl;
    std::cout << "  PASSED!" << std::endl;
}

void test_outliers_and_csr() {
    std::cout << "\n[Test 3] CSR Sparse Outlier Bypass Test..." << std::endl;
    const int in_f = 64;
    const int out_f = 32;

    ojasx::SplitManifoldLinear layer(in_f, out_f, 16, 256);

    // Setup CSR outliers: 2 outliers per row
    layer.outliers.out_features = out_f;
    layer.outliers.in_features = in_f;
    layer.outliers.nnz = out_f * 2;
    layer.outliers.row_ptrs.resize(out_f + 1);

    for (int m = 0; m < out_f; ++m) {
        layer.outliers.row_ptrs[m] = m * 2;
        layer.outliers.col_indices.push_back((m * 2) % in_f);
        layer.outliers.col_indices.push_back((m * 2 + 1) % in_f);
        layer.outliers.values.push_back(10.0f);
        layer.outliers.values.push_back(-5.0f);
    }
    layer.outliers.row_ptrs[out_f] = out_f * 2;

    std::vector<float> x(in_f, 1.0f);
    std::vector<float> y(out_f, 0.0f);

    layer.forward_cpu(x.data(), y.data(), 1, 1);

    // Outlier contribution should be 10.0 - 5.0 = 5.0 added to each row
    std::cout << "  Sample output y[0]: " << y[0] << std::endl;
    assert(y[0] >= 4.5f); // 5.0 outlier + manifold
    std::cout << "  PASSED!" << std::endl;
}

void test_kv_cache() {
    std::cout << "\n[Test 4] Zero-Unpack Compressed KV-Cache Test..." << std::endl;
    const int head_dim = 128;
    const int max_seq = 2048;

    ojasx::HeadKVCache kv(head_dim, max_seq, 16);

    // Simulate appending 1000 tokens
    std::vector<float> k(head_dim, 0.5f);
    std::vector<float> v(head_dim, 1.0f);

    for (int t = 0; t < 1000; ++t) {
        k[0] = (float)t * 0.01f;
        v[0] = (float)t * 0.02f;
        kv.append_token(k.data(), v.data());
    }

    assert(kv.current_seq_len == 1000);
    std::cout << "  Appended 1000 tokens successfully." << std::endl;
    std::cout << "  Dense Memory Baseline: " << kv.get_dense_byte_size() / 1024 << " KB" << std::endl;
    std::cout << "  Compressed KV Memory:  " << kv.get_byte_size() / 1024 << " KB" << std::endl;
    std::cout << "  KV Compression Ratio:  " << kv.get_compression_ratio() << "x smaller" << std::endl;
    assert(kv.get_compression_ratio() >= 12.0f);

    // Compute attention logits with Query vector
    std::vector<float> q(head_dim, 0.2f);
    std::vector<float> logits(1000);
    kv.compute_attention_scores_zero_unpack(q.data(), logits.data());

    std::cout << "  Logits computed. First logit: " << logits[0] << ", Last logit: " << logits[999] << std::endl;
    assert(!std::isnan(logits[0]));
    assert(!std::isnan(logits[999]));
    std::cout << "  PASSED!" << std::endl;
}

void test_c_api() {
    std::cout << "\n[Test 5] C-ABI Dynamic Library Export Test..." << std::endl;
    // Real-world layer dimensions (e.g. 1024 x 512)
    OjasxLinearHandle handle = ojasx_create_linear(1024, 512, 16, 256, nullptr, nullptr, nullptr, nullptr);
    assert(handle != nullptr);

    size_t comp_size = ojasx_get_compressed_size(handle);
    size_t dense_size = ojasx_get_dense_size(handle);
    float ratio = ojasx_get_compression_ratio(handle);

    std::cout << "  Linear Handle created. Dense: " << dense_size / 1024 << " KB, Compressed: " << comp_size / 1024 << " KB, Ratio: " << ratio << "x" << std::endl;
    assert(dense_size > comp_size);
    assert(ratio > 10.0f);

    ojasx_destroy_linear(handle);

    OjasxKVHandle kv_handle = ojasx_create_kv_cache(4, 128, 512, 16);
    assert(kv_handle != nullptr);
    ojasx_destroy_kv_cache(kv_handle);

    std::cout << "  PASSED!" << std::endl;
}

void test_moe_engine() {
    std::cout << "\n[Test 6] Native C++ MoE 512-Expert Zero-Arithmetic Gather Test..." << std::endl;
    const int num_experts = 512;
    const int hidden_dim = 256;
    const int intermediate_dim = 64;
    const int top_k = 10;
    const int block_dim = 16;

    ojasx::SplitManifoldMoE moe(num_experts, hidden_dim, intermediate_dim, top_k, block_dim);

    // Initialize codebooks
    for (int k = 0; k < 256; ++k) {
        for (int d = 0; d < block_dim; ++d) {
            moe.codebook[k * block_dim + d] = 0.05f * (float)((k % 11) - 5);
            moe.down_codebook[k * block_dim + d] = 0.03f * (float)((k % 7) - 3);
        }
    }

    // Set scales and deterministic codes
    std::fill(moe.gate_up_scales.begin(), moe.gate_up_scales.end(), 1.0f);
    std::fill(moe.down_scales.begin(), moe.down_scales.end(), 1.0f);
    for (size_t i = 0; i < moe.gate_up_codes.size(); ++i) {
        moe.gate_up_codes[i] = (uint8_t)(i % 256);
    }
    for (size_t i = 0; i < moe.down_codes.size(); ++i) {
        moe.down_codes[i] = (uint8_t)((i * 3) % 256);
    }

    std::cout << "  512-Expert MoE Initialized." << std::endl;
    std::cout << "  Dense Baseline Size:     " << moe.get_uncompressed_dense_byte_size() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Compressed Memory in RAM: " << moe.get_compressed_byte_size() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Compression Ratio:        " << moe.get_compression_ratio() << "x" << std::endl;
    assert(moe.get_compression_ratio() > 20.0f);

    // Test routing top-10 experts
    std::vector<float> x(hidden_dim, 0.5f);
    std::vector<int> top_k_indices = {4, 12, 57, 103, 199, 255, 340, 412, 480, 511};
    std::vector<float> top_k_weights = {0.25f, 0.20f, 0.15f, 0.10f, 0.08f, 0.07f, 0.05f, 0.04f, 0.03f, 0.03f};
    std::vector<float> y_out(hidden_dim, 0.0f);

    moe.forward_cpu(x.data(), top_k_indices.data(), top_k_weights.data(), y_out.data(), top_k);

    std::cout << "  Forward gather completed. y_out[0]: " << y_out[0] << ", y_out[127]: " << y_out[127] << std::endl;
    assert(!std::isnan(y_out[0]));
    assert(!std::isnan(y_out[127]));

    // Verify C-API export for MoE
    OjasxMoEHandle moe_handle = ojasx_create_moe(
        num_experts, hidden_dim, intermediate_dim, top_k, block_dim,
        moe.codebook.data(), moe.down_codebook.data(),
        moe.gate_up_codes.data(), moe.gate_up_scales.data(),
        moe.down_codes.data(), moe.down_scales.data()
    );
    assert(moe_handle != nullptr);

    std::vector<float> y_c_api(hidden_dim, 0.0f);
    int rc = ojasx_moe_forward(moe_handle, x.data(), top_k_indices.data(), top_k_weights.data(), y_c_api.data(), top_k);
    assert(rc == 0);
    assert(std::abs(y_out[0] - y_c_api[0]) < 1e-6f);
    ojasx_destroy_moe(moe_handle);

    std::cout << "  PASSED!" << std::endl;
}

void test_ggml_adapter() {
    std::cout << "\n[Test 7] llama.cpp / GGML Ecosystem Adapter Dispatch Test..." << std::endl;
    int rc = ojasx_ggml_init();
    assert(rc == 0);

    const int in_f = 128;
    const int out_f = 64;
    ojasx::SplitManifoldLinear layer(in_f, out_f, 16, 256);
    std::fill(layer.codebook.begin(), layer.codebook.end(), 0.1f);
    std::fill(layer.scales.begin(), layer.scales.end(), 1.0f);

    // Create GGML tensors
    struct ojasx_ggml_tensor src0;
    src0.type = 0; // GGML_TYPE_F32
    src0.ne[0] = in_f;
    src0.ne[1] = 1;
    std::vector<float> x(in_f, 1.0f);
    src0.data = x.data();
    src0.extra = nullptr;

    struct ojasx_ggml_tensor src1;
    src1.type = GGML_TYPE_OJASX_LUT;
    src1.ne[0] = in_f;
    src1.ne[1] = out_f;
    src1.data = nullptr;
    src1.extra = (void*)&layer; // Pass layer directly

    struct ojasx_ggml_tensor dst;
    dst.type = 0; // GGML_TYPE_F32
    dst.ne[0] = out_f;
    dst.ne[1] = 1;
    std::vector<float> y(out_f, 0.0f);
    dst.data = y.data();
    dst.extra = nullptr;

    // Dispatch via GGML adapter
    rc = ojasx_ggml_forward_linear(&dst, &src0, &src1, 0, 1, (void*)&layer);
    assert(rc == 0);

    std::cout << "  GGML Linear dispatch returned code: " << rc << " | Output y[0]: " << y[0] << std::endl;
    assert(!std::isnan(y[0]));
    std::cout << "  PASSED!" << std::endl;
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "STARTING NATIVE C++ SPLIT-MANIFOLD TEST SUITE" << std::endl;
    std::cout << "========================================================" << std::endl;

    test_metadata();
    test_linear_correctness();
    test_outliers_and_csr();
    test_kv_cache();
    test_c_api();
    test_moe_engine();
    test_ggml_adapter();

    std::cout << "\n========================================================" << std::endl;
    std::cout << "ALL 7 NATIVE TESTS PASSED WITH 0 ERRORS!" << std::endl;
    std::cout << "========================================================" << std::endl;
    return 0;
}

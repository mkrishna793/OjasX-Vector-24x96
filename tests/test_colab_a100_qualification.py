"""Unit tests for Colab A100 GPU Qualification, CUDA Kernel Verification & Local Smoke Test."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON_PATH = REPO_ROOT / "benchmarks" / "colab_a100_results.json"
VALIDATION_SCRIPT_PATH = REPO_ROOT / "benchmarks" / "colab_a100_validation.py"


class TestColabA100QualificationResults:
    """Validate the empirical qualification results from Google Colab 80GB A100 GPU run."""

    def test_colab_a100_results_json_structure_and_hardware(self):
        assert RESULTS_JSON_PATH.exists(), f"Missing {RESULTS_JSON_PATH}"
        data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))

        # Hardware Probe Verification
        assert "hardware" in data
        hw = data["hardware"]
        assert "A100" in hw["gpu_name"]
        assert hw["gpu_vram_gb"] >= 75.0, f"Expected 80GB A100, got {hw['gpu_vram_gb']} GB"
        assert hw["cuda_available"] is True
        assert hw["nvcc_available"] is True
        assert hw["cpu_ram_gb"] >= 30.0

    def test_cuda_kernel_compilation_and_accuracy(self):
        data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        assert "test_1_cuda_kernels" in data
        t1 = data["test_1_cuda_kernels"]
        assert t1["compiled_successfully"] is True
        assert t1["compile_time_sec"] > 0.0

        # RVQ CUDA Fused GEMV verification
        rvq = t1["rvq_fused_gemv"]
        assert rvq["passed_accuracy_check"] is True
        assert rvq["max_abs_diff_vs_reference"] < 1e-4
        assert rvq["latency_us"] > 0.0
        assert rvq["throughput_gflops"] > 0.0

        # Leech24 CUDA Fused GEMV verification
        leech = t1["leech24_fused_gemv"]
        assert leech["passed_accuracy_check"] is True
        assert leech["max_abs_diff_vs_reference"] < 1e-4
        assert leech["latency_us"] > 0.0
        assert leech["throughput_gflops"] > 0.0

    def test_large_matrix_quantization_metrics(self):
        data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        assert "test_2_large_matrix_quantization" in data
        t2 = data["test_2_large_matrix_quantization"]

        # [4096, 4096] (33.5 MB dense)
        assert "matrix_4096x4096_bf16" in t2
        m4k = t2["matrix_4096x4096_bf16"]
        assert m4k["original_dense_bytes"] == 33554432
        assert m4k["leech24"]["compression_ratio"] >= 3.0
        assert m4k["leech24"]["measured_bits_per_weight"] <= 4.0
        assert m4k["leech24"]["relative_reconstruction_error"] <= 0.65
        assert m4k["rvq"]["compression_ratio"] >= 3.0
        assert m4k["rvq"]["measured_bits_per_weight"] <= 4.0

        # [4096, 11008] (90.1 MB dense)
        assert "matrix_4096x11008_bf16" in t2
        m11k = t2["matrix_4096x11008_bf16"]
        assert m11k["original_dense_bytes"] == 90177536
        assert m11k["leech24"]["compression_ratio"] >= 3.0
        assert m11k["rvq"]["compression_ratio"] >= 3.0

    def test_pretrained_hf_model_pipeline_metrics(self):
        data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        assert "test_3_pretrained_hf_model" in data
        t3 = data["test_3_pretrained_hf_model"]

        # M0 Introspection & 7-Bucket Admission
        m0 = t3["m0_introspection"]
        assert m0["hidden_size"] > 0
        assert m0["num_hidden_layers"] > 0
        assert m0["admission_80gb"]["admitted"] is True
        assert m0["admission_24gb"]["admitted"] is True

        # M2 Streaming Safetensors Ingestion
        m2 = t3["m2_streaming_ingest"]
        assert m2["quantized_tensors"] > 0
        assert m2["compression_ratio"] >= 1.0

        # M1 OQPack Container Verification
        m1 = t3["m1_oqpack"]
        assert m1["verified_integrity"] is True
        assert m1["reloaded_exact_parity"] is True

        # M3 Quality Certificate
        m3 = t3["m3_quality_certificate"]
        assert m3["passed"] is True
        assert abs(m3["perplexity_delta"]) <= 5.0

        # M4 KV Engine Parity & Chat
        m4 = t3["m4_kv_engine"]
        assert m4["cached_vs_uncached_parity"] is True
        assert m4["max_logit_abs_diff"] < 1e-4
        assert len(m4["turn_1"]["generated_text"]) > 0
        assert len(m4["turn_2"]["generated_text"]) > 0


class TestColabValidationLocalSmoke:
    """Smoke test to verify colab_a100_validation.py executes locally without error."""

    def test_local_smoke_run(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                sys.executable,
                str(VALIDATION_SCRIPT_PATH),
                "--smoke-local",
                "--output",
                str(tmp_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            assert proc.returncode == 0, f"Script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            assert tmp_path.exists()
            data = json.loads(tmp_path.read_text(encoding="utf-8"))
            assert "hardware" in data
            assert "test_2_large_matrix_quantization" in data
            assert "test_3_pretrained_hf_model" in data
        finally:
            tmp_path.unlink(missing_ok=True)

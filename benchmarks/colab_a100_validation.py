"""Large-Scale Validation and CUDA Kernel Execution for OjasX-24X96 on Google Colab A100.

Performs:
1. Hardware & Environment Probe (GPU name, VRAM, CPU RAM, CUDA/PyTorch versions).
2. Test 1: Real NVIDIA nvcc compilation of vq_kernels_cuda.cu & CUDA execution (RVQ & Leech24).
3. Test 2: Large-Scale BF16 Matrix Quantization ([4096, 4096] and [4096, 11008]).
4. Test 3: Real Pretrained Hugging Face LLM End-to-End Pipeline (M0-M8 with Qwen2.5-0.5B or fallback).
   - M0: Architecture Introspection & 7-Bucket Memory Admission (80GB A100 & 24GB budgets).
   - M2: Streaming Checkpoint Ingestion of real .safetensors shards.
   - M1: Canonical OQPack Container Save, Checksum Verification & Reload Parity.
   - M3: Quality Certificate Generation (PPL, KL, Top-1/Top-5 agreement).
   - M4: KV-Cache Parity (use_cache=True vs False) & Multi-Turn Chat Generation.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure ojasx_24x96 is importable from repo root or Colab /content
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ojasx_24x96 import (
    LeechQuantizationConfig,
    OjasXInferenceEngine,
    PlanningConstraints,
    QualityCertificateThresholds,
    QuantizationConfig,
    SamplingParams,
    cuda_kernel_source_status,
    dequantize_leech_weight,
    dequantize_weight,
    evaluate_memory_admission,
    generate_quality_certificate,
    introspect_architecture,
    load_quantized_model_from_oqpack,
    max_admissible_context_length,
    quantize_leech_weight,
    quantize_model_generalized,
    quantize_weight,
    rotate_activation_for_leech,
    save_oqpack,
    stream_convert_checkpoint,
    verify_oqpack_integrity,
)
from ojasx_24x96.leech24 import golay_codewords, octads, pair_indices


def enable_cuda_leech_acceleration():
    """Accelerate nearest_shell2 using CUDA when a GPU is available."""
    if not torch.cuda.is_available():
        return
    import ojasx_24x96.leech24 as l24
    import ojasx_24x96.leech_codec as lc

    orig_nearest_shell2 = l24.nearest_shell2

    def _cuda_accel_nearest_shell2(vectors: torch.Tensor, *, chunk_size: int = 16384):
        if not torch.cuda.is_available() or vectors.device.type == "cuda":
            return orig_nearest_shell2(vectors, chunk_size=chunk_size)
        if vectors.ndim != 2 or vectors.shape[1] != 24:
            raise ValueError("Shell search expects [vectors, 24]")
        indices_list = []
        dots_list = []
        step = max(chunk_size, 16384)
        for start in range(0, vectors.shape[0], step):
            sub = vectors[start : start + step].cuda()
            sub_idx, sub_dot = l24._nearest_shell2_chunk(sub)
            indices_list.append(sub_idx.to(device=vectors.device))
            dots_list.append(sub_dot.to(device=vectors.device))
        if not indices_list:
            return orig_nearest_shell2(vectors, chunk_size=chunk_size)
        return torch.cat(indices_list), torch.cat(dots_list)

    lc.nearest_shell2 = _cuda_accel_nearest_shell2
    l24.nearest_shell2 = _cuda_accel_nearest_shell2


def probe_hardware() -> Dict[str, Any]:
    """Probe hardware, GPU VRAM, CPU memory, and compiler tools."""
    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "cpu_ram_bytes": psutil.virtual_memory().total,
        "cpu_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    if torch.cuda.is_available():
        dev_idx = 0
        props = torch.cuda.get_device_properties(dev_idx)
        cap = torch.cuda.get_device_capability(dev_idx)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(dev_idx),
                "gpu_count": torch.cuda.device_count(),
                "gpu_vram_bytes": props.total_memory,
                "gpu_vram_gb": round(props.total_memory / (1024**3), 2),
                "gpu_sm_capability": list(cap),
                "gpu_multi_processor_count": getattr(props, "multi_processor_count", None),
            }
        )
    else:
        info.update(
            {
                "gpu_name": "None (CPU Execution)",
                "gpu_count": 0,
                "gpu_vram_bytes": 0,
                "gpu_vram_gb": 0.0,
                "gpu_sm_capability": None,
            }
        )

    nvcc_path = shutil.which("nvcc")
    info["nvcc_available"] = nvcc_path is not None
    info["nvcc_path"] = nvcc_path
    if nvcc_path:
        try:
            res = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, check=True)
            info["nvcc_version"] = res.stdout.strip().split("\n")[-1]
        except Exception as e:
            info["nvcc_version"] = f"Error: {e}"

    return info


def test_1_compile_and_execute_cuda_kernels(
    cuda_src_path: Path, output_lib_path: Path = Path("/tmp/libojasx_vq_cuda.so")
) -> Dict[str, Any]:
    """Test 1: Compile vq_kernels_cuda.cu with nvcc and execute RVQ & Leech24 kernels."""
    result: Dict[str, Any] = {
        "cuda_source_path": str(cuda_src_path),
        "cuda_source_exists": cuda_src_path.exists(),
        "output_lib_path": str(output_lib_path),
        "compiled_successfully": False,
        "compile_time_sec": None,
        "rvq_fused_gemv": {},
        "leech24_fused_gemv": {},
    }

    if not torch.cuda.is_available() or not shutil.which("nvcc"):
        result["skipped"] = True
        result["skip_reason"] = "CUDA or nvcc not available in environment"
        return result

    # 1. Compile with nvcc
    major, minor = torch.cuda.get_device_capability(0)
    arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        arch_flag,
        str(cuda_src_path),
        "-o",
        str(output_lib_path),
    ]

    t0 = time.perf_counter()
    comp_res = subprocess.run(cmd, capture_output=True, text=True)
    compile_time = time.perf_counter() - t0
    result["compile_time_sec"] = round(compile_time, 3)

    if comp_res.returncode != 0:
        result["compile_error"] = comp_res.stderr.strip()
        return result

    result["compiled_successfully"] = True

    # 2. Load shared library via ctypes
    lib = ctypes.CDLL(str(output_lib_path))

    # Setup signatures
    lib.ojasx_cuda_rvq_fused_gemm.argtypes = [
        ctypes.c_void_p,  # packed_codes
        ctypes.c_void_p,  # codebooks
        ctypes.c_void_p,  # row_scales
        ctypes.c_void_p,  # bias
        ctypes.c_void_p,  # inputs
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # stages
        ctypes.c_int,     # bits_per_stage
        ctypes.c_int,     # block_size
        ctypes.c_int,     # rows
        ctypes.c_int,     # columns
        ctypes.c_int,     # row_start
        ctypes.c_int,     # row_stop
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # stream
    ]
    lib.ojasx_cuda_rvq_fused_gemm.restype = ctypes.c_int

    lib.ojasx_cuda_leech24_fused_gemm.argtypes = [
        ctypes.c_void_p,  # packed_indices
        ctypes.c_void_p,  # packed_gains
        ctypes.c_void_p,  # presence
        ctypes.c_void_p,  # gain_codebooks
        ctypes.c_void_p,  # row_scales
        ctypes.c_void_p,  # bias
        ctypes.c_void_p,  # rotated_inputs
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # stages
        ctypes.c_int,     # gain_bits
        ctypes.c_int,     # adaptive_stages
        ctypes.c_int,     # rows
        ctypes.c_int,     # columns
        ctypes.c_int,     # row_start
        ctypes.c_int,     # row_stop
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # d_words
        ctypes.c_void_p,  # d_octads
        ctypes.c_void_p,  # d_pairs
        ctypes.c_void_p,  # stream
    ]
    lib.ojasx_cuda_leech24_fused_gemm.restype = ctypes.c_int

    # 3. Test RVQ Fused GEMV
    torch.manual_seed(20261104)
    batch_size, rows, cols = 4, 128, 256
    weight_rvq = torch.randn(rows, cols, dtype=torch.float32)
    bias_rvq = torch.randn(rows, dtype=torch.float32)
    inputs_rvq = torch.randn(batch_size, cols, dtype=torch.float32, device="cuda")

    cfg_rvq = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        kmeans_iterations=4,
        min_weight_elements=1,
        max_relative_error=None,
        require_compression=False,
    )
    enc_rvq, _ = quantize_weight(weight_rvq, cfg_rvq)
    assert enc_rvq is not None

    d_packed_codes = enc_rvq.packed_codes.cuda().contiguous()
    d_codebooks = enc_rvq.codebooks.cuda().contiguous()
    d_row_scales = enc_rvq.row_scales.cuda().contiguous()
    d_bias_rvq = bias_rvq.cuda().contiguous()
    d_output_rvq = torch.zeros(batch_size, rows, dtype=torch.float32, device="cuda")

    status_rvq = lib.ojasx_cuda_rvq_fused_gemm(
        ctypes.c_void_p(d_packed_codes.data_ptr()),
        ctypes.c_void_p(d_codebooks.data_ptr()),
        ctypes.c_void_p(d_row_scales.data_ptr()),
        ctypes.c_void_p(d_bias_rvq.data_ptr()),
        ctypes.c_void_p(inputs_rvq.data_ptr()),
        batch_size,
        enc_rvq.stages,
        enc_rvq.bits_per_stage,
        enc_rvq.block_size,
        rows,
        cols,
        0,
        rows,
        ctypes.c_void_p(d_output_rvq.data_ptr()),
        None,
    )
    torch.cuda.synchronize()
    assert status_rvq == 0, f"RVQ CUDA kernel failed with code {status_rvq}"

    ref_w_rvq = dequantize_weight(enc_rvq).cuda()
    ref_out_rvq = F.linear(inputs_rvq, ref_w_rvq, d_bias_rvq)
    max_diff_rvq = float((d_output_rvq - ref_out_rvq).abs().max().item())

    # Warmup and benchmark
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        lib.ojasx_cuda_rvq_fused_gemm(
            ctypes.c_void_p(d_packed_codes.data_ptr()),
            ctypes.c_void_p(d_codebooks.data_ptr()),
            ctypes.c_void_p(d_row_scales.data_ptr()),
            ctypes.c_void_p(d_bias_rvq.data_ptr()),
            ctypes.c_void_p(inputs_rvq.data_ptr()),
            batch_size,
            enc_rvq.stages,
            enc_rvq.bits_per_stage,
            enc_rvq.block_size,
            rows,
            cols,
            0,
            rows,
            ctypes.c_void_p(d_output_rvq.data_ptr()),
            None,
        )
    torch.cuda.synchronize()

    start_ev.record()
    bench_iters = 50
    for _ in range(bench_iters):
        lib.ojasx_cuda_rvq_fused_gemm(
            ctypes.c_void_p(d_packed_codes.data_ptr()),
            ctypes.c_void_p(d_codebooks.data_ptr()),
            ctypes.c_void_p(d_row_scales.data_ptr()),
            ctypes.c_void_p(d_bias_rvq.data_ptr()),
            ctypes.c_void_p(inputs_rvq.data_ptr()),
            batch_size,
            enc_rvq.stages,
            enc_rvq.bits_per_stage,
            enc_rvq.block_size,
            rows,
            cols,
            0,
            rows,
            ctypes.c_void_p(d_output_rvq.data_ptr()),
            None,
        )
    end_ev.record()
    torch.cuda.synchronize()
    rvq_ms = start_ev.elapsed_time(end_ev) / bench_iters
    rvq_flops = 2 * batch_size * rows * cols
    rvq_gflops = (rvq_flops / (rvq_ms * 1e-3)) / 1e9

    result["rvq_fused_gemv"] = {
        "batch_size": batch_size,
        "rows": rows,
        "columns": cols,
        "max_abs_diff_vs_reference": max_diff_rvq,
        "passed_accuracy_check": bool(max_diff_rvq < 1e-4),
        "latency_us": round(rvq_ms * 1000.0, 2),
        "throughput_gflops": round(rvq_gflops, 2),
    }

    # 4. Test Leech24 Fused GEMV
    cols_leech = 240
    weight_leech = torch.randn(rows, cols_leech, dtype=torch.float32)
    bias_leech = torch.randn(rows, dtype=torch.float32)
    inputs_leech = torch.randn(batch_size, cols_leech, dtype=torch.float32, device="cuda")

    cfg_leech = LeechQuantizationConfig(
        max_stages=2,
        gain_bits=2,
        rotation_reflections=2,
        adaptive_stages=False,
        min_weight_elements=1,
        max_relative_error=None,
        require_compression=False,
    )
    enc_leech, _ = quantize_leech_weight(weight_leech, cfg_leech)
    assert enc_leech is not None

    cuda_indices = [t.cuda().contiguous() for t in enc_leech.packed_indices]
    cuda_gains = [t.cuda().contiguous() for t in enc_leech.packed_gains]
    cuda_presence = [t.cuda().contiguous() for t in enc_leech.presence]

    d_packed_indices = torch.tensor([t.data_ptr() for t in cuda_indices], dtype=torch.int64, device="cuda")
    d_packed_gains = torch.tensor([t.data_ptr() for t in cuda_gains], dtype=torch.int64, device="cuda")
    d_presence = torch.tensor([t.data_ptr() for t in cuda_presence], dtype=torch.int64, device="cuda")

    d_gain_codebooks = enc_leech.gain_codebooks.cuda().contiguous()
    d_row_scales_l = enc_leech.row_scales.cuda().contiguous()
    d_bias_l = bias_leech.cuda().contiguous()

    d_words = torch.tensor(golay_codewords(), dtype=torch.int32, device="cuda").contiguous()
    d_octads = torch.tensor(octads(), dtype=torch.int32, device="cuda").contiguous()
    d_pairs = torch.tensor(pair_indices(), dtype=torch.int32, device="cuda").contiguous()

    rotated_inputs = rotate_activation_for_leech(
        inputs_leech,
        cols_leech,
        enc_leech.rotation_reflections,
        enc_leech.rotation_seed,
    ).cuda().contiguous()

    d_output_leech = torch.zeros(batch_size, rows, dtype=torch.float32, device="cuda")

    status_leech = lib.ojasx_cuda_leech24_fused_gemm(
        ctypes.c_void_p(d_packed_indices.data_ptr()),
        ctypes.c_void_p(d_packed_gains.data_ptr()),
        ctypes.c_void_p(d_presence.data_ptr()),
        ctypes.c_void_p(d_gain_codebooks.data_ptr()),
        ctypes.c_void_p(d_row_scales_l.data_ptr()),
        ctypes.c_void_p(d_bias_l.data_ptr()),
        ctypes.c_void_p(rotated_inputs.data_ptr()),
        batch_size,
        enc_leech.stages,
        enc_leech.gain_bits,
        int(enc_leech.adaptive_stages),
        rows,
        cols_leech,
        0,
        rows,
        ctypes.c_void_p(d_output_leech.data_ptr()),
        ctypes.c_void_p(d_words.data_ptr()),
        ctypes.c_void_p(d_octads.data_ptr()),
        ctypes.c_void_p(d_pairs.data_ptr()),
        None,
    )
    torch.cuda.synchronize()
    assert status_leech == 0, f"Leech24 CUDA kernel failed with code {status_leech}"

    ref_w_leech = dequantize_leech_weight(enc_leech).cuda()
    ref_out_leech = F.linear(inputs_leech, ref_w_leech, d_bias_l)
    max_diff_leech = float((d_output_leech - ref_out_leech).abs().max().item())

    # Warmup and benchmark
    for _ in range(10):
        lib.ojasx_cuda_leech24_fused_gemm(
            ctypes.c_void_p(d_packed_indices.data_ptr()),
            ctypes.c_void_p(d_packed_gains.data_ptr()),
            ctypes.c_void_p(d_presence.data_ptr()),
            ctypes.c_void_p(d_gain_codebooks.data_ptr()),
            ctypes.c_void_p(d_row_scales_l.data_ptr()),
            ctypes.c_void_p(d_bias_l.data_ptr()),
            ctypes.c_void_p(rotated_inputs.data_ptr()),
            batch_size,
            enc_leech.stages,
            enc_leech.gain_bits,
            int(enc_leech.adaptive_stages),
            rows,
            cols_leech,
            0,
            rows,
            ctypes.c_void_p(d_output_leech.data_ptr()),
            ctypes.c_void_p(d_words.data_ptr()),
            ctypes.c_void_p(d_octads.data_ptr()),
            ctypes.c_void_p(d_pairs.data_ptr()),
            None,
        )
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(bench_iters):
        lib.ojasx_cuda_leech24_fused_gemm(
            ctypes.c_void_p(d_packed_indices.data_ptr()),
            ctypes.c_void_p(d_packed_gains.data_ptr()),
            ctypes.c_void_p(d_presence.data_ptr()),
            ctypes.c_void_p(d_gain_codebooks.data_ptr()),
            ctypes.c_void_p(d_row_scales_l.data_ptr()),
            ctypes.c_void_p(d_bias_l.data_ptr()),
            ctypes.c_void_p(rotated_inputs.data_ptr()),
            batch_size,
            enc_leech.stages,
            enc_leech.gain_bits,
            int(enc_leech.adaptive_stages),
            rows,
            cols_leech,
            0,
            rows,
            ctypes.c_void_p(d_output_leech.data_ptr()),
            ctypes.c_void_p(d_words.data_ptr()),
            ctypes.c_void_p(d_octads.data_ptr()),
            ctypes.c_void_p(d_pairs.data_ptr()),
            None,
        )
    end_ev.record()
    torch.cuda.synchronize()
    leech_ms = start_ev.elapsed_time(end_ev) / bench_iters
    leech_flops = 2 * batch_size * rows * cols_leech
    leech_gflops = (leech_flops / (leech_ms * 1e-3)) / 1e9

    result["leech24_fused_gemv"] = {
        "batch_size": batch_size,
        "rows": rows,
        "columns": cols_leech,
        "max_abs_diff_vs_reference": max_diff_leech,
        "passed_accuracy_check": bool(max_diff_leech < 1e-4),
        "latency_us": round(leech_ms * 1000.0, 2),
        "throughput_gflops": round(leech_gflops, 2),
    }

    return result


def test_2_large_matrix_quantization(
    matrix_shapes: Optional[List[Tuple[int, int]]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Test 2: Quantize 7B/14B-scale BF16 projection matrices (4096x4096 and 4096x11008)."""
    if matrix_shapes is None:
        matrix_shapes = [(4096, 4096), (4096, 11008)]

    results: Dict[str, Any] = {}
    target_dev = torch.device(device)

    for rows, cols in matrix_shapes:
        key = f"matrix_{rows}x{cols}_bf16"
        dense_bytes = rows * cols * 2  # BF16 = 2 bytes per element
        dense_mb = round(dense_bytes / (1024**2), 2)

        torch.manual_seed(20261104)
        weight = torch.randn(rows, cols, dtype=torch.bfloat16, device=target_dev)

        # 1. Leech24 Quantization
        leech_cfg = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=2,
            rotation_reflections=2,
            chunk_size=4096,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        t0 = time.perf_counter()
        enc_leech, meas_leech = quantize_leech_weight(weight, leech_cfg)
        leech_quant_time = time.perf_counter() - t0

        assert enc_leech is not None
        leech_stored = enc_leech.storage_bytes
        leech_ratio = round(dense_bytes / max(1, leech_stored), 3)
        leech_bpw = round(enc_leech.measured_bits_per_weight, 3)

        # Relative reconstruction error
        deq_leech = dequantize_leech_weight(enc_leech).to(dtype=torch.float32, device=target_dev)
        ref_float = weight.to(dtype=torch.float32)
        leech_rel_err = float((deq_leech - ref_float).norm() / ref_float.norm().clamp_min(1e-8))

        # 2. RVQ Quantization
        rvq_cfg = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            kmeans_iterations=4,
            chunk_size=4096,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        t0 = time.perf_counter()
        enc_rvq, meas_rvq = quantize_weight(weight, rvq_cfg)
        rvq_quant_time = time.perf_counter() - t0

        assert enc_rvq is not None
        rvq_stored = enc_rvq.storage_bytes
        rvq_ratio = round(dense_bytes / max(1, rvq_stored), 3)
        rvq_bpw = round(enc_rvq.measured_bits_per_weight, 3)

        deq_rvq = dequantize_weight(enc_rvq).to(dtype=torch.float32, device=target_dev)
        rvq_rel_err = float((deq_rvq - ref_float).norm() / ref_float.norm().clamp_min(1e-8))

        results[key] = {
            "rows": rows,
            "columns": cols,
            "original_dense_bytes": dense_bytes,
            "original_dense_mb": dense_mb,
            "leech24": {
                "stored_bytes": leech_stored,
                "compression_ratio": leech_ratio,
                "measured_bits_per_weight": leech_bpw,
                "relative_reconstruction_error": round(leech_rel_err, 4),
                "quantization_time_sec": round(leech_quant_time, 3),
            },
            "rvq": {
                "stored_bytes": rvq_stored,
                "compression_ratio": rvq_ratio,
                "measured_bits_per_weight": rvq_bpw,
                "relative_reconstruction_error": round(rvq_rel_err, 4),
                "quantization_time_sec": round(rvq_quant_time, 3),
            },
        }

        del weight, deq_leech, deq_rvq, ref_float
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def test_3_pretrained_hf_model_pipeline(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    fallback_if_offline: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Test 3: Real Pretrained Hugging Face LLM End-to-End Pipeline (M0-M8)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from huggingface_hub import snapshot_download

    target_dev = torch.device(device)
    result: Dict[str, Any] = {"model_id": model_id, "mode": "real_pretrained"}

    download_dir: Optional[Path] = None
    if model_id in {"fallback", "mock", "local"}:
        print("Using local synthetic Llama checkpoint for local smoke test...")
        result["mode"] = "synthetic_local"
        tokenizer = None
        base_model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=256,
                hidden_size=96,
                intermediate_size=192,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        ).eval().to(target_dev)
    else:
        try:
            print(f"Downloading checkpoint files for {model_id}...")
            raw_download_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt", "tokenizer*"],
            )
            download_dir = Path(raw_download_path)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).eval().to(target_dev)
        except Exception as exc:
            if not fallback_if_offline:
                raise
            print(f"Failed to download/load {model_id} ({exc}); falling back to local Llama checkpoint...")
            result["mode"] = "offline_fallback"
            result["fallback_reason"] = str(exc)
            tokenizer = None
            base_model = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=256,
                    hidden_size=96,
                    intermediate_size=192,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
            ).eval().to(target_dev)

    # 1. M0: Architecture Introspection & 7-Bucket Memory Admission
    arch_profile = introspect_architecture(base_model)
    result["m0_introspection"] = {
        "model_type": arch_profile.model_type,
        "architecture_class": arch_profile.architecture_class,
        "hidden_size": arch_profile.hidden_size,
        "intermediate_size": arch_profile.intermediate_size,
        "num_hidden_layers": arch_profile.num_hidden_layers,
        "num_attention_heads": arch_profile.num_attention_heads,
        "num_key_value_heads": arch_profile.num_key_value_heads,
        "head_dim": arch_profile.head_dim,
        "vocab_size": arch_profile.vocab_size,
        "is_moe": arch_profile.is_moe,
        "is_hybrid_attention": arch_profile.is_hybrid_attention,
    }

    # Evaluate memory admission for 80GB A100 and 24GB laptop budgets
    dense_param_bytes = sum(p.numel() * p.element_size() for p in base_model.parameters())
    # Estimate 4.2x compression
    est_encoded_bytes = int(dense_param_bytes / 4.2)
    admission_80gb = evaluate_memory_admission(
        arch_profile,
        encoded_weight_bytes=est_encoded_bytes,
        dense_weight_bytes=0,
        context_length=4096,
        batch_size=1,
        kv_dtype=torch.bfloat16,
        budget_bytes=80 * 1024**3,  # 80 GB A100 budget
    )
    admission_24gb = evaluate_memory_admission(
        arch_profile,
        encoded_weight_bytes=est_encoded_bytes,
        dense_weight_bytes=0,
        context_length=4096,
        batch_size=1,
        kv_dtype=torch.bfloat16,
        budget_bytes=24 * 1024**3,  # 24 GB laptop budget
    )
    headroom_80 = (admission_80gb.budget_bytes - admission_80gb.total_required_bytes) if admission_80gb.budget_bytes else 0
    headroom_24 = (admission_24gb.budget_bytes - admission_24gb.total_required_bytes) if admission_24gb.budget_bytes else 0
    result["m0_introspection"]["admission_80gb"] = {
        "budget_gb": 80.0,
        "total_required_gb": round(admission_80gb.total_required_bytes / (1024**3), 3),
        "headroom_gb": round(headroom_80 / (1024**3), 3),
        "admitted": admission_80gb.admitted,
        "buckets": {
            "encoded_weights_mb": round(admission_80gb.encoded_weight_bytes / (1024**2), 2),
            "dense_weights_mb": round(admission_80gb.dense_weight_bytes / (1024**2), 2),
            "kv_cache_mb": round(admission_80gb.kv_cache_bytes / (1024**2), 2),
            "activation_prefill_mb": round(admission_80gb.activation_prefill_bytes / (1024**2), 2),
            "activation_decode_mb": round(admission_80gb.activation_decode_bytes / (1024**2), 2),
            "workspace_mb": round(admission_80gb.workspace_bytes / (1024**2), 2),
            "runtime_overhead_mb": round(admission_80gb.runtime_overhead_bytes / (1024**2), 2),
        },
    }
    result["m0_introspection"]["admission_24gb"] = {
        "budget_gb": 24.0,
        "total_required_gb": round(admission_24gb.total_required_bytes / (1024**3), 3),
        "headroom_gb": round(headroom_24 / (1024**3), 3),
        "admitted": admission_24gb.admitted,
    }

    # 2. M2: Streaming Checkpoint Ingestion
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_ckpt = download_dir if download_dir is not None else tmp_path / "mock_ckpt"
        if download_dir is None:
            source_ckpt.mkdir()
            from safetensors.torch import save_file

            sd = {k: v.contiguous().cpu() for k, v in base_model.state_dict().items()}
            save_file(sd, str(source_ckpt / "model.safetensors"))

        stream_out = tmp_path / "stream_out"
        # Ingest with Leech24
        stream_report = stream_convert_checkpoint(
            source_ckpt,
            stream_out,
            codec="leech24_shell2_residual",
            leech_config=LeechQuantizationConfig(
                max_stages=2,
                gain_bits=2,
                rotation_reflections=2,
                chunk_size=4096,
                min_weight_elements=64,
                max_relative_error=None,
                require_compression=False,
            ),
            include_globs=("*layers.0.*", "*layers.1.*") if result["mode"] == "real_pretrained" else ("*",),
            resume=True,
        )
        result["m2_streaming_ingest"] = {
            "total_tensors": stream_report.total_tensors,
            "quantized_tensors": stream_report.quantized_tensors,
            "passthrough_tensors": stream_report.passthrough_tensors,
            "compression_ratio": round(stream_report.compression_ratio, 3),
            "peak_single_tensor_bytes": stream_report.peak_single_tensor_bytes,
        }

    # Prepare quantized model for M1, M3, M4
    # We quantize the first 2 layers for fast full-pipeline certification
    candidate_configs = [
        LeechQuantizationConfig(
            max_stages=2,
            gain_bits=2,
            rotation_reflections=2,
            chunk_size=4096,
            min_weight_elements=64,
            max_relative_error=None,
            require_compression=False,
        ),
        QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            kmeans_iterations=4,
            chunk_size=4096,
            min_weight_elements=64,
            max_relative_error=None,
            require_compression=False,
        ),
    ]

    # Evaluation token sentences
    sample_texts = [
        "The Leech lattice is an exceptional unimodular lattice in 24 dimensions.",
        "Vector quantization enables highly compressed deep learning inference on edge devices.",
        "High-performance CUDA kernels provide fused low-precision GEMV execution without dense weights.",
    ]
    if tokenizer is not None:
        eval_batches = [
            tokenizer(text, return_tensors="pt")["input_ids"].to(target_dev) for text in sample_texts
        ]
    else:
        eval_batches = [
            torch.randint(0, 256, (1, 16), dtype=torch.long, device=target_dev) for _ in range(3)
        ]

    working_model = copy.deepcopy(base_model)
    q_result = quantize_model_generalized(
        working_model,
        candidate_configs=candidate_configs,
        calibration_inputs=eval_batches[:1],
        include=("*layers.0.*", "*layers.1.*") if result["mode"] == "real_pretrained" else ("*",),
        constraints=PlanningConstraints(
            target_state_ratio=0.999,
            max_tensor_weight_error=1.0,
            runtime_profile="memory",
            architecture_profile=arch_profile,
            max_total_runtime_bytes=80 * 1024**3,
            context_length=4096,
        ),
    )
    q_model = q_result.model.eval().to(target_dev)

    # 3. M1: Canonical OQPack Container Save, Checksum Verification & Reload Parity
    with tempfile.TemporaryDirectory() as tmpdir:
        oq_dir = Path(tmpdir) / "oqpack_bundle"
        save_oqpack(
            q_model,
            oq_dir,
            plan=q_result.plan,
            architecture_profile=arch_profile,
            memory_admission=admission_80gb,
            model_id=str(model_id),
            max_shard_bytes=128 * 1024 * 1024,
        )
        verified_manifest = verify_oqpack_integrity(oq_dir, verify_checksums=True)
        skeleton = copy.deepcopy(base_model)
        reloaded_model, _ = load_quantized_model_from_oqpack(skeleton, oq_dir, verify_checksums=True)
        reloaded_model.eval().to(target_dev)

        with torch.inference_mode():
            out_q = q_model(eval_batches[0])
            out_r = reloaded_model(eval_batches[0])
            logits_q = out_q.logits if hasattr(out_q, "logits") else out_q
            logits_r = out_r.logits if hasattr(out_r, "logits") else out_r
            oqpack_parity = bool(torch.equal(logits_q, logits_r))

        result["m1_oqpack"] = {
            "verified_integrity": True,
            "manifest_schema": verified_manifest.format_version,
            "shard_count": len(verified_manifest.shards),
            "reloaded_exact_parity": oqpack_parity,
        }

    # 4. M3: Quality Certificate Generation
    cert = generate_quality_certificate(
        base_model,
        q_model,
        eval_batches,
        plan=q_result.plan,
        thresholds=QualityCertificateThresholds(
            max_perplexity_delta=50.0,
            max_kl_divergence=2.0,
            min_top1_agreement=0.10,
            min_top5_agreement=0.25,
            max_long_context_kl_drift=2.0,
        ),
    )
    result["m3_quality_certificate"] = {
        "passed": cert.passed,
        "failed_checks": list(cert.failed_checks),
        "baseline_perplexity": round(cert.baseline_perplexity, 2),
        "quantized_perplexity": round(cert.quantized_perplexity, 2),
        "perplexity_delta": round(cert.perplexity_delta, 2),
        "mean_kl_divergence": round(cert.mean_kl_divergence, 4),
        "max_kl_divergence": round(cert.max_kl_divergence, 4),
        "top1_agreement": round(cert.top1_agreement, 3),
        "top5_agreement": round(cert.top5_agreement, 3),
        "compression_ratio": round(cert.compression_ratio, 3),
    }

    # 5. M4: KV-Cache Parity & Multi-Turn Chat Generation
    engine = OjasXInferenceEngine(
        q_model,
        max_context_length=512,
        memory_budget_bytes=4 * 1024**3,
        kv_dtype=torch.float32,
    )

    test_prompt = eval_batches[0][0].tolist()[:8]
    parity_report = engine.verify_cached_vs_uncached_parity(test_prompt, max_new_tokens=4, atol=1e-4)

    # Multi-turn chat
    turn1_prompt = eval_batches[0][0].tolist()[:10]
    t1 = engine.chat_turn(turn1_prompt, SamplingParams(max_new_tokens=5, temperature=0.0))
    turn2_prompt = eval_batches[1][0].tolist()[:6]
    t2 = engine.chat_turn(turn2_prompt, SamplingParams(max_new_tokens=5, temperature=0.0))

    t1_text = tokenizer.decode(t1.generated_token_ids) if tokenizer is not None else str(t1.generated_token_ids)
    t2_text = tokenizer.decode(t2.generated_token_ids) if tokenizer is not None else str(t2.generated_token_ids)

    result["m4_kv_engine"] = {
        "cached_vs_uncached_parity": parity_report.matches_exactly,
        "max_logit_abs_diff": parity_report.max_logit_abs_diff,
        "turn_1": {
            "prefill_tokens": t1.prefill_tokens,
            "decode_steps": t1.decode_steps,
            "generated_text": t1_text.strip(),
            "kv_cache_bytes": t1.kv_cache_bytes,
        },
        "turn_2": {
            "prefill_tokens": t2.prefill_tokens,
            "decode_steps": t2.decode_steps,
            "generated_text": t2_text.strip(),
            "kv_cache_bytes": t2.kv_cache_bytes,
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="OjasX-24X96 Large-Scale & CUDA Kernel Validation on Colab A100")
    parser.add_argument("--output", type=str, default=None, help="Path to save colab_a100_results.json")
    parser.add_argument("--smoke-local", action="store_true", help="Run quick local smoke test with smaller matrices")
    parser.add_argument("--cuda-src", type=str, default=None, help="Path to vq_kernels_cuda.cu")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    if args.output:
        out_json_path = Path(args.output).resolve()
    else:
        out_json_path = repo_dir / "benchmarks" / "colab_a100_results.json"

    if args.cuda_src:
        cuda_src = Path(args.cuda_src).resolve()
    else:
        cuda_src = repo_dir / "ojasx_24x96" / "native" / "vq_kernels_cuda.cu"

    print("================================================================================")
    print("OjasX-24X96 Large-Scale Validation & CUDA Kernel Execution (Google Colab A100)")
    print("================================================================================")

    enable_cuda_leech_acceleration()

    # 1. Probe Hardware
    hw = probe_hardware()
    print("\n--- Hardware Probe ---")
    print(f"Platform: {hw['platform']}")
    print(f"GPU: {hw['gpu_name']} ({hw.get('gpu_vram_gb', 0.0)} GB VRAM)")
    print(f"CPU RAM: {hw['cpu_ram_gb']} GB, Cores: {hw['cpu_count']}")
    print(f"PyTorch: {hw['pytorch_version']}, CUDA: {hw['cuda_version']}, NVCC: {hw.get('nvcc_version', 'N/A')}")

    # 2. Test 1: Real NVIDIA nvcc compilation & CUDA execution
    print("\n--- Test 1: nvcc Compilation & CUDA Kernel Execution ---")
    test1 = test_1_compile_and_execute_cuda_kernels(cuda_src)
    if test1.get("compiled_successfully"):
        print(f"CUDA Kernels compiled in {test1['compile_time_sec']}s")
        rvq = test1["rvq_fused_gemv"]
        print(f"RVQ CUDA GEMV: max diff vs ref = {rvq['max_abs_diff_vs_reference']:.2e}, latency = {rvq['latency_us']}us ({rvq['throughput_gflops']} GFLOPS)")
        leech = test1["leech24_fused_gemv"]
        print(f"Leech24 CUDA GEMV: max diff vs ref = {leech['max_abs_diff_vs_reference']:.2e}, latency = {leech['latency_us']}us ({leech['throughput_gflops']} GFLOPS)")
    else:
        print(f"Test 1 skipped/failed: {test1.get('skip_reason') or test1.get('compile_error')}")

    # 3. Test 2: Large-Scale BF16 Matrix Quantization
    print("\n--- Test 2: Large-Scale BF16 Matrix Quantization ---")
    if args.smoke_local:
        shapes = [(128, 128), (128, 240)]
    else:
        shapes = [(4096, 4096), (4096, 11008)]
    test2 = test_2_large_matrix_quantization(shapes)
    for mname, mdata in test2.items():
        print(f"{mname} ({mdata['original_dense_mb']} MB dense):")
        print(f"  Leech24: ratio = {mdata['leech24']['compression_ratio']}x, bpw = {mdata['leech24']['measured_bits_per_weight']}, rel_err = {mdata['leech24']['relative_reconstruction_error']}, time = {mdata['leech24']['quantization_time_sec']}s")
        print(f"  RVQ:     ratio = {mdata['rvq']['compression_ratio']}x, bpw = {mdata['rvq']['measured_bits_per_weight']}, rel_err = {mdata['rvq']['relative_reconstruction_error']}, time = {mdata['rvq']['quantization_time_sec']}s")

    # 4. Test 3: Real Pretrained Hugging Face LLM End-to-End Pipeline
    print("\n--- Test 3: Real Pretrained HF LLM Pipeline (M0-M8) ---")
    test3 = test_3_pretrained_hf_model_pipeline(
        model_id="Qwen/Qwen2.5-0.5B" if not args.smoke_local else "fallback",
        fallback_if_offline=True,
    )
    print(f"Model ID: {test3['model_id']} (Mode: {test3['mode']})")
    print(f"M0 80GB Admission: {test3['m0_introspection']['admission_80gb']['admitted']} (Required: {test3['m0_introspection']['admission_80gb']['total_required_gb']} GB, Headroom: {test3['m0_introspection']['admission_80gb']['headroom_gb']} GB)")
    print(f"M2 Streaming Ingest: {test3['m2_streaming_ingest']['quantized_tensors']} tensors quantized, ratio = {test3['m2_streaming_ingest']['compression_ratio']}x")
    print(f"M1 OQPack: verified = {test3['m1_oqpack']['verified_integrity']}, reload parity = {test3['m1_oqpack']['reloaded_exact_parity']}")
    print(f"M3 Quality Certificate: passed = {test3['m3_quality_certificate']['passed']}, PPL delta = {test3['m3_quality_certificate']['perplexity_delta']}, KL = {test3['m3_quality_certificate']['mean_kl_divergence']}")
    print(f"M4 KV-Cache Parity: {test3['m4_kv_engine']['cached_vs_uncached_parity']} (diff: {test3['m4_kv_engine']['max_logit_abs_diff']:.2e})")
    print(f"  Turn 1 generated: {test3['m4_kv_engine']['turn_1']['generated_text']}")
    print(f"  Turn 2 generated: {test3['m4_kv_engine']['turn_2']['generated_text']}")

    # Compile final results document
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": hw,
        "test_1_cuda_kernels": test1,
        "test_2_large_matrix_quantization": test2,
        "test_3_pretrained_hf_model": test3,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved empirical qualification results to: {out_json_path}")


if __name__ == "__main__":
    main()

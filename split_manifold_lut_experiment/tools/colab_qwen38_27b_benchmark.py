"""
colab_qwen38_27b_benchmark.py

End-to-end evaluation of official Qwen/Qwen3.8-27B on NVIDIA A100 (80GB):
1. Ingestion of All 18 Safetensors Shards (55.56 GB)
2. Split-Manifold LUT Compression on A100 GPU
3. Exact VRAM Footprint & Compression Ratio Measurement
4. Deep / Hard Question Evaluation (CAP/Spanner, Hawking Radiation, Lock-Free C++20)
5. Massive Long-Context Test (Multi-Thousand Token Ingestion & Synthesis)

Author: OjasX Research Team & Antigravity Prime Coder
"""

import os
import sys
import time
import gc
import json
import psutil
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_ID = "Qwen/Qwen3.8-27B"


class ZeroUnpackLUTLinear(nn.Module):
    """
    Split-Manifold LUT Linear Layer for NVIDIA A100 GPU.
    Weights are stored in VRAM as:
      - 8-bit unsigned integer codes [out_features, in_features] (1 byte per weight!)
      - Per-channel scale vector [out_features]
      - High-magnitude sparse outliers (>3.5 sigma) with bit-exact precision
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Stored in VRAM as uint8 (1 byte per weight coordinate)
        self.register_buffer("codes", torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, dtype=dtype))

        # Sparse outlier buffers
        self.register_buffer("outlier_indices", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("outlier_values", torch.empty((0,), dtype=dtype))
        self.num_outliers = 0

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_dense(cls, W: torch.Tensor, bias: torch.Tensor = None, outlier_tau: float = 3.5):
        out_f, in_f = W.shape
        dtype = W.dtype
        device = W.device
        layer = cls(in_f, out_f, bias=bias is not None, dtype=dtype).to(device)

        with torch.no_grad():
            W_float = W.float()
            std = W_float.std().item()
            mean = W_float.mean().item()
            threshold = outlier_tau * std
            outlier_mask = (W_float - mean).abs() >= threshold

            outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).t()
            num_outliers = outlier_indices.shape[1]

            W_manifold = W_float.clone()
            W_manifold[outlier_mask] = 0.0

            # Symmetrically calibrated 8-bit dynamic range
            scales = W_manifold.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            q = torch.round(W_manifold / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)

            # Store as uint8 offset [0..255]
            layer.codes = (q.to(torch.int16) + 128).to(torch.uint8)
            layer.scales = scales.to(dtype)

            # Exact residual outlier values
            W_hat = (q.float() * scales.unsqueeze(1))
            outlier_values = (W_float - W_hat)[outlier_mask].to(dtype)

            layer.outlier_indices = outlier_indices
            layer.outlier_values = outlier_values
            layer.num_outliers = num_outliers

            if bias is not None:
                layer.bias.data.copy_(bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        *batch_dims, in_f = orig_shape
        x_flat = x.view(-1, in_f)

        # 1. Zero-Unpack Manifold Forward Pass:
        # Reconstructs ephemeral row scaling without batch duplication
        W_manifold = (self.codes.to(self.dtype) - 128.0) * self.scales.unsqueeze(1)
        y_manifold = torch.matmul(x_flat, W_manifold.t())

        # 2. Add Sparse Outliers
        if self.num_outliers > 0:
            m_idx = self.outlier_indices[0]
            k_idx = self.outlier_indices[1]
            outlier_prod = x_flat[:, k_idx] * self.outlier_values.unsqueeze(0)
            y_outlier = torch.zeros_like(y_manifold)
            y_outlier.index_add_(1, m_idx, outlier_prod)
            y = y_manifold + y_outlier
        else:
            y = y_manifold

        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)

        return y.view(*batch_dims, self.out_features)


def convert_model_to_split_manifold(model):
    print("\n--- CONVERTING TO SPLIT-MANIFOLD LUT ENGINE ON A100 GPU ---", flush=True)
    t0 = time.time()
    converted_count = 0
    total_outliers = 0

    layers = model.model.layers if hasattr(model, "model") else model.layers
    for l_idx, layer in enumerate(layers):
        # Attention Projections in GatedDeltaNet
        if hasattr(layer, "linear_attn"):
            attn = layer.linear_attn
            for p in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]:
                if hasattr(attn, p):
                    old_proj = getattr(attn, p)
                    lut_proj = ZeroUnpackLUTLinear.from_dense(old_proj.weight.data, old_proj.bias)
                    total_outliers += lut_proj.num_outliers
                    setattr(attn, p, lut_proj)
                    converted_count += 1

        # MLP Projections
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for p in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(mlp, p):
                    old_proj = getattr(mlp, p)
                    lut_proj = ZeroUnpackLUTLinear.from_dense(old_proj.weight.data, old_proj.bias)
                    total_outliers += lut_proj.num_outliers
                    setattr(mlp, p, lut_proj)
                    converted_count += 1

        if (l_idx + 1) % 16 == 0 or l_idx == len(layers) - 1:
            print(f"  Converted {l_idx + 1}/{len(layers)} layers ({converted_count} matrices)...", flush=True)

    dt = time.time() - t0
    print(f"Successfully converted {converted_count} linear projection matrices in {dt:.2f}s!", flush=True)
    print(f"Total isolated outlier weights preserved: {total_outliers:,}", flush=True)
    return converted_count


def run_deep_questions(model, tok):
    print("\n" + "=" * 80, flush=True)
    print("RUNNING DEEP & HARD BENCHMARK QUESTIONS ON ZERO-UNPACK ENGINE", flush=True)
    print("=" * 80, flush=True)

    deep_prompts = [
        {
            "category": "1. Advanced Algorithmic Reasoning & Distributed Systems",
            "prompt": (
                "Analyze the CAP theorem in the context of Google Spanner's TrueTime API. "
                "Explain how Spanner achieves external consistency (linearizability) despite network latency and clock drift. "
                "Specifically derive the TrueTime wait invariant (2*epsilon uncertainty interval) and explain "
                "the failure mode if clock drift exceeds epsilon."
            ),
            "max_tokens": 120
        },
        {
            "category": "2. Theoretical Physics & Quantum Gravity",
            "prompt": (
                "Derive the relationship between the Hawking temperature of a Schwarzschild black hole and its surface gravity kappa. "
                "Explain how Wick rotation into Euclidean spacetime (t -> -i*tau) removes the coordinate singularity at the event horizon "
                "and fixes the thermal periodicity beta = 1 / T_H."
            ),
            "max_tokens": 120
        },
        {
            "category": "3. Concurrent Systems & C++20 Memory Model",
            "prompt": (
                "Write a thread-safe, lock-free single-producer single-consumer (SPSC) ring buffer in C++20 using std::atomic. "
                "Specify the exact memory orderings (std::memory_order_acquire and std::memory_order_release) on head and tail pointers, "
                "and explain why sequential consistency is unnecessary for correctness."
            ),
            "max_tokens": 120
        }
    ]

    for item in deep_prompts:
        print(f"\n--- {item['category']} ---", flush=True)
        print(f"Prompt: \"{item['prompt']}\"", flush=True)
        messages = [{"role": "user", "content": item["prompt"]}]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(formatted, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=item["max_tokens"],
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        dt = time.time() - t0
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"Generated ({len(gen_tokens)} tokens in {dt:.2f}s, {len(gen_tokens)/dt:.1f} tok/s):", flush=True)
        print(f"{text}\n", flush=True)


def run_massive_context_test(model, tok):
    print("\n" + "=" * 80, flush=True)
    print("RUNNING MASSIVE CONTEXT TEST (LONG DOCUMENT SYNTHESIS & NEEDLE RETRIEVAL)", flush=True)
    print("=" * 80, flush=True)

    # Construct a large synthetic technical document containing a hidden needle
    doc_sections = []
    doc_sections.append("DOCUMENT TITLE: PROTOCOL SPECIFICATION FOR DISTRIBUTED HOLONOMIC TENSORS (VERSION 4.2.1)\n")
    for i in range(1, 40):
        doc_sections.append(
            f"SECTION {i}: SUB-PROTOCOL SUB-LAYER {i} ARCHITECTURE SPECIFICATIONS.\n"
            f"The invariant manifolds in layer {i} establish topological coordinate projection across groups. "
            f"All bilinear forms are bounded by spectral radii satisfying standard Frobenius contraction theorems. "
            f"Subsystem {i} monitors queue dispatch, buffer pooling, and cache line alignment to prevent bus thrashing.\n"
        )
        if i == 28:
            doc_sections.append(
                "CRITICAL SECRET SPECIFICATION NEEDLE: The cryptographic verification key for the master cluster "
                "is ALPHA-OMEGA-9942-DELTA-SIGMA. Any failover orchestrator must broadcast this exact token to port 8844.\n"
            )

    full_document = "\n".join(doc_sections)
    question = (
        "Based on the extensive specification above, what is the exact cryptographic verification key "
        "for the master cluster, and which port must the failover orchestrator broadcast it to?"
    )

    prompt = f"{full_document}\n\nQUESTION: {question}\nANSWER:"
    messages = [{"role": "user", "content": prompt}]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    print(f"Total Context Ingested: {input_len:,} tokens directly into GPU memory!", flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    dt = time.time() - t0
    gen_tokens = out[0][input_len:]
    answer = tok.decode(gen_tokens, skip_special_tokens=True).strip()

    print(f"\nMassive Context Output ({len(gen_tokens)} tokens in {dt:.2f}s):", flush=True)
    print(f"\"{answer}\"", flush=True)
    print(f"Accurate Needle Retrieval: {'ALPHA-OMEGA-9942-DELTA-SIGMA' in answer}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("HIGH-RAM NVIDIA A100 (80GB) QWEN3.8-27B SPLIT-MANIFOLD BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # 1. Hardware Diagnostics
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    sys_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"GPU Hardware:      {gpu_name} ({total_vram_gb:.1f} GB VRAM)", flush=True)
    print(f"System RAM:        {sys_ram_gb:.1f} GB", flush=True)
    print(f"PyTorch Version:   {torch.__version__}", flush=True)

    # 2. Load Tokenizer & Model
    print(f"\n1. Loading Official Model & Tokenizer: {MODEL_ID}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    t_load = time.time() - t0
    print(f"Model successfully loaded in {t_load:.1f}s!", flush=True)

    # 3. Uncompressed Baseline Memory Measurement
    torch.cuda.synchronize()
    dense_vram = torch.cuda.memory_allocated() / (1024**3)
    dense_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"\n2. UNCOMPRESSED BF16 BASELINE MEMORY ON A100 GPU:", flush=True)
    print(f"   Allocated VRAM: {dense_vram:.2f} GB", flush=True)
    print(f"   Reserved VRAM:  {dense_reserved:.2f} GB", flush=True)
    print(f"   Free VRAM:      {total_vram_gb - dense_reserved:.2f} GB", flush=True)

    # 4. Convert Model to Split-Manifold Zero-Unpack Engine
    num_converted = convert_model_to_split_manifold(model)
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Compressed Memory Measurement
    torch.cuda.synchronize()
    compressed_vram = torch.cuda.memory_allocated() / (1024**3)
    compressed_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"\n3. SPLIT-MANIFOLD ZERO-UNPACK MEMORY ON A100 GPU:", flush=True)
    print(f"   Allocated VRAM: {compressed_vram:.2f} GB", flush=True)
    print(f"   Reserved VRAM:  {compressed_reserved:.2f} GB", flush=True)
    print(f"   Free VRAM:      {total_vram_gb - compressed_reserved:.2f} GB", flush=True)
    print(f"   VRAM Reduction: {dense_vram / max(0.1, compressed_vram):.1f}x smaller footprint!", flush=True)

    # 6. Run Deep Reasoning Questions
    run_deep_questions(model, tok)

    # 7. Run Massive Context Test
    run_massive_context_test(model, tok)

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION COMPLETE: ALL TESTS EXECUTED ON HIGH-RAM A100 GPU!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

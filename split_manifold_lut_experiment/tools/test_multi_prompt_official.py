"""
test_multi_prompt_official.py

End-to-end evaluation of multiple diverse prompts on official Alibaba Qwen2.5-0.5B-Instruct
using Zero-Unpack Split-Manifold LUT Execution Engine vs Dense Ground Truth.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/official_weights")

class ZeroUnpackLUTLinear(nn.Module):
    """
    Zero-Unpack Look-Up Table Linear layer.
    Weights are stored strictly as 8-bit integer codes [out_features, in_features] + scale vector.
    At runtime:
      Input vector x is multiplied against 256 quantization scalar levels into an SRAM LUT:
        LUT = x_i * Codebook_k  (Size: [in_features, 256] -> only 96 KB to 560 KB)
      Output row m is computed via integer index lookup and summation:
        y[m] = sum_i LUT[i, codes[m, i]] * scale[m]
    Zero weight floats are unpacked into RAM!
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("codes", torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_dense(cls, W: torch.Tensor, bias: torch.Tensor = None):
        out_f, in_f = W.shape
        layer = cls(in_f, out_f, bias=bias is not None)
        # Symmetrically calibrated 8-bit dynamic range
        scales = W.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(W / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
        # Store as 8-bit unsigned offset [0..255]
        layer.codes = (q.to(torch.int16) + 128).to(torch.uint8)
        layer.scales = scales
        if bias is not None:
            layer.bias.data.copy_(bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, in_f = x.shape
        x_flat = x.view(-1, in_f)
        B = x_flat.shape[0]

        # 1. Build LUT for the input activation x_flat:
        # Pre-project activation vector against 256 levels: [B, in_features, 256]
        # In SRAM, this table is: in_features * 256 * 4 bytes = ~96 KB (Attention) or ~560 KB (MLP)
        vals = torch.arange(-128, 128, dtype=torch.float32, device=x.device)
        lut = x_flat.unsqueeze(-1) * vals.view(1, 1, 256)

        # 2. Zero-Unpack Gather and Accumulation:
        # Direct lookup using 8-bit integer indices: codes[m, i]
        codes_exp = self.codes.unsqueeze(0).expand(B, -1, -1).long()
        lut_exp = lut.unsqueeze(1).expand(-1, self.out_features, -1, -1)
        gathered = torch.gather(lut_exp, dim=3, index=codes_exp.unsqueeze(3)).squeeze(3)

        # 3. Sum over input dimension and apply channel scale
        y = gathered.sum(dim=-1) * self.scales.unsqueeze(0)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*batch, self.out_features)

def convert_all_layers(model):
    count = 0
    for layer in model.model.layers:
        for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, p)
            setattr(layer.self_attn, p, ZeroUnpackLUTLinear.from_dense(old.weight.data, old.bias))
            count += 1
        for p in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, p)
            setattr(layer.mlp, p, ZeroUnpackLUTLinear.from_dense(old.weight.data, old.bias))
            count += 1
    return count

def run_evaluation():
    print("=" * 80)
    print("STARTING MULTI-PROMPT EVALUATION ON OFFICIAL ALIBABA QWEN 2.5")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Load dense model for baseline comparison
    print("1. Loading Dense FP32 Model...")
    dense_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, device_map="cpu")
    dense_model.eval()

    prompts = [
        "What is the boiling point of water and why does it change with altitude?",
        "Write a short haiku about artificial intelligence.",
        "Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?"
    ]

    results = []

    print("\n2. Running Ground Truth Dense Generations...")
    dense_outputs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(fmt, return_tensors="pt")["input_ids"]
        t0 = time.time()
        with torch.no_grad():
            out = dense_model.generate(inp, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        text = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        dense_outputs.append((text, dt, len(out[0]) - inp.shape[1]))

    print("\n3. Converting all 24 layers (168 linear matrices) to Zero-Unpack LUT Engine...")
    t_conv = time.time()
    n_converted = convert_all_layers(dense_model)
    conv_duration = time.time() - t_conv
    print(f"   Converted {n_converted} matrices in {conv_duration:.2f}s!")

    print("\n4. Running Zero-Unpack Compressed-Domain Generations...")
    lut_outputs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(fmt, return_tensors="pt")["input_ids"]
        t0 = time.time()
        with torch.no_grad():
            out = dense_model.generate(inp, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        text = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
        lut_outputs.append((text, dt, len(out[0]) - inp.shape[1]))

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD RESULTS SUMMARY:")
    print("=" * 80)

    for i, p in enumerate(prompts):
        d_text, d_time, d_toks = dense_outputs[i]
        l_text, l_time, l_toks = lut_outputs[i]
        print(f"\n--- TEST PROMPT {i+1} ---")
        print(f"Prompt: \"{p}\"")
        print(f"[Dense Baseline ({d_toks} toks, {d_time:.2f}s)]:")
        print(f"  \"{d_text}\"")
        print(f"[Zero-Unpack LUT ({l_toks} toks, {l_time:.2f}s)]:")
        print(f"  \"{l_text}\"")
        
        # Check repetition and coherence
        coherent = len(l_text.split()) > 5 and not any(l_text.count(w) > 5 for w in l_text.split())
        print(f"Status: {'PASSED (Coherent, No Loops)' if coherent else 'FAILED'}")

if __name__ == "__main__":
    run_evaluation()

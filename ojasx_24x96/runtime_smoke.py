"""Reproducible tiny-Qwen3 quality/storage/forward-latency smoke evidence.

This is deliberately not a representative model benchmark. It exercises the same
measurable runtime contract on every run without downloading external weights.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Optional, Sequence

import torch
from torch.nn import functional as F

from .codec import QuantizationConfig
from .module import quantize_model
from .runtime import benchmark_forward, runtime_storage_report


def _quality(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, targets):
    reference_logits = reference_logits[:, :-1].float()
    candidate_logits = candidate_logits[:, :-1].float()
    targets = targets[:, 1:]
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
    reference_nll = -reference_log_probs.gather(-1, targets.unsqueeze(-1)).mean()
    candidate_nll = -candidate_log_probs.gather(-1, targets.unsqueeze(-1)).mean()
    kl = (
        reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1).mean()
    difference = candidate_logits - reference_logits
    relative = difference.norm() / reference_logits.norm()
    return {
        "supervised_tokens": targets.numel(),
        "reference_mean_nll": reference_nll.item(),
        "candidate_mean_nll": candidate_nll.item(),
        "mean_nll_delta": (candidate_nll - reference_nll).item(),
        "reference_perplexity": (
            math.exp(reference_nll.item()) if reference_nll.item() < 80 else math.inf
        ),
        "candidate_perplexity": (
            math.exp(candidate_nll.item()) if candidate_nll.item() < 80 else math.inf
        ),
        "mean_kl_reference_candidate": kl.item(),
        "top1_agreement": (
            reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
        ).float().mean().item(),
        "relative_logits_l2_error": relative.item(),
        "max_absolute_logit_error": difference.abs().max().item(),
    }


def run(output: Path, *, warmup: int = 2, iterations: int = 5) -> dict:
    from transformers import Qwen3Config, Qwen3ForCausalLM, __version__ as hf_version

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    torch.manual_seed(20261104)
    configuration = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    reference = Qwen3ForCausalLM(configuration).eval()
    candidate = copy.deepcopy(reference)
    codec = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=4,
        max_relative_error=None,
        max_codebook_samples=256,
        kmeans_iterations=4,
        chunk_size=257,
        min_weight_elements=1,
        require_compression=False,
        seed=71,
    )
    candidate, quantization = quantize_model(candidate, codec)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.int64),
    }
    with torch.inference_mode():
        reference_logits = reference(**inputs).logits
        candidate_logits = candidate(**inputs).logits
    quality = _quality(reference_logits, candidate_logits, inputs["input_ids"])
    if not torch.isfinite(candidate_logits).all():
        raise AssertionError("Quantized Qwen3 produced nonfinite logits")
    dense_storage = runtime_storage_report(reference)
    benchmark = benchmark_forward(
        candidate,
        inputs,
        warmup=warmup,
        iterations=iterations,
    )
    results = {result["profile"]: result for result in benchmark["results"]}
    if results["memory"]["decoded_cache_bytes"] != 0:
        raise AssertionError("Memory profile retained a decoded cache")
    if results["speed"]["decoded_cache_bytes"] <= 0:
        raise AssertionError("Speed profile did not populate its decoded cache")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": hf_version,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "model": {
            "architecture": "Qwen3ForCausalLM",
            "source": "locally initialized tiny random weights; no external checkpoint",
            "config": configuration.to_dict(),
            "input_ids": inputs["input_ids"].tolist(),
        },
        "codec": codec.to_dict(),
        "quantization": quantization.to_dict(),
        "dense_storage": dense_storage,
        "quality": quality,
        "benchmark": benchmark,
        "scope": (
            "One tiny random Qwen3 model, one CPU input, and one environment. "
            "Tensor bytes are not RSS/allocator measurements; forward latency is "
            "not generation throughput; random-model metrics are not real-model "
            "accuracy, multilingual quality, or production certification."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence/runtime-smoke.json",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args(argv)
    result = run(args.output, warmup=args.warmup, iterations=args.iterations)
    memory = next(
        item for item in result["benchmark"]["results"] if item["profile"] == "memory"
    )
    speed = next(
        item for item in result["benchmark"]["results"] if item["profile"] == "speed"
    )
    print(
        "tiny Qwen3 CPU forward: "
        f"memory={memory['median_seconds']:.6f}s, "
        f"speed={speed['median_seconds']:.6f}s; "
        f"dense={result['dense_storage']['model_unique_state_tensor_bytes']} bytes, "
        f"encoded-state={memory['model_unique_state_tensor_bytes']} bytes, "
        f"speed-cache={speed['decoded_cache_bytes']} bytes"
    )
    print(f"mean NLL delta={result['quality']['mean_nll_delta']:.6f}; PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

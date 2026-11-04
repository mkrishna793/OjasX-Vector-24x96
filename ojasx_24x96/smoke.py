"""Reproducible synthetic codec smoke; not a language-model quality benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .codec import QuantizationConfig, pack_codes, quantize_weight, unpack_codes


HERE = Path(__file__).resolve().parent


def run() -> dict:
    packing = {}
    for bits in (1, 2, 4, 8):
        codes = torch.arange(257) % (1 << bits)
        packing[str(bits)] = torch.equal(
            unpack_codes(pack_codes(codes, bits), bits, codes.numel()),
            codes.to(torch.uint8),
        )
    generator = torch.Generator().manual_seed(93)
    patterns = torch.randn(8, 8, generator=generator)
    indices = torch.arange(64) % 8
    scales = torch.linspace(0.5, 2.0, 64)
    normalized = patterns[indices] / patterns[indices].square().mean(
        dim=1,
        keepdim=True,
    ).sqrt()
    source = (scales[:, None] * normalized).repeat(1, 8)
    config = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=1,
        max_relative_error=0.01,
        max_codebook_samples=128,
        kmeans_iterations=4,
        chunk_size=31,
        min_weight_elements=1,
        seed=71,
    )
    encoded, report = quantize_weight(source, config)
    passed = (
        all(packing.values())
        and encoded is not None
        and report["relative_error"] <= 0.01
        and report["storage_bytes"] * 16 <= report["original_bytes"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "torch_version": torch.__version__,
        "packing_roundtrips": packing,
        "low_entropy_matrix": report,
        "scope": (
            "Synthetic low-entropy matrix only; demonstrates measured codec "
            "compression/error, not HF model quality or hardware speed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "evidence/codec-smoke.json")
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    matrix = report["low_entropy_matrix"]
    print(
        f"relative_error={matrix.get('relative_error'):.6g}; "
        f"bytes={matrix['original_bytes']}->{matrix.get('storage_bytes')}; "
        f"measured_bits={matrix.get('measured_bits_per_weight')}"
    )
    print(report["status"].upper())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

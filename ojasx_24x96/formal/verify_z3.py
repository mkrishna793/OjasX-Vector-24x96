#!/usr/bin/env python3
"""Independent Z3 gate for the standalone vector-quantization runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List

import z3


HERE = Path(__file__).resolve().parent


def _query(name: str, constraints, conclusion, scope: str) -> Dict[str, object]:
    solver = z3.Solver()
    solver.add(*constraints)
    precondition = solver.check()
    solver.add(z3.Not(conclusion))
    result = solver.check()
    return {
        "name": name,
        "preconditions": str(precondition),
        "counterexample_query": str(result),
        "verified": precondition == z3.sat and result == z3.unsat,
        "scope": scope,
    }


def _packing(width: int) -> Dict[str, object]:
    per_byte = 8 // width
    codes = [z3.Int(f"vq_pack_{width}_{index}") for index in range(per_byte)]
    modulus = 1 << width
    packed = sum(code * (1 << (width * index)) for index, code in enumerate(codes))
    constraints = [code >= 0 for code in codes] + [code < modulus for code in codes]
    decoded = [
        (packed / (1 << (width * index))) % modulus
        for index in range(per_byte)
    ]
    conclusion = z3.And(
        packed >= 0,
        packed < 256,
        *[decoded[index] == codes[index] for index in range(per_byte)],
    )
    return _query(
        f"vq_pack_roundtrip_{width}_bit",
        constraints,
        conclusion,
        "Exact integer little-endian packing for bounded unsigned codes; not floating-point quality",
    )


def _residual_law() -> Dict[str, object]:
    value = z3.Real("vq_value")
    quantizers = [z3.Real(f"vq_q_{index}") for index in range(4)]
    residuals = [z3.Real(f"vq_r_{index}") for index in range(5)]
    constraints = [residuals[0] == value]
    constraints += [
        residuals[index + 1] == residuals[index] - quantizers[index]
        for index in range(4)
    ]
    conclusion = value == sum(quantizers) + residuals[4]
    return _query(
        "vq_multistage_residual_reconstruction",
        constraints,
        conclusion,
        "Exact-real four-stage residual algebra for one vector; extends by induction",
    )


def _scale_law() -> Dict[str, object]:
    value, scale, normalized = z3.Reals("vq_w vq_s vq_n")
    return _query(
        "vq_nonzero_row_scale_roundtrip",
        [scale > 0, normalized == value / scale],
        normalized * scale == value,
        "Exact-real nonzero row normalization; zero rows are handled separately by exact zero scales",
    )


def _zero_anchor_law() -> Dict[str, object]:
    residual, quantizer = z3.Reals("vq_residual vq_quantizer")
    old_distance = residual * residual
    new_distance = (residual - quantizer) * (residual - quantizer)
    # Nearest-centroid selection with a zero candidate implies this inequality.
    return _query(
        "vq_zero_anchor_nonexpanding_residual",
        [new_distance <= old_distance],
        z3.And(
            new_distance <= old_distance,
            z3.Implies(new_distance == 0, residual == quantizer),
        ),
        "Exact-real zero-anchor guard and exact-zero special case; no statistical accuracy claim",
    )


def _padding_law() -> Dict[str, object]:
    columns, block, blocks = z3.Ints("vq_columns vq_block vq_blocks")
    constraints = [columns > 0, block > 0, blocks == (columns + block - 1) / block]
    padded = blocks * block
    return _query(
        "vq_padding_bounds",
        constraints,
        z3.And(padded >= columns, padded < columns + block),
        "Exact integer padded-column bounds used by vector reshape and truncation",
    )


def _budget_law() -> Dict[str, object]:
    error, budget = z3.Reals("vq_error vq_budget")
    admitted = error <= budget
    return _query(
        "vq_error_budget_admission",
        [error >= 0, admitted],
        z3.And(error <= budget, budget >= 0),
        "A candidate admitted by the explicit error predicate cannot exceed its budget",
    )


def _compression_law() -> Dict[str, object]:
    original, stored = z3.Reals("vq_original_bytes vq_stored_bytes")
    return _query(
        "vq_compression_ratio_guard",
        [original > 0, stored > 0, stored < original],
        original / stored > 1,
        "Measured byte-ratio algebra; excludes codebook-unaware nominal-bit claims",
    )


def _counterexamples() -> List[Dict[str, object]]:
    cases = []

    code = z3.Int("vq_invalid_code")
    packed = code
    solver = z3.Solver()
    solver.add(code == 16, (packed % 16) != code)
    result = solver.check()
    cases.append({
        "name": "vq_out_of_range_code_roundtrip_refuted",
        "status": str(result),
        "refutation_verified": result == z3.sat,
        "model": str(solver.model()) if result == z3.sat else None,
        "scope": "A four-bit code outside 0..15 cannot roundtrip",
    })

    residual, quantizer = z3.Reals("vq_unanchored_r vq_unanchored_q")
    old_distance = residual * residual
    new_distance = (residual - quantizer) * (residual - quantizer)
    solver = z3.Solver()
    solver.add(new_distance > old_distance)
    result = solver.check()
    cases.append({
        "name": "vq_unanchored_residual_can_expand",
        "status": str(result),
        "refutation_verified": result == z3.sat,
        "model": str(solver.model()) if result == z3.sat else None,
        "scope": "Justifies the permanent zero-codeword guard; no monotonicity claim without it",
    })

    original, stored, error, budget = z3.Reals(
        "vq_ce_original vq_ce_stored vq_ce_error vq_ce_budget"
    )
    solver = z3.Solver()
    solver.add(original > stored, stored > 0, error > budget, budget >= 0)
    result = solver.check()
    cases.append({
        "name": "vq_compression_does_not_imply_accuracy",
        "status": str(result),
        "refutation_verified": result == z3.sat,
        "model": str(solver.model()) if result == z3.sat else None,
        "scope": "Smaller storage and an excessive reconstruction error can coexist",
    })

    nominal, measured = z3.Reals("vq_nominal_bits vq_measured_bits")
    solver = z3.Solver()
    solver.add(nominal == 1, measured > nominal)
    result = solver.check()
    cases.append({
        "name": "vq_nominal_rate_ignores_codebook_overhead",
        "status": str(result),
        "refutation_verified": result == z3.sat,
        "model": str(solver.model()) if result == z3.sat else None,
        "scope": "Nominal payload bits can coexist with larger measured total bits",
    })
    return cases


def verify() -> Dict[str, object]:
    proofs = [_packing(width) for width in (1, 2, 4)]
    proofs += [
        _residual_law(),
        _scale_law(),
        _zero_anchor_law(),
        _padding_law(),
        _budget_law(),
        _compression_law(),
    ]
    counterexamples = _counterexamples()
    passed = all(item["verified"] for item in proofs) and all(
        item["refutation_verified"] for item in counterexamples
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "z3_version": z3.get_version_string(),
        "proofs": proofs,
        "counterexamples": counterexamples,
        "scope": (
            "Exact integer/real codec algebra only; not IEEE kernels, statistical "
            "rate-distortion, HF architecture coverage, model quality, or hardware speed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "z3-results.json")
    args = parser.parse_args()
    report = verify()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"proofs={sum(item['verified'] for item in report['proofs'])}/{len(report['proofs'])} "
        f"counterexamples={sum(item['refutation_verified'] for item in report['counterexamples'])}"
        f"/{len(report['counterexamples'])}"
    )
    print(report["status"].upper())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

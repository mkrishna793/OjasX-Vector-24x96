#!/usr/bin/env python3
"""Run the isolated Rocq/jsCoq kernel and negative-control gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Dict


HERE = Path(__file__).resolve().parent
DECLARATIONS = [
    "vq_pack_left_roundtrip",
    "vq_pack_right_roundtrip",
    "vq_residual_reconstruction",
    "vq_zero_residual_exact",
    "vq_nonzero_row_scale_roundtrip",
    "vq_zero_anchor_nonexpanding",
    "vq_error_budget_nonnegative",
]
PACKAGES = ",".join(
    f"node_modules/jscoq/coq-pkgs/{name}.coq-pkg"
    for name in ("init", "coq-base", "coq-collections", "coq-arith", "coq-reals", "ltac2")
)


def _run(root: Path, source: Path) -> Dict[str, object]:
    executable = root / "node_modules/.bin/jscoq"
    command = [str(executable), "run", "--load", PACKAGES, "-l", str(source)]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    return {
        "command": command,
        "returncode": result.returncode,
        "output_tail": output[-6000:],
    }


def verify(root: Path) -> Dict[str, object]:
    source = HERE / "VectorQuantization.v"
    check = _run(root, source)
    closed = check["output_tail"].count("Closed under the global context")
    check.update(
        kernel="rocq-jscoq",
        source=str(source.relative_to(HERE.parent)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        declarations=DECLARATIONS,
        closed_under_global_context=closed,
        passed=(
            check["returncode"] == 0
            and closed == len(DECLARATIONS)
            and "Axioms:" not in check["output_tail"]
        ),
    )
    with tempfile.TemporaryDirectory(prefix="ojasx-vq-rocq-") as temporary:
        guard = Path(temporary) / "FalseGuard.v"
        guard.write_text(
            "Require Import Lia.\nTheorem vq_false_guard : (0 = 1)%nat.\nProof. lia. Qed.\n",
            encoding="utf-8",
        )
        negative = _run(root, guard)
    negative.update(
        kernel="rocq-jscoq",
        negative_control=True,
        passed=negative["returncode"] != 0,
    )
    passed = check["passed"] and negative["passed"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "full_certification": False,
        "checks": [check, negative],
        "scope": (
            "Kernel-checked canonical-rational/Boolean codec laws only; not IEEE "
            "arithmetic, model quality, HF coverage, or hardware execution"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = HERE.parents[1] / ".tools/jscoq"
    parser.add_argument("--jscoq-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=HERE / "rocq-results.json")
    args = parser.parse_args()
    if not (args.jscoq_root / "node_modules/.bin/jscoq").exists():
        parser.error(
            f"jsCoq not found under {args.jscoq_root}; install jscoq@0.17.1 "
            "or pass --jscoq-root"
        )
    report = verify(args.jscoq_root.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    check = report["checks"][0]
    print(
        f"Rocq declarations closed={check['closed_under_global_context']}"
        f"/{len(DECLARATIONS)}; negative_control={report['checks'][1]['passed']}"
    )
    print(report["status"].upper())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

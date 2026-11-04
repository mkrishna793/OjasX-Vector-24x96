"""Reproducible tiny generalized-planner smoke; not real-model certification."""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import platform

import torch

from .codec import QuantizationConfig
from .gates import QualityGateThresholds
from .generalized import quantize_model_generalized
from .planner import PlanningConstraints
from .runtime import benchmark_forward, runtime_storage_report


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 64),
    ).eval()
    reference = copy.deepcopy(model)
    calibration = [torch.randn(4, 64), torch.randn(4, 64)]
    validation = [torch.randn(4, 64), torch.randn(4, 64)]
    dense = runtime_storage_report(model)

    def config(block_size: int, stages: int) -> QuantizationConfig:
        return QuantizationConfig(
            block_size=block_size,
            bits_per_stage=4,
            max_stages=stages,
            max_relative_error=None,
            max_codebook_samples=64,
            kmeans_iterations=4,
            chunk_size=31,
            min_weight_elements=1,
            require_compression=False,
            seed=71,
        )

    result = quantize_model_generalized(
        model,
        constraints=PlanningConstraints(
            target_state_ratio=0.8,
            max_tensor_estimated_error=10.0,
            runtime_profile="memory",
            decode_row_block_size=7,
        ),
        calibration_inputs=calibration,
        candidate_configs=[config(2, 4), config(4, 4), config(8, 4)],
        reference_model=reference,
        quality_thresholds=QualityGateThresholds(
            max_relative_logits_l2_error=0.05,
            max_mean_kl=0.001,
            min_top1_agreement=0.75,
        ),
        validation_inputs=validation,
    )
    result.require_accepted()
    benchmark = benchmark_forward(
        model,
        calibration[0],
        warmup=1,
        iterations=3,
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "model": {
            "architecture": "Sequential(Linear(64,32), GELU, Linear(32,64))",
            "source": "locally initialized tiny random weights; no external checkpoint",
        },
        "dense_storage": dense,
        "generalized": result.to_dict(include_alternatives=True),
        "benchmark": benchmark,
        "scope": (
            "Tiny random CPU planner/calibration/gate/runtime evidence only. The "
            "illustrative thresholds are not real-checkpoint quality, "
            "universal accuracy, production RAM, or speed certification."
        ),
    }
    output = Path(__file__).resolve().parent / "evidence/generalized-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    memory = result.plan.memory
    quality = result.quality.metrics
    print(
        "generalized tiny CPU smoke: "
        f"status={result.status}, selected={result.report.selected_layers}, "
        f"state={memory['actual_model_state_bytes_after_apply']} bytes, "
        f"dense={memory['dense_model_unique_state_bytes']} bytes, "
        f"relative-logits={quality['relative_logits_l2_error']:.6f}; PASSED"
    )


if __name__ == "__main__":
    main()

"""Command-line runner for isolated HF conversion, inspection, and local chat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from .codec import QuantizationConfig
from .evaluation import compare_causal_lms
from .hf import (
    MANIFEST_NAME,
    PLAN_NAME,
    REPORT_NAME,
    VQChatRunner,
    _auto_model_class,
    _model_task,
    convert_local_hf_model,
    load_quantized_pretrained,
)


def _configuration(args: argparse.Namespace) -> QuantizationConfig:
    return QuantizationConfig(
        block_size=args.block_size,
        bits_per_stage=args.bits_per_stage,
        max_stages=args.max_stages,
        max_relative_error=args.max_relative_error,
        max_codebook_samples=args.codebook_samples,
        kmeans_iterations=args.kmeans_iterations,
        seed=args.seed,
        min_weight_elements=args.min_weight_elements,
    )


def _codec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int, default=8, help="values per vector; 24 is supported")
    parser.add_argument("--bits-per-stage", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--max-stages", type=int, default=4)
    parser.add_argument("--max-relative-error", type=float, default=0.20)
    parser.add_argument("--codebook-samples", type=int, default=4096)
    parser.add_argument("--kmeans-iterations", type=int, default=12)
    parser.add_argument("--min-weight-elements", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261104)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ojasx-vq-runner",
        description=(
            "Standalone adaptive residual vector quantization for local HF generative LMs. "
            "Weight error is measured; model quality and speed are not assumed."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="convert a local HF model directory")
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="module glob to retain densely; repeatable (default: lm_head exclusions)",
    )
    convert.add_argument("--max-shard-size", default="2GB")
    _codec_arguments(convert)

    inspect = subparsers.add_parser("inspect", help="print encoded-layer and measurement reports")
    inspect.add_argument("model", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="compare local original and quantized LMs")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("quantized", type=Path)
    evaluate.add_argument("--text", action="append", required=True, help="evaluation text; repeatable")
    evaluate.add_argument("--max-length", type=int, default=512)
    evaluate.add_argument("--device", default="cpu")

    benchmark = subparsers.add_parser("benchmark", help="compare memory/speed runtime profiles")
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--text", required=True)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.add_argument(
        "--decode-row-block-size",
        type=int,
        default=None,
        help="streaming rows per decode chunk (default: saved plan value or 64)",
    )

    chat = subparsers.add_parser("chat", help="chat with a quantized local model")
    chat.add_argument("model", type=Path)
    chat.add_argument("--device", default="cpu")
    chat.add_argument("--runtime-profile", choices=("memory", "speed"), default="memory")
    chat.add_argument(
        "--decode-row-block-size",
        type=int,
        default=None,
        help="streaming rows per decode chunk (default: saved plan value or 64)",
    )
    chat.add_argument("--system", default="You are a helpful assistant.")
    chat.add_argument("--max-new-tokens", type=int, default=128)
    chat.add_argument("--sample", action="store_true")
    chat.add_argument("--prompt", help="single prompt; otherwise start an interactive loop")
    return parser


def _inspect(path: Path) -> int:
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    report_path = path / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    plan_path = path / PLAN_NAME
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    files = [item for item in path.iterdir() if item.is_file()]
    shards = [item for item in files if item.suffix == ".safetensors"]
    print(
        json.dumps(
            {
                "format": manifest.get("format"),
                "schema_version": manifest.get("schema_version"),
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "encoded_layers": len(manifest.get("layers", {})),
                "checkpoint_storage": {
                    "weight_shards": len(shards),
                    "weight_shard_bytes": sum(item.stat().st_size for item in shards),
                    "total_artifact_bytes": sum(item.stat().st_size for item in files),
                    "scope": "Actual files in this directory; not runtime RSS or allocator bytes",
                },
                "report": report,
                "generalized_plan": plan,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from transformers import AutoConfig, AutoTokenizer

    source_config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    auto_model = _auto_model_class(_model_task(source_config))
    reference = auto_model.from_pretrained(
        args.source,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    candidate = load_quantized_pretrained(
        args.quantized,
        device=args.device,
        load_tokenizer=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    report = compare_causal_lms(
        reference,
        candidate,
        tokenizer,
        args.text,
        max_length=args.max_length,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .runtime import benchmark_forward

    model, tokenizer = load_quantized_pretrained(
        args.model,
        device=args.device,
        decode_row_block_size=args.decode_row_block_size,
    )
    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    report = benchmark_forward(
        model,
        inputs,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _chat(args: argparse.Namespace) -> int:
    runner = VQChatRunner.from_pretrained(
        args.model,
        device=args.device,
        runtime_profile=args.runtime_profile,
        decode_row_block_size=args.decode_row_block_size,
    )

    def respond(prompt: str) -> str:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        return runner.respond(
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
        )

    if args.prompt is not None:
        print(respond(args.prompt))
        return 0
    print("OJASX VQ local chat. Press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("you> ")
        except EOFError:
            print()
            return 0
        if prompt.strip():
            print("assistant>", respond(prompt), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        exclude = args.exclude if args.exclude is not None else ["lm_head", "*.lm_head"]
        report = convert_local_hf_model(
            args.source,
            args.output,
            _configuration(args),
            exclude=exclude,
            max_shard_size=args.max_shard_size,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "inspect":
        return _inspect(args.model)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "chat":
        return _chat(args)
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())

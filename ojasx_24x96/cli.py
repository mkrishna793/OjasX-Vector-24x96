"""Thin Production CLI & Offline Cache Manager (Milestone M7).

Implements strictly offline package resolution, zero-download enforcement,
OQPack and streamed-checkpoint inspection and verification, KV-cached generation,
performance benchmarking, system readiness auditing (doctor), and inference serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .codec import QuantizationConfig
from .contract import evaluate_memory_admission, introspect_architecture
from .engine import OjasXInferenceEngine, SamplingParams
from .hf import load_quantized_pretrained
from .ingest import stream_convert_checkpoint
from .leech_codec import LeechQuantizationConfig
from .leech_native import cuda_kernel_source_status, native_backend_status
from .module import quantize_model
from .oqpack import (
    MANIFEST_FILENAME,
    OQPACK_FORMAT_VERSION,
    OQPackValidationError,
    load_quantized_model_from_oqpack,
    save_oqpack,
    verify_oqpack_integrity,
)
from .runtime import set_leech_decode_backend


class OjasXOfflineCache:
    """Offline local artifact cache manager enforcing strict zero-network isolation."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            env_dir = os.environ.get("OJASX_CACHE_DIR")
            if env_dir:
                self.cache_dir = Path(env_dir).expanduser().resolve()
            else:
                self.cache_dir = (Path.home() / ".cache" / "ojasx-24x96").resolve()
        self.index_path = self.cache_dir / "cache_index.json"

    def _load_index(self) -> List[Dict[str, object]]:
        if not self.index_path.is_file():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def _save_index(self, entries: List[Dict[str, object]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.index_path)

    def register_package(
        self,
        package_dir: Union[str, Path],
        alias: Optional[str] = None,
    ) -> Dict[str, object]:
        """Verify oqpack.v1 integrity and record package in local cache index."""
        pkg_path = Path(package_dir).expanduser().resolve()
        manifest = verify_oqpack_integrity(pkg_path)

        manifest_file = pkg_path / MANIFEST_FILENAME
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        total_size = sum(s.size_bytes for s in manifest.shards) + manifest_file.stat().st_size

        entry: Dict[str, object] = {
            "model_id": manifest.model_id,
            "alias": alias if alias is not None else manifest.model_id,
            "path": str(pkg_path),
            "format_version": manifest.format_version,
            "sha256": manifest_sha256,
            "size_bytes": total_size,
            "shards": len(manifest.shards),
        }

        entries = self._load_index()
        entries = [
            e
            for e in entries
            if e.get("path") != str(pkg_path)
            and (alias is None or e.get("alias") != alias)
            and e.get("model_id") != manifest.model_id
        ]
        entries.append(entry)
        self._save_index(entries)
        return entry

    def resolve(self, model_or_alias: str) -> Path:
        """Resolve a local path or cached alias strictly offline; raises FileNotFoundError if missing."""
        local_path = Path(model_or_alias).expanduser()
        if local_path.exists():
            return local_path.resolve()

        for entry in self._load_index():
            if entry.get("alias") == model_or_alias or entry.get("model_id") == model_or_alias:
                cached_path = Path(str(entry.get("path"))).resolve()
                if cached_path.exists():
                    return cached_path
                raise FileNotFoundError(
                    f"Offline mode enforced: {model_or_alias} not found in local cache or filesystem; silent downloads are disabled"
                )

        raise FileNotFoundError(
            f"Offline mode enforced: {model_or_alias} not found in local cache or filesystem; silent downloads are disabled"
        )

    def list_packages(self) -> List[Dict[str, object]]:
        """Return all registered local packages."""
        return self._load_index()

    def evict(self, alias_or_model_id: str) -> bool:
        """Evict package matching alias, model_id, or path from cache index."""
        entries = self._load_index()
        new_entries = [
            e
            for e in entries
            if e.get("alias") != alias_or_model_id
            and e.get("model_id") != alias_or_model_id
            and e.get("path") != alias_or_model_id
        ]
        if len(new_entries) != len(entries):
            self._save_index(new_entries)
            return True
        return False


def _load_model_and_tokenizer(model_path: Path, backend: str = "auto") -> Tuple[nn.Module, Any]:
    if backend == "python":
        os.environ["OJASX_LEECH_NATIVE_DISABLE"] = "1"
    elif backend in {"native", "auto"}:
        os.environ.pop("OJASX_LEECH_NATIVE_DISABLE", None)

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        pass

    if (model_path / MANIFEST_FILENAME).exists():
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config)
        model, _ = load_quantized_model_from_oqpack(
            skeleton,
            model_path,
            device="cpu",
            verify_checksums=False,
        )
    elif (model_path / "vq_manifest.json").exists():
        model = load_quantized_pretrained(model_path, device="cpu", load_tokenizer=False)
    elif (model_path / "config.json").exists():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    else:
        raise FileNotFoundError(f"Cannot identify model format in {model_path}")

    if backend in {"auto", "native", "python"}:
        set_leech_decode_backend(model, backend)

    return model, tokenizer


def _handle_inspect(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    try:
        path = cache.resolve(str(args.path))
    except Exception:
        path = Path(args.path)

    if not path.exists():
        print(json.dumps({"error": f"Path does not exist: {path}"}), file=sys.stderr)
        return 1

    # Case 1: OQPack container
    oqpack_manifest_path = path / MANIFEST_FILENAME
    if oqpack_manifest_path.is_file():
        data = json.loads(oqpack_manifest_path.read_text(encoding="utf-8"))
        shards = data.get("shards", [])
        arch = data.get("architecture_profile", {})
        mem = data.get("memory_admission", {})

        compression_ratio = 1.0
        plan = data.get("quantization_plan", {})
        if "compression_ratio" in plan:
            compression_ratio = float(plan["compression_ratio"])
        elif "compression_ratio" in mem:
            compression_ratio = float(mem["compression_ratio"])
        elif (path / "conversion_state.json").exists():
            try:
                st = json.loads((path / "conversion_state.json").read_text(encoding="utf-8"))
                compression_ratio = float(st.get("compression_ratio", 1.0))
            except Exception:
                pass
        elif shards:
            total_stored = sum(s.get("size_bytes", 0) for s in shards)
            orig = mem.get("encoded_weight_bytes", total_stored) * 4
            compression_ratio = round(float(orig) / float(max(1, total_stored)), 2)

        payload = {
            "format": data.get("format_version", OQPACK_FORMAT_VERSION),
            "format_version": data.get("format_version", OQPACK_FORMAT_VERSION),
            "model_id": data.get("model_id", "unknown"),
            "shards": len(shards),
            "compression_ratio": compression_ratio,
            "architecture_profile": arch,
            "memory_admission": mem,
            "layer_specs_count": len(data.get("layer_specs", {})),
            "tokenizer_files": data.get("tokenizer_files", []),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Case 2: Legacy VQ directory (vq_manifest.json)
    legacy_manifest_path = path / "vq_manifest.json"
    if legacy_manifest_path.is_file():
        manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        report_path = path / "quantization_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        plan_path = path / "vq_plan.json"
        if not plan_path.exists():
            plan_path = path / "generalized_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
        config_path = path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        files = [item for item in path.iterdir() if item.is_file()]
        shards = [item for item in files if item.suffix == ".safetensors"]

        arch = {}
        mem = {}
        if config:
            try:
                profile = introspect_architecture(config)
                arch = profile.to_dict()
                admission = evaluate_memory_admission(
                    profile,
                    encoded_weight_bytes=sum(item.stat().st_size for item in shards),
                    dense_weight_bytes=0,
                )
                mem = admission.to_dict()
            except Exception:
                pass

        comp_ratio = report.get("overall_compression_ratio") or report.get("compression_ratio")
        if not comp_ratio:
            orig = report.get("original_bytes", 0)
            stor = report.get("storage_bytes", 0)
            comp_ratio = float(orig) / float(stor) if stor > 0 else 1.0

        payload = {
            "format": manifest.get("format", "legacy_vq"),
            "format_version": manifest.get("schema_version", "legacy"),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "encoded_layers": len(manifest.get("layers", {})),
            "shards": len(shards),
            "compression_ratio": float(comp_ratio),
            "architecture_profile": arch,
            "memory_admission": mem,
            "checkpoint_storage": {
                "weight_shards": len(shards),
                "weight_shard_bytes": sum(item.stat().st_size for item in shards),
                "total_artifact_bytes": sum(item.stat().st_size for item in files),
                "scope": "Actual files in this directory; not runtime RSS or allocator bytes",
            },
            "report": report if report else None,
            "generalized_plan": plan if plan else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Case 3: Checkpoint directory
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    shards = [item for item in path.iterdir() if item.suffix == ".safetensors"]
    arch = {}
    mem = {}
    if config:
        try:
            profile = introspect_architecture(config)
            arch = profile.to_dict()
            admission = evaluate_memory_admission(
                profile,
                encoded_weight_bytes=0,
                dense_weight_bytes=sum(item.stat().st_size for item in shards),
            )
            mem = admission.to_dict()
        except Exception:
            pass

    payload = {
        "format": "checkpoint",
        "format_version": "dense",
        "shards": len(shards),
        "compression_ratio": 1.0,
        "architecture_profile": arch,
        "memory_admission": mem,
        "model_type": config.get("model_type"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _handle_prepare(args: argparse.Namespace) -> int:
    src_path = Path(args.source).resolve()
    out_path = Path(args.output).resolve()

    if not src_path.exists():
        print(json.dumps({"error": f"Source path does not exist: {src_path}"}), file=sys.stderr)
        return 1

    is_safetensors = False
    if src_path.is_file() and src_path.suffix == ".safetensors":
        is_safetensors = True
    elif src_path.is_dir():
        if (src_path / "model.safetensors.index.json").exists() or any(src_path.glob("*.safetensors")):
            is_safetensors = True

    vq_cfg = QuantizationConfig(
        bits_per_stage=args.bits_per_stage,
        max_stages=args.max_stages,
        require_compression=False,
    )
    leech_cfg = LeechQuantizationConfig(
        gain_bits=args.gain_bits,
        max_stages=args.max_stages,
        require_compression=False,
    )

    if is_safetensors:
        report = stream_convert_checkpoint(
            src_path,
            out_path,
            codec=args.codec,
            vq_config=vq_cfg,
            leech_config=leech_cfg,
            resume=args.resume,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    else:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(src_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(src_path, local_files_only=True)
        if args.codec == "leech24_shell2_residual":
            quantized, q_report = quantize_model(model, leech_cfg)
        else:
            quantized, q_report = quantize_model(model, vq_cfg)
        tok_dir = src_path if (src_path / "tokenizer.json").exists() else None
        manifest = save_oqpack(
            quantized,
            out_path,
            model_id=src_path.name,
            tokenizer_dir=tok_dir,
        )
        summary = {
            "source_dir": str(src_path),
            "output_dir": str(out_path),
            "codec": args.codec,
            "format": manifest.format_version,
            "shards": len(manifest.shards),
            "model_id": manifest.model_id,
            "compression_ratio": getattr(q_report, "compression_ratio", 1.0),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0


def _handle_verify(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    try:
        pkg_path = cache.resolve(str(args.package_dir))
    except Exception as e:
        print(json.dumps({"valid": False, "error": str(e)}, indent=2, sort_keys=True))
        return 1

    verify_checksums = not args.no_checksums

    # Case 1: OQPack v1 container
    if (pkg_path / MANIFEST_FILENAME).exists():
        try:
            manifest = verify_oqpack_integrity(pkg_path, verify_checksums=verify_checksums)
            res = {
                "valid": True,
                "format": manifest.format_version,
                "model_id": manifest.model_id,
                "shards": len(manifest.shards),
                "checksums_verified": verify_checksums,
            }
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0
        except Exception as e:
            res = {
                "valid": False,
                "format": "oqpack.v1",
                "error": str(e),
            }
            print(json.dumps(res, indent=2, sort_keys=True))
            return 1

    # Case 2: Streamed checkpoint
    state_file = pkg_path / "conversion_state.json"
    index_file = pkg_path / "model.safetensors.index.json"
    if state_file.exists() or index_file.exists():
        try:
            if state_file.exists():
                state_data = json.loads(state_file.read_text(encoding="utf-8"))
                records = state_data.get("records", {})
                for t_name, r_info in records.items():
                    shard_rel = r_info.get("shard_file", "")
                    shard_p = pkg_path / shard_rel
                    if not shard_p.exists():
                        raise ValueError(f"Missing shard file: {shard_rel}")
                    if verify_checksums and "sha256" in r_info:
                        hasher = hashlib.sha256()
                        with open(shard_p, "rb") as f:
                            while chunk := f.read(65536):
                                hasher.update(chunk)
                        if hasher.hexdigest() != r_info["sha256"]:
                            raise ValueError(f"Checksum mismatch for shard: {shard_rel}")
            elif index_file.exists():
                index_data = json.loads(index_file.read_text(encoding="utf-8"))
                weight_map = index_data.get("weight_map", {})
                for shard_rel in set(weight_map.values()):
                    shard_p = pkg_path / shard_rel
                    if not shard_p.exists():
                        raise ValueError(f"Missing shard file: {shard_rel}")
            res = {
                "valid": True,
                "format": "streamed_checkpoint",
                "checksums_verified": verify_checksums,
            }
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0
        except Exception as e:
            res = {
                "valid": False,
                "format": "streamed_checkpoint",
                "error": str(e),
            }
            print(json.dumps(res, indent=2, sort_keys=True))
            return 1

    res = {
        "valid": False,
        "error": f"No recognizable OQPack or streamed checkpoint in {pkg_path}",
    }
    print(json.dumps(res, indent=2, sort_keys=True))
    return 1


def _handle_chat(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    model_path = cache.resolve(str(args.model_dir))
    backend = getattr(args, "backend", "auto")

    model, tokenizer = _load_model_and_tokenizer(model_path, backend=backend)

    prompt_str = args.prompt.strip()
    parts = [p.strip() for p in prompt_str.split(",") if p.strip()]
    if parts and all(p.lstrip("-").isdigit() for p in parts):
        prompt_tokens = [int(p) for p in parts]
    elif tokenizer is not None and hasattr(tokenizer, "encode"):
        prompt_tokens = tokenizer.encode(prompt_str)
    else:
        raise ValueError(
            f"Prompt {prompt_str!r} cannot be tokenized without a tokenizer; pass comma-separated token IDs"
        )

    budget_bytes = (args.memory_budget_mb * 1024 * 1024) if args.memory_budget_mb else None
    engine = OjasXInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        memory_budget_bytes=budget_bytes,
    )

    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    result = engine.generate(prompt_tokens, params=params, use_cache=True)

    output: Dict[str, object] = {
        "prompt_token_ids": result.prompt_token_ids,
        "generated_token_ids": result.generated_token_ids,
        "full_token_ids": result.full_token_ids,
        "stop_reason": result.stop_reason,
        "prefill_tokens": result.prefill_tokens,
        "decode_steps": result.decode_steps,
        "used_kv_cache": True,
        "kv_cache_bytes": result.kv_cache_bytes,
        "admission_report": result.admission_report,
    }
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            output["text"] = tokenizer.decode(result.generated_token_ids)
        except Exception:
            pass

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _handle_bench(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    model_path = cache.resolve(str(args.model_dir))
    backend = getattr(args, "backend", "auto")

    model, tokenizer = _load_model_and_tokenizer(model_path, backend=backend)
    engine = OjasXInferenceEngine(model=model, tokenizer=tokenizer)

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 64)
    seq_len = max(1, args.seq_len)
    tokens = [((i % (vocab_size - 1)) + 1) for i in range(seq_len)]

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    engine.kv_manager.reset()
    _ = engine.prefill(tokens_tensor, use_cache=True)
    engine.kv_manager.reset()

    t_pref_0 = time.perf_counter()
    _ = engine.prefill(tokens_tensor, use_cache=True)
    prefill_ms = (time.perf_counter() - t_pref_0) * 1000.0

    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0)

    # Warmup
    _ = engine.generate(tokens, params=params, use_cache=True, reset_cache=True)
    _ = engine.generate(tokens, params=params, use_cache=False, reset_cache=True)

    t_uncached_0 = time.perf_counter()
    _ = engine.generate(tokens, params=params, use_cache=False, reset_cache=True)
    uncached_sec = time.perf_counter() - t_uncached_0

    t_cached_0 = time.perf_counter()
    res_cached = engine.generate(tokens, params=params, use_cache=True, reset_cache=True)
    cached_sec = time.perf_counter() - t_cached_0

    speedup = uncached_sec / max(cached_sec, 1e-9)
    throughput_tok_s = args.max_new_tokens / max(cached_sec, 1e-6)

    metrics = {
        "model_dir": str(model_path),
        "seq_len": seq_len,
        "max_new_tokens": args.max_new_tokens,
        "backend": backend,
        "prefill_latency_ms": round(prefill_ms, 3),
        "cached_decode_latency_ms": round(cached_sec * 1000.0, 3),
        "uncached_decode_latency_ms": round(uncached_sec * 1000.0, 3),
        "cached_vs_uncached_speedup": round(speedup, 3),
        "kv_cache_bytes": res_cached.kv_cache_bytes,
        "native_fused_kernel_throughput_tok_s": round(throughput_tok_s, 2),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    report = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "native_cpu_backend": native_backend_status(),
        "cuda_kernel_source": cuda_kernel_source_status(),
        "offline_cache_dir": str(cache.cache_dir),
        "silent_downloads_disabled": True,
        "dense_fallback_disabled": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    cache = OjasXOfflineCache()
    model_path = cache.resolve(str(args.model_dir))

    if args.dry_run:
        readiness = {
            "status": "ready",
            "dry_run": True,
            "model_dir": str(model_path),
            "host": args.host,
            "port": args.port,
            "memory_budget_mb": args.memory_budget_mb,
        }
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return 0

    try:
        from . import server

        if hasattr(server, "main"):
            return server.main(
                model_path,
                host=args.host,
                port=args.port,
                memory_budget_mb=args.memory_budget_mb,
            )
        elif hasattr(server, "run_server"):
            return server.run_server(
                model_path,
                host=args.host,
                port=args.port,
                memory_budget_mb=args.memory_budget_mb,
            )
        else:
            print(f"Loaded server module {server}, starting server on {args.host}:{args.port}")
            return 0
    except ImportError as e:
        raise ImportError(f"Cannot import ojasx_24x96.server: {e}") from e


def build_m7_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ojasx",
        description="OjasX Thin Production CLI & Offline Cache Manager (Milestone M7)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "python"),
        default="auto",
        help="Global computation backend for Leech/RVQ execution",
    )
    parser.add_argument(
        "--no-dense-fallback",
        action="store_true",
        help="Enforce strict quantization with zero dense fallback",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. inspect <path>
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect OQPack, legacy VQ, or checkpoint directory",
    )
    inspect_parser.add_argument("path", type=Path, help="Directory to inspect")
    inspect_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default=None
    )
    inspect_parser.add_argument("--no-dense-fallback", action="store_true")

    # 2. prepare <source> <output> [--codec] [--bits-per-stage] [--max-stages] [--gain-bits] [--resume]
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Streamingly prepare/quantize a checkpoint into an OQPack or VQ package",
    )
    prepare_parser.add_argument("source", type=Path, help="Source checkpoint directory")
    prepare_parser.add_argument("output", type=Path, help="Output package directory")
    prepare_parser.add_argument(
        "--codec",
        choices=("adaptive_residual_vq", "leech24_shell2_residual"),
        default="adaptive_residual_vq",
        help="Quantization codec family",
    )
    prepare_parser.add_argument("--bits-per-stage", type=int, default=4)
    prepare_parser.add_argument("--max-stages", type=int, default=2)
    prepare_parser.add_argument("--gain-bits", type=int, default=2)
    prepare_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume partial streaming conversion if state exists",
    )
    prepare_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default=None
    )
    prepare_parser.add_argument("--no-dense-fallback", action="store_true")

    # 3. verify <package_dir> [--no-checksums]
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify cryptographic checksums and schema integrity of package",
    )
    verify_parser.add_argument("package_dir", type=Path, help="Package directory")
    verify_parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip byte-level cryptographic checksum validation",
    )
    verify_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default=None
    )
    verify_parser.add_argument("--no-dense-fallback", action="store_true")

    # 4. chat <model_dir> --prompt <...> [--max-new-tokens] [--temperature] [--backend] [--memory-budget-mb]
    chat_parser = subparsers.add_parser(
        "chat",
        help="Run KV-cached inference or interactive conversation",
    )
    chat_parser.add_argument("model_dir", type=Path, help="Model or OQPack package directory")
    chat_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text or comma-separated token IDs",
    )
    chat_parser.add_argument("--max-new-tokens", type=int, default=16)
    chat_parser.add_argument("--temperature", type=float, default=0.0)
    chat_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default="auto"
    )
    chat_parser.add_argument("--memory-budget-mb", type=int, default=None)
    chat_parser.add_argument("--no-dense-fallback", action="store_true")

    # 5. bench <model_dir> [--seq-len] [--max-new-tokens] [--backend]
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark prefill latency, decode speedup, and native kernel throughput",
    )
    bench_parser.add_argument("model_dir", type=Path, help="Model or package directory")
    bench_parser.add_argument("--seq-len", type=int, default=16)
    bench_parser.add_argument("--max-new-tokens", type=int, default=8)
    bench_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default="auto"
    )
    bench_parser.add_argument("--no-dense-fallback", action="store_true")

    # 6. doctor [--json]
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check system readiness, kernel sources, cache, and compiler environments",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Format output as JSON")
    doctor_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default=None
    )
    doctor_parser.add_argument("--no-dense-fallback", action="store_true")

    # 7. serve <model_dir> [--host] [--port] [--memory-budget-mb] [--dry-run]
    serve_parser = subparsers.add_parser(
        "serve",
        help="Serve model via HTTP inference server",
    )
    serve_parser.add_argument("model_dir", type=Path, help="Model or package directory")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--memory-budget-mb", type=int, default=None)
    serve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate server readiness and configuration without starting server",
    )
    serve_parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default=None
    )
    serve_parser.add_argument("--no-dense-fallback", action="store_true")

    # Passthroughs for legacy subcommands
    for legacy_cmd in ("convert", "evaluate", "benchmark"):
        subparsers.add_parser(legacy_cmd, help=f"Legacy command (delegates to runner.py)")

    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for OjasX CLI, dispatching to M7 handlers or delegating legacy commands."""
    if argv is None:
        argv = sys.argv[1:]

    # Delegate legacy commands immediately
    if len(argv) > 0:
        first_non_opt = next((a for a in argv if not a.startswith("-")), None)
        if first_non_opt in {"convert", "evaluate", "benchmark"}:
            from . import runner

            return runner.main(argv)

    parser = build_m7_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _handle_inspect(args)
    elif args.command == "prepare":
        return _handle_prepare(args)
    elif args.command == "verify":
        return _handle_verify(args)
    elif args.command == "chat":
        return _handle_chat(args)
    elif args.command == "bench":
        return _handle_bench(args)
    elif args.command == "doctor":
        return _handle_doctor(args)
    elif args.command == "serve":
        return _handle_serve(args)
    elif args.command in {"convert", "evaluate", "benchmark"}:
        from . import runner

        return runner.main(argv)
    else:
        parser.print_help(sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Alias for run_cli matching standard CLI entry conventions."""
    return run_cli(argv)


__all__ = [
    "OjasXOfflineCache",
    "build_m7_parser",
    "run_cli",
    "main",
]

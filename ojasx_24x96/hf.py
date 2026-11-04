"""Optional Hugging Face persistence and local-chat integration.

Heavy HF imports are deliberately local to functions so the codec remains usable
with only PyTorch. Loading a quantized checkpoint constructs the architecture on
meta tensors and assigns saved shards; it does not load a dense model first.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .codec import QuantizationConfig
from .ir import QuantizationPlan
from .module import (
    ModelQuantizationReport,
    quantization_manifest,
    quantize_model,
    restore_vq_modules,
)
from .runtime import runtime_storage_report, set_decode_row_block_size, set_runtime_profile


MANIFEST_NAME = "vq_manifest.json"
REPORT_NAME = "quantization_report.json"
PLAN_NAME = "vq_plan.json"


def _model_task(config) -> str:
    vision = getattr(config, "vision_config", None)
    if vision is not None and not getattr(config, "language_model_only", False):
        return "multimodal_lm"
    return "causal_lm"


def _auto_model_class(task: str):
    if task == "causal_lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    if task == "multimodal_lm":
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    raise ValueError(f"Unsupported model task {task!r}")


def _json_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_quantized_pretrained(
    model,
    output_directory,
    *,
    tokenizer=None,
    report: Optional[ModelQuantizationReport] = None,
    plan: Optional[QuantizationPlan] = None,
    max_shard_size: str = "2GB",
) -> Dict[str, object]:
    """Save model shards, HF configuration, tokenizer, and VQ reconstruction data."""

    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("Saving HF models requires transformers and safetensors") from error
    if plan is not None:
        if not isinstance(plan, QuantizationPlan):
            raise TypeError("plan must be a generalized QuantizationPlan")
        actual_state = runtime_storage_report(model)["model_unique_state_tensor_bytes"]
        predicted_state = plan.memory.get("predicted_model_state_bytes")
        if predicted_state != actual_state:
            raise ValueError(
                f"Generalized plan is stale or unrelated: predicted {predicted_state} "
                f"persistent bytes but model has {actual_state}"
            )
        plan.memory["actual_model_state_bytes_after_apply"] = actual_state
        plan.memory["actual_matches_prediction"] = True
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Quantized output directory must be empty to avoid stale tensor shards")
    if not output.is_dir():
        output.mkdir(parents=True)
    manifest = quantization_manifest(model)
    manifest["model_task"] = _model_task(getattr(model, "config", None))
    if not manifest["layers"]:
        raise ValueError("The model contains no admitted residual-VQ layers")
    save_options = {}
    if "save_original_format" in inspect.signature(model.save_pretrained).parameters:
        # Transformers 5 can reverse fused-weight conversions while saving. VQ
        # buffers are runtime tensors, not source fused weights, so persist the
        # exact internal state_dict consumed by restore_vq_modules instead.
        save_options["save_original_format"] = False
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=max_shard_size,
        **save_options,
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    manifest["library_versions"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    _json_write(output / MANIFEST_NAME, manifest)
    if report is not None:
        _json_write(output / REPORT_NAME, report.to_dict())
    if plan is not None:
        _json_write(output / PLAN_NAME, plan.to_dict())
    return manifest


def _weight_shards(directory: Path) -> Tuple[List[Path], Dict[str, str]]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = dict(payload["weight_map"])
        files = sorted({directory / filename for filename in weight_map.values()})
        return files, weight_map
    single = directory / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No safetensors model found in {directory}")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    return [single], {key: single.name for key in keys}


def _ignored_missing(model, key: str) -> bool:
    patterns: List[str] = []
    for attribute in ("_tied_weights_keys", "_keys_to_ignore_on_load_missing"):
        value = getattr(model, attribute, None) or []
        if isinstance(value, dict):
            value = list(value)
        if isinstance(value, str):
            value = [value]
        patterns.extend(str(item) for item in value)
    return any(re.search(pattern, key) for pattern in patterns)


def load_quantized_pretrained(
    model_directory,
    *,
    device="cpu",
    load_tokenizer: bool = True,
    runtime_profile: str = "memory",
    decode_row_block_size: Optional[int] = None,
):
    """Load an encoded checkpoint without first materializing a dense model."""

    try:
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Quantized HF loading requires transformers, accelerate, and safetensors"
        ) from error
    directory = Path(model_directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    task = _model_task(config)
    if manifest.get("model_task", task) != task:
        raise ValueError("Manifest model task is incompatible with the saved HF config")
    auto_model = _auto_model_class(task)
    with init_empty_weights():
        model = auto_model.from_config(config)
    restore_vq_modules(model, manifest)
    expected = set(model.state_dict())
    files, _ = _weight_shards(directory)
    loaded = set()
    for shard_path in files:
        shard = load_file(str(shard_path), device="cpu")
        duplicates = sorted(loaded.intersection(shard))
        if duplicates:
            raise ValueError(
                f"Tensors occur in more than one shard at {shard_path.name}: {duplicates[:8]}"
            )
        unexpected = sorted(set(shard) - expected)
        if unexpected:
            raise ValueError(f"Unexpected tensors in {shard_path.name}: {unexpected[:8]}")
        model.load_state_dict(shard, strict=False, assign=True)
        loaded.update(shard)
        del shard
    missing = sorted(expected - loaded)
    unexpected_missing = [key for key in missing if not _ignored_missing(model, key)]
    if unexpected_missing:
        raise ValueError(f"Checkpoint is missing tensors: {unexpected_missing[:8]}")
    model.tie_weights()
    meta_tensors = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise ValueError(f"Some model tensors were not restored: {meta_tensors[:8]}")
    model.to(device=device)
    set_runtime_profile(model, runtime_profile)
    if decode_row_block_size is None:
        decode_row_block_size = 64
        plan_path = directory / PLAN_NAME
        if plan_path.exists():
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            decode_row_block_size = int(
                plan_payload.get("constraints", {}).get("decode_row_block_size", 64)
            )
    set_decode_row_block_size(model, decode_row_block_size)
    model.eval()
    tokenizer = (
        AutoTokenizer.from_pretrained(directory, local_files_only=True)
        if load_tokenizer
        else None
    )
    return (model, tokenizer) if load_tokenizer else model


def convert_local_hf_model(
    source_directory,
    output_directory,
    config: QuantizationConfig = QuantizationConfig(),
    *,
    include: Iterable[str] = ("*",),
    exclude: Iterable[str] = ("lm_head", "*.lm_head"),
    max_shard_size: str = "2GB",
) -> ModelQuantizationReport:
    """Convert a local HF generative LM, preserving tokenizer and chat metadata.

    This offline conversion loads the source model once. Subsequent inference uses
    :func:`load_quantized_pretrained`, which never loads the dense source weights.
    """

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("HF conversion requires transformers") from error
    source = Path(source_directory)
    output = Path(output_directory)
    if source.resolve() == output.resolve():
        raise ValueError("Quantized output must differ from the source directory")
    hf_config = AutoConfig.from_pretrained(source, local_files_only=True)
    auto_model = _auto_model_class(_model_task(hf_config))
    model = auto_model.from_pretrained(
        source,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model, report = quantize_model(model, config, include=include, exclude=exclude)
    if not report.selected_layers:
        raise RuntimeError("No layer met the compression and error-budget requirements")
    save_quantized_pretrained(
        model,
        output,
        tokenizer=tokenizer,
        report=report,
        max_shard_size=max_shard_size,
    )
    return report


class VQChatRunner:
    """Local-only causal-LM chat runner that uses the tokenizer's own template."""

    def __init__(self, model, tokenizer, device="cpu") -> None:
        if tokenizer is None:
            raise ValueError("Chat requires a tokenizer")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_directory,
        *,
        device="cpu",
        runtime_profile: str = "memory",
        decode_row_block_size: Optional[int] = None,
    ) -> "VQChatRunner":
        model, tokenizer = load_quantized_pretrained(
            model_directory,
            device=device,
            runtime_profile=runtime_profile,
            decode_row_block_size=decode_row_block_size,
        )
        return cls(model, tokenizer, device)

    def render_chat(self, messages: Sequence[Dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError(
                "This tokenizer has no chat template. Refusing to guess one; "
                "provide the original instruct-model tokenizer metadata."
            )
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def respond(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prompt = self.render_chat(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        if do_sample:
            options.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(**encoded, **options)
        continuation = output[0, input_length:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)

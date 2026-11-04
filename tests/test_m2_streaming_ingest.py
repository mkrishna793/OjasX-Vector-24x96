"""Unit tests for Milestone M2: Bounded-RAM Streaming Ingestion & Resumable Converter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Dict, Tuple

import pytest
import safetensors.torch
from safetensors.torch import save_file
import torch
import torch.nn as nn

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.ingest import (
    CheckpointIngestionError,
    StreamingConversionReport,
    StreamingTensorRecord,
    discover_checkpoint_safetensors,
    load_streamed_checkpoint_into_model,
    should_quantize_tensor_name,
    stream_convert_checkpoint,
)
from ojasx_24x96.leech_codec import LeechQuantizationConfig
from ojasx_24x96.module import ResidualVQLinear


class MiniTransformerBlock(nn.Module):
    """Minimal multi-layer Transformer block for sharded streaming tests."""

    def __init__(self, hidden_dim: int = 48, intermediate_dim: int = 96):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden_dim, hidden_dim, bias=True),
                "k_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "v_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "o_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
            }
        )
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden_dim, intermediate_dim, bias=False),
                "up_proj": nn.Linear(hidden_dim, intermediate_dim, bias=False),
                "down_proj": nn.Linear(intermediate_dim, hidden_dim, bias=False),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.input_layernorm(x)
        q = self.self_attn["q_proj"](norm_x)
        k = self.self_attn["k_proj"](norm_x)
        v = self.self_attn["v_proj"](norm_x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn_out = self.self_attn["o_proj"](torch.matmul(torch.softmax(attn, dim=-1), v))
        x = x + attn_out

        norm_x2 = self.post_attention_layernorm(x)
        gate = torch.relu(self.mlp["gate_proj"](norm_x2))
        up = self.mlp["up_proj"](norm_x2)
        mlp_out = self.mlp["down_proj"](gate * up)
        return x + mlp_out


class MiniTransformer(nn.Module):
    """Transformer with embedding, multiple layers, layernorm, and lm_head."""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 48,
        intermediate_dim: int = 96,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MiniTransformerBlock(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def _create_sharded_checkpoint(
    model: nn.Module,
    dest_dir: Path,
    num_shards: int = 3,
) -> Tuple[Path, Dict[str, str]]:
    """Save model state_dict into multi-shard safetensors with model.safetensors.index.json."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    keys = list(state.keys())

    # Distribute keys evenly across shards
    shard_tensors: Dict[str, Dict[str, torch.Tensor]] = {
        f"model-{i+1:05d}-of-{num_shards:05d}.safetensors": {} for i in range(num_shards)
    }
    shard_names = list(shard_tensors.keys())
    weight_map: Dict[str, str] = {}

    for idx, key in enumerate(keys):
        s_name = shard_names[idx % num_shards]
        shard_tensors[s_name][key] = state[key]
        weight_map[key] = s_name

    for s_name, tensors in shard_tensors.items():
        save_file(tensors, str(dest_dir / s_name))

    index_payload = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": weight_map,
    }
    (dest_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )
    (dest_dir / "config.json").write_text(
        json.dumps({"model_type": "mini_transformer", "hidden_size": 48}), encoding="utf-8"
    )
    return dest_dir, weight_map


def test_should_quantize_tensor_name():
    """Verify 2D weight filtering vs 1D vectors and glob exclusions."""
    assert should_quantize_tensor_name("layers.0.self_attn.q_proj.weight", (48, 48)) is True
    assert should_quantize_tensor_name("layers.0.mlp.gate_proj.weight", (96, 48)) is True

    # 1D tensors must NOT be quantized
    assert should_quantize_tensor_name("layers.0.self_attn.q_proj.bias", (48,)) is False
    assert should_quantize_tensor_name("layers.0.input_layernorm.weight", (48,)) is False
    assert should_quantize_tensor_name("norm.weight", (48,)) is False

    # Excluded globs
    assert should_quantize_tensor_name("embed_tokens.weight", (100, 48)) is False
    assert should_quantize_tensor_name("model.embed_tokens.weight", (100, 48)) is False
    assert should_quantize_tensor_name("lm_head.weight", (100, 48)) is False
    assert should_quantize_tensor_name("model.lm_head.weight", (100, 48)) is False
    assert should_quantize_tensor_name("layers.0.self_attn.q_proj.bias", (48, 48)) is False

    # Custom include/exclude
    assert (
        should_quantize_tensor_name(
            "custom.layer.weight", (32, 32), include_globs=("*custom*",), exclude_globs=()
        )
        is True
    )
    assert (
        should_quantize_tensor_name(
            "other.layer.weight", (32, 32), include_globs=("*custom*",), exclude_globs=()
        )
        is False
    )


def test_discover_checkpoint_safetensors_validation(tmp_path: Path):
    """Test checkpoint discovery and all CheckpointIngestionError failure modes."""
    # 1. Non-existent path
    with pytest.raises(CheckpointIngestionError, match="does not exist"):
        discover_checkpoint_safetensors(tmp_path / "nonexistent")

    # 2. Invalid file type
    txt_file = tmp_path / "model.txt"
    txt_file.write_text("not safetensors")
    with pytest.raises(CheckpointIngestionError, match="must be a .safetensors"):
        discover_checkpoint_safetensors(txt_file)

    # 3. Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(CheckpointIngestionError, match="No .safetensors checkpoint found"):
        discover_checkpoint_safetensors(empty_dir)

    # 4. Valid single safetensors file
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    single_path = single_dir / "model.safetensors"
    save_file({"w": torch.randn(4, 4)}, str(single_path))
    shards, wmap = discover_checkpoint_safetensors(single_dir)
    assert len(shards) == 1 and shards[0].name == "model.safetensors"
    assert wmap == {"w": "model.safetensors"}

    # Pass file directly
    shards_f, wmap_f = discover_checkpoint_safetensors(single_path)
    assert len(shards_f) == 1 and shards_f[0] == single_path
    assert wmap_f == {"w": "model.safetensors"}

    # 5. Missing shard referenced in weight_map
    broken_dir = tmp_path / "broken_index"
    broken_dir.mkdir()
    save_file({"w1": torch.randn(4, 4)}, str(broken_dir / "shard1.safetensors"))
    index_payload = {
        "weight_map": {"w1": "shard1.safetensors", "w2": "shard_missing.safetensors"}
    }
    (broken_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))
    with pytest.raises(CheckpointIngestionError, match="Missing shard file"):
        discover_checkpoint_safetensors(broken_dir)

    # 6. Orphan shard not listed in weight_map
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    save_file({"w1": torch.randn(4, 4)}, str(orphan_dir / "shard1.safetensors"))
    save_file({"w_orphan": torch.randn(4, 4)}, str(orphan_dir / "orphan.safetensors"))
    index_payload = {"weight_map": {"w1": "shard1.safetensors"}}
    (orphan_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))
    with pytest.raises(CheckpointIngestionError, match="Orphan shard"):
        discover_checkpoint_safetensors(orphan_dir)

    # 7. Weight in weight_map missing from target shard
    corrupt_map_dir = tmp_path / "corrupt_map"
    corrupt_map_dir.mkdir()
    save_file({"w1": torch.randn(4, 4)}, str(corrupt_map_dir / "shard1.safetensors"))
    index_payload = {"weight_map": {"ghost_weight": "shard1.safetensors"}}
    (corrupt_map_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))
    with pytest.raises(CheckpointIngestionError, match="not found in file"):
        discover_checkpoint_safetensors(corrupt_map_dir)

    # 8. Malformed JSON
    bad_json_dir = tmp_path / "bad_json"
    bad_json_dir.mkdir()
    (bad_json_dir / "model.safetensors.index.json").write_text("{broken json")
    with pytest.raises(CheckpointIngestionError, match="Malformed"):
        discover_checkpoint_safetensors(bad_json_dir)

    # 9. Multiple shards without index
    multi_no_index = tmp_path / "multi_no_index"
    multi_no_index.mkdir()
    save_file({"a": torch.randn(2, 2)}, str(multi_no_index / "a.safetensors"))
    save_file({"b": torch.randn(2, 2)}, str(multi_no_index / "b.safetensors"))
    with pytest.raises(CheckpointIngestionError, match="Multiple .safetensors shards found"):
        discover_checkpoint_safetensors(multi_no_index)


def test_stream_convert_checkpoint_adaptive_residual_vq(tmp_path: Path):
    """Test streaming ingestion with adaptive_residual_vq, verifying memory and model loading."""
    torch.manual_seed(20261104)
    model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    ckpt_dir, _ = _create_sharded_checkpoint(model, tmp_path / "source_ckpt", num_shards=3)

    out_dir = tmp_path / "converted_adaptive"
    vq_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=0.99,
        require_compression=False,
    )

    report = stream_convert_checkpoint(
        ckpt_dir,
        out_dir,
        codec="adaptive_residual_vq",
        vq_config=vq_cfg,
    )

    # Verification: Bounded RAM condition (strictly < 35% of total weight bytes)
    assert report.total_tensors == len(model.state_dict())
    assert report.quantized_tensors > 0
    assert report.passthrough_tensors > 0
    assert report.total_tensors == report.quantized_tensors + report.passthrough_tensors
    assert report.resumed_tensors == 0

    peak_fraction = report.peak_single_tensor_bytes / report.original_bytes
    assert peak_fraction < 0.35, f"Peak tensor fraction {peak_fraction:.2%} exceeds 35%"
    assert report.compression_ratio > 1.0

    # Verify generated manifest and index files
    assert (out_dir / "conversion_state.json").exists()
    assert (out_dir / "model.safetensors.index.json").exists()
    assert (out_dir / "vq_manifest.json").exists()
    assert (out_dir / "config.json").exists()  # Metadata preserved

    # Load converted checkpoint into a clean model architecture
    new_model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    loaded_model = load_streamed_checkpoint_into_model(new_model, out_dir, device="cpu")

    # Verify linear layers were replaced by ResidualVQLinear
    for layer in loaded_model.layers:
        assert isinstance(layer.self_attn["q_proj"], ResidualVQLinear)
        assert isinstance(layer.self_attn["k_proj"], ResidualVQLinear)
        assert isinstance(layer.self_attn["v_proj"], ResidualVQLinear)
        assert isinstance(layer.self_attn["o_proj"], ResidualVQLinear)
        assert isinstance(layer.mlp["gate_proj"], ResidualVQLinear)
        assert isinstance(layer.mlp["up_proj"], ResidualVQLinear)
        assert isinstance(layer.mlp["down_proj"], ResidualVQLinear)

    # Passthrough layers remain dense
    assert isinstance(loaded_model.embed_tokens, nn.Embedding)
    assert isinstance(loaded_model.norm, nn.LayerNorm)
    assert isinstance(loaded_model.lm_head, nn.Linear)

    # Forward pass verification
    test_input = torch.randint(0, 100, (2, 8))
    output = loaded_model(test_input)
    assert output.shape == (2, 8, 100)
    assert torch.isfinite(output).all()


def test_stream_convert_checkpoint_leech24_residual(tmp_path: Path):
    """Test streaming ingestion with leech24_shell2_residual codec."""
    torch.manual_seed(20261104)
    model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    ckpt_dir, _ = _create_sharded_checkpoint(model, tmp_path / "source_ckpt_leech", num_shards=3)

    out_dir = tmp_path / "converted_leech"
    leech_cfg = LeechQuantizationConfig(
        max_stages=2,
        gain_bits=2,
        min_weight_elements=64,
        max_relative_error=0.99,
        require_compression=False,
    )

    report = stream_convert_checkpoint(
        ckpt_dir,
        out_dir,
        codec="leech24_shell2_residual",
        leech_config=leech_cfg,
    )

    assert report.total_tensors == len(model.state_dict())
    assert report.quantized_tensors > 0
    assert report.resumed_tensors == 0
    peak_fraction = report.peak_single_tensor_bytes / report.original_bytes
    assert peak_fraction < 0.35, f"Leech peak tensor fraction {peak_fraction:.2%} exceeds 35%"

    # Load converted model
    new_model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    loaded_model = load_streamed_checkpoint_into_model(new_model, out_dir, device="cpu")

    for layer in loaded_model.layers:
        assert isinstance(layer.self_attn["q_proj"], ResidualVQLinear)
        assert layer.self_attn["q_proj"].codec == "leech24_shell2_residual"

    test_input = torch.randint(0, 100, (2, 8))
    output = loaded_model(test_input)
    assert output.shape == (2, 8, 100)
    assert torch.isfinite(output).all()


def test_crash_and_resumable_conversion(tmp_path: Path):
    """Test simulated interruption and recovery with identical model output verification."""
    torch.manual_seed(20261104)
    model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    ckpt_dir, _ = _create_sharded_checkpoint(model, tmp_path / "source_ckpt", num_shards=3)

    vq_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=0.99,
        require_compression=False,
    )

    # 1. Non-interrupted baseline conversion
    clean_out = tmp_path / "clean_out"
    report_clean = stream_convert_checkpoint(
        ckpt_dir,
        clean_out,
        codec="adaptive_residual_vq",
        vq_config=vq_cfg,
    )
    model_clean = load_streamed_checkpoint_into_model(
        copy.deepcopy(model), clean_out, device="cpu"
    )

    test_input = torch.randint(0, 100, (2, 12))
    output_clean = model_clean(test_input)

    # 2. Crash run with fail_after_tensors=2
    crash_out = tmp_path / "resumed_out"
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        stream_convert_checkpoint(
            ckpt_dir,
            crash_out,
            codec="adaptive_residual_vq",
            vq_config=vq_cfg,
            fail_after_tensors=2,
        )

    # Verify partial state
    state_file = crash_out / "conversion_state.json"
    assert state_file.exists()
    partial_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(partial_state["records"]) == 2

    # 3. Resume run
    report_resumed = stream_convert_checkpoint(
        ckpt_dir,
        crash_out,
        codec="adaptive_residual_vq",
        vq_config=vq_cfg,
        resume=True,
    )

    # Verify resume metrics
    assert report_resumed.resumed_tensors == 2
    assert report_resumed.total_tensors == report_clean.total_tensors
    assert report_resumed.quantized_tensors == report_clean.quantized_tensors
    assert report_resumed.passthrough_tensors == report_clean.passthrough_tensors

    # 4. Load resumed model and verify outputs are identical
    model_resumed = load_streamed_checkpoint_into_model(
        copy.deepcopy(model), crash_out, device="cpu"
    )
    output_resumed = model_resumed(test_input)

    assert torch.equal(output_clean, output_resumed)


def test_resume_recovers_from_corrupted_shard(tmp_path: Path):
    """Verify that if an output shard is corrupted, resume detects sha256 mismatch and re-quantizes."""
    torch.manual_seed(20261104)
    model = MiniTransformer(vocab_size=100, hidden_dim=48, intermediate_dim=96, num_layers=2)
    ckpt_dir, _ = _create_sharded_checkpoint(model, tmp_path / "source_ckpt", num_shards=2)

    vq_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=0.99,
        require_compression=False,
    )

    run_dir = tmp_path / "corrupt_test_out"
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        stream_convert_checkpoint(
            ckpt_dir,
            run_dir,
            codec="adaptive_residual_vq",
            vq_config=vq_cfg,
            fail_after_tensors=3,
        )

    # Corrupt one of the converted shards on disk
    state_data = json.loads((run_dir / "conversion_state.json").read_text(encoding="utf-8"))
    first_record = list(state_data["records"].values())[0]
    corrupted_shard = run_dir / first_record["shard_file"]
    corrupted_shard.write_bytes(b"corrupted bytes!")

    # Resume: the corrupted shard should NOT be resumed, but re-converted
    report = stream_convert_checkpoint(
        ckpt_dir,
        run_dir,
        codec="adaptive_residual_vq",
        vq_config=vq_cfg,
        resume=True,
    )
    # Since 1 of the 3 was corrupted, only 2 should have been resumed!
    assert report.resumed_tensors == 2
    assert report.total_tensors == len(model.state_dict())

    # Model can be successfully loaded and executed
    loaded_model = load_streamed_checkpoint_into_model(copy.deepcopy(model), run_dir)
    test_input = torch.randint(0, 100, (2, 8))
    output = loaded_model(test_input)
    assert output.shape == (2, 8, 100)
    assert torch.isfinite(output).all()

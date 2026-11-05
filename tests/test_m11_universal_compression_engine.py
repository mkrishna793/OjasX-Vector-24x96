"""Milestone M11: Universal Compress-Once Engine (2D + 3D MoE + Conv) Verification.

Tests cover:
1. Structural Duck-Typing & Universal 3D MoE Expert Bank Support:
   - Defining a custom 3D MoE Expert class CustomNovel3DExperts (NOT in _FUSED_EXPERT_CONTRACTS)
     with gate_up_proj [E, 2*I, H] and down_proj [E, H, I], plus a router and attention block.
   - Verifying UniversalCompressionEngine automatically detects and replaces CustomNovel3DExperts
     with ResidualVQFusedExperts and 2D projections with ResidualVQLinear without any class
     registration or model-specific allowlist.
   - Testing forward pass execution for both streaming_decode=True and streaming_decode=False.
2. Structural Duck-Typing for Novel 2D Linear Projections:
   - Custom 2D module without nn.Linear subclassing automatically detected and converted.
3. Hybrid Role-Aware Codec Selection (select_codec_for_tensor):
   - mode="hybrid_leech_rvq" routes sensitive attention, recurrent/DeltaNet, and vision layers
     to leech24_shell2_residual, while massive 3D MoE expert banks and standard FFN route to
     adaptive_residual_vq.
   - Fixed modes (adaptive_residual_vq, leech24_shell2_residual) route uniformly.
4. In-Place Hybrid Module Compression:
   - Verifies hybrid compression end-to-end where attention projections use Leech24 and
     3D experts use Adaptive RVQ.
5. compress_model_to_oqpack with 3D MoE Experts:
   - Compression to verified .oqpack container and bit-exact reconstruction via
     load_quantized_model_from_oqpack.
6. stream_compress_checkpoint & load_streamed_checkpoint_into_model with 3D MoE Tensors:
   - Sharding model containing 2D projections and 3D MoE expert weights across multi-shard safetensors.
   - Streaming compression verifies 3D expert tensors are quantized (not skipped!).
   - Ingestion restores clean model into ResidualVQFusedExperts producing finite forward outputs
     matching the in-place quantized model.
7. Smooth Channel Conditioning & Configuration Options:
   - Integration with SmoothChannelConditioner and glob filter exclusions.
"""

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
import torch.nn.functional as F

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.compression_engine import (
    UniversalCompressionConfig,
    UniversalCompressionEngine,
    UniversalCompressionResult,
    select_codec_for_tensor,
)
from ojasx_24x96.ingest import (
    load_streamed_checkpoint_into_model,
    should_quantize_tensor_name,
)
from ojasx_24x96.leech_codec import LeechQuantizationConfig
from ojasx_24x96.module import (
    _FUSED_EXPERT_CONTRACTS,
    ResidualVQFusedExperts,
    ResidualVQLinear,
    _expert_contract_key,
)
from ojasx_24x96.oqpack import (
    load_quantized_model_from_oqpack,
    save_oqpack,
)


class CustomNovel3DExperts(nn.Module):
    """Completely novel, unregistered 3D MoE Expert class.

    Notice: This class is NOT listed in `_FUSED_EXPERT_CONTRACTS` or any allowlist.
    It duck-types the 3D expert tensor contract:
      - gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
      - down_proj: [num_experts, hidden_dim, intermediate_dim]
      - act_fn: SwiGLU / SiLU activation
    """

    def __init__(
        self,
        num_experts: int = 4,
        hidden_dim: int = 48,
        intermediate_dim: int = 48,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.02
        )
        self.act_fn = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Standard MoE forward matching ResidualVQFusedExperts mathematical semantics."""
        final_states = torch.zeros_like(hidden_states)
        for i in range(hidden_states.shape[0]):
            for k in range(top_k_index.shape[1]):
                exp = int(top_k_index[i, k].item())
                w = top_k_weights[i, k]
                x = hidden_states[i : i + 1]
                gu = F.linear(x, self.gate_up_proj[exp])
                gate, up = gu.chunk(2, dim=-1)
                h = self.act_fn(gate) * up
                out = F.linear(h, self.down_proj[exp])
                final_states[i : i + 1] += out * w
        return final_states


class CustomAttentionBlock(nn.Module):
    """Custom multi-head projection block with 2D Linear layers."""

    def __init__(self, hidden_dim: int = 48) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class CustomNovelLinear(nn.Module):
    """Duck-typed 2D Linear projection without subclassing nn.Linear."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class FullNovelMoEBlock(nn.Module):
    """Complete block integrating custom attention, router, and novel 3D experts."""

    def __init__(
        self,
        hidden_dim: int = 48,
        intermediate_dim: int = 48,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.attn = CustomAttentionBlock(hidden_dim)
        # Router linear gate projection
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = CustomNovel3DExperts(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Attention residual
        h = x + self.attn(x)
        # 2. Router top-k gating
        logits = self.gate(h)
        scores = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(scores, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # 3. MoE Expert routing
        expert_out = self.experts(h, top_k_index, top_k_weights)
        return h + expert_out


def test_custom_novel_3d_experts_structural_duck_typing():
    """Verify that CustomNovel3DExperts is not registered and automatically converted."""
    torch.manual_seed(20261105)
    model = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)

    # 1. Assert explicitly that this class is NOT in the allowlist contracts
    contract_key = _expert_contract_key(model.experts)
    assert contract_key not in _FUSED_EXPERT_CONTRACTS, (
        f"{contract_key} unexpectedly found in _FUSED_EXPERT_CONTRACTS!"
    )

    # 2. Compress model in place with UniversalCompressionEngine
    cfg = UniversalCompressionConfig(
        mode="adaptive_residual_vq",
        vq_config=QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
        ),
    )
    engine = UniversalCompressionEngine(cfg)
    compressed_model, report = engine.compress_module_in_place(model)

    # 3. Verify structural replacement
    assert isinstance(compressed_model.experts, ResidualVQFusedExperts), (
        "CustomNovel3DExperts was not replaced with ResidualVQFusedExperts!"
    )
    assert isinstance(compressed_model.experts.gate_up_proj, ResidualVQLinear)
    assert isinstance(compressed_model.experts.down_proj, ResidualVQLinear)
    assert isinstance(compressed_model.attn.q_proj, ResidualVQLinear)
    assert isinstance(compressed_model.attn.k_proj, ResidualVQLinear)
    assert isinstance(compressed_model.attn.v_proj, ResidualVQLinear)
    assert isinstance(compressed_model.attn.o_proj, ResidualVQLinear)

    # Router was excluded due to *.gate.weight in default exclude_globs
    assert isinstance(compressed_model.gate, nn.Linear)

    # 4. Forward pass verification in both speed and streaming memory modes
    x = torch.randn(3, 48)

    # Streaming decode (memory mode)
    compressed_model.experts.streaming_decode = True
    out_streaming = compressed_model(x)
    assert out_streaming.shape == (3, 48)
    assert torch.isfinite(out_streaming).all()

    # Cached bank decode (speed mode)
    compressed_model.experts.streaming_decode = False
    out_cached = compressed_model(x)
    assert out_cached.shape == (3, 48)
    assert torch.isfinite(out_cached).all()
    assert torch.allclose(out_streaming, out_cached, atol=1e-5)


def test_duck_typed_novel_linear_projections():
    """Verify duck-typed 2D custom linear layers are admitted and compressed."""
    torch.manual_seed(20261105)

    class ModelWithDuckLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.duck_proj = CustomNovelLinear(48, 48)
            self.std_proj = nn.Linear(48, 48, bias=False)

        def forward(self, x):
            return self.std_proj(self.duck_proj(x))

    model = ModelWithDuckLinear()
    engine = UniversalCompressionEngine(
        UniversalCompressionConfig(
            mode="adaptive_residual_vq",
            vq_config=QuantizationConfig(
                block_size=8,
                bits_per_stage=4,
                max_stages=2,
                require_compression=False,
                max_relative_error=None,
            ),
        )
    )
    comp_model, report = engine.compress_module_in_place(model)

    assert isinstance(comp_model.duck_proj, ResidualVQLinear)
    assert isinstance(comp_model.std_proj, ResidualVQLinear)
    x = torch.randn(2, 48)
    out = comp_model(x)
    assert out.shape == (2, 48)
    assert torch.isfinite(out).all()


def test_select_codec_for_tensor_role_awareness():
    """Verify hybrid role-aware codec selection assigns Leech24 vs Adaptive RVQ correctly."""
    # 1. Hybrid Mode
    hybrid = "hybrid_leech_rvq"

    # Attention projections -> Leech24
    assert select_codec_for_tensor("layers.0.self_attn.q_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("layers.0.self_attn.k_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("layers.0.self_attn.v_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("layers.0.self_attn.o_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("model.attention.dense.weight", (48, 48), hybrid) == "leech24_shell2_residual"

    # Recurrent / DeltaNet / Linear Attention -> Leech24
    assert select_codec_for_tensor("layers.0.linear_attn.q_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("layers.0.gated_delta.in_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"

    # Vision & Audio projections -> Leech24
    assert select_codec_for_tensor("vision_tower.patch_embed.proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"
    assert select_codec_for_tensor("audio_model.encoder.layers.0.self_attn.q_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"

    # Shared Expert -> Leech24
    assert select_codec_for_tensor("layers.0.mlp.shared_expert.gate_proj.weight", (48, 48), hybrid) == "leech24_shell2_residual"

    # 3D MoE Routed Expert Banks (shape len == 3) -> Adaptive RVQ
    assert select_codec_for_tensor("layers.0.experts.gate_up_proj", (8, 96, 48), hybrid) == "adaptive_residual_vq"
    assert select_codec_for_tensor("layers.0.experts.down_proj", (8, 48, 48), hybrid) == "adaptive_residual_vq"
    # Even if named with 'attn', 3D tensors are routed to adaptive_residual_vq
    assert select_codec_for_tensor("layers.0.attn_experts.gate_up_proj", (8, 96, 48), hybrid) == "adaptive_residual_vq"

    # Standard FFN 2D projections -> Adaptive RVQ
    assert select_codec_for_tensor("layers.0.mlp.gate_proj.weight", (96, 48), hybrid) == "adaptive_residual_vq"
    assert select_codec_for_tensor("layers.0.mlp.up_proj.weight", (96, 48), hybrid) == "adaptive_residual_vq"
    assert select_codec_for_tensor("layers.0.mlp.down_proj.weight", (48, 96), hybrid) == "adaptive_residual_vq"

    # 2. Uniform Modes
    for mode in ("adaptive_residual_vq", "leech24_shell2_residual"):
        assert select_codec_for_tensor("layers.0.self_attn.q_proj.weight", (48, 48), mode) == mode
        assert select_codec_for_tensor("layers.0.experts.gate_up_proj", (8, 96, 48), mode) == mode


def test_hybrid_mode_compress_module_in_place():
    """Verify hybrid compression applies Leech24 to attention and RVQ to 3D MoE."""
    torch.manual_seed(20261105)
    model = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)

    cfg = UniversalCompressionConfig(
        mode="hybrid_leech_rvq",
        leech_config=LeechQuantizationConfig(
            max_stages=2,
            gain_bits=2,
            require_compression=False,
            max_relative_error=None,
        ),
        vq_config=QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
        ),
    )
    engine = UniversalCompressionEngine(cfg)
    comp_model, report = engine.compress_module_in_place(model)

    # Attention projections must use leech24_shell2_residual
    assert comp_model.attn.q_proj.codec == "leech24_shell2_residual"
    assert comp_model.attn.k_proj.codec == "leech24_shell2_residual"
    assert comp_model.attn.v_proj.codec == "leech24_shell2_residual"
    assert comp_model.attn.o_proj.codec == "leech24_shell2_residual"

    # 3D MoE expert projections must use adaptive_residual_vq
    assert comp_model.experts.gate_up_proj.codec == "adaptive_residual_vq"
    assert comp_model.experts.down_proj.codec == "adaptive_residual_vq"

    # Forward pass executes cleanly
    x = torch.randn(2, 48)
    out = comp_model(x)
    assert out.shape == (2, 48)
    assert torch.isfinite(out).all()


def test_compress_model_to_oqpack_with_3d_moe_experts(tmp_path: Path):
    """Verify packaging of 3D MoE models into .oqpack and perfect deserialization."""
    torch.manual_seed(20261105)
    model = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)
    model.eval()

    cfg = UniversalCompressionConfig(
        mode="adaptive_residual_vq",
        vq_config=QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
        ),
    )
    engine = UniversalCompressionEngine(cfg)
    oq_dir = tmp_path / "novel_moe.oqpack"

    q_model, manifest, result = engine.compress_model_to_oqpack(
        model,
        oq_dir,
        model_id="custom-novel-moe",
    )

    # Result summary validation
    assert isinstance(result, UniversalCompressionResult)
    assert result.model_id == "custom-novel-moe"
    assert result.quantized_tensors == 6  # 4 attn + 2 expert projs
    assert result.passthrough_tensors >= 1  # gate router
    assert result.compression_ratio > 0.0
    assert result.effective_bpw > 0.0
    assert Path(result.manifest_path).exists()

    res_dict = result.to_dict()
    assert res_dict["model_id"] == "custom-novel-moe"

    # Manifest checks
    assert "experts" in manifest.layer_specs
    assert manifest.layer_specs["experts"]["module_type"] == "fused_experts"
    assert manifest.layer_specs["experts"]["num_experts"] == 4

    # Load back using load_quantized_model_from_oqpack
    clean_skeleton = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)
    clean_skeleton.eval()
    restored_model, loaded_manifest = load_quantized_model_from_oqpack(clean_skeleton, oq_dir)

    assert isinstance(restored_model.experts, ResidualVQFusedExperts)
    assert isinstance(restored_model.attn.q_proj, ResidualVQLinear)

    x = torch.randn(2, 48)
    expected_out = q_model(x)
    restored_out = restored_model(x)

    assert torch.allclose(expected_out, restored_out, atol=1e-5)
    assert torch.isfinite(restored_out).all()


def test_stream_compress_checkpoint_and_load_with_3d_moe(tmp_path: Path):
    """Verify streaming compression quantizes 3D expert weights and loads correctly."""
    torch.manual_seed(20261105)
    orig_model = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)
    orig_model.eval()

    # 1. Build a multi-shard safetensors checkpoint
    ckpt_dir = tmp_path / "source_sharded_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = orig_model.state_dict()

    shard1_tensors = {}
    shard2_tensors = {}
    weight_map = {}

    for idx, (k, v) in enumerate(state.items()):
        shard_name = "model-00001-of-00002.safetensors" if idx % 2 == 0 else "model-00002-of-00002.safetensors"
        if idx % 2 == 0:
            shard1_tensors[k] = v
        else:
            shard2_tensors[k] = v
        weight_map[k] = shard_name

    save_file(shard1_tensors, str(ckpt_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2_tensors, str(ckpt_dir / "model-00002-of-00002.safetensors"))
    index_payload = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": weight_map,
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload), encoding="utf-8")
    (ckpt_dir / "config.json").write_text(json.dumps({"model_type": "novel_moe"}), encoding="utf-8")

    # 2. In-place quantized model as reference
    model_for_inplace = copy.deepcopy(orig_model)
    cfg = UniversalCompressionConfig(mode="adaptive_residual_vq")
    engine = UniversalCompressionEngine(cfg)
    inplace_model, in_report = engine.compress_module_in_place(model_for_inplace)

    # 3. Stream-compress the sharded checkpoint
    stream_out_dir = tmp_path / "stream_compressed_out"
    stream_report = engine.stream_compress_checkpoint(ckpt_dir, stream_out_dir)

    # Verify 3D expert tensors are quantized (NOT skipped or marked passthrough)
    assert stream_report.total_tensors == len(state)
    assert "experts.gate_up_proj" in stream_report.records
    assert "experts.down_proj" in stream_report.records
    assert stream_report.records["experts.gate_up_proj"].status == "quantized"
    assert stream_report.records["experts.down_proj"].status == "quantized"
    assert stream_report.records["experts.gate_up_proj"].codec == "adaptive_residual_vq"
    assert stream_report.records["experts.down_proj"].codec == "adaptive_residual_vq"

    # Router gate must remain passthrough
    assert stream_report.records["gate.weight"].status == "passthrough"

    # 4. Ingest streamed checkpoint into a clean model instance
    clean_model = copy.deepcopy(orig_model)
    clean_model.eval()
    loaded_model = load_streamed_checkpoint_into_model(clean_model, stream_out_dir, device="cpu")

    # Verify structural restoration of 3D expert bank
    assert isinstance(loaded_model.experts, ResidualVQFusedExperts)
    assert isinstance(loaded_model.experts.gate_up_proj, ResidualVQLinear)
    assert isinstance(loaded_model.experts.down_proj, ResidualVQLinear)
    assert isinstance(loaded_model.attn.q_proj, ResidualVQLinear)

    # 5. Forward outputs comparison
    x = torch.randn(4, 48)
    inplace_out = inplace_model(x)
    loaded_out = loaded_model(x)

    assert loaded_out.shape == (4, 48)
    assert torch.isfinite(loaded_out).all()
    assert torch.isfinite(inplace_out).all()

    # The forward pass outputs must closely match the in-place model
    cos_sim = F.cosine_similarity(inplace_out.flatten(), loaded_out.flatten(), dim=0).item()
    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.5f} is below 0.999"


def test_smooth_channels_conditioning_with_custom_moe():
    """Verify smooth channel conditioning runs on custom 3D expert model."""
    torch.manual_seed(20261105)
    model = FullNovelMoEBlock(hidden_dim=48, intermediate_dim=48, num_experts=4)

    cfg = UniversalCompressionConfig(
        smooth_channels=True,
        smooth_alpha=0.5,
        mode="adaptive_residual_vq",
        vq_config=QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=2,
            require_compression=False,
            max_relative_error=None,
        ),
    )
    engine = UniversalCompressionEngine(cfg)
    comp_model, report = engine.compress_module_in_place(model)

    assert isinstance(comp_model.experts, ResidualVQFusedExperts)
    x = torch.randn(2, 48)
    out = comp_model(x)
    assert out.shape == (2, 48)
    assert torch.isfinite(out).all()


def test_allow_3d_experts_toggle():
    """Verify that allow_3d_experts=False skips 3D expert tensors."""
    assert should_quantize_tensor_name(
        "experts.gate_up_proj",
        (4, 96, 48),
        allow_3d_experts=True,
    ) is True

    assert should_quantize_tensor_name(
        "experts.gate_up_proj",
        (4, 96, 48),
        allow_3d_experts=False,
    ) is False

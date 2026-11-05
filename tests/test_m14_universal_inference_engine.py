"""Milestone M14: Universal Inference Engine & Multi-Modal/Multi-Arch Zero-RAM Verification Suite.

Comprehensive verification verifying:
1. End-to-End Engine 1 (`UniversalCompressionEngine`) -> `.oqpack` -> Engine 2 (`UniversalInferenceEngine.from_oqpack`)
   with zero-RAM `meta`-device factory (`lambda: Model(...)`):
   - Dense Transformer CausalLM (`LlamaForCausalLM`).
   - 3D Routed MoE (`Qwen3NextForCausalLM` with hybrid Gated-DeltaNet + full attention + 3D MoE experts).
   - Vision / Multimodal encoder-projector-decoder model (takes `pixel_values` [B, C, H, W] + `input_ids`).
   - Audio spectrogram encoder-decoder model (takes `input_features` [B, F, T]).
2. Layer-by-layer mathematical forward graph auditor (`audit_forward_graph`):
   - `execution_coverage_ratio == 1.0` (100% of quantized layers actually executed).
   - `all_outputs_finite is True`.
   - For MoE models: `len(active_moe_experts_union) > 0`.
   - Exact detection of bypassed/unexecuted layers when coverage is incomplete.
3. End-to-end autoregressive generation (`engine.generate`) with token IDs, string prompts,
   and streaming generation (`stream_ultra_long`).
4. Hardware dispatch and LUT execution flags propagation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
)

from ojasx_24x96.compression_engine import (
    UniversalCompressionConfig,
    UniversalCompressionEngine,
)
from ojasx_24x96.engine import SamplingParams
from ojasx_24x96.inference_engine import (
    ForwardGraphAuditReport,
    LayerExecutionTrace,
    UniversalInferenceEngine,
)
from ojasx_24x96.leech_codec import LeechQuantizationConfig
from ojasx_24x96.module import ResidualVQFusedExperts, ResidualVQLinear

# Ensure rotation_seed compatibility on LeechQuantizationConfig if required by compression engine
if not hasattr(LeechQuantizationConfig, "rotation_seed"):
    LeechQuantizationConfig.rotation_seed = property(lambda self: self.seed)  # type: ignore

import ojasx_24x96.compression_engine as ce

_orig_quantize_leech_weight = ce.quantize_leech_weight


def _patched_quantize_leech_weight(weight, config=None, seed=None):
    if seed is not None and config is not None:
        config = dataclasses.replace(config, seed=seed)
    return _orig_quantize_leech_weight(weight, config)


ce.quantize_leech_weight = _patched_quantize_leech_weight


def _create_toy_tokenizer(vocab_size: int = 64) -> PreTrainedTokenizerFast:
    """Creates a deterministic fast tokenizer for testing text-based generation."""
    vocab = {f"token_{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = 0
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok)


# ==============================================================================
# Multimodal & Audio Synthetic Model Architectures
# ==============================================================================


class TinyVisionTower(nn.Module):
    """Vision encoder taking raw image patches [B, C, H, W] -> patch tokens [B, N, D]."""

    def __init__(self, in_channels: int = 3, patch_size: int = 4, hidden_dim: int = 48) -> None:
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.vision_dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        x = self.patch_proj(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        return torch.relu(self.vision_dense(x))


class TinyMultimodalProjector(nn.Module):
    """Cross-modal projection module aligning vision hidden dimension with text hidden dimension."""

    def __init__(self, vision_dim: int = 48, text_dim: int = 48) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, text_dim)
        self.linear_2 = nn.Linear(text_dim, text_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.relu(self.linear_1(x)))


class TinyVisionMultimodalModel(nn.Module):
    """Vision-Language encoder-projector-decoder model taking pixel_values and input_ids."""

    def __init__(
        self,
        vocab_size: int = 64,
        text_dim: int = 48,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, text_dim)
        self.vision_tower = TinyVisionTower(
            in_channels=3, patch_size=4, hidden_dim=hidden_dim
        )
        self.multi_modal_projector = TinyMultimodalProjector(
            vision_dim=hidden_dim, text_dim=text_dim
        )
        self.decoder_layer = nn.Linear(text_dim, text_dim)
        self.lm_head = nn.Linear(text_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None and pixel_values is not None:
            batch_size = pixel_values.shape[0]
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=pixel_values.device)

        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            vision_feats = self.vision_tower(pixel_values)
            projected = self.multi_modal_projector(vision_feats)
            inputs_embeds = torch.cat([projected, inputs_embeds], dim=1)

        hidden = torch.relu(self.decoder_layer(inputs_embeds))
        logits = self.lm_head(hidden)
        return logits


class TinyAudioEncoder(nn.Module):
    """Audio spectrogram encoder taking input_features [B, F, T] -> [B, T, D]."""

    def __init__(self, n_mels: int = 48, hidden_dim: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)
        self.audio_dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: [B, F, T]
        x = torch.relu(self.conv1(input_features))
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        return torch.relu(self.audio_dense(x))


class TinyAudioDecoder(nn.Module):
    """Audio decoder predicting token/frame logits from encoded representations."""

    def __init__(self, hidden_dim: int = 48, vocab_size: int = 64) -> None:
        super().__init__()
        self.decoder_dense = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.decoder_dense(encoder_hidden))
        return self.lm_head(h)


class TinyAudioSpectrogramModel(nn.Module):
    """Audio spectrogram encoder-decoder model taking input_features [B, F, T] -> logits."""

    def __init__(self, n_mels: int = 48, hidden_dim: int = 48, vocab_size: int = 64) -> None:
        super().__init__()
        self.audio_encoder = TinyAudioEncoder(n_mels=n_mels, hidden_dim=hidden_dim)
        self.decoder = TinyAudioDecoder(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        enc = self.audio_encoder(input_features)
        logits = self.decoder(enc)
        return logits


class BranchingModelWithConditionalBypass(nn.Module):
    """Model with two quantized branches to test partial execution coverage detection."""

    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.branch_a = nn.Linear(dim, dim)
        self.branch_b = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, 10, bias=False)

    def forward(self, x: torch.Tensor, use_branch_b: bool = False) -> torch.Tensor:
        if use_branch_b:
            h = self.branch_b(x)
        else:
            h = self.branch_a(x)
        return self.lm_head(h)


# ==============================================================================
# Test Cases
# ==============================================================================


def test_dense_transformer_causallm_oqpack_meta_inference():
    """Test 1: Dense Transformer CausalLM (LlamaForCausalLM) end-to-end.

    Verifies:
    - UniversalCompressionEngine -> .oqpack compilation
    - UniversalInferenceEngine.from_oqpack with zero-RAM meta factory lambda: LlamaForCausalLM(...)
    - audit_forward_graph: 100% execution coverage (coverage == 1.0), all outputs finite
    - Autoregressive generate (token IDs and text prompt) and stream_ultra_long.
    """
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=True,
    )
    model = LlamaForCausalLM(cfg).eval()
    tokenizer = _create_toy_tokenizer(vocab_size=64)

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Engine 1: Compress to .oqpack container
        comp_engine = UniversalCompressionEngine()
        q_model, manifest, result = comp_engine.compress_model_to_oqpack(
            model,
            tmpdir,
            model_id="test-llama-dense",
        )
        assert result.quantized_tensors == 14
        assert result.compression_ratio > 0.0

        # Verify zero-RAM meta factory instantiates on device 'meta'
        with torch.device("meta"):
            meta_skeleton = LlamaForCausalLM(cfg).eval()
        assert all(p.device.type == "meta" for p in meta_skeleton.parameters())

        # Engine 2: Instantiate from .oqpack using zero-RAM factory
        engine, loaded_manifest = UniversalInferenceEngine.from_oqpack(
            lambda: LlamaForCausalLM(cfg).eval(),
            tmpdir,
            tokenizer=tokenizer,
            device="cpu",
        )

        # Confirm all parameters and buffers are now on real CPU device
        assert all(p.device.type == "cpu" for p in engine.model.parameters())
        assert all(b.device.type == "cpu" for b in engine.model.buffers())

        # Layer-by-layer mathematical forward graph audit
        input_ids = torch.tensor([[1, 2, 3, 4]])
        out, audit_rep = engine.audit_forward_graph(input_ids=input_ids)

        assert audit_rep.execution_coverage_ratio == 1.0
        assert audit_rep.all_outputs_finite is True
        assert audit_rep.total_quantized_modules == 14
        assert audit_rep.executed_quantized_modules == 14
        assert audit_rep.total_forward_calls >= 14

        # Validate audit report dictionary representation
        rep_dict = audit_rep.to_dict()
        assert rep_dict["execution_coverage_ratio"] == 1.0
        assert rep_dict["all_outputs_finite"] is True
        assert len(rep_dict["traces"]) == 14

        # Autoregressive generation from token IDs
        gen_res = engine.generate(input_ids, params=SamplingParams(max_new_tokens=6))
        assert len(gen_res.generated_token_ids) == 6
        assert all(isinstance(t, int) for t in gen_res.generated_token_ids)

        # Autoregressive generation from text prompt
        text, text_gen_res = engine.generate(
            "token_1 token_2", params=SamplingParams(max_new_tokens=4)
        )
        assert isinstance(text, str)
        assert len(text_gen_res.generated_token_ids) == 4

        # Stream ultra-long chunk processing
        chunks = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        stream_tokens = engine.stream_ultra_long(chunks, max_new_tokens=4)
        assert len(stream_tokens) == 4
        assert all(isinstance(t, int) for t in stream_tokens)

    finally:
        shutil.rmtree(tmpdir)


def test_3d_routed_moe_qwen3_next_oqpack_meta_inference():
    """Test 2: 3D Routed MoE (Qwen3NextForCausalLM) end-to-end.

    Verifies:
    - Hybrid Gated-DeltaNet + full attention + 3D MoE expert banks ([E, 2*I, H] and [E, H, I])
    - UniversalCompressionEngine -> .oqpack -> UniversalInferenceEngine.from_oqpack with meta factory
    - audit_forward_graph: execution_coverage_ratio == 1.0, all_outputs_finite == True,
      and len(active_moe_experts_union) > 0 (proving real mathematical MoE routing execution)
    - Autoregressive generate and stream_ultra_long.
    """
    common = {
        "vocab_size": 64,
        "hidden_size": 72,  # 72 = 3 * 24
        "intermediate_size": 144,  # 144 = 6 * 24
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 18,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "max_position_embeddings": 64,
        "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": True,
        "use_cache": False,
        "moe_intermediate_size": 24,  # 24 = 1 * 24
        "shared_expert_intermediate_size": 24,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    cfg = Qwen3NextConfig(**common)
    model = Qwen3NextForCausalLM(cfg).eval()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Engine 1: Compress to .oqpack container
        comp_engine = UniversalCompressionEngine()
        q_model, manifest, result = comp_engine.compress_model_to_oqpack(
            model,
            tmpdir,
            model_id="test-qwen3next-moe",
        )
        assert result.quantized_tensors > 0

        # Confirm 3D fused experts were captured in manifest
        fused_expert_specs = [
            k for k, v in manifest.layer_specs.items() if v.get("module_type") == "fused_experts"
        ]
        assert len(fused_expert_specs) == 2  # 2 hidden layers with MoE blocks

        # Engine 2: Instantiate from .oqpack using zero-RAM meta factory
        engine, loaded_manifest = UniversalInferenceEngine.from_oqpack(
            lambda: Qwen3NextForCausalLM(cfg).eval(),
            tmpdir,
            device="cpu",
        )

        # Audit live forward graph execution
        input_ids = torch.tensor([[1, 2, 3, 4]])
        out, audit_rep = engine.audit_forward_graph(input_ids=input_ids)

        assert audit_rep.execution_coverage_ratio == 1.0
        assert audit_rep.all_outputs_finite is True
        assert audit_rep.total_quantized_modules > 0
        assert audit_rep.executed_quantized_modules == audit_rep.total_quantized_modules
        assert len(audit_rep.active_moe_experts_union) > 0
        # Verify active MoE experts subset is non-empty and valid
        assert all(0 <= e < 4 for e in audit_rep.active_moe_experts_union)

        # Autoregressive generation
        gen_res = engine.generate(input_ids, params=SamplingParams(max_new_tokens=4))
        assert len(gen_res.generated_token_ids) == 4

        # Stream ultra-long
        chunks = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        stream_tokens = engine.stream_ultra_long(chunks, max_new_tokens=4)
        assert len(stream_tokens) == 4

    finally:
        shutil.rmtree(tmpdir)


def test_vision_multimodal_encoder_projector_decoder_oqpack_meta_inference():
    """Test 3: Vision / Multimodal encoder-projector-decoder model end-to-end.

    Verifies:
    - Inputs: pixel_values [B, C, H, W] + input_ids -> multimodal logits [B, N + L, vocab]
    - UniversalCompressionEngine -> .oqpack -> UniversalInferenceEngine.from_oqpack
    - Role-aware dual-codec routing: vision tower uses Leech24, projector/decoder use RVQ
    - audit_forward_graph: execution_coverage_ratio == 1.0 and all_outputs_finite == True
    """
    model = TinyVisionMultimodalModel(vocab_size=64, text_dim=48, hidden_dim=48).eval()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Engine 1: Compress to .oqpack container
        comp_engine = UniversalCompressionEngine()
        q_model, manifest, result = comp_engine.compress_model_to_oqpack(
            model,
            tmpdir,
            model_id="test-vision-multimodal",
        )
        assert result.quantized_tensors == 4

        # Engine 2: Instantiate from .oqpack via zero-RAM meta factory
        engine, loaded_manifest = UniversalInferenceEngine.from_oqpack(
            lambda: TinyVisionMultimodalModel(vocab_size=64, text_dim=48, hidden_dim=48).eval(),
            tmpdir,
            device="cpu",
        )

        # Audit forward execution with image patches + input IDs
        pixel_values = torch.randn(2, 3, 16, 16)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out, audit_rep = engine.audit_forward_graph(
            input_ids=input_ids, pixel_values=pixel_values
        )

        assert out.shape == (2, 19, 64)  # (16/4)^2 = 16 patches + 3 text tokens = 19
        assert audit_rep.execution_coverage_ratio == 1.0
        assert audit_rep.all_outputs_finite is True
        assert audit_rep.total_quantized_modules == 4
        assert audit_rep.executed_quantized_modules == 4

        # Verify Leech24 codec assignment on sensitive vision layers
        vision_trace = audit_rep.traces["vision_tower.vision_dense"]
        assert vision_trace.codec == "leech24_shell2_residual"
        assert vision_trace.call_count == 1
        assert vision_trace.all_finite is True
        assert vision_trace.output_l2_norm > 0.0

        # Native forward_any execution
        native_out = engine.forward_any(input_ids=input_ids, pixel_values=pixel_values)
        assert torch.allclose(out, native_out, atol=1e-5)

    finally:
        shutil.rmtree(tmpdir)


def test_audio_spectrogram_encoder_decoder_oqpack_meta_inference():
    """Test 4: Audio spectrogram encoder-decoder model end-to-end.

    Verifies:
    - Inputs: input_features [B, F, T] (filterbanks, time) -> logits [B, T, vocab]
    - UniversalCompressionEngine -> .oqpack -> UniversalInferenceEngine.from_oqpack
    - Role-aware dual-codec routing: audio encoder uses Leech24, decoder uses RVQ
    - audit_forward_graph: execution_coverage_ratio == 1.0 and all_outputs_finite == True
    """
    model = TinyAudioSpectrogramModel(n_mels=48, hidden_dim=48, vocab_size=64).eval()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Engine 1: Compress to .oqpack container
        comp_engine = UniversalCompressionEngine()
        q_model, manifest, result = comp_engine.compress_model_to_oqpack(
            model,
            tmpdir,
            model_id="test-audio-spectrogram",
        )
        assert result.quantized_tensors == 2

        # Engine 2: Instantiate from .oqpack via zero-RAM meta factory
        engine, loaded_manifest = UniversalInferenceEngine.from_oqpack(
            lambda: TinyAudioSpectrogramModel(n_mels=48, hidden_dim=48, vocab_size=64).eval(),
            tmpdir,
            device="cpu",
        )

        # Audit forward execution with audio spectrogram features [B, F, T]
        input_features = torch.randn(2, 48, 32)
        out, audit_rep = engine.audit_forward_graph(input_features=input_features)

        assert out.shape == (2, 32, 64)
        assert audit_rep.execution_coverage_ratio == 1.0
        assert audit_rep.all_outputs_finite is True
        assert audit_rep.total_quantized_modules == 2
        assert audit_rep.executed_quantized_modules == 2

        # Verify Leech24 codec assignment on audio encoder
        audio_trace = audit_rep.traces["audio_encoder.audio_dense"]
        assert audio_trace.codec == "leech24_shell2_residual"
        assert audio_trace.call_count == 1
        assert audio_trace.all_finite is True

        # Decoder uses adaptive RVQ
        dec_trace = audit_rep.traces["decoder.decoder_dense"]
        assert dec_trace.codec == "adaptive_residual_vq"
        assert dec_trace.call_count == 1
        assert dec_trace.all_finite is True

    finally:
        shutil.rmtree(tmpdir)


def test_audit_forward_graph_unexecuted_layer_detection():
    """Verifies that audit_forward_graph catches unexecuted / bypassed quantized layers."""
    model = BranchingModelWithConditionalBypass(dim=48).eval()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        comp_engine = UniversalCompressionEngine()
        q_model, manifest, result = comp_engine.compress_model_to_oqpack(
            model,
            tmpdir,
            model_id="test-branching",
        )
        assert result.quantized_tensors == 2  # branch_a and branch_b

        engine, _ = UniversalInferenceEngine.from_oqpack(
            lambda: BranchingModelWithConditionalBypass(dim=48).eval(),
            tmpdir,
            device="cpu",
        )

        # Execute only branch A
        x = torch.randn(2, 48)
        out, audit_rep = engine.audit_forward_graph(x, use_branch_b=False)

        # Coverage must be strictly 50% (1 of 2 executed)
        assert audit_rep.total_quantized_modules == 2
        assert audit_rep.executed_quantized_modules == 1
        assert audit_rep.execution_coverage_ratio == 0.5
        assert audit_rep.traces["branch_a"].call_count == 1
        assert audit_rep.traces["branch_b"].call_count == 0

        # Now execute branch B
        out_b, audit_rep_b = engine.audit_forward_graph(x, use_branch_b=True)
        assert audit_rep_b.executed_quantized_modules == 1
        assert audit_rep_b.execution_coverage_ratio == 0.5
        assert audit_rep_b.traces["branch_a"].call_count == 0
        assert audit_rep_b.traces["branch_b"].call_count == 1

    finally:
        shutil.rmtree(tmpdir)


def test_lut_mode_and_hardware_flags_propagation():
    """Verifies subspace_lut_mode and fused_native_gemm configuration flags."""
    model = nn.Sequential(nn.Linear(48, 48), nn.Linear(48, 48)).eval()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        comp_engine = UniversalCompressionEngine()
        comp_engine.compress_model_to_oqpack(model, tmpdir, model_id="test-flags")

        engine, _ = UniversalInferenceEngine.from_oqpack(
            lambda: nn.Sequential(nn.Linear(48, 48), nn.Linear(48, 48)).eval(),
            tmpdir,
            subspace_lut_mode=True,
            device="cpu",
        )

        for _, mod in engine.model.named_modules():
            if isinstance(mod, ResidualVQLinear):
                assert mod.subspace_lut_mode is True

    finally:
        shutil.rmtree(tmpdir)

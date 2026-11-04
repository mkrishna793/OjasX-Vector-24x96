"""Comprehensive verification suite for Canonical M1 OQPack Container (oqpack.v1).

Tests:
1. Multi-shard roundtrip with adaptive_residual_vq (bit-identical output verification).
2. Multi-shard roundtrip with leech24_shell2_residual (bit-identical output verification).
3. Fused expert bank roundtrip through OQPack.
4. Tokenizer files and chat_template packaging.
5. Zero-copy mmap tensor dictionary access.
6. Negative security and integrity failure modes:
   - Byte-level corruption in safetensors shard
   - Missing shard file
   - Unsupported format version
   - Tampered tensor shape
   - Tampered tensor dtype
   - Tampered shard size
   - Missing declared tensor
   - Missing declared tokenizer file
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from ojasx_24x96.codec import QuantizationConfig, quantize_weight
from ojasx_24x96.contract import introspect_architecture
from ojasx_24x96.leech_codec import LeechQuantizationConfig, quantize_leech_weight
from ojasx_24x96.module import (
    ResidualVQConv1D,
    ResidualVQFusedExperts,
    ResidualVQLinear,
)
from ojasx_24x96.oqpack import (
    MANIFEST_FILENAME,
    OQPACK_FORMAT_VERSION,
    OQPackValidationError,
    load_quantized_model_from_oqpack,
    open_oqpack_mmap,
    save_oqpack,
    verify_oqpack_integrity,
)


class MultiLayerModel(nn.Module):
    """Synthetic model mixing quantized linear layers and dense components."""

    def __init__(self, in_features: int = 32, hidden: int = 32, out_features: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.norm(h + self.fc2(h))
        return self.head(h)


class LeechModel(nn.Module):
    """Model with dimensions compatible with Leech24 block size (multiples of 24)."""

    def __init__(self, dim: int = 48, out_features: int = 24):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, out_features, bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(torch.relu(self.fc1(x)))
        return self.fc2(h)


def test_adaptive_residual_vq_roundtrip_multi_shard():
    """Verify bit-identical outputs and multi-shard splitting for adaptive_residual_vq."""
    torch.manual_seed(20261104)
    model = MultiLayerModel(in_features=32, hidden=32, out_features=16)

    # Quantize fc1 and fc2 with adaptive residual VQ
    cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=1.0,
    )
    enc1, _ = quantize_weight(model.fc1.weight, cfg)
    enc2, _ = quantize_weight(model.fc2.weight, cfg)

    model.fc1 = ResidualVQLinear.from_encoded(
        enc1, bias=model.fc1.bias, source_dtype=torch.float32, device=torch.device("cpu")
    )
    model.fc2 = ResidualVQLinear.from_encoded(
        enc2, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
    )
    model.eval()

    inputs = torch.randn(4, 32)
    expected_outputs = model(inputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "adaptive_vq_pkg"

        # Force multiple shards with small max_shard_bytes=2048
        manifest = save_oqpack(
            model,
            package_dir,
            model_id="test-adaptive-vq",
            max_shard_bytes=2048,
        )

        assert manifest.format_version == OQPACK_FORMAT_VERSION
        assert len(manifest.shards) >= 2, f"Expected multiple shards, got {len(manifest.shards)}"
        assert manifest.model_id == "test-adaptive-vq"
        assert "fc1" in manifest.layer_specs
        assert "fc2" in manifest.layer_specs
        assert manifest.layer_specs["fc1"]["codec"] == "adaptive_residual_vq"

        # Check shard file naming convention
        for i, shard in enumerate(manifest.shards):
            expected_name = f"shard-{i + 1:05d}-of-{len(manifest.shards):05d}.safetensors"
            assert shard.filename == expected_name
            assert (package_dir / shard.filename).is_file()

        # Cryptographic and structural verification
        verified_manifest = verify_oqpack_integrity(package_dir, verify_checksums=True)
        assert verified_manifest.format_version == OQPACK_FORMAT_VERSION

        # Reconstruction into empty skeleton model
        skeleton = MultiLayerModel(in_features=32, hidden=32, out_features=16)
        restored_model, _ = load_quantized_model_from_oqpack(
            skeleton,
            package_dir,
            streaming_decode=True,
            weight_cache=False,
        )
        restored_outputs = restored_model(inputs)

        # Bit-identical assertion
        assert torch.equal(restored_outputs, expected_outputs), "Forward output changed after roundtrip"

        # Also test with weight_cache=True (speed mode)
        restored_model, _ = load_quantized_model_from_oqpack(
            skeleton,
            package_dir,
            streaming_decode=False,
            weight_cache=True,
        )
        cached_outputs = restored_model(inputs)
        assert torch.equal(cached_outputs, expected_outputs), "Cached speed mode output mismatch"


def test_leech24_shell2_residual_roundtrip_multi_shard():
    """Verify bit-identical outputs and multi-shard splitting for leech24_shell2_residual."""
    torch.manual_seed(20261104)
    model = LeechModel(dim=96, out_features=72)

    cfg = LeechQuantizationConfig(
        max_stages=2,
        max_relative_error=1.0,
        gain_bits=1,
        rotation_reflections=2,
    )
    enc1, _ = quantize_leech_weight(model.fc1.weight, cfg)
    enc2, _ = quantize_leech_weight(model.fc2.weight, cfg)

    model.fc1 = ResidualVQLinear.from_encoded(
        enc1, bias=model.fc1.bias, source_dtype=torch.float32, device=torch.device("cpu")
    )
    model.fc2 = ResidualVQLinear.from_encoded(
        enc2, bias=model.fc2.bias, source_dtype=torch.float32, device=torch.device("cpu")
    )
    model.eval()

    inputs = torch.randn(4, 96)
    expected_outputs = model(inputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "leech_pkg"

        # Force multi-shard splitting with max_shard_bytes=2048
        manifest = save_oqpack(
            model,
            package_dir,
            model_id="test-leech24-model",
            max_shard_bytes=2048,
        )

        assert manifest.format_version == OQPACK_FORMAT_VERSION
        assert len(manifest.shards) >= 2, f"Expected multiple shards, got {len(manifest.shards)}"
        assert manifest.layer_specs["fc1"]["codec"] == "leech24_shell2_residual"
        assert manifest.layer_specs["fc1"]["active_counts"] is not None

        # Verify integrity
        verify_oqpack_integrity(package_dir, verify_checksums=True)

        # Restore into fresh skeleton
        skeleton = LeechModel(dim=96, out_features=72)
        restored_model, _ = load_quantized_model_from_oqpack(skeleton, package_dir)
        restored_outputs = restored_model(inputs)

        # Bit-identical assertion
        assert torch.equal(restored_outputs, expected_outputs), "Leech24 forward output changed"


def test_fused_experts_roundtrip():
    """Verify OQPack packaging and direct reconstruction of ResidualVQFusedExperts."""
    torch.manual_seed(20261104)
    num_experts = 4
    hidden_dim = 16
    intermediate_dim = 32

    w_gate_up = torch.randn(num_experts * 2 * intermediate_dim, hidden_dim)
    cfg = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, max_relative_error=1.0)
    enc_gate_up, _ = quantize_weight(w_gate_up, cfg)
    gate_up_proj = ResidualVQLinear.from_encoded(
        enc_gate_up, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
    )

    w_down = torch.randn(num_experts * hidden_dim, intermediate_dim)
    enc_down, _ = quantize_weight(w_down, cfg)
    down_proj = ResidualVQLinear.from_encoded(
        enc_down, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
    )

    class ExpertModel(nn.Module):
        def __init__(self, g_proj, d_proj):
            super().__init__()
            self.experts = ResidualVQFusedExperts(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                gate_up_proj=g_proj,
                down_proj=d_proj,
                act_fn=nn.SiLU(),
            )

        def forward(self, h, top_k_idx, top_k_wt):
            return self.experts(h, top_k_idx, top_k_wt)

    model = ExpertModel(gate_up_proj, down_proj)
    model.eval()

    h = torch.randn(2, hidden_dim)
    top_k_idx = torch.tensor([[0, 1], [2, 3]])
    top_k_wt = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    expected_out = model(h, top_k_idx, top_k_wt)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "expert_pkg"
        save_oqpack(model, package_dir)

        # Skeleton with dummy dense layers
        dummy_g = nn.Linear(hidden_dim, 2 * intermediate_dim)
        dummy_d = nn.Linear(intermediate_dim, hidden_dim)
        skel = ExpertModel(dummy_g, dummy_d)

        restored, manifest = load_quantized_model_from_oqpack(skel, package_dir)
        restored_out = restored(h, top_k_idx, top_k_wt)

        assert "experts" in manifest.layer_specs
        assert manifest.layer_specs["experts"]["module_type"] == "fused_experts"
        assert torch.equal(restored_out, expected_out)


def test_tokenizer_and_mmap_tensor_access():
    """Verify tokenizer files, chat template, and zero-copy mmap tensor dictionary."""
    torch.manual_seed(20261104)
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = Path(tmpdir) / "dummy_tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text('{"dummy": true}', encoding="utf-8")
        (tok_dir / "tokenizer_config.json").write_text('{"bos_token": "<s>"}', encoding="utf-8")
        (tok_dir / "special_tokens_map.json").write_text('{"bos_token": "<s>"}', encoding="utf-8")

        chat_tmpl = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        package_dir = Path(tmpdir) / "pkg_with_tok"

        manifest = save_oqpack(
            model,
            package_dir,
            tokenizer_dir=tok_dir,
            chat_template=chat_tmpl,
        )

        assert "tokenizer.json" in manifest.tokenizer_files
        assert "tokenizer_config.json" in manifest.tokenizer_files
        assert "special_tokens_map.json" in manifest.tokenizer_files
        assert "chat_template.jinja" in manifest.tokenizer_files
        assert manifest.chat_template == chat_tmpl

        assert (package_dir / "tokenizer.json").is_file()
        assert (package_dir / "chat_template.jinja").read_text(encoding="utf-8") == chat_tmpl

        # Open mmap tensors
        mmap_tensors = open_oqpack_mmap(package_dir, device="cpu", verify_checksums=True)
        assert len(mmap_tensors) == len(manifest.tensors)
        for name, tensor in model.state_dict().items():
            assert name in mmap_tensors
            assert torch.equal(mmap_tensors[name], tensor)


def test_corrupt_shard_byte_raises_validation_error():
    """Negative test: flipping a single byte in any shard must fail verification."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        manifest = save_oqpack(model, package_dir, max_shard_bytes=256)
        assert len(manifest.shards) >= 2

        # Corrupt one byte in the first shard
        shard_path = package_dir / manifest.shards[0].filename
        data = bytearray(shard_path.read_bytes())
        data[20] ^= 0xAA
        shard_path.write_bytes(data)

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir, verify_checksums=True)
        assert "checksum mismatch" in str(exc_info.value) or "Corrupt" in str(exc_info.value)


def test_missing_shard_file_raises_validation_error():
    """Negative test: unlinking a shard file must raise OQPackValidationError."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        manifest = save_oqpack(model, package_dir, max_shard_bytes=256)

        # Unlink one shard
        shard_path = package_dir / manifest.shards[1].filename
        shard_path.unlink()

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "missing" in str(exc_info.value).lower()


def test_unsupported_format_version_raises_validation_error():
    """Negative test: unsupported manifest format_version must fail validation."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        save_oqpack(model, package_dir)

        manifest_file = package_dir / MANIFEST_FILENAME
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        payload["format_version"] = "oqpack.v999"
        manifest_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "Unsupported format_version" in str(exc_info.value)


def test_tampered_tensor_shape_raises_validation_error():
    """Negative test: modifying tensor shape in manifest must fail validation."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        save_oqpack(model, package_dir)

        manifest_file = package_dir / MANIFEST_FILENAME
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        first_tensor_name = list(payload["tensors"].keys())[0]
        payload["tensors"][first_tensor_name]["shape"] = [999, 999]
        manifest_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "shape mismatch" in str(exc_info.value).lower()


def test_tampered_tensor_dtype_raises_validation_error():
    """Negative test: modifying tensor dtype in manifest must fail validation."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        save_oqpack(model, package_dir)

        manifest_file = package_dir / MANIFEST_FILENAME
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        first_tensor_name = list(payload["tensors"].keys())[0]
        payload["tensors"][first_tensor_name]["dtype"] = "int8"
        manifest_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "dtype mismatch" in str(exc_info.value).lower()


def test_tampered_shard_size_raises_validation_error():
    """Negative test: modifying shard size in manifest must fail validation."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "pkg"
        save_oqpack(model, package_dir)

        manifest_file = package_dir / MANIFEST_FILENAME
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        payload["shards"][0]["size_bytes"] = 123456789
        manifest_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "size mismatch" in str(exc_info.value).lower()


def test_conv1d_roundtrip():
    """Verify OQPack roundtrip with ResidualVQConv1D layers."""
    torch.manual_seed(20261104)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(32, 32)
            self.c2 = nn.Linear(32, 16)

        def forward(self, x):
            return self.c2(torch.relu(self.c1(x)))

    model = ConvNet()
    cfg = QuantizationConfig(block_size=8, bits_per_stage=4, max_stages=2, max_relative_error=1.0)
    enc, _ = quantize_weight(model.c1.weight.T, cfg)
    model.c1 = ResidualVQConv1D.from_encoded(
        enc, bias=model.c1.bias, source_dtype=torch.float32, device=torch.device("cpu")
    )
    model.eval()

    x = torch.randn(2, 32)
    expected_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = Path(tmpdir) / "conv_pkg"
        manifest = save_oqpack(model, package_dir)
        assert manifest.layer_specs["c1"]["module_type"] == "conv1d"

        skel = ConvNet()
        restored, _ = load_quantized_model_from_oqpack(skel, package_dir)
        restored_out = restored(x)
        assert torch.equal(restored_out, expected_out)


def test_missing_tokenizer_file_raises_validation_error():
    """Negative test: declared tokenizer file removed from package directory must fail verification."""
    model = MultiLayerModel(in_features=16, hidden=16, out_features=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = Path(tmpdir) / "tok"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

        package_dir = Path(tmpdir) / "pkg"
        save_oqpack(model, package_dir, tokenizer_dir=tok_dir)

        # Delete tokenizer.json from package
        (package_dir / "tokenizer.json").unlink()

        with pytest.raises(OQPackValidationError) as exc_info:
            verify_oqpack_integrity(package_dir)
        assert "tokenizer file missing" in str(exc_info.value).lower()


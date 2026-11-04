"""Unit and conformance tests for Milestone M6: Stable Backend C ABI & llama.cpp Adapter."""

import ctypes
import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx_24x96.backend_abi import (
    OJASX_BACKEND_ABI_VERSION,
    OJASX_LEECH24_DIM,
    OJASX_MAX_STAGES,
    CBackendVTable,
    CLeech24LayerDesc,
    CRVQLayerDesc,
    LlamaCppGGMLAdapter,
    OjasxStatus,
    abi_header_status,
    build_c_layer_descriptor,
    run_backend_conformance_suite,
)
from ojasx_24x96.codec import QuantizationConfig, dequantize_weight, quantize_weight
from ojasx_24x96.leech_codec import (
    LeechQuantizationConfig,
    dequantize_leech_weight,
    quantize_leech_weight,
)
from ojasx_24x96.leech_native import _load_library, native_backend_status
from ojasx_24x96.module import ResidualVQLinear
from ojasx_24x96.oqpack import save_oqpack


def _skip_if_no_native():
    status = native_backend_status()
    if not status.get("available", False):
        pytest.skip(f"Native backend unavailable: {status.get('reason')}")


class TestBackendABIHeader:
    """Validate ojasx_backend_abi.h integrity, constants, and struct layouts."""

    def test_abi_header_presence_and_version(self):
        status = abi_header_status()
        assert status["exists"] is True
        assert status["verified"] is True
        assert status["abi_version"] == 1
        assert status["expected_abi_version"] == 1
        assert status["matches_python_version"] is True
        assert isinstance(status["sha256"], str)
        assert len(status["sha256"]) == 64

    def test_abi_constants_and_status_enum(self):
        assert OJASX_BACKEND_ABI_VERSION == 1
        assert OJASX_LEECH24_DIM == 24
        assert OJASX_MAX_STAGES == 8

        assert OjasxStatus.OK == 0
        assert OjasxStatus.INVALID_ARGUMENT == -1
        assert OjasxStatus.NULL_POINTER == -2
        assert OjasxStatus.MISSING_PRESENCE == -3
        assert OjasxStatus.GAIN_OUT_OF_RANGE == -5
        assert OjasxStatus.DECODE_POINT_FAILED == -6
        assert OjasxStatus.ABI_MISMATCH == -7

    def test_ctypes_struct_sizes_and_alignments(self):
        # CLeech24LayerDesc: 27 pointers (216B) + 7 int32s (28B) -> 244B + 4B pad = 248B
        assert ctypes.sizeof(CLeech24LayerDesc) == 248
        # CRVQLayerDesc: 4 pointers (32B) + 5 int32s (20B) -> 52B + 4B pad = 56B
        assert ctypes.sizeof(CRVQLayerDesc) == 56
        # CBackendVTable: 1 uint32 (4B) + 1 char* (8B, 4B pad) + 5 fn ptrs (40B) -> 56B
        assert ctypes.sizeof(CBackendVTable) == 56


class TestBuildCLayerDescriptor:
    """Validate construction of C ABI layer descriptors holding raw pointers to contiguous buffers."""

    def test_build_c_descriptor_leech24(self):
        torch.manual_seed(20261104)
        rows, cols = 16, 70  # non-24 multiple
        weight = torch.randn(rows, cols)
        bias = torch.randn(rows)

        config = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            adaptive_stages=True,
            seed=42,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        desc, metadata = build_c_layer_descriptor(module)
        assert isinstance(desc, CLeech24LayerDesc)
        assert desc.rows == 16
        assert desc.columns == 70
        assert desc.stages == 2
        assert desc.gain_bits == 1
        assert desc.adaptive_stages == 1
        assert desc.rotation_reflections == 2
        assert desc.rotation_seed == 42

        # Pointers must be valid non-null ints
        assert desc.packed_indices[0] is not None
        assert desc.packed_indices[1] is not None
        assert desc.packed_gains[0] is not None
        assert desc.packed_gains[1] is not None
        # Stage 0 has no presence mask because all base vectors are present; stage 1 has presence
        assert desc.presence[0] is None
        assert desc.presence[1] is not None
        assert desc.gain_codebooks is not None
        assert desc.row_scales is not None
        assert desc.bias is not None

        # Unused stages must be null (represented as None in ctypes)
        for s in range(2, OJASX_MAX_STAGES):
            assert desc.packed_indices[s] is None
            assert desc.packed_gains[s] is None
            assert desc.presence[s] is None

        # Metadata validation
        assert metadata["codec"] == "OJASX_LEECH24"
        assert metadata["abi_version"] == OJASX_BACKEND_ABI_VERSION
        assert metadata["workspace_bytes"] == 0
        assert metadata["has_bias"] is True
        assert len(metadata["tensors"]) > 0

    def test_build_c_descriptor_rvq(self):
        torch.manual_seed(12345)
        rows, cols = 14, 30  # non-block-8 multiple
        weight = torch.randn(rows, cols)

        config = QuantizationConfig(
            block_size=8,
            bits_per_stage=4,
            max_stages=3,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=None,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        desc, metadata = build_c_layer_descriptor(module)
        assert isinstance(desc, CRVQLayerDesc)
        assert desc.rows == 14
        assert desc.columns == 30
        assert desc.stages == 3
        assert desc.bits_per_stage == 4
        assert desc.block_size == 8

        assert desc.packed_codes is not None
        assert desc.codebooks is not None
        assert desc.row_scales is not None
        assert desc.bias is None  # no bias

        assert metadata["codec"] == "OJASX_RVQ"
        assert metadata["abi_version"] == OJASX_BACKEND_ABI_VERSION
        assert metadata["workspace_bytes"] == 0
        assert metadata["has_bias"] is False

    def test_build_c_descriptor_from_encoded_weights(self):
        torch.manual_seed(999)
        w_leech = torch.randn(12, 48)
        enc_leech, _ = quantize_leech_weight(
            w_leech,
            LeechQuantizationConfig(
                max_stages=1,
                gain_bits=1,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        assert enc_leech is not None
        desc_l, meta_l = build_c_layer_descriptor(enc_leech)
        assert isinstance(desc_l, CLeech24LayerDesc)
        assert desc_l.rows == 12
        assert desc_l.columns == 48
        assert desc_l.bias is None

        w_rvq = torch.randn(10, 32)
        enc_rvq, _ = quantize_weight(
            w_rvq,
            QuantizationConfig(
                block_size=8,
                bits_per_stage=2,
                max_stages=2,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        assert enc_rvq is not None
        desc_r, meta_r = build_c_layer_descriptor(enc_rvq)
        assert isinstance(desc_r, CRVQLayerDesc)
        assert desc_r.rows == 10
        assert desc_r.columns == 32
        assert desc_r.bias is None

    def test_build_c_descriptor_invalid_type(self):
        with pytest.raises(TypeError, match="Unsupported module"):
            build_c_layer_descriptor(nn.Linear(10, 10))


class TestLlamaCppGGMLAdapter:
    """Validate llama.cpp / ggml custom operator manifest export and dispatch."""

    def test_export_manifest_from_module(self):
        torch.manual_seed(42)
        # Create a model with one Leech24 layer and one RVQ layer
        w1 = torch.randn(16, 70)
        enc1, _ = quantize_leech_weight(
            w1,
            LeechQuantizationConfig(
                max_stages=2,
                gain_bits=1,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        mod1 = ResidualVQLinear.from_encoded(
            enc1, bias=torch.randn(16), source_dtype=torch.float32, device=torch.device("cpu")
        )

        w2 = torch.randn(12, 30)
        enc2, _ = quantize_weight(
            w2,
            QuantizationConfig(
                block_size=8,
                bits_per_stage=4,
                max_stages=2,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        mod2 = ResidualVQLinear.from_encoded(
            enc2, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )

        model = nn.Sequential()
        model.add_module("attn_q", mod1)
        model.add_module("attn_k", mod2)
        model.model_id = "test_two_layer_model"

        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "ggml_adapter.json"
            manifest = LlamaCppGGMLAdapter.export_ggml_adapter_manifest(model, out_file)

            assert out_file.exists()
            assert manifest["format"] == "ggml_ojasx_adapter.v1"
            assert manifest["abi_version"] == OJASX_BACKEND_ABI_VERSION
            assert manifest["model_id"] == "test_two_layer_model"
            assert manifest["total_quantized_tensors"] == 2
            assert manifest["default_workspace_bytes"] == 0

            tensors = manifest["quantized_tensors"]
            assert "attn_q.weight" in tensors
            assert "attn_k.weight" in tensors

            q_spec = tensors["attn_q.weight"]
            assert q_spec["codec"] == "OJASX_LEECH24"
            assert q_spec["shape"] == [16, 70]
            assert q_spec["workspace_bytes"] == 0
            assert q_spec["has_bias"] is True

            k_spec = tensors["attn_k.weight"]
            assert k_spec["codec"] == "OJASX_RVQ"
            assert k_spec["shape"] == [12, 30]
            assert k_spec["workspace_bytes"] == 0
            assert k_spec["has_bias"] is False

    def test_export_manifest_from_oqpack_directory(self):
        torch.manual_seed(101)
        w = torch.randn(16, 48)
        enc, _ = quantize_leech_weight(
            w,
            LeechQuantizationConfig(
                max_stages=1,
                gain_bits=1,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        mod = ResidualVQLinear.from_encoded(
            enc, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )

        class ModelHolder(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.proj = layer

        container_model = ModelHolder(mod)

        with tempfile.TemporaryDirectory() as td:
            oqpack_dir = Path(td) / "oqpack_artifact"
            save_oqpack(container_model, oqpack_dir, model_id="oqpack_export_test")

            # Export manifest directly passing the OQPack directory
            manifest = LlamaCppGGMLAdapter.export_ggml_adapter_manifest(
                oqpack_dir, td
            )
            assert manifest["model_id"] == "oqpack_export_test"
            assert "proj.weight" in manifest["quantized_tensors"]
            target_manifest = Path(td) / "ggml_ojasx_adapter.json"
            assert target_manifest.exists()

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("has_bias", [False, True])
    def test_dispatch_ggml_custom_matmul_leech(self, batch_size: int, has_bias: bool):
        _skip_if_no_native()
        torch.manual_seed(20261104)
        rows, cols = 16, 70  # non-24 multiple
        weight = torch.randn(rows, cols)
        bias = torch.randn(rows) if has_bias else None
        inputs = torch.randn(batch_size, cols)

        config = LeechQuantizationConfig(
            max_stages=2,
            gain_bits=1,
            rotation_reflections=2,
            adaptive_stages=True,
            seed=20261104,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_leech_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        ref_w = dequantize_leech_weight(encoded)
        ref_out = F.linear(inputs, ref_w, bias)

        dispatched = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(module, inputs)
        assert module.cached_weight_bytes == 0
        assert torch.allclose(ref_out, dispatched, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("bits_per_stage", [1, 2, 4, 8])
    def test_dispatch_ggml_custom_matmul_rvq(self, bits_per_stage: int):
        _skip_if_no_native()
        torch.manual_seed(55 + bits_per_stage)
        rows, cols = 14, 30  # non-block-8 multiple
        weight = torch.randn(rows, cols)
        bias = torch.randn(rows)
        inputs = torch.randn(3, cols)

        config = QuantizationConfig(
            block_size=8,
            bits_per_stage=bits_per_stage,
            max_stages=2,
            min_weight_elements=1,
            max_relative_error=None,
            require_compression=False,
        )
        encoded, _ = quantize_weight(weight, config)
        assert encoded is not None

        module = ResidualVQLinear.from_encoded(
            encoded,
            bias=bias,
            source_dtype=torch.float32,
            device=torch.device("cpu"),
        )

        ref_w = dequantize_weight(encoded)
        ref_out = F.linear(inputs, ref_w, bias)

        dispatched = LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(module, inputs)
        assert module.cached_weight_bytes == 0
        assert torch.allclose(ref_out, dispatched, atol=1e-5, rtol=1e-4)

    def test_dispatch_ggml_custom_matmul_dimension_mismatch(self):
        w = torch.randn(10, 24)
        enc, _ = quantize_weight(
            w,
            QuantizationConfig(
                block_size=8,
                bits_per_stage=4,
                max_stages=1,
                min_weight_elements=1,
                max_relative_error=None,
                require_compression=False,
            ),
        )
        mod = ResidualVQLinear.from_encoded(
            enc, bias=None, source_dtype=torch.float32, device=torch.device("cpu")
        )
        with pytest.raises(ValueError, match="Expected inputs with 24 features"):
            LlamaCppGGMLAdapter.dispatch_ggml_custom_matmul(mod, torch.randn(4, 32))


class TestBackendConformanceSuite:
    """Validate full M6 C ABI & runtime conformance suite."""

    def test_full_conformance_suite_execution(self):
        _skip_if_no_native()
        report = run_backend_conformance_suite(include_native=True)

        assert report["passed"] is True
        assert report["abi_version"] == OJASX_BACKEND_ABI_VERSION

        checks = report["checks"]
        assert checks["abi_header_and_symbols"]["passed"] is True
        assert checks["leech24_matmul_conformance"]["passed"] is True
        assert checks["rvq_matmul_conformance"]["passed"] is True
        assert checks["zero_dense_weight_cache"]["passed"] is True
        assert checks["negative_boundary_error_codes"]["passed"] is True

        assert checks["leech24_matmul_conformance"]["max_abs_diff"] <= 1e-5
        assert checks["rvq_matmul_conformance"]["max_abs_diff"] <= 1e-5
        assert checks["zero_dense_weight_cache"]["leech_cached_bytes"] == 0
        assert checks["zero_dense_weight_cache"]["rvq_cached_bytes"] == 0

    def test_conformance_suite_without_native(self):
        report = run_backend_conformance_suite(include_native=False)
        assert report["passed"] is True
        assert report["abi_version"] == OJASX_BACKEND_ABI_VERSION

    def test_negative_boundary_direct_error_codes(self):
        _skip_if_no_native()
        lib, _ = _load_library()
        assert lib is not None

        # 1. Null pointers in decode_indices returns -1 (OJASX_STATUS_INVALID_ARGUMENT)
        assert lib.ojasx_leech24_decode_indices(None, 10, None) == OjasxStatus.INVALID_ARGUMENT

        # 2. Invalid row range returns -1
        arr_type = ctypes.c_void_p * 1
        dummy_buf = torch.zeros(100, dtype=torch.uint8)
        buf_ptr = ctypes.c_void_p(dummy_buf.data_ptr())
        gc = torch.zeros(1, 1, dtype=torch.float32)
        out = torch.zeros(1, 24, dtype=torch.float32)
        res_inv = lib.ojasx_leech24_decode_rows(
            arr_type(buf_ptr),
            arr_type(buf_ptr),
            arr_type(buf_ptr),
            ctypes.c_void_p(gc.data_ptr()),
            1,
            0,
            0,
            10,
            24,
            8,
            4,  # row_start=8, row_stop=4
            ctypes.c_void_p(out.data_ptr()),
        )
        assert res_inv == OjasxStatus.INVALID_ARGUMENT

        # 3. Null stage pointer inside packed_indices returns -2 (OJASX_STATUS_NULL_POINTER)
        res_null_st = lib.ojasx_leech24_decode_rows(
            arr_type(None),
            arr_type(buf_ptr),
            arr_type(buf_ptr),
            ctypes.c_void_p(gc.data_ptr()),
            1,
            0,
            0,
            1,
            24,
            0,
            1,
            ctypes.c_void_p(out.data_ptr()),
        )
        assert res_null_st == OjasxStatus.NULL_POINTER

        # 4. Missing presence in adaptive stage returns -3 (OJASX_STATUS_MISSING_PRESENCE)
        arr_type2 = ctypes.c_void_p * 2
        gc2 = torch.zeros(2, 1, dtype=torch.float32)
        res_missing_pr = lib.ojasx_leech24_decode_rows(
            arr_type2(buf_ptr, buf_ptr),
            arr_type2(buf_ptr, buf_ptr),
            arr_type2(buf_ptr, None),  # stage 1 presence is NULL
            ctypes.c_void_p(gc2.data_ptr()),
            2,
            0,
            1,  # adaptive_stages = 1
            1,
            24,
            0,
            1,
            ctypes.c_void_p(out.data_ptr()),
        )
        assert res_missing_pr == OjasxStatus.MISSING_PRESENCE

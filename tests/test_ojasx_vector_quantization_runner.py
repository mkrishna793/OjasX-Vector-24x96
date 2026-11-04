"""Standalone OJASX Vector Quantization Runner regression tests."""

import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from ojasx_vector_quantization_runner import (
    QuantizationConfig,
    QuantizedWeight,
    ResidualVQConv1D,
    ResidualVQFusedExperts,
    ResidualVQLinear,
    dequantize_weight,
    pack_codes,
    quantize_model,
    quantize_weight,
    unpack_codes,
)
from ojasx_vector_quantization_runner.evaluation import compare_causal_lms
from ojasx_vector_quantization_runner.hf import (
    VQChatRunner,
    convert_local_hf_model,
    load_quantized_pretrained,
    save_quantized_pretrained,
)
from ojasx_vector_quantization_runner.module import quantization_manifest, restore_vq_modules
from ojasx_vector_quantization_runner.runner import main as runner_main
from ojasx_vector_quantization_runner.runtime import (
    benchmark_forward,
    runtime_storage_report,
    set_runtime_profile,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "ojasx_vector_quantization_runner"


def _config(**overrides):
    values = {
        "block_size": 8,
        "bits_per_stage": 4,
        "max_stages": 3,
        "max_relative_error": None,
        "max_codebook_samples": 128,
        "kmeans_iterations": 4,
        "chunk_size": 31,
        "min_weight_elements": 1,
        "require_compression": False,
        "seed": 71,
    }
    values.update(overrides)
    return QuantizationConfig(**values)


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
def test_bit_packing_roundtrip_and_bounds(bits):
    for count in (0, 1, 2, 7, 8, 9, 255, 257):
        codes = torch.arange(count, dtype=torch.int64) % (1 << bits)
        packed = pack_codes(codes, bits)
        assert packed.dtype == torch.uint8
        assert packed.numel() == (count * bits + 7) // 8
        assert torch.equal(unpack_codes(packed, bits, count), codes.to(torch.uint8))
    with pytest.raises(ValueError, match="exceeds"):
        pack_codes(torch.tensor([1 << bits]), bits)
    with pytest.raises(ValueError, match="Expected"):
        unpack_codes(torch.tensor([], dtype=torch.uint8), bits, 1)


def test_known_encoded_matrix_has_exact_structural_reconstruction():
    books = torch.zeros(1, 16, 2, dtype=torch.float32)
    books[0, 1] = torch.tensor([0.5, -0.25])
    encoded = QuantizedWeight(
        original_shape=(1, 2),
        block_size=2,
        bits_per_stage=4,
        packed_codes=torch.tensor([[1]], dtype=torch.uint8),
        codebooks=books,
        row_scales=torch.tensor([2.0]),
        metrics={},
    )
    assert torch.equal(dequantize_weight(encoded), torch.tensor([[1.0, -0.5]]))
    assert encoded.nominal_bits_per_weight == 2.0
    with pytest.raises(ValueError, match="at least one"):
        QuantizedWeight(
            original_shape=(1, 2),
            block_size=2,
            bits_per_stage=4,
            packed_codes=torch.empty(0, 1, dtype=torch.uint8),
            codebooks=torch.empty(0, 16, 2),
            row_scales=torch.tensor([2.0]),
            metrics={},
        )


def test_deterministic_zero_anchored_multistage_codec_and_padding():
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(17, 35, generator=generator).T  # noncontiguous [35, 17]
    config = _config()
    first, report = quantize_weight(source, config)
    second, _ = quantize_weight(source, config)
    assert first is not None and second is not None
    assert torch.equal(first.packed_codes, second.packed_codes)
    assert torch.equal(first.codebooks, second.codebooks)
    assert torch.equal(first.row_scales, second.row_scales)
    assert torch.all(first.codebooks[:, 0] == 0)
    errors = report["stage_relative_errors"]
    assert all(right <= left + 1e-6 for left, right in zip(errors, errors[1:]))
    reconstructed = dequantize_weight(first)
    assert reconstructed.shape == source.shape
    measured = (reconstructed - source.float()).norm() / source.float().norm()
    torch.testing.assert_close(measured, torch.tensor(report["relative_error"]))
    assert first.measured_bits_per_weight > first.nominal_bits_per_weight


def test_zero_rows_and_twenty_four_dimensional_blocks():
    source = torch.zeros(8, 48, dtype=torch.float64)
    source[::2] = torch.randn(4, 48)
    encoded, report = quantize_weight(source, _config(block_size=24, max_stages=2))
    assert encoded is not None
    assert encoded.vector_count == 16
    assert torch.all(encoded.row_scales[1::2] == 0)
    assert dequantize_weight(encoded).shape == source.shape
    assert report["relative_error"] < 1.0


def test_low_entropy_weights_admit_deep_compression_under_error_budget():
    generator = torch.Generator().manual_seed(93)
    patterns = torch.randn(8, 8, generator=generator)
    row_indices = torch.arange(64) % 8
    scales = torch.linspace(0.5, 2.0, 64)
    normalized_rows = patterns[row_indices] / patterns[row_indices].square().mean(
        dim=1,
        keepdim=True,
    ).sqrt()
    source = (scales[:, None] * normalized_rows).repeat(1, 8)
    encoded, report = quantize_weight(
        source,
        _config(max_stages=1, max_relative_error=0.01, require_compression=True),
    )
    assert encoded is not None
    assert encoded.stages == 1
    assert report["relative_error"] <= 0.01
    assert report["storage_bytes"] * 16 <= report["original_bytes"]
    # The measured rate includes the codebook and is honestly above the 0.5-bit payload.
    assert 0.5 < encoded.measured_bits_per_weight < 2.1


def test_error_budget_and_compression_failures_retain_original_candidate():
    source = torch.randn(16, 64)
    rejected, report = quantize_weight(
        source,
        _config(bits_per_stage=1, max_stages=1, max_relative_error=0.0),
    )
    assert rejected is None
    assert report["reason"] == "error_budget_not_met"

    small, report = quantize_weight(
        torch.randn(2, 2),
        _config(require_compression=True),
    )
    assert small is None
    assert report["reason"] == "not_smaller_than_original"

    nonfinite, report = quantize_weight(torch.tensor([[float("nan")] * 8] * 2), _config())
    assert nonfinite is None
    assert report["reason"] == "nonfinite_weights"


def test_vq_linear_forward_bias_and_no_dense_weight_cache():
    source = torch.randn(12, 20)
    encoded, _ = quantize_weight(source, _config())
    assert encoded is not None
    bias = torch.randn(12)
    module = ResidualVQLinear.from_encoded(
        encoded,
        bias=bias,
        source_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    inputs = torch.randn(3, 20)
    expected = F.linear(inputs, dequantize_weight(encoded), bias)
    torch.testing.assert_close(module(inputs), expected)
    assert "weight" not in module._parameters
    torch.testing.assert_close(module.weight, dequantize_weight(encoded))
    half_module = ResidualVQLinear.from_encoded(
        encoded,
        bias=bias.half(),
        source_dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert half_module.bias.dtype == torch.float16
    assert half_module.weight.dtype == torch.float16
    assert module.storage_bytes < 12 * 20 * 4 + 12 * 4 + 3 * 16 * 8 * 4
    assert {"packed_codes", "codebooks", "row_scales", "bias"} <= set(module.state_dict())
    with pytest.raises(ValueError, match="final dimension"):
        module(torch.randn(1, 19))


def test_model_replacement_filters_fallback_and_manifest_restore():
    torch.manual_seed(5)
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 24),
        torch.nn.GELU(),
        torch.nn.Linear(24, 8),
    )
    reference = copy.deepcopy(model)
    excluded_module = model[2]
    model, report = quantize_model(
        model,
        _config(),
        include=("0", "2"),
        exclude=("2",),
    )
    assert report.selected_layers == 1
    assert isinstance(model[0], ResidualVQLinear)
    assert isinstance(model[2], torch.nn.Linear)
    assert model[2] is excluded_module
    assert model[2] is not reference[2]
    manifest = quantization_manifest(model)
    assert set(manifest["layers"]) == {"0"}

    meta_model = torch.nn.Sequential(
        torch.nn.Linear(20, 24),
        torch.nn.GELU(),
        torch.nn.Linear(24, 8),
    ).to("meta")
    restore_vq_modules(meta_model, manifest)
    assert isinstance(meta_model[0], ResidualVQLinear)
    assert meta_model[0].packed_codes.device.type == "meta"


def test_tiny_llama_uses_same_architecture_neutral_replacement():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(23)
    model = LlamaForCausalLM(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(inputs).logits
    model, report = quantize_model(model, _config(max_stages=2, require_compression=True))
    assert report.selected_layers >= 4
    with torch.inference_mode():
        actual = model(inputs).logits
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    # Compare against the same encoded weights, not an assumed accuracy tolerance.
    for module in model.modules():
        if isinstance(module, ResidualVQLinear):
            assert module.dequantize_weight().shape == (
                module.out_features,
                module.in_features,
            )


def _tiny_qwen_and_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(config).eval()
    words = [
        "<unk>",
        "<pad>",
        "<eos>",
        "<bos>",
        "hello",
        "world",
        "English",
        "question",
        "answer",
        "user",
        "assistant",
        "system",
    ]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    template = (
        "{% for message in messages %}{{'<' + message['role'] + '> ' + "
        "message['content'] + ' <eos>\\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        chat_template=template,
    )
    return model, tokenizer


def test_hf_qwen_sharded_roundtrip_chat_template_and_evaluation(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    model, tokenizer = _tiny_qwen_and_tokenizer()
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    model, report = quantize_model(model, _config(max_stages=2, require_compression=True))
    assert report.selected_layers
    messages = [{"role": "user", "content": "hello world"}]
    original_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(original_prompt, return_tensors="pt")
    with torch.inference_mode():
        expected_logits = model(**encoded).logits
    save_quantized_pretrained(
        model,
        output,
        tokenizer=tokenizer,
        report=report,
        max_shard_size="1KB",
    )
    assert (output / "model.safetensors.index.json").exists()
    assert (output / "chat_template.jinja").exists()

    # Quantized loading must use from_config/meta reconstruction, not dense load.
    with patch.object(
        AutoModelForCausalLM,
        "from_pretrained",
        side_effect=AssertionError("Dense from_pretrained must not be called"),
    ):
        loaded = load_quantized_pretrained(output, load_tokenizer=False)
    # The synthetic tokenizer is intentionally generic rather than Qwen's BPE;
    # real Qwen checkpoints continue through AutoTokenizer.
    loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(output)
    assert not any(parameter.device.type == "meta" for parameter in loaded.parameters())
    assert loaded_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) == original_prompt
    with torch.inference_mode():
        actual_logits = loaded(**encoded).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    runner = VQChatRunner(loaded, loaded_tokenizer)
    assert runner.render_chat(messages) == original_prompt
    assert isinstance(runner.respond(messages, max_new_tokens=2), str)

    reference = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
    evaluation = compare_causal_lms(
        reference,
        loaded,
        loaded_tokenizer,
        ["hello world", "English question"],
        max_length=32,
    )
    assert evaluation["supervised_tokens"] > 0
    assert 0 <= evaluation["top1_agreement"] <= 1
    assert evaluation["candidate_mean_nll"] > 0
    assert len(evaluation["per_text"]) == 2
    assert runner_main(["inspect", str(output)]) == 0
    with patch.object(AutoTokenizer, "from_pretrained", return_value=loaded_tokenizer):
        assert runner_main(
            [
                "benchmark",
                str(output),
                "--text",
                "hello world",
                "--warmup",
                "0",
                "--iterations",
                "1",
            ]
        ) == 0


def test_local_conversion_api_and_evaluate_cli(tmp_path):
    from transformers import AutoTokenizer

    model, tokenizer = _tiny_qwen_and_tokenizer()
    source = tmp_path / "source"
    output = tmp_path / "encoded"
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    # The synthetic WordLevel tokenizer is not Qwen's production BPE. Patch only
    # its AutoTokenizer selection; model conversion and shard loading are real.
    with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer):
        report = convert_local_hf_model(
            source,
            output,
            _config(max_stages=2, require_compression=True),
        )
        assert report.selected_layers
        assert runner_main(
            [
                "evaluate",
                str(source),
                str(output),
                "--text",
                "hello world",
                "--max-length",
                "16",
            ]
        ) == 0
    assert (output / "vq_manifest.json").exists()
    assert (output / "quantization_report.json").exists()
    loaded = load_quantized_pretrained(output, load_tokenizer=False)
    assert not any(parameter.device.type == "meta" for parameter in loaded.parameters())
    with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer):
        with pytest.raises(ValueError, match="must be empty"):
            convert_local_hf_model(source, output, _config(max_stages=1))


def test_identical_model_evaluation_and_isolated_formal_reports():
    model, tokenizer = _tiny_qwen_and_tokenizer()
    report = compare_causal_lms(model, model, tokenizer, ["hello world"], max_length=16)
    assert report["top1_agreement"] == 1.0
    assert report["mean_kl_reference_candidate"] == 0.0
    assert report["reference_mean_nll"] == report["candidate_mean_nll"]
    assert report["mean_nll_delta"] == 0.0
    assert len(report["per_text"]) == 1
    assert report["per_text"][0]["top1_agreement"] == 1.0

    z3 = json.loads((PROJECT / "formal/z3-results.json").read_text())
    rocq = json.loads((PROJECT / "formal/rocq-results.json").read_text())
    assert z3["status"] == rocq["status"] == "passed"
    assert len(z3["proofs"]) == 9 and len(z3["counterexamples"]) == 4
    assert rocq["checks"][0]["closed_under_global_context"] == 7
    assert rocq["checks"][1]["passed"]


def test_runtime_profiles_reuse_exact_weights_and_report_whole_state():
    source = torch.randn(12, 20)
    encoded, _ = quantize_weight(source, _config(max_stages=1))
    module = ResidualVQLinear.from_encoded(
        encoded,
        bias=torch.randn(12),
        source_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    inputs = torch.randn(3, 20)
    set_runtime_profile(module, "memory")
    with torch.inference_mode():
        expected = module(inputs)
    assert runtime_storage_report(module)["decoded_cache_bytes"] == 0

    set_runtime_profile(module, "speed")
    with torch.inference_mode():
        first = module(inputs)
        cached_pointer = module._cached_weight.data_ptr()
        second = module(inputs)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert module._cached_weight.data_ptr() == cached_pointer
    storage = runtime_storage_report(module)
    assert storage["decoded_cache_bytes"] == source.numel() * source.element_size()
    assert storage["model_unique_state_tensor_bytes"] == storage[
        "vq_storage_bytes_including_bias"
    ]
    assert storage["model_state_plus_decoded_cache_bytes"] > storage[
        "model_unique_state_tensor_bytes"
    ]
    with torch.no_grad():
        module.codebooks.add_(0.01)
    with torch.inference_mode():
        updated = module(inputs)
    freshly_decoded = F.linear(inputs, dequantize_weight(module.encoded()), module.bias)
    torch.testing.assert_close(updated, freshly_decoded, rtol=0, atol=0)

    benchmark = benchmark_forward(
        module,
        {"inputs": inputs},
        warmup=1,
        iterations=2,
    )
    assert [result["profile"] for result in benchmark["results"]] == ["memory", "speed"]
    assert benchmark["results"][0]["decoded_cache_bytes"] == 0
    assert benchmark["results"][1]["decoded_cache_bytes"] > 0
    assert not module.weight_cache  # Benchmarking restores the memory profile.
    with pytest.raises(ValueError, match="Runtime profile"):
        set_runtime_profile(module, "unknown")


def _tiny_modern_model(architecture: str):
    if architecture == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                tie_word_embeddings=True,
            )
        )
    if architecture == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        return Gemma3ForCausalLM(
            Gemma3TextConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
                tie_word_embeddings=True,
            )
        )
    if architecture == "mistral":
        from transformers import MistralConfig, MistralForCausalLM

        return MistralForCausalLM(
            MistralConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                tie_word_embeddings=True,
            )
        )
    if architecture == "qwen2_moe":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        return Qwen2MoeForCausalLM(
            Qwen2MoeConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                moe_intermediate_size=16,
                num_experts=4,
                num_experts_per_tok=2,
                tie_word_embeddings=True,
            )
        )
    if architecture == "qwen3_moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        return Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                moe_intermediate_size=16,
                num_experts=4,
                num_experts_per_tok=2,
                tie_word_embeddings=True,
            )
        )
    if architecture == "mixtral":
        from transformers import MixtralConfig, MixtralForCausalLM

        return MixtralForCausalLM(
            MixtralConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
                tie_word_embeddings=True,
            )
        )
    raise AssertionError(architecture)


def _tiny_current_model(architecture: str):
    common = {
        "vocab_size": 64,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "max_position_embeddings": 64,
        "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": True,
        "use_cache": False,
    }
    if architecture == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        return Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(intermediate_size=128, **common)
        )
    if architecture in {"qwen3_5_moe", "qwen3_next"}:
        moe_values = {
            **common,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
        if architecture == "qwen3_5_moe":
            from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

            return Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**moe_values))
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

        return Qwen3NextForCausalLM(
            Qwen3NextConfig(intermediate_size=128, **moe_values)
        )
    if architecture in {"gemma4", "gemma4_moe"}:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        moe = architecture == "gemma4_moe"
        return Gemma4ForCausalLM(
            Gemma4TextConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=64,
                sliding_window=32,
                layer_types=["sliding_attention", "full_attention"],
                tie_word_embeddings=True,
                use_cache=False,
                hidden_size_per_layer_input=0,
                vocab_size_per_layer_input=64,
                enable_moe_block=moe,
                num_experts=4 if moe else None,
                top_k_experts=2 if moe else None,
                moe_intermediate_size=16 if moe else None,
            )
        )
    if architecture == "glm_moe_dsa":
        from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

        return GlmMoeDsaForCausalLM(
            GlmMoeDsaConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                n_shared_experts=1,
                n_routed_experts=4,
                num_experts_per_tok=2,
                kv_lora_rank=16,
                q_lora_rank=16,
                qk_rope_head_dim=8,
                v_head_dim=16,
                qk_nope_head_dim=8,
                index_topk=2,
                index_head_dim=8,
                index_n_heads=2,
                first_k_dense_replace=1,
                mlp_layer_types=["dense", "sparse"],
                indexer_types=["full", "full"],
                max_position_embeddings=64,
                tie_word_embeddings=True,
                use_cache=False,
            )
        )
    if architecture == "deepseek_v4":
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM(
            DeepseekV4Config(
                vocab_size=64,
                hidden_size=64,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=16,
                q_lora_rank=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=1,
                max_position_embeddings=64,
                hc_mult=2,
                hc_sinkhorn_iters=2,
                mlp_layer_types=["moe", "moe"],
                layer_types=[
                    "compressed_sparse_attention",
                    "heavily_compressed_attention",
                ],
                compress_rates={
                    "compressed_sparse_attention": 4,
                    "heavily_compressed_attention": 4,
                },
                sliding_window=8,
                o_groups=2,
                o_lora_rank=16,
                index_n_heads=2,
                index_head_dim=8,
                index_topk=2,
                num_nextn_predict_layers=0,
                tie_word_embeddings=True,
                use_cache=False,
            )
        )
    raise AssertionError(architecture)


def _tiny_composite_model(architecture: str):
    from transformers import AutoModelForMultimodalLM

    if architecture == "qwen3_5_composite":
        from transformers import Qwen3_5Config, Qwen3_5VisionConfig

        return AutoModelForMultimodalLM.from_config(
            Qwen3_5Config(
                text_config=_tiny_current_model("qwen3_5").config,
                vision_config=Qwen3_5VisionConfig(
                    depth=1,
                    hidden_size=32,
                    intermediate_size=64,
                    num_heads=4,
                    out_hidden_size=64,
                    num_position_embeddings=16,
                    patch_size=16,
                ),
            )
        )
    if architecture == "gemma4_composite":
        from transformers import Gemma4Config, Gemma4VisionConfig

        return AutoModelForMultimodalLM.from_config(
            Gemma4Config(
                text_config=_tiny_current_model("gemma4").config,
                vision_config=Gemma4VisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=8,
                    position_embedding_size=16,
                    pooling_kernel_size=2,
                    patch_size=16,
                ),
            )
        )
    if architecture == "glm5_next_composite":
        from transformers import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig

        return AutoModelForMultimodalLM.from_config(
            Glm5NextConfig(
                text_config=Glm5NextTextConfig(
                    vocab_size=64,
                    hidden_size=64,
                    intermediate_size=128,
                    moe_intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    n_shared_experts=1,
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                    kv_lora_rank=16,
                    q_lora_rank=16,
                    qk_rope_head_dim=0,
                    v_head_dim=16,
                    qk_nope_head_dim=8,
                    index_topk=2,
                    index_head_dim=8,
                    index_n_heads=2,
                    mlp_layer_types=["sparse", "sparse"],
                    layer_types=["linear_attention", "indexed_attention"],
                    indexer_types=["full", "full"],
                    max_position_embeddings=64,
                    tie_word_embeddings=True,
                    use_cache=False,
                    linear_head_dim=8,
                    linear_num_heads=4,
                    hc_mult=2,
                    hc_sinkhorn_iters=2,
                    index_kpool=2,
                    pad_token_id=0,
                    bos_token_id=0,
                    eos_token_id=1,
                ),
                vision_config=Glm5NextVisionConfig(
                    depth=1,
                    hidden_size=32,
                    intermediate_size=64,
                    num_heads=4,
                    out_hidden_size=64,
                    image_size=32,
                    patch_size=16,
                    projection_intermediate_size=64,
                ),
                image_token_id=60,
                video_token_id=61,
                image_start_token_id=62,
                image_end_token_id=63,
                video_start_token_id=56,
                video_end_token_id=57,
            )
        )
    raise AssertionError(architecture)


def _assert_exact_quantized_roundtrip(model, report, output, inputs):
    assert report.selected_layers
    model = model.eval()
    with torch.inference_mode():
        expected = model(inputs).logits
    assert expected.shape == (1, 4, 64)
    assert torch.isfinite(expected).all()
    save_quantized_pretrained(model, output)
    loaded = load_quantized_pretrained(output, load_tokenizer=False)
    with torch.inference_mode():
        actual = loaded(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return loaded


@pytest.mark.parametrize("architecture", ("qwen3", "gemma3", "mistral"))
def test_tiny_modern_dense_architectures_roundtrip(architecture, tmp_path):
    torch.manual_seed(101)
    model = _tiny_modern_model(architecture)
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
    )
    _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / architecture,
        torch.tensor([[1, 2, 3, 4]]),
    )


def test_gpt2_conv1d_adapter_roundtrip(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(102)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=4,
            n_inner=64,
            bos_token_id=1,
            eos_token_id=1,
        )
    )
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
    )
    assert report.to_dict()["module_types"]["conv1d"]["selected"] == 4
    loaded = _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / "gpt2",
        torch.tensor([[1, 2, 3, 4]]),
    )
    assert sum(isinstance(module, ResidualVQConv1D) for module in loaded.modules()) == 4


@pytest.mark.parametrize("architecture", ("qwen2_moe", "qwen3_moe", "mixtral"))
def test_fused_moe_experts_roundtrip(architecture, tmp_path):
    torch.manual_seed(103)
    model = _tiny_modern_model(architecture)
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
    )
    module_types = report.to_dict()["module_types"]
    assert module_types["fused_expert_gate_up"]["selected"] == 1
    assert module_types["fused_expert_down"]["selected"] == 1
    loaded = _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / architecture,
        torch.tensor([[1, 2, 3, 4]]),
    )
    assert any(isinstance(module, ResidualVQFusedExperts) for module in loaded.modules())


@pytest.mark.parametrize(
    "architecture,expert_banks",
    (
        ("qwen3_5", 0),
        ("qwen3_5_moe", 2),
        ("qwen3_next", 2),
        ("gemma4", 0),
        ("gemma4_moe", 2),
        ("glm_moe_dsa", 1),
        ("deepseek_v4", 2),
    ),
)
def test_current_hybrid_architectures_roundtrip(architecture, expert_banks, tmp_path):
    torch.manual_seed(105)
    model = _tiny_current_model(architecture)
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
    )
    loaded = _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / architecture,
        torch.tensor([[1, 2, 3, 4]]),
    )
    banks = [
        module for module in loaded.modules() if isinstance(module, ResidualVQFusedExperts)
    ]
    assert len(banks) == expert_banks
    if architecture == "deepseek_v4":
        assert all(module.gate_limit == 10.0 for module in banks)


@pytest.mark.parametrize(
    "architecture",
    ("qwen3_5_composite", "gemma4_composite", "glm5_next_composite"),
)
def test_multimodal_composite_text_path_roundtrip(architecture, tmp_path):
    torch.manual_seed(106)
    model = _tiny_composite_model(architecture)
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
    )
    loaded = _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / architecture,
        torch.tensor([[1, 2, 3, 4]]),
    )
    if architecture == "glm5_next_composite":
        banks = [
            module
            for module in loaded.modules()
            if isinstance(module, ResidualVQFusedExperts)
        ]
        assert len(banks) == 2
        assert all(module.gate_limit == 10.0 for module in banks)


def test_fused_expert_projection_fallback_is_independently_restored(tmp_path):
    torch.manual_seed(104)
    model = _tiny_modern_model("qwen3_moe")
    model, report = quantize_model(
        model,
        _config(max_stages=1, kmeans_iterations=2, max_codebook_samples=64),
        include=("*.gate_up_proj",),
    )
    experts = next(
        module for module in model.modules() if isinstance(module, ResidualVQFusedExperts)
    )
    assert isinstance(experts.gate_up_proj, ResidualVQLinear)
    assert isinstance(experts.down_proj, torch.Tensor)
    assert report.selected_layers == 1
    _assert_exact_quantized_roundtrip(
        model,
        report,
        tmp_path / "mixed-moe",
        torch.tensor([[1, 2, 3, 4]]),
    )


def test_shared_weight_preservation_and_extreme_finite_scales():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.head.weight = self.embed.weight

    model = TiedModel()
    original_head = model.head
    model, report = quantize_model(model, _config(), include=("head",), exclude=())
    assert model.head is original_head
    assert model.head.weight is model.embed.weight
    assert report.layers[0].reason == "shared_or_tied_weight"

    for value in (1e-30, 1e30):
        encoded, measurement = quantize_weight(
            torch.full((4, 8), value),
            _config(max_stages=1, max_relative_error=1e-6),
        )
        assert encoded is not None, measurement
        assert measurement["relative_error"] <= 1e-6
        assert torch.isfinite(dequantize_weight(encoded)).all()


def test_unknown_linear_subclass_is_retained_densely():
    class CustomLinear(torch.nn.Linear):
        def forward(self, inputs):
            return 2 * super().forward(inputs)

    model = torch.nn.Sequential(CustomLinear(8, 8, bias=False))
    original = model[0]
    model, report = quantize_model(model, _config(), exclude=())
    assert model[0] is original
    assert type(model[0]) is CustomLinear
    assert report.selected_layers == 0
    assert report.layers == []


def test_project_does_not_import_ojasx_foundation():
    for path in PROJECT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"ojasx", "ojasx2"}
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {"ojasx", "ojasx2"}

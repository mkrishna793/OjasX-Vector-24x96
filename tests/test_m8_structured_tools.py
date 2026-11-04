"""Milestone M8: Structured Output Constrained Decoding & Sandboxed Tool-Using Agent Runtime Tests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.engine import (
    GenerationResult,
    OjasXInferenceEngine,
    SamplingParams,
)
from ojasx_24x96.module import quantize_model
from ojasx_24x96.structured_tools import (
    OjasXToolUsingAgent,
    SandboxedToolRunner,
    StructuredOutputError,
    TokenTrie,
    TokenTrieLogitProcessor,
    ToolCall,
    ToolResult,
    ToolSpec,
    generate_structured_json,
    synthesize_sample_from_schema,
    validate_json_schema,
)


def _create_tiny_quantized_engine(vocab_size: int = 64) -> OjasXInferenceEngine:
    """Create a miniature, fast quantized model and inference engine for testing."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=True,
    )
    model = LlamaForCausalLM(cfg).eval()
    q_cfg = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=None,
        require_compression=False,
        min_weight_elements=1,
    )
    quantized_model, _ = quantize_model(model, q_cfg)
    return OjasXInferenceEngine(quantized_model, max_context_length=128)


# ==============================================================================
# 1. Tests for validate_json_schema
# ==============================================================================


class TestValidateJsonSchema:
    def test_primitive_types(self) -> None:
        # String
        valid, _ = validate_json_schema("hello", {"type": "string"})
        assert valid is True
        valid, errs = validate_json_schema(123, {"type": "string"})
        assert valid is False
        assert any("expected type 'string'" in e for e in errs)

        # Integer vs Boolean
        valid, _ = validate_json_schema(42, {"type": "integer"})
        assert valid is True
        # In JSON schema, booleans are not integers
        valid, errs = validate_json_schema(True, {"type": "integer"})
        assert valid is False

        # Number accepts float and int
        valid, _ = validate_json_schema(3.14, {"type": "number"})
        assert valid is True
        valid, _ = validate_json_schema(100, {"type": "number"})
        assert valid is True
        valid, _ = validate_json_schema(True, {"type": "number"})
        assert valid is False

        # Boolean
        valid, _ = validate_json_schema(False, {"type": "boolean"})
        assert valid is True
        valid, _ = validate_json_schema("True", {"type": "boolean"})
        assert valid is False

        # Null
        valid, _ = validate_json_schema(None, {"type": "null"})
        assert valid is True
        valid, _ = validate_json_schema(0, {"type": "null"})
        assert valid is False

    def test_numeric_bounds(self) -> None:
        schema = {"type": "number", "minimum": 10.0, "maximum": 50.0}
        assert validate_json_schema(25.0, schema)[0] is True
        assert validate_json_schema(10.0, schema)[0] is True
        assert validate_json_schema(50.0, schema)[0] is True

        valid, errs = validate_json_schema(9.9, schema)
        assert valid is False
        assert any("< minimum 10.0" in e for e in errs)

        valid, errs = validate_json_schema(50.1, schema)
        assert valid is False
        assert any("> maximum 50.0" in e for e in errs)

    def test_string_length_and_enum(self) -> None:
        schema = {
            "type": "string",
            "minLength": 3,
            "maxLength": 6,
            "enum": ["cat", "dog", "turtle"],
        }
        assert validate_json_schema("cat", schema)[0] is True
        assert validate_json_schema("turtle", schema)[0] is True

        # Enum mismatch
        valid, errs = validate_json_schema("bat", schema)
        assert valid is False
        assert any("is not in enum" in e for e in errs)

        # Length mismatch
        len_schema = {"type": "string", "minLength": 4, "maxLength": 8}
        assert validate_json_schema("hi", len_schema)[0] is False
        assert validate_json_schema("supercalifragilistic", len_schema)[0] is False

    def test_array_validation(self) -> None:
        schema = {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 0},
        }
        assert validate_json_schema([1, 2, 3], schema)[0] is True

        # Items failure
        valid, errs = validate_json_schema([1, -2, 3], schema)
        assert valid is False
        assert any("< minimum 0" in e for e in errs)

        # minItems failure
        valid, errs = validate_json_schema([1], schema)
        assert valid is False
        assert any("items count 1 < minItems 2" in e for e in errs)

        # maxItems failure
        valid, errs = validate_json_schema([1, 2, 3, 4, 5], schema)
        assert valid is False
        assert any("items count 5 > maxItems 4" in e for e in errs)

    def test_nested_objects_and_missing_required(self) -> None:
        schema = {
            "type": "object",
            "required": ["user_id", "profile"],
            "properties": {
                "user_id": {"type": "integer"},
                "profile": {
                    "type": "object",
                    "required": ["username", "email"],
                    "properties": {
                        "username": {"type": "string", "minLength": 3},
                        "email": {"type": "string"},
                        "location": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "integer"},
                            },
                            "required": ["city"],
                        },
                    },
                },
            },
        }

        valid_data = {
            "user_id": 101,
            "profile": {
                "username": "alice",
                "email": "alice@example.com",
                "location": {"city": "New York", "zip": 10001},
            },
        }
        assert validate_json_schema(valid_data, schema)[0] is True

        # Missing required at top-level
        invalid_top = {"profile": {"username": "alice", "email": "alice@example.com"}}
        valid, errs = validate_json_schema(invalid_top, schema)
        assert valid is False
        assert any("missing required property 'user_id'" in e for e in errs)

        # Missing required in nested object
        invalid_nested = {
            "user_id": 101,
            "profile": {
                "username": "al",  # too short
                "location": {},  # missing city
            },
        }
        valid, errs = validate_json_schema(invalid_nested, schema)
        assert valid is False
        assert any("missing required property 'email'" in e for e in errs)
        assert any("length 2 < minLength 3" in e for e in errs)
        assert any("missing required property 'city'" in e for e in errs)

    def test_additional_properties_constraint(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        assert validate_json_schema({"name": "item"}, schema)[0] is True

        valid, errs = validate_json_schema({"name": "item", "extra": 123}, schema)
        assert valid is False
        assert any("additional properties not allowed" in e for e in errs)

    def test_additional_properties_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        assert validate_json_schema({"name": "item", "count": 10}, schema)[0] is True
        valid, errs = validate_json_schema({"name": "item", "count": "not_int"}, schema)
        assert valid is False
        assert any("expected type 'integer'" in e for e in errs)

    def test_empty_schema_and_nested_array_of_objects(self) -> None:
        assert validate_json_schema({"any": "thing"}, {})[0] is True

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sku", "qty"],
                "properties": {
                    "sku": {"type": "string"},
                    "qty": {"type": "integer", "minimum": 1},
                },
            },
        }
        valid_items = [{"sku": "A1", "qty": 5}, {"sku": "B2", "qty": 1}]
        assert validate_json_schema(valid_items, schema)[0] is True

        invalid_items = [{"sku": "A1", "qty": 0}]
        valid, errs = validate_json_schema(invalid_items, schema)
        assert valid is False
        assert any("< minimum 1" in e for e in errs)


# ==============================================================================
# 2. Tests for TokenTrie and TokenTrieLogitProcessor
# ==============================================================================


class TestTokenTrieAndLogitProcessor:
    def test_token_trie_prefix_matching(self) -> None:
        trie = TokenTrie()
        trie.insert([1, 2, 3])
        trie.insert([1, 2, 4])
        trie.insert([1, 5, 6])

        # At root prefix []
        assert trie.get_allowed_tokens([]) == {1}
        # After [1]
        assert trie.get_allowed_tokens([1]) == {2, 5}
        # After [1, 2]
        assert trie.get_allowed_tokens([1, 2]) == {3, 4}
        # After [1, 5]
        assert trie.get_allowed_tokens([1, 5]) == {6}
        # Non-existent prefix
        assert trie.get_allowed_tokens([9, 9]) == set()

    def test_logit_processor_masking(self) -> None:
        processor = TokenTrieLogitProcessor(allowed_tokens=[3, 7])
        logits = torch.zeros(10)
        masked = processor.mask_logits([], logits)

        # Disallowed tokens should be -inf
        assert masked[3] == 0.0
        assert masked[7] == 0.0
        assert masked[0] == float("-inf")
        assert masked[1] == float("-inf")
        assert masked[9] == float("-inf")

    def test_logit_processor_2d_and_callback(self) -> None:
        # Dynamic callback allowing tokens based on step count
        processor = TokenTrieLogitProcessor(
            allowed_tokens_fn=lambda toks: [toks[-1] + 1] if toks else [10]
        )
        logits_2d = torch.zeros(1, 20)
        masked_0 = processor.mask_logits([], logits_2d)
        assert masked_0[0, 10] == 0.0
        assert masked_0[0, 9] == float("-inf")

        masked_1 = processor.mask_logits([10], logits_2d)
        assert masked_1[0, 11] == 0.0
        assert masked_1[0, 10] == float("-inf")

    def test_logit_processor_with_engine_attachment(self) -> None:
        engine = _create_tiny_quantized_engine(vocab_size=64)
        # Constrain to strict sequence [15, 25, 35]
        processor = TokenTrieLogitProcessor(allowed_sequences=[[15, 25, 35]])

        with processor.attach(engine):
            res = engine.generate(
                [1, 2],
                SamplingParams(max_new_tokens=3, temperature=0.0),
                use_cache=True,
                reset_cache=True,
            )

        assert res.generated_token_ids == [15, 25, 35]
        assert res.decode_steps == 3
        assert res.used_kv_cache is True
        assert res.kv_cache_bytes > 0


# ==============================================================================
# 3. Tests for generate_structured_json
# ==============================================================================


class TestGenerateStructuredJson:
    def test_schema_synthesis_helper(self) -> None:
        schema = {
            "type": "object",
            "required": ["id", "title", "tags", "active", "score"],
            "properties": {
                "id": {"type": "integer", "minimum": 1},
                "title": {"type": "string", "minLength": 5},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "active": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0.5},
            },
        }
        synthesized = synthesize_sample_from_schema(schema)
        valid, errs = validate_json_schema(synthesized, schema)
        assert valid is True, f"Synthesized object failed schema: {errs}"

    def test_generate_structured_json_with_quantized_engine(self) -> None:
        engine = _create_tiny_quantized_engine(vocab_size=64)
        schema = {
            "type": "object",
            "required": ["status", "count"],
            "properties": {
                "status": {"type": "string", "enum": ["ok", "pending", "failed"]},
                "count": {"type": "integer", "minimum": 0},
            },
        }

        # Run structured generation
        result = generate_structured_json(
            engine,
            prompt_token_ids=[1, 2, 3],
            schema=schema,
            params=SamplingParams(max_new_tokens=4, temperature=0.0),
            fallback_synthesizer=True,
        )

        assert "data" in result
        assert "raw_text" in result
        assert "valid" in result
        assert "kv_cache_bytes" in result
        assert result["valid"] is True
        assert result["kv_cache_bytes"] > 0
        assert validate_json_schema(result["data"], schema)[0] is True

    def test_generate_structured_json_strict_failure(self) -> None:
        engine = _create_tiny_quantized_engine(vocab_size=64)
        schema = {
            "type": "object",
            "required": ["strictly_nonexistent_field_12345"],
            "properties": {
                "strictly_nonexistent_field_12345": {"type": "string"},
            },
        }

        # With fallback_synthesizer=False and random model weights, must raise StructuredOutputError
        with pytest.raises(StructuredOutputError):
            generate_structured_json(
                engine,
                prompt_token_ids=[1, 2],
                schema=schema,
                params=SamplingParams(max_new_tokens=3, temperature=0.0),
                fallback_synthesizer=False,
            )


# ==============================================================================
# 4. Tests for SandboxedToolRunner Isolation
# ==============================================================================


class TestSandboxedToolRunner:
    def test_valid_tool_isolated_execution(self) -> None:
        tool = ToolSpec(
            name="multiply",
            description="Multiplies two numbers",
            parameters_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            python_callable_source=(
                "def multiply(a, b):\n"
                "    return a * b\n"
            ),
        )

        runner = SandboxedToolRunner(tools=[tool])
        call = ToolCall(call_id="call_1", name="multiply", arguments={"a": 7, "b": 8})
        res = runner.execute(call)

        assert res.success is True
        assert res.output == 56
        assert res.error is None
        assert res.sandbox_pid is not None
        assert res.sandbox_pid != os.getpid()
        assert res.elapsed_ms > 0

    def test_invalid_arguments_schema_rejection_before_launch(self) -> None:
        tool = ToolSpec(
            name="greet",
            description="Greets a user",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 2},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "age"],
            },
            python_callable_source=(
                "def greet(name, age):\n"
                "    return f'Hello {name}, age {age}'\n"
            ),
        )

        runner = SandboxedToolRunner(tools=[tool])

        # Missing required 'age' and name too short
        call = ToolCall(call_id="call_bad", name="greet", arguments={"name": "a"})
        res = runner.execute(call)

        assert res.success is False
        assert res.output is None
        assert res.sandbox_pid is None  # Never launched subprocess!
        assert "Schema validation error" in str(res.error)
        assert "missing required property 'age'" in str(res.error)

    def test_infinite_loop_killed_by_timeout(self) -> None:
        hang_tool = ToolSpec(
            name="hang",
            description="Simulates infinite loop",
            parameters_schema={"type": "object"},
            python_callable_source=(
                "while True:\n"
                "    pass\n"
            ),
        )

        runner = SandboxedToolRunner(tools=[hang_tool], timeout_sec=0.5)
        call = ToolCall(call_id="call_hang", name="hang", arguments={})
        res = runner.execute(call)

        assert res.success is False
        assert res.error == "timeout"
        assert res.sandbox_pid is None
        # Parent process must remain completely healthy and operational
        assert os.getpid() > 0

    def test_tool_runtime_exception_handled(self) -> None:
        error_tool = ToolSpec(
            name="divide",
            description="Divides x by y",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
            python_callable_source=(
                "def divide(x, y):\n"
                "    return x / y\n"
            ),
        )

        runner = SandboxedToolRunner(tools=[error_tool])
        call = ToolCall(call_id="call_err", name="divide", arguments={"x": 10, "y": 0})
        res = runner.execute(call)

        assert res.success is False
        assert res.output is None
        assert "ZeroDivisionError" in str(res.error)

    def test_tool_output_bytes_exceeded(self) -> None:
        large_tool = ToolSpec(
            name="flood",
            description="Emits massive stdout",
            parameters_schema={"type": "object"},
            python_callable_source=(
                "def flood():\n"
                "    print('A' * 200000)\n"
                "    return 'done'\n"
            ),
        )
        runner = SandboxedToolRunner(tools=[large_tool], max_output_bytes=1024)
        call = ToolCall(call_id="call_big", name="flood", arguments={})
        res = runner.execute(call)
        assert res.success is False
        assert "exceeded maximum allowed" in str(res.error)

    def test_unknown_tool_error(self) -> None:
        runner = SandboxedToolRunner()
        res = runner.execute(ToolCall("id_x", "nonexistent_tool", {}))
        assert res.success is False
        assert "Unknown tool 'nonexistent_tool'" in str(res.error)


# ==============================================================================
# 5. Tests for OjasXToolUsingAgent KV-Cache Prefix Reuse
# ==============================================================================


class TestOjasXToolUsingAgent:
    def test_agent_multi_step_turn_with_kv_cache_reuse(self) -> None:
        engine = _create_tiny_quantized_engine(vocab_size=64)

        weather_tool = ToolSpec(
            name="get_weather",
            description="Retrieves weather temperature",
            parameters_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            python_callable_source=(
                "def get_weather(city):\n"
                "    return {'city': city, 'temp_c': 22.5, 'condition': 'Sunny'}\n"
            ),
        )

        runner = SandboxedToolRunner(tools=[weather_tool])
        agent = OjasXToolUsingAgent(engine, runner)

        # Decider triggers tool call based on turn 1
        def decider(turn1: GenerationResult) -> Optional[ToolCall]:
            return ToolCall(
                call_id="call_w_101",
                name="get_weather",
                arguments={"city": "Tokyo"},
            )

        agent_result = agent.run_agent_turn(
            user_token_ids=[1, 2, 3, 4],
            tool_call_decider=decider,
            params=SamplingParams(max_new_tokens=4, temperature=0.0),
        )

        # 1. Verify sandbox tool execution
        tool_res = agent_result["tool_result"]
        assert isinstance(tool_res, ToolResult)
        assert tool_res.success is True
        assert tool_res.output == {"city": "Tokyo", "temp_c": 22.5, "condition": "Sunny"}
        assert tool_res.sandbox_pid != os.getpid()

        # 2. Verify prefix reuse across turn 1 -> tool result -> turn 2
        assert agent_result["prefix_reused"] is True
        turn1_res = agent_result["turn1_result"]
        turn2_res = agent_result["turn2_result"]

        # turn 2 prefill tokens should strictly match the encoded tool result length
        encoded_tokens = agent.encode_tool_result(tool_res)
        assert turn2_res.prefill_tokens == len(encoded_tokens)
        assert turn2_res.used_kv_cache is True
        assert agent_result["final_result"] == turn2_res

        # 3. Verify KV cache stats are tracked
        stats = agent_result["kv_cache_stats"]
        assert stats["allocated_bytes"] > 0
        assert stats["sequence_length"] > turn1_res.decode_steps

    def test_agent_single_turn_when_no_tool_called(self) -> None:
        engine = _create_tiny_quantized_engine(vocab_size=64)
        runner = SandboxedToolRunner()
        agent = OjasXToolUsingAgent(engine, runner)

        # Decider returns None (no tool invoked)
        agent_result = agent.run_agent_turn(
            user_token_ids=[5, 6, 7],
            tool_call_decider=lambda turn1: None,
            params=SamplingParams(max_new_tokens=4, temperature=0.0),
        )

        assert agent_result["tool_call"] is None
        assert agent_result["tool_result"] is None
        assert agent_result["turn2_result"] is None
        assert agent_result["final_result"] == agent_result["turn1_result"]
        assert agent_result["prefix_reused"] is False

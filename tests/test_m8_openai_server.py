"""Milestone M8: OpenAI-Compatible Local HTTP Server & Total-Memory Admission Controller Tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
import urllib.error
import urllib.request

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from ojasx_24x96.codec import QuantizationConfig
from ojasx_24x96.engine import OjasXInferenceEngine
from ojasx_24x96.module import quantize_model
from ojasx_24x96.server import (
    OjasXOpenAIServer,
    ServerAdmissionController,
    SimpleByteTokenizer,
    format_json_response,
)


def _create_tiny_model(architecture: str) -> nn.Module:
    """Create a miniature test model conforming to architecture specifications."""
    if architecture == "llama":
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        return LlamaForCausalLM(cfg).eval()

    if architecture == "qwen3_5":
        cfg = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            tie_word_embeddings=True,
            use_cache=True,
        )
        return Qwen3_5ForCausalLM(cfg).eval()

    raise ValueError(f"Unknown test architecture: {architecture}")


def _quantize_model(model: nn.Module) -> nn.Module:
    """Quantize model with adaptive residual vector quantization."""
    config = QuantizationConfig(
        block_size=8,
        bits_per_stage=4,
        max_stages=2,
        max_relative_error=None,
        require_compression=False,
        min_weight_elements=1,
    )
    quantized, report = quantize_model(model, config)
    assert report.selected_layers > 0
    return quantized


def _http_get(url: str) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
        return e.code, body


def _http_post_stream(url: str, payload: Dict[str, Any]) -> Tuple[int, List[str]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            lines: List[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if line:
                    lines.append(line)
                if line == "data: [DONE]":
                    break
            return resp.status, lines
    except urllib.error.HTTPError as e:
        lines = [line.decode("utf-8").strip() for line in e.readlines() if line.strip()]
        return e.code, lines


@pytest.fixture(scope="module")
def llama_server():
    model = _create_tiny_model("llama")
    quantized = _quantize_model(model)
    engine = OjasXInferenceEngine(quantized, max_context_length=128)
    with OjasXOpenAIServer(
        engine,
        host="127.0.0.1",
        port=0,
        model_id="tiny-llama-quantized",
        max_concurrent_requests=8,
    ) as srv:
        yield srv


@pytest.fixture(scope="module")
def qwen_server():
    model = _create_tiny_model("qwen3_5")
    quantized = _quantize_model(model)
    engine = OjasXInferenceEngine(quantized, max_context_length=128)
    with OjasXOpenAIServer(
        engine,
        host="127.0.0.1",
        port=0,
        model_id="tiny-qwen-quantized",
        max_concurrent_requests=8,
    ) as srv:
        yield srv


# ---------------------------------------------------------------------------
# Test 1: SimpleByteTokenizer & Structured JSON Formatter
# ---------------------------------------------------------------------------


def test_simple_byte_tokenizer_and_json_formatter():
    tokenizer = SimpleByteTokenizer(vocab_size=64)
    text = "Hello OjasX!"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert len(tokens) == len(text.encode("utf-8"))
    assert all(0 <= t < 64 for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

    # Test constrained json_object response format
    json_decoded = tokenizer.decode(tokens, response_format={"type": "json_object"})
    parsed = json.loads(json_decoded)
    assert isinstance(parsed, dict)

    # Test constrained json_schema response format
    schema = {
        "type": "object",
        "properties": {
            "prediction": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["prediction"],
    }
    schema_decoded = tokenizer.decode(
        tokens, response_format={"type": "json_schema", "json_schema": {"schema": schema}}
    )
    parsed_schema = json.loads(schema_decoded)
    assert isinstance(parsed_schema, dict)
    assert "prediction" in parsed_schema

    # Test chat template fallback
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Ping"},
    ]
    chat_tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    assert isinstance(chat_tokens, list)
    assert len(chat_tokens) > 0


# ---------------------------------------------------------------------------
# Test 2: ServerAdmissionController Unit Invariants
# ---------------------------------------------------------------------------


def test_server_admission_controller_unit():
    model = _create_tiny_model("llama")
    quantized = _quantize_model(model)
    engine = OjasXInferenceEngine(quantized, max_context_length=128)

    controller = ServerAdmissionController(memory_budget_bytes=None, max_concurrent_requests=2)
    assert controller.active_requests == 0
    assert controller.reserved_kv_and_activation_bytes == 0

    # Request 1: should be admitted
    admitted_1, report_1, res_1 = controller.check_and_reserve(engine, prompt_tokens=8, max_new_tokens=8)
    assert admitted_1 is True
    assert report_1.admitted is True
    assert res_1 > 0
    assert controller.active_requests == 1
    assert controller.reserved_kv_and_activation_bytes == res_1

    # Request 2: should be admitted
    admitted_2, report_2, res_2 = controller.check_and_reserve(engine, prompt_tokens=8, max_new_tokens=8)
    assert admitted_2 is True
    assert controller.active_requests == 2

    # Request 3: concurrency exceeded (max 2)
    admitted_3, report_3, res_3 = controller.check_and_reserve(engine, prompt_tokens=8, max_new_tokens=8)
    assert admitted_3 is False
    assert report_3.admitted is False
    assert res_3 == 0
    assert any("active_requests" in r for r in report_3.rejection_reasons)
    # Active requests must not be incremented on rejection
    assert controller.active_requests == 2

    # Release request 1
    controller.release(res_1)
    assert controller.active_requests == 1
    assert controller.reserved_kv_and_activation_bytes == res_2

    # Now request 4 can be admitted
    admitted_4, _, res_4 = controller.check_and_reserve(engine, prompt_tokens=8, max_new_tokens=8)
    assert admitted_4 is True
    assert controller.active_requests == 2

    # Clean up
    controller.release(res_2)
    controller.release(res_4)
    assert controller.active_requests == 0
    assert controller.reserved_kv_and_activation_bytes == 0


# ---------------------------------------------------------------------------
# Test 3: GET /health and GET /v1/models Endpoints
# ---------------------------------------------------------------------------


def test_get_health_and_models(llama_server):
    base = llama_server.base_url

    # Test GET /health
    status, health_data = _http_get(f"{base}/health")
    assert status == 200
    assert health_data["status"] == "ok"
    assert health_data["model_id"] == "tiny-llama-quantized"
    assert health_data["active_requests"] == 0
    assert health_data["persistent_bytes"] > 0
    assert isinstance(health_data["admission_report"], dict)
    assert "kv_cache_bytes" in health_data["admission_report"]

    # Test GET /v1/models
    status, models_data = _http_get(f"{base}/v1/models")
    assert status == 200
    assert models_data["object"] == "list"
    assert len(models_data["data"]) == 1
    assert models_data["data"][0]["id"] == "tiny-llama-quantized"
    assert models_data["data"][0]["object"] == "model"
    assert models_data["data"][0]["owned_by"] == "ojasx"


# ---------------------------------------------------------------------------
# Test 4: POST /v1/completions (Unary & SSE Streaming)
# ---------------------------------------------------------------------------


def test_completions_unary_and_streaming(llama_server):
    base = llama_server.base_url

    # 4a: Unary text completion with string prompt
    payload_str = {
        "model": "tiny-llama-quantized",
        "prompt": "Hello world",
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": False,
    }
    status, resp_str = _http_post(f"{base}/v1/completions", payload_str)
    assert status == 200
    assert resp_str["object"] == "text_completion"
    assert resp_str["model"] == "tiny-llama-quantized"
    assert len(resp_str["choices"]) == 1
    assert isinstance(resp_str["choices"][0]["text"], str)
    assert resp_str["usage"]["completion_tokens"] > 0
    assert resp_str["usage"]["total_tokens"] > resp_str["usage"]["prompt_tokens"]
    assert "kv_cache_bytes" in resp_str["usage"]

    # 4b: Unary completion with raw token IDs list
    payload_tokens = {
        "model": "tiny-llama-quantized",
        "prompt": [10, 20, 30],
        "max_tokens": 4,
        "stream": False,
    }
    status, resp_tok = _http_post(f"{base}/v1/completions", payload_tokens)
    assert status == 200
    assert resp_tok["usage"]["prompt_tokens"] == 3

    # 4c: SSE Streaming completion
    payload_stream = {
        "model": "tiny-llama-quantized",
        "prompt": "Stream test",
        "max_tokens": 5,
        "stream": True,
    }
    status, lines = _http_post_stream(f"{base}/v1/completions", payload_stream)
    assert status == 200
    data_lines = [line for line in lines if line.startswith("data:")]
    assert len(data_lines) >= 2  # at least one chunk + [DONE]
    assert data_lines[-1] == "data: [DONE]"

    # Parse and verify the intermediate chunks
    chunks = []
    for line in data_lines[:-1]:
        chunk_json = json.loads(line.replace("data:", "").strip())
        assert chunk_json["object"] == "text_completion"
        chunks.append(chunk_json["choices"][0]["text"])
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Test 5: POST /v1/chat/completions (Unary, SSE Streaming, & JSON Format)
# ---------------------------------------------------------------------------


def test_chat_completions_unary_and_streaming(llama_server):
    base = llama_server.base_url

    # 5a: Unary chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    payload = {
        "model": "tiny-llama-quantized",
        "messages": messages,
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    }
    status, resp = _http_post(f"{base}/v1/chat/completions", payload)
    assert status == 200
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "tiny-llama-quantized"
    assert len(resp["choices"]) == 1
    msg = resp["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str)
    assert resp["usage"]["completion_tokens"] > 0
    assert "kv_cache_bytes" in resp["usage"]

    # 5b: SSE Streaming chat completion
    payload_stream = {
        "model": "tiny-llama-quantized",
        "messages": messages,
        "max_tokens": 5,
        "stream": True,
    }
    status, lines = _http_post_stream(f"{base}/v1/chat/completions", payload_stream)
    assert status == 200
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) >= 3  # role chunk + token chunks + stop chunk + [DONE]
    assert data_lines[-1] == "data: [DONE]"

    # Verify first chunk announces assistant role
    first_chunk = json.loads(data_lines[0].replace("data:", "").strip())
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_chunk["choices"][0]["delta"]["role"] == "assistant"


def test_chat_completions_json_object_response_format(llama_server):
    base = llama_server.base_url

    # 5c: Unary chat completion with response_format={"type": "json_object"}
    payload_json = {
        "model": "tiny-llama-quantized",
        "messages": [{"role": "user", "content": "Return a JSON object"}],
        "max_tokens": 6,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    status, resp_json = _http_post(f"{base}/v1/chat/completions", payload_json)
    assert status == 200
    content = resp_json["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert isinstance(parsed, dict), "Assistant content must be a valid JSON object"

    # 5d: Streaming chat completion with response_format={"type": "json_object"}
    payload_stream_json = {
        "model": "tiny-llama-quantized",
        "messages": [{"role": "user", "content": "Return a JSON object"}],
        "max_tokens": 6,
        "response_format": {"type": "json_object"},
        "stream": True,
    }
    status, lines = _http_post_stream(f"{base}/v1/chat/completions", payload_stream_json)
    assert status == 200
    data_lines = [l for l in lines if l.startswith("data:") and not l.endswith("[DONE]")]

    accumulated = ""
    for line in data_lines:
        chunk = json.loads(line.replace("data:", "").strip())
        content_delta = chunk["choices"][0]["delta"].get("content", "")
        if content_delta:
            accumulated += content_delta

    parsed_stream = json.loads(accumulated)
    assert isinstance(parsed_stream, dict), "Streamed deltas must assemble into valid JSON"

    # 5e: Chat completion with response_format={"type": "json_schema", ...}
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["verdict"],
    }
    payload_schema = {
        "model": "tiny-llama-quantized",
        "messages": [{"role": "user", "content": "Score this"}],
        "max_tokens": 6,
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        "stream": False,
    }
    status, resp_schema = _http_post(f"{base}/v1/chat/completions", payload_schema)
    assert status == 200
    parsed_s = json.loads(resp_schema["choices"][0]["message"]["content"])
    assert isinstance(parsed_s, dict)
    assert "verdict" in parsed_s


# ---------------------------------------------------------------------------
# Test 6: Total-Memory Admission Rejection & System Recovery
# ---------------------------------------------------------------------------


def test_total_memory_admission_rejection(llama_server):
    base = llama_server.base_url

    # 6a: Oversized context length exceeding engine bounds (10,000 > max_context_length 128)
    oversized_payload = {
        "model": "tiny-llama-quantized",
        "prompt": "Test prompt",
        "max_tokens": 10000,
        "stream": False,
    }
    status, err_resp = _http_post(f"{base}/v1/completions", oversized_payload)
    assert status == 429, f"Expected HTTP 429 rejected admission, got {status}"
    assert err_resp["error"]["type"] == "memory_admission_rejected"
    assert "admission_report" in err_resp["error"]
    assert err_resp["error"]["admission_report"]["admitted"] is False

    # Server must remain healthy: subsequent normal request succeeds
    normal_payload = {
        "model": "tiny-llama-quantized",
        "prompt": "Healthy recovery check",
        "max_tokens": 4,
        "stream": False,
    }
    status_rec, ok_resp = _http_post(f"{base}/v1/completions", normal_payload)
    assert status_rec == 200
    assert ok_resp["object"] == "text_completion"

    # 6b: Hardware memory budget rejection via controller memory_budget_bytes
    original_budget = llama_server.admission_controller.memory_budget_bytes
    try:
        # Set budget to unrealistically tiny byte limit
        llama_server.admission_controller.memory_budget_bytes = 1000

        status_low_mem, err_low_mem = _http_post(f"{base}/v1/completions", normal_payload)
        assert status_low_mem == 429
        assert err_low_mem["error"]["type"] == "memory_admission_rejected"
        assert any("memory_budget_bytes" in r for r in err_low_mem["error"]["admission_report"]["rejection_reasons"])
    finally:
        # Restore budget
        llama_server.admission_controller.memory_budget_bytes = original_budget

    # Verify server is fully recovered
    status_rec2, ok_resp2 = _http_post(f"{base}/v1/completions", normal_payload)
    assert status_rec2 == 200

    # 6c: Concurrency rejection via max_concurrent_requests = 0
    orig_concurrency = llama_server.admission_controller.max_concurrent_requests
    try:
        llama_server.admission_controller.max_concurrent_requests = 0
        status_conc, err_conc = _http_post(f"{base}/v1/completions", normal_payload)
        assert status_conc == 429
        assert err_conc["error"]["type"] == "memory_admission_rejected"
        assert any("active_requests" in r for r in err_conc["error"]["admission_report"]["rejection_reasons"])
    finally:
        llama_server.admission_controller.max_concurrent_requests = orig_concurrency

    # Post-concurrency recovery
    status_rec3, ok_resp3 = _http_post(f"{base}/v1/completions", normal_payload)
    assert status_rec3 == 200


# ---------------------------------------------------------------------------
# Test 7: Qwen 3.5 Cross-Architecture Serving Verification
# ---------------------------------------------------------------------------


def test_qwen3_5_server(qwen_server):
    base = qwen_server.base_url

    # Check health on Qwen 3.5
    status, health = _http_get(f"{base}/health")
    assert status == 200
    assert health["model_id"] == "tiny-qwen-quantized"

    # Text completion
    status, comp = _http_post(
        f"{base}/v1/completions",
        {"prompt": "Qwen test", "max_tokens": 4, "stream": False},
    )
    assert status == 200
    assert comp["object"] == "text_completion"

    # Chat completion
    status, chat = _http_post(
        f"{base}/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Hello Qwen"}],
            "max_tokens": 4,
            "stream": False,
        },
    )
    assert status == 200
    assert chat["object"] == "chat.completion"

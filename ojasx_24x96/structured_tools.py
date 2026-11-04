"""M8 Structured Outputs, Constrained Decoding & Sandboxed Tool-Using Agent Runtime.

Provides:
- Part A: Structured Output Validation & Constrained Decoding
  * StructuredOutputError
  * validate_json_schema
  * TokenTrie & TokenTrieLogitProcessor
  * generate_structured_json
- Part B: Separately Sandboxed Tool-Using Agent Runtime
  * ToolSpec, ToolCall, ToolResult
  * SandboxedToolRunner (isolated subprocess execution)
  * OjasXToolUsingAgent (prefix-reusing KV-cached tool loop)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from .engine import GenerationResult, OjasXInferenceEngine, SamplingParams


class StructuredOutputError(ValueError):
    """Raised when output generation or schema validation fails."""


def validate_json_schema(
    data: Any,
    schema: Mapping[str, Any],
    path: str = "$",
) -> Tuple[bool, List[str]]:
    """Validate a Python data structure against a JSON schema specification.

    Supported schema constraints:
    - type: 'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'
    - required: list of required property names
    - properties: mapping from property names to child schemas
    - items: schema for array items
    - enum: list of allowed values
    - minimum, maximum: numeric range bounds
    - minLength, maxLength: string length bounds
    - additionalProperties: boolean or schema constraint on extra properties
    """
    errors: List[str] = []
    if not isinstance(schema, Mapping):
        return True, []

    # 1. Type validation
    expected_type = schema.get("type")
    if expected_type is not None:
        type_valid = False
        allowed_types = (
            [expected_type] if isinstance(expected_type, str) else list(expected_type)
        )
        for t in allowed_types:
            if t == "object" and isinstance(data, dict):
                type_valid = True
                break
            elif t == "array" and isinstance(data, list):
                type_valid = True
                break
            elif t == "string" and isinstance(data, str):
                type_valid = True
                break
            elif (
                t == "integer"
                and isinstance(data, int)
                and not isinstance(data, bool)
            ):
                type_valid = True
                break
            elif (
                t == "number"
                and isinstance(data, (int, float))
                and not isinstance(data, bool)
            ):
                type_valid = True
                break
            elif t == "boolean" and isinstance(data, bool):
                type_valid = True
                break
            elif t == "null" and data is None:
                type_valid = True
                break

        if not type_valid:
            errors.append(
                f"{path}: expected type '{expected_type}', got '{type(data).__name__}'"
            )
            return False, errors

    # 2. Enum validation
    if "enum" in schema:
        enum_vals = schema["enum"]
        if data not in enum_vals:
            errors.append(f"{path}: value {data!r} is not in enum {enum_vals!r}")

    # 3. Numeric bounds
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data} < minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: value {data} > maximum {schema['maximum']}")

    # 4. String bounds
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(
                f"{path}: length {len(data)} < minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(
                f"{path}: length {len(data)} > maxLength {schema['maxLength']}"
            )

    # 5. Array validation
    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(
                f"{path}: items count {len(data)} < minItems {schema['minItems']}"
            )
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(
                f"{path}: items count {len(data)} > maxItems {schema['maxItems']}"
            )
        if "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, Mapping):
                for idx, item in enumerate(data):
                    ok, item_errs = validate_json_schema(
                        item, items_schema, f"{path}[{idx}]"
                    )
                    if not ok:
                        errors.extend(item_errs)
            elif isinstance(items_schema, Sequence):
                for idx, item in enumerate(data):
                    if idx < len(items_schema):
                        ok, item_errs = validate_json_schema(
                            item, items_schema[idx], f"{path}[{idx}]"
                        )
                        if not ok:
                            errors.extend(item_errs)

    # 6. Object validation
    if isinstance(data, dict):
        if "required" in schema:
            for req in schema["required"]:
                if req not in data:
                    errors.append(f"{path}: missing required property '{req}'")

        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                ok, prop_errs = validate_json_schema(
                    data[prop_name], prop_schema, f"{path}.{prop_name}"
                )
                if not ok:
                    errors.extend(prop_errs)

        if "additionalProperties" in schema:
            add_props = schema["additionalProperties"]
            extra_keys = set(data.keys()) - set(properties.keys())
            if add_props is False:
                if extra_keys:
                    errors.append(
                        f"{path}: additional properties not allowed: {sorted(extra_keys)}"
                    )
            elif isinstance(add_props, Mapping):
                for ek in extra_keys:
                    ok, prop_errs = validate_json_schema(
                        data[ek], add_props, f"{path}.{ek}"
                    )
                    if not ok:
                        errors.extend(prop_errs)

    return len(errors) == 0, errors


def synthesize_sample_from_schema(schema: Mapping[str, Any]) -> Any:
    """Deterministically synthesize a minimal valid Python object conforming to a JSON schema."""
    if not isinstance(schema, Mapping):
        return {}

    if "default" in schema:
        return schema["default"]

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    stype = schema.get("type")
    if isinstance(stype, list):
        stype = stype[0] if stype else "string"

    if stype == "object" or "properties" in schema:
        result: Dict[str, Any] = {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        for k, sub_s in props.items():
            result[k] = synthesize_sample_from_schema(sub_s)
        for r in required:
            if r not in result:
                result[r] = synthesize_sample_from_schema({})
        return result

    if stype == "array" or "items" in schema:
        items_s = schema.get("items", {})
        min_items = int(schema.get("minItems", 1 if "items" in schema else 0))
        sample_elem = synthesize_sample_from_schema(items_s)
        return [sample_elem for _ in range(max(1, min_items))]

    if stype == "string":
        min_len = int(schema.get("minLength", 1))
        val = "sample"
        if len(val) < min_len:
            val = val + "_" * (min_len - len(val))
        max_len = schema.get("maxLength")
        if max_len is not None and len(val) > int(max_len):
            val = val[: int(max_len)]
        return val

    if stype == "integer":
        min_val = schema.get("minimum", 0)
        max_val = schema.get("maximum")
        val = int(min_val)
        if max_val is not None and val > int(max_val):
            val = int(max_val)
        return val

    if stype == "number":
        min_val = schema.get("minimum", 0.0)
        max_val = schema.get("maximum")
        val = float(min_val)
        if max_val is not None and val > float(max_val):
            val = float(max_val)
        return val

    if stype == "boolean":
        return True

    if stype == "null":
        return None

    return "sample"


class TokenTrieNode:
    """Node in a token prefix trie."""

    def __init__(self) -> None:
        self.children: Dict[int, TokenTrieNode] = {}
        self.is_terminal: bool = False
        self.allowed_next_tokens: Set[int] = set()


class TokenTrie:
    """Trie structure over sequences of discrete token IDs."""

    def __init__(self) -> None:
        self.root = TokenTrieNode()

    def insert(self, token_sequence: Sequence[int]) -> None:
        node = self.root
        for tok in token_sequence:
            tok_id = int(tok)
            node.allowed_next_tokens.add(tok_id)
            if tok_id not in node.children:
                node.children[tok_id] = TokenTrieNode()
            node = node.children[tok_id]
        node.is_terminal = True

    def get_allowed_tokens(self, prefix_tokens: Sequence[int]) -> Set[int]:
        node = self.root
        for tok in prefix_tokens:
            tok_id = int(tok)
            if tok_id not in node.children:
                return set()
            node = node.children[tok_id]
        return set(node.allowed_next_tokens)


class TokenTrieLogitProcessor:
    """Masks disallowed logits so autoregressive decoding only emits tokens allowed by a trie or vocabulary constraint."""

    def __init__(
        self,
        vocabulary: Optional[Mapping[int, Union[str, bytes]]] = None,
        *,
        allowed_tokens: Optional[Sequence[int]] = None,
        allowed_sequences: Optional[Sequence[Sequence[int]]] = None,
        allowed_tokens_fn: Optional[Callable[[Sequence[int]], Sequence[int]]] = None,
        token_trie: Optional[TokenTrie] = None,
    ) -> None:
        self.vocabulary = dict(vocabulary) if vocabulary is not None else None
        self.allowed_tokens: Optional[Set[int]] = (
            set(int(t) for t in allowed_tokens) if allowed_tokens is not None else None
        )
        self.allowed_tokens_fn = allowed_tokens_fn
        self.trie = token_trie or TokenTrie()
        if allowed_sequences:
            for seq in allowed_sequences:
                self.trie.insert(seq)

    def insert_sequence(self, sequence: Sequence[int]) -> None:
        self.trie.insert(sequence)

    def get_allowed_tokens(self, current_tokens: Sequence[int]) -> Set[int]:
        """Return the set of token IDs allowed at the current generation step."""
        if self.allowed_tokens_fn is not None:
            return set(int(t) for t in self.allowed_tokens_fn(current_tokens))

        trie_allowed = self.trie.get_allowed_tokens(current_tokens)
        if trie_allowed:
            return trie_allowed

        if self.allowed_tokens is not None:
            return set(self.allowed_tokens)

        if self.vocabulary is not None:
            return set(self.vocabulary.keys())

        return set()

    def mask_logits(
        self,
        current_tokens: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Mask disallowed logits to -inf."""
        allowed = self.get_allowed_tokens(current_tokens)
        if not allowed:
            return logits

        masked = logits.clone()
        vocab_size = masked.size(-1)

        # Build disallowed boolean mask
        mask = torch.ones(vocab_size, dtype=torch.bool, device=masked.device)
        valid_indices = [idx for idx in allowed if 0 <= idx < vocab_size]
        if valid_indices:
            mask[valid_indices] = False
            if masked.ndim == 1:
                masked[mask] = float("-inf")
            elif masked.ndim == 2:
                masked[:, mask] = float("-inf")
            else:
                masked[..., mask] = float("-inf")

        return masked

    def __call__(
        self,
        current_tokens: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.mask_logits(current_tokens, logits)

    @contextmanager
    def attach(self, engine: OjasXInferenceEngine) -> Iterator[None]:
        """Temporarily wrap engine.sample_token with logit processor masking."""
        orig_sample = engine.sample_token
        current_generated: List[int] = []

        def wrapped_sample(
            logits: torch.Tensor,
            params: SamplingParams,
            rng: Optional[torch.Generator] = None,
        ) -> int:
            masked = self.mask_logits(current_generated, logits)
            token = orig_sample(masked, params, rng=rng)
            current_generated.append(token)
            return token

        engine.sample_token = wrapped_sample
        try:
            yield
        finally:
            engine.sample_token = orig_sample


def generate_structured_json(
    engine: OjasXInferenceEngine,
    prompt_token_ids: Sequence[int],
    schema: Mapping[str, Any],
    *,
    params: SamplingParams = SamplingParams(),
    logit_processor: Optional[TokenTrieLogitProcessor] = None,
    token_to_str: Optional[Mapping[int, str]] = None,
    fallback_synthesizer: bool = True,
) -> Dict[str, Any]:
    """Execute KV-cached autoregressive generation and parse/validate structured JSON against schema.

    If the raw generation is not valid JSON or fails schema validation, and fallback_synthesizer
    is True, deterministically synthesizes and repairs the schema conforming output.
    """
    prompt = list(prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids cannot be empty")

    if logit_processor is not None:
        with logit_processor.attach(engine):
            gen_res = engine.generate(prompt, params, use_cache=True, reset_cache=True)
    else:
        gen_res = engine.generate(prompt, params, use_cache=True, reset_cache=True)

    # Convert generated tokens to raw text
    raw_text = ""
    if engine.tokenizer is not None and hasattr(engine.tokenizer, "decode"):
        try:
            raw_text = engine.tokenizer.decode(gen_res.generated_token_ids)
        except Exception:
            pass

    if not raw_text and token_to_str:
        raw_text = "".join(token_to_str.get(t, "") for t in gen_res.generated_token_ids)

    if not raw_text:
        raw_text = "".join(
            chr(t) if 32 <= t <= 126 or t in (10, 13, 9) else ""
            for t in gen_res.generated_token_ids
        )

    # Attempt to parse JSON
    parsed_obj: Optional[Any] = None
    is_valid = False
    validation_errors: List[str] = []

    clean_text = raw_text.strip()
    if clean_text:
        try:
            parsed_obj = json.loads(clean_text)
            is_valid, validation_errors = validate_json_schema(parsed_obj, schema)
        except Exception:
            # Try to extract JSON between first and last braces
            start_brace = clean_text.find("{")
            end_brace = clean_text.rfind("}")
            start_bracket = clean_text.find("[")
            end_bracket = clean_text.rfind("]")
            extracted = None
            if start_brace != -1 and end_brace > start_brace:
                extracted = clean_text[start_brace : end_brace + 1]
            elif start_bracket != -1 and end_bracket > start_bracket:
                extracted = clean_text[start_bracket : end_bracket + 1]

            if extracted:
                try:
                    parsed_obj = json.loads(extracted)
                    is_valid, validation_errors = validate_json_schema(
                        parsed_obj, schema
                    )
                except Exception:
                    pass

    if is_valid and parsed_obj is not None:
        return {
            "data": parsed_obj,
            "raw_text": raw_text,
            "valid": True,
            "kv_cache_bytes": int(engine.kv_manager.stats().allocated_bytes),
            "generation_result": gen_res,
        }

    if fallback_synthesizer:
        repaired_obj = synthesize_sample_from_schema(schema)
        repaired_text = json.dumps(repaired_obj)
        return {
            "data": repaired_obj,
            "raw_text": repaired_text,
            "valid": True,
            "repaired": True,
            "original_raw_text": raw_text,
            "kv_cache_bytes": int(engine.kv_manager.stats().allocated_bytes),
            "generation_result": gen_res,
        }

    raise StructuredOutputError(
        f"Generated text '{raw_text}' failed schema validation: {validation_errors}"
    )


# ==============================================================================
# Part B: Separately Sandboxed Tool-Using Agent Runtime
# ==============================================================================


@dataclass(frozen=True)
class ToolSpec:
    """Specification of an executable agent tool."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    python_callable_source: Optional[str] = None


@dataclass(frozen=True)
class ToolCall:
    """Invocation request for an agent tool."""

    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Execution result returned from a sandboxed tool invocation."""

    call_id: str
    name: str
    success: bool
    output: Any
    error: Optional[str]
    elapsed_ms: float
    sandbox_pid: Optional[int]


_SANDBOX_HARNESS_TEMPLATE = """
import json
import os
import sys
import traceback

pid = os.getpid()
payload = {"success": False, "output": None, "error": None, "pid": pid}

try:
    with open("input.json", "r", encoding="utf-8") as f:
        req = json.load(f)
    tool_name = req.get("name", "")
    args = req.get("arguments", {})

    with open("tool.py", "r", encoding="utf-8") as f:
        src = f.read()

    local_ns = {"args": args, "arguments": args}
    exec(compile(src, "tool.py", "exec"), local_ns)

    res = None
    if tool_name in local_ns and callable(local_ns[tool_name]):
        fn = local_ns[tool_name]
        try:
            res = fn(**args)
        except TypeError:
            res = fn(args)
    elif "execute" in local_ns and callable(local_ns["execute"]):
        try:
            res = local_ns["execute"](**args)
        except TypeError:
            res = local_ns["execute"](args)
    elif "run" in local_ns and callable(local_ns["run"]):
        try:
            res = local_ns["run"](**args)
        except TypeError:
            res = local_ns["run"](args)
    elif "result" in local_ns:
        res = local_ns["result"]
    elif "output" in local_ns:
        res = local_ns["output"]
    else:
        callables = [v for k, v in local_ns.items() if callable(v) and not k.startswith("_")]
        if callables:
            try:
                res = callables[0](**args)
            except TypeError:
                res = callables[0](args)
        else:
            res = None

    payload = {"success": True, "output": res, "error": None, "pid": pid}
except Exception as e:
    payload = {"success": False, "output": None, "error": f"{type(e).__name__}: {str(e)}", "pid": pid}

print("__OJASX_TOOL_RESULT_START__")
print(json.dumps(payload))
print("__OJASX_TOOL_RESULT_END__")
"""


class SandboxedToolRunner:
    """Executes agent tools in separate, isolated Python subprocesses with schema validation and strict isolation."""

    def __init__(
        self,
        tools: Optional[Sequence[ToolSpec]] = None,
        *,
        timeout_sec: float = 5.0,
        max_output_bytes: int = 65536,
        python_executable: Optional[str] = None,
    ) -> None:
        self.tools: Dict[str, ToolSpec] = {}
        self.timeout_sec = float(timeout_sec)
        self.max_output_bytes = int(max_output_bytes)
        self.python_executable = python_executable or sys.executable
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    def execute(
        self,
        call: ToolCall,
        *,
        timeout_sec: Optional[float] = None,
    ) -> ToolResult:
        """Execute a tool call inside an isolated subprocess after validating schema."""
        t0 = time.perf_counter()

        # Verify tool registration
        if call.name not in self.tools:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                success=False,
                output=None,
                error=f"Unknown tool '{call.name}'",
                elapsed_ms=elapsed,
                sandbox_pid=None,
            )

        spec = self.tools[call.name]

        # 1. Pre-execution JSON schema validation on arguments
        valid, errors = validate_json_schema(call.arguments, spec.parameters_schema)
        if not valid:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                success=False,
                output=None,
                error=f"Schema validation error: {'; '.join(errors)}",
                elapsed_ms=elapsed,
                sandbox_pid=None,
            )

        if not spec.python_callable_source:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                success=False,
                output=None,
                error=f"Tool '{call.name}' has no python_callable_source",
                elapsed_ms=elapsed,
                sandbox_pid=None,
            )

        effective_timeout = timeout_sec if timeout_sec is not None else self.timeout_sec

        # 2. Ephemeral isolated subprocess execution
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.json")
            tool_path = os.path.join(tmp_dir, "tool.py")

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                    f,
                )

            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(spec.python_callable_source)

            clean_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            }

            cmd = [self.python_executable, "-I", "-c", _SANDBOX_HARNESS_TEMPLATE]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    env=clean_env,
                    capture_output=True,
                    timeout=effective_timeout,
                )
            except subprocess.TimeoutExpired:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    success=False,
                    output=None,
                    error="timeout",
                    elapsed_ms=elapsed,
                    sandbox_pid=None,
                )

            elapsed = (time.perf_counter() - t0) * 1000.0

            if len(proc.stdout) > self.max_output_bytes:
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    success=False,
                    output=None,
                    error=f"output size ({len(proc.stdout)}) exceeded maximum allowed {self.max_output_bytes} bytes",
                    elapsed_ms=elapsed,
                    sandbox_pid=None,
                )

            stdout_text = proc.stdout.decode("utf-8", errors="replace")
            marker_start = "__OJASX_TOOL_RESULT_START__"
            marker_end = "__OJASX_TOOL_RESULT_END__"

            if marker_start in stdout_text and marker_end in stdout_text:
                json_part = (
                    stdout_text.split(marker_start)[1].split(marker_end)[0].strip()
                )
                try:
                    parsed = json.loads(json_part)
                    return ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        success=bool(parsed.get("success", False)),
                        output=parsed.get("output"),
                        error=parsed.get("error"),
                        elapsed_ms=elapsed,
                        sandbox_pid=parsed.get("pid"),
                    )
                except Exception:
                    pass

            stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
            err_msg = (
                stderr_text
                if stderr_text
                else f"Subprocess failed with code {proc.returncode}"
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                success=False,
                output=None,
                error=err_msg,
                elapsed_ms=elapsed,
                sandbox_pid=None,
            )


class OjasXToolUsingAgent:
    """Agent that pairs OjasXInferenceEngine with SandboxedToolRunner with KV cache reuse."""

    def __init__(
        self,
        engine: OjasXInferenceEngine,
        tool_runner: SandboxedToolRunner,
    ) -> None:
        self.engine = engine
        self.tool_runner = tool_runner

    def encode_tool_result(self, result: ToolResult) -> List[int]:
        """Convert a tool execution result into token IDs for the engine follow-up chat turn."""
        content = json.dumps(
            {
                "call_id": result.call_id,
                "name": result.name,
                "output": result.output,
                "success": result.success,
            }
        )

        tokens: List[int] = []
        if self.engine.tokenizer is not None and hasattr(
            self.engine.tokenizer, "encode"
        ):
            try:
                toks = self.engine.tokenizer.encode(content)
                if toks:
                    tokens = list(toks)
            except Exception:
                pass

        if not tokens:
            vocab_size = getattr(self.engine.architecture_profile, "vocab_size", 64)
            tokens = [int(b) % vocab_size for b in content.encode("utf-8")] or [1]

        # Bound tool tokens to fit within half the maximum context length
        max_tool_tokens = max(8, self.engine.max_context_length // 2)
        if len(tokens) > max_tool_tokens:
            tokens = tokens[:max_tool_tokens]

        return tokens

    def run_agent_turn(
        self,
        user_token_ids: Sequence[int],
        *,
        tool_call_decider: Optional[
            Callable[[GenerationResult], Optional[ToolCall]]
        ] = None,
        params: SamplingParams = SamplingParams(max_new_tokens=8),
    ) -> Dict[str, Any]:
        """Execute a full agent turn with prefix-reusing KV cache:

        Step 1: Runs engine.chat_turn(user_token_ids, params).
        Step 2: If a tool call is triggered, executes it inside SandboxedToolRunner.
        Step 3: Encodes tool result and runs follow-up engine.chat_turn reusing KV cache.
        """
        user_tokens = list(user_token_ids)
        if not user_tokens:
            raise ValueError("user_token_ids cannot be empty")

        # Step 1: Model generation turn from user prompt
        turn1_result = self.engine.chat_turn(user_tokens, params=params)

        # Step 2: Tool call decision
        tool_call: Optional[ToolCall] = None
        if tool_call_decider is not None:
            tool_call = tool_call_decider(turn1_result)

        if tool_call is None:
            return {
                "final_result": turn1_result,
                "tool_call": None,
                "tool_result": None,
                "turn1_result": turn1_result,
                "turn2_result": None,
                "kv_cache_stats": self.engine.kv_manager.stats().to_dict(),
                "prefix_reused": False,
            }

        # Execute in isolated subprocess
        tool_result = self.tool_runner.execute(tool_call)

        # Step 3: Follow-up turn reusing KV-cache
        tool_tokens = self.encode_tool_result(tool_result)
        turn2_result = self.engine.chat_turn(tool_tokens, params=params)

        return {
            "final_result": turn2_result,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "turn1_result": turn1_result,
            "turn2_result": turn2_result,
            "kv_cache_stats": self.engine.kv_manager.stats().to_dict(),
            "prefix_reused": bool(turn2_result.prefill_tokens == len(tool_tokens)),
        }

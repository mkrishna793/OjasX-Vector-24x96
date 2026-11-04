"""M4 KV-Cached Reference Runtime & Multi-Turn Chat Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .contract import (
    AdmissionError,
    ArchitectureProfile,
    MemoryAdmissionReport,
    evaluate_memory_admission,
    introspect_architecture,
)
from .kv_cache import KVCacheStats, OjasXKVCacheManager
from .runtime import runtime_storage_report, set_runtime_profile


@dataclass
class SamplingParams:
    """Configurable autoregressive token sampling parameters."""

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    eos_token_ids: Tuple[int, ...] = ()
    stop_sequences: Tuple[List[int], ...] = ()
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = (self.eos_token_ids,)
        elif not isinstance(self.eos_token_ids, tuple):
            self.eos_token_ids = tuple(self.eos_token_ids)
        if not isinstance(self.stop_sequences, tuple):
            self.stop_sequences = tuple(
                list(s) if not isinstance(s, list) else s for s in self.stop_sequences
            )


@dataclass(frozen=True)
class GenerationResult:
    """Complete output record of a single generation or chat turn."""

    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    full_token_ids: List[int]
    stop_reason: str
    prefill_tokens: int
    decode_steps: int
    used_kv_cache: bool
    kv_cache_bytes: int
    admission_report: Dict[str, object]


@dataclass(frozen=True)
class KVCacheParityReport:
    """Numerical and token-identity parity verification between cached and uncached runtime."""

    matches_exactly: bool
    max_logit_abs_diff: float
    cached_token_ids: List[int]
    uncached_token_ids: List[int]
    cached_kv_bytes: int


@dataclass(frozen=True)
class ChatTurn:
    """Record of one conversation turn."""

    role: str
    content: str
    token_ids: List[int]


def apply_execution_profile(model: nn.Module, profile: str = "memory") -> Dict[str, object]:
    """Apply the memory or speed execution profile across model weights."""
    return set_runtime_profile(model, profile)


class OjasXInferenceEngine:
    """M4 Reference KV-Cached Autoregressive Inference & Multi-Turn Chat Engine."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        max_context_length: int = 2048,
        memory_budget_bytes: Optional[int] = None,
        kv_dtype: torch.dtype = torch.float16,
        execution_profile: str = "memory",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_context_length = int(max_context_length)
        self.memory_budget_bytes = (
            int(memory_budget_bytes) if memory_budget_bytes is not None else None
        )
        self.kv_dtype = kv_dtype
        self.execution_profile = execution_profile

        # Architecture introspection and 7-bucket production memory admission
        cfg = getattr(model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "allow_global_per_layer_attribute_access"):
                setattr(cfg, "allow_global_per_layer_attribute_access", True)
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None and hasattr(text_cfg, "allow_global_per_layer_attribute_access"):
                setattr(text_cfg, "allow_global_per_layer_attribute_access", True)

        self.architecture_profile: ArchitectureProfile = introspect_architecture(model)
        storage = runtime_storage_report(model)
        self.admission_report: MemoryAdmissionReport = evaluate_memory_admission(
            self.architecture_profile,
            encoded_weight_bytes=storage["vq_storage_bytes_including_bias"],
            dense_weight_bytes=storage["non_vq_state_tensor_bytes"],
            codebooks_and_scales_bytes=0,
            decoded_weight_cache_bytes=storage["decoded_cache_bytes"],
            batch_size=1,
            context_length=self.max_context_length,
            kv_dtype=self.kv_dtype,
            budget_bytes=self.memory_budget_bytes,
        )
        if self.memory_budget_bytes is not None and not self.admission_report.admitted:
            raise AdmissionError(self.admission_report)

        # Configure execution profile (streaming vs cached dense weight matrices)
        self.execution_profile_report = apply_execution_profile(
            self.model, profile=self.execution_profile
        )

        # KV Cache Manager & multi-turn history tracking
        self.kv_manager = OjasXKVCacheManager(
            max_context_length=self.max_context_length,
            default_dtype=self.kv_dtype,
        )
        self.chat_history: List[ChatTurn] = []
        self.conversation_tokens: List[int] = []

    def _get_device(self) -> torch.device:
        param = next(self.model.parameters(), None)
        if param is not None:
            return param.device
        buf = next(self.model.buffers(), None)
        if buf is not None:
            return buf.device
        return torch.device("cpu")

    def prefill(self, input_ids: torch.Tensor, *, use_cache: bool = True) -> torch.Tensor:
        """Run the prompt forward pass, update kv_manager, and return last-token logits."""
        device = self._get_device()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=device, dtype=torch.long)

        if use_cache:
            pkv = self.kv_manager.past_key_values
            output = self.model(input_ids, past_key_values=pkv, use_cache=True)
            new_pkv = getattr(output, "past_key_values", None)
            if new_pkv is None and isinstance(output, tuple) and len(output) > 1:
                new_pkv = output[1]
            if new_pkv is not None:
                self.kv_manager.update(new_pkv)
        else:
            output = self.model(input_ids, use_cache=False)

        logits = getattr(output, "logits", output)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, -1, :]

    def decode_step(self, next_token_id: torch.Tensor) -> torch.Tensor:
        """Run a 1-token forward step using past_key_values from kv_manager and return next-token logits."""
        device = self._get_device()
        if next_token_id.ndim == 0:
            next_token_id = next_token_id.view(1, 1)
        elif next_token_id.ndim == 1:
            next_token_id = next_token_id.unsqueeze(0)
        next_token_id = next_token_id.to(device=device, dtype=torch.long)

        output = self.model(
            next_token_id,
            past_key_values=self.kv_manager.past_key_values,
            use_cache=True,
        )
        new_pkv = getattr(output, "past_key_values", None)
        if new_pkv is None and isinstance(output, tuple) and len(output) > 1:
            new_pkv = output[1]
        if new_pkv is not None:
            self.kv_manager.update(new_pkv)

        logits = getattr(output, "logits", output)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, -1, :]

    def sample_token(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        rng: Optional[torch.Generator] = None,
    ) -> int:
        """Sample a single token via greedy, temperature, top_k, or top_p sampling."""
        logits = logits.detach().float()
        if logits.ndim > 1:
            logits = logits[-1]
        logits = logits.flatten()

        if params.temperature <= 0.0:
            return int(torch.argmax(logits).item())

        scaled = logits / max(params.temperature, 1e-8)

        if params.top_k > 0 and params.top_k < scaled.size(-1):
            top_k_values, _ = torch.topk(scaled, params.top_k)
            min_val = top_k_values[-1]
            scaled = torch.where(scaled < min_val, torch.tensor(float("-inf"), device=scaled.device), scaled)

        if 0.0 < params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled[indices_to_remove] = float("-inf")

        probs = torch.softmax(scaled, dim=-1)
        if rng is None and params.seed is not None:
            rng = torch.Generator(device=probs.device).manual_seed(params.seed)

        token = torch.multinomial(probs, num_samples=1, generator=rng)
        return int(token.item())

    def _normalize_prompt(self, prompt_token_ids: Union[Sequence[int], torch.Tensor]) -> List[int]:
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt = prompt_token_ids.flatten().tolist()
        else:
            prompt = list(prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids cannot be empty")
        if len(prompt) > self.max_context_length:
            raise ValueError(
                f"Prompt token length ({len(prompt)}) exceeds max_context_length ({self.max_context_length})"
            )
        return prompt

    def generate(
        self,
        prompt_token_ids: Union[Sequence[int], torch.Tensor],
        params: SamplingParams = SamplingParams(),
        *,
        use_cache: bool = True,
        reset_cache: bool = True,
    ) -> GenerationResult:
        """Autoregressively generate tokens with or without genuine KV caching."""
        prompt = self._normalize_prompt(prompt_token_ids)
        device = self._get_device()

        if params.max_new_tokens <= 0:
            return GenerationResult(
                prompt_token_ids=prompt,
                generated_token_ids=[],
                full_token_ids=prompt,
                stop_reason="max_new_tokens",
                prefill_tokens=0,
                decode_steps=0,
                used_kv_cache=use_cache,
                kv_cache_bytes=self.kv_manager.stats().allocated_bytes if use_cache else 0,
                admission_report=self.admission_report.to_dict(),
            )

        rng: Optional[torch.Generator] = None
        if params.seed is not None and params.temperature > 0.0:
            rng = torch.Generator(device=device).manual_seed(params.seed)

        cached_len = self.kv_manager.sequence_length
        can_reuse_prefix = not reset_cache and cached_len > 0 and len(prompt) >= cached_len

        if use_cache:
            if can_reuse_prefix:
                delta = prompt[cached_len:]
                if delta:
                    prefill_tokens = len(delta)
                    input_tensor = torch.tensor([delta], device=device, dtype=torch.long)
                    logits = self.prefill(input_tensor, use_cache=True)
                else:
                    self.kv_manager.crop(cached_len - 1)
                    prefill_tokens = 0
                    input_tensor = torch.tensor([[prompt[-1]]], device=device, dtype=torch.long)
                    logits = self.decode_step(input_tensor)
            else:
                if reset_cache or cached_len > 0:
                    self.kv_manager.reset()
                prefill_tokens = len(prompt)
                input_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
                logits = self.prefill(input_tensor, use_cache=True)
        else:
            self.kv_manager.reset()
            prefill_tokens = len(prompt)
            input_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
            out = self.model(input_tensor, use_cache=False)
            logits = getattr(out, "logits", out)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :]

        generated: List[int] = []
        curr_seq = list(prompt)
        stop_reason = "max_new_tokens"

        for step in range(params.max_new_tokens):
            token = self.sample_token(logits, params, rng=rng)
            generated.append(token)
            curr_seq.append(token)

            if token in params.eos_token_ids:
                stop_reason = "eos"
                if use_cache and self.kv_manager.sequence_length < self.max_context_length:
                    tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                    _ = self.decode_step(tok_tensor)
                break

            matched_stop = False
            for seq in params.stop_sequences:
                if seq and len(generated) >= len(seq) and generated[-len(seq):] == seq:
                    stop_reason = "stop_sequence"
                    matched_stop = True
                    break
            if matched_stop:
                if use_cache and self.kv_manager.sequence_length < self.max_context_length:
                    tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                    _ = self.decode_step(tok_tensor)
                break

            if len(curr_seq) >= self.max_context_length:
                stop_reason = "max_context_length"
                break

            if use_cache:
                tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                logits = self.decode_step(tok_tensor)
            else:
                if step + 1 < params.max_new_tokens:
                    input_tensor = torch.tensor([curr_seq], device=device, dtype=torch.long)
                    out = self.model(input_tensor, use_cache=False)
                    logits = getattr(out, "logits", out)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    logits = logits[:, -1, :]

            if step + 1 >= params.max_new_tokens:
                break

        return GenerationResult(
            prompt_token_ids=prompt,
            generated_token_ids=generated,
            full_token_ids=curr_seq,
            stop_reason=stop_reason,
            prefill_tokens=prefill_tokens,
            decode_steps=len(generated),
            used_kv_cache=use_cache,
            kv_cache_bytes=self.kv_manager.stats().allocated_bytes if use_cache else 0,
            admission_report=self.admission_report.to_dict(),
        )

    def stream_generate(
        self,
        prompt_token_ids: Union[Sequence[int], torch.Tensor],
        params: SamplingParams = SamplingParams(),
        *,
        use_cache: bool = True,
        reset_cache: bool = True,
    ) -> Iterator[int]:
        """Yield generated token IDs one by one as they are autoregressively decoded."""
        prompt = self._normalize_prompt(prompt_token_ids)
        device = self._get_device()

        if params.max_new_tokens <= 0:
            return

        rng: Optional[torch.Generator] = None
        if params.seed is not None and params.temperature > 0.0:
            rng = torch.Generator(device=device).manual_seed(params.seed)

        cached_len = self.kv_manager.sequence_length
        can_reuse_prefix = not reset_cache and cached_len > 0 and len(prompt) >= cached_len

        if use_cache:
            if can_reuse_prefix:
                delta = prompt[cached_len:]
                if delta:
                    input_tensor = torch.tensor([delta], device=device, dtype=torch.long)
                    logits = self.prefill(input_tensor, use_cache=True)
                else:
                    self.kv_manager.crop(cached_len - 1)
                    input_tensor = torch.tensor([[prompt[-1]]], device=device, dtype=torch.long)
                    logits = self.decode_step(input_tensor)
            else:
                if reset_cache or cached_len > 0:
                    self.kv_manager.reset()
                input_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
                logits = self.prefill(input_tensor, use_cache=True)
        else:
            self.kv_manager.reset()
            input_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
            out = self.model(input_tensor, use_cache=False)
            logits = getattr(out, "logits", out)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :]

        generated: List[int] = []
        curr_seq = list(prompt)

        for step in range(params.max_new_tokens):
            token = self.sample_token(logits, params, rng=rng)
            generated.append(token)
            curr_seq.append(token)
            yield token

            if token in params.eos_token_ids:
                if use_cache and self.kv_manager.sequence_length < self.max_context_length:
                    tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                    _ = self.decode_step(tok_tensor)
                break

            matched_stop = False
            for seq in params.stop_sequences:
                if seq and len(generated) >= len(seq) and generated[-len(seq):] == seq:
                    matched_stop = True
                    break
            if matched_stop:
                if use_cache and self.kv_manager.sequence_length < self.max_context_length:
                    tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                    _ = self.decode_step(tok_tensor)
                break

            if len(curr_seq) >= self.max_context_length:
                break

            if use_cache:
                tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                logits = self.decode_step(tok_tensor)
            else:
                if step + 1 < params.max_new_tokens:
                    input_tensor = torch.tensor([curr_seq], device=device, dtype=torch.long)
                    out = self.model(input_tensor, use_cache=False)
                    logits = getattr(out, "logits", out)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    logits = logits[:, -1, :]

            if step + 1 >= params.max_new_tokens:
                break

    def verify_cached_vs_uncached_parity(
        self,
        prompt_token_ids: Union[Sequence[int], torch.Tensor],
        *,
        max_new_tokens: int = 8,
        atol: float = 1e-4,
    ) -> KVCacheParityReport:
        """Verify token-level and numerical logit equivalence between cached and uncached runtime."""
        prompt = self._normalize_prompt(prompt_token_ids)
        greedy_params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
        device = self._get_device()

        # Step 1: Run uncached greedy generation
        res_uncached = self.generate(prompt, greedy_params, use_cache=False, reset_cache=True)

        # Step 2: Run cached greedy generation
        res_cached = self.generate(prompt, greedy_params, use_cache=True, reset_cache=True)

        # Step 3: Run step-by-step logits check
        self.kv_manager.reset()
        input_0 = torch.tensor([prompt], device=device, dtype=torch.long)
        cached_l0 = self.prefill(input_0, use_cache=True)

        out_u0 = self.model(input_0, use_cache=False)
        uncached_l0 = getattr(out_u0, "logits", out_u0)
        if isinstance(uncached_l0, tuple):
            uncached_l0 = uncached_l0[0]
        uncached_l0 = uncached_l0[:, -1, :]

        diffs = [(cached_l0 - uncached_l0).abs().max().item()]

        curr_seq = list(prompt)
        for tok in res_cached.generated_token_ids[:-1]:
            tok_tensor = torch.tensor([[tok]], device=device, dtype=torch.long)
            c_step_l = self.decode_step(tok_tensor)
            curr_seq.append(tok)
            out_u = self.model(torch.tensor([curr_seq], device=device, dtype=torch.long), use_cache=False)
            u_step_l = getattr(out_u, "logits", out_u)
            if isinstance(u_step_l, tuple):
                u_step_l = u_step_l[0]
            u_step_l = u_step_l[:, -1, :]
            diffs.append((c_step_l - u_step_l).abs().max().item())

        max_diff = max(diffs) if diffs else 0.0
        tokens_match = res_cached.generated_token_ids == res_uncached.generated_token_ids
        matches_exactly = bool(tokens_match and (max_diff <= atol))

        return KVCacheParityReport(
            matches_exactly=matches_exactly,
            max_logit_abs_diff=float(max_diff),
            cached_token_ids=res_cached.generated_token_ids,
            uncached_token_ids=res_uncached.generated_token_ids,
            cached_kv_bytes=res_cached.kv_cache_bytes,
        )

    def chat_turn(
        self,
        user_token_ids: Sequence[int],
        params: SamplingParams = SamplingParams(),
    ) -> GenerationResult:
        """Append user turn, reuse existing KV cache without recomputing history, decode, and update history."""
        user_tokens = list(user_token_ids)
        if not user_tokens:
            raise ValueError("user_token_ids cannot be empty")

        user_content = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                user_content = self.tokenizer.decode(user_tokens)
            except Exception:
                pass
        self.chat_history.append(ChatTurn(role="user", content=user_content, token_ids=user_tokens))
        self.conversation_tokens.extend(user_tokens)

        res = self.generate(
            self.conversation_tokens,
            params,
            use_cache=True,
            reset_cache=False,
        )

        assistant_tokens = res.generated_token_ids
        assistant_content = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                assistant_content = self.tokenizer.decode(assistant_tokens)
            except Exception:
                pass
        self.chat_history.append(
            ChatTurn(role="assistant", content=assistant_content, token_ids=assistant_tokens)
        )
        self.conversation_tokens.extend(assistant_tokens)
        return res

    def reset_chat(self) -> None:
        """Clear conversation history, purge KV cache, and verify zero live allocated bytes."""
        self.chat_history.clear()
        self.conversation_tokens.clear()
        self.kv_manager.reset()
        allocated = self.kv_manager.stats().allocated_bytes
        if allocated != 0:
            raise RuntimeError(f"KV cache reset verification failed: {allocated} bytes remain allocated")

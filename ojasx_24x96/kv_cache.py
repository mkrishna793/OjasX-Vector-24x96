"""M4 Standard BF16/FP16/FP32 KV-Cache Management & Accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch


@dataclass(frozen=True)
class KVCacheStats:
    """Live KV cache and recurrent state accounting statistics."""

    allocated_bytes: int
    sequence_length: int
    max_context_length: int
    num_cached_layers: int
    dtype: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "sequence_length": self.sequence_length,
            "max_context_length": self.max_context_length,
            "num_cached_layers": self.num_cached_layers,
            "dtype": self.dtype,
        }


def _collect_unique_live_tensors(obj: Any) -> List[torch.Tensor]:
    """Recursively discover all unique live tensors inside past_key_values structures."""
    tensors: List[torch.Tensor] = []
    seen: Set[Tuple[str, int]] = set()

    def add_tensor(t: Any) -> None:
        if isinstance(t, torch.Tensor) and t.device.type != "meta":
            ptr = (str(t.device), t.untyped_storage().data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                tensors.append(t)

    def recurse(current: Any, depth: int = 0) -> None:
        if depth > 12 or current is None:
            return
        if isinstance(current, torch.Tensor):
            add_tensor(current)
            return
        if isinstance(current, (list, tuple)):
            for item in current:
                recurse(item, depth + 1)
            return
        if isinstance(current, dict):
            for v in current.values():
                recurse(v, depth + 1)
            return

        # Handle DynamicCache and Cache Layer structures
        if hasattr(current, "layers"):
            recurse(getattr(current, "layers"), depth + 1)

        for attr in (
            "key_cache",
            "value_cache",
            "keys",
            "values",
            "conv_states",
            "recurrent_states",
            "state",
            "ssm_states",
        ):
            if hasattr(current, attr):
                recurse(getattr(current, attr), depth + 1)

        # Catch-all for layer dictionary or state slots
        if hasattr(current, "__dict__"):
            for val in current.__dict__.values():
                if isinstance(val, (torch.Tensor, list, tuple, dict)):
                    recurse(val, depth + 1)

    recurse(obj)
    return tensors


class OjasXKVCacheManager:
    """Manages Hugging Face DynamicCache or tuple past_key_values across architectures."""

    def __init__(
        self,
        max_context_length: int = 2048,
        default_dtype: Union[torch.dtype, str] = torch.float16,
    ) -> None:
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        self.max_context_length = int(max_context_length)
        if isinstance(default_dtype, str):
            clean_name = default_dtype.lower().replace("torch.", "")
            self.default_dtype = getattr(torch, clean_name, torch.float16)
        else:
            self.default_dtype = default_dtype
        self.past_key_values: Optional[Any] = None
        self._seen_tokens: int = 0

    def _infer_sequence_length(self, pkv: Any) -> Optional[int]:
        if pkv is None:
            return 0
        if hasattr(pkv, "get_seq_length"):
            try:
                seq = pkv.get_seq_length()
                if seq is not None and int(seq) >= 0:
                    return int(seq)
            except Exception:
                pass
        if hasattr(pkv, "layers"):
            for layer in getattr(pkv, "layers"):
                if hasattr(layer, "get_seq_length"):
                    try:
                        seq = layer.get_seq_length()
                        if seq is not None and int(seq) >= 0:
                            return int(seq)
                    except Exception:
                        pass
                for attr in ("keys", "key_cache"):
                    if hasattr(layer, attr):
                        t = getattr(layer, attr)
                        if isinstance(t, torch.Tensor) and t.ndim >= 2:
                            return int(t.shape[-2])
        if hasattr(pkv, "key_cache") and getattr(pkv, "key_cache"):
            t = getattr(pkv, "key_cache")[0]
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                return int(t.shape[-2])
        if isinstance(pkv, (tuple, list)) and pkv:
            layer0 = pkv[0]
            if isinstance(layer0, (tuple, list)) and layer0:
                k0 = layer0[0]
                if isinstance(k0, torch.Tensor) and k0.ndim >= 2:
                    return int(k0.shape[-2])
        return None

    @property
    def sequence_length(self) -> int:
        inferred = self._infer_sequence_length(self.past_key_values)
        return max(self._seen_tokens, inferred or 0)

    @property
    def allocated_bytes(self) -> int:
        if self.past_key_values is None:
            return 0
        tensors = self.extract_live_tensors()
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def num_cached_layers(self) -> int:
        if self.past_key_values is None:
            return 0
        if hasattr(self.past_key_values, "layers"):
            return len(getattr(self.past_key_values, "layers"))
        if hasattr(self.past_key_values, "key_cache"):
            return len(getattr(self.past_key_values, "key_cache"))
        if isinstance(self.past_key_values, (tuple, list)):
            return len(self.past_key_values)
        return 0

    @property
    def dtype(self) -> str:
        tensors = self.extract_live_tensors()
        if tensors:
            return str(tensors[0].dtype).replace("torch.", "")
        return str(self.default_dtype).replace("torch.", "")

    def extract_live_tensors(self) -> List[torch.Tensor]:
        if self.past_key_values is None:
            return []
        return _collect_unique_live_tensors(self.past_key_values)

    def update(self, past_key_values: Any, sequence_length: Optional[int] = None) -> None:
        self.past_key_values = past_key_values
        if sequence_length is not None:
            self._seen_tokens = int(sequence_length)
        else:
            inferred = self._infer_sequence_length(past_key_values)
            if inferred is not None and inferred > 0:
                self._seen_tokens = inferred
        if self.sequence_length > self.max_context_length:
            raise ValueError(
                f"KV cache sequence length ({self.sequence_length}) exceeds admission bound "
                f"max_context_length ({self.max_context_length})"
            )

    def reset(self) -> None:
        if self.past_key_values is not None and hasattr(self.past_key_values, "reset"):
            try:
                self.past_key_values.reset()
            except Exception:
                pass
        self.past_key_values = None
        self._seen_tokens = 0

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError("max_length must be nonnegative")
        if self.past_key_values is None:
            self._seen_tokens = min(self._seen_tokens, max_length)
            return

        # Handle DynamicCache layers
        if hasattr(self.past_key_values, "layers"):
            for layer in getattr(self.past_key_values, "layers"):
                for k_attr in ("keys", "key_cache"):
                    if hasattr(layer, k_attr):
                        val = getattr(layer, k_attr)
                        if (
                            isinstance(val, torch.Tensor)
                            and val is not None
                            and val.ndim >= 2
                            and val.shape[-2] > max_length
                        ):
                            setattr(layer, k_attr, val[..., :max_length, :])
                for v_attr in ("values", "value_cache"):
                    if hasattr(layer, v_attr):
                        val = getattr(layer, v_attr)
                        if (
                            isinstance(val, torch.Tensor)
                            and val is not None
                            and val.ndim >= 2
                            and val.shape[-2] > max_length
                        ):
                            setattr(layer, v_attr, val[..., :max_length, :])

        # Handle legacy key_cache / value_cache on DynamicCache
        if hasattr(self.past_key_values, "key_cache") and hasattr(self.past_key_values, "value_cache"):
            kc = getattr(self.past_key_values, "key_cache")
            vc = getattr(self.past_key_values, "value_cache")
            if isinstance(kc, list) and isinstance(vc, list):
                for idx in range(min(len(kc), len(vc))):
                    if isinstance(kc[idx], torch.Tensor) and kc[idx].ndim >= 2 and kc[idx].shape[-2] > max_length:
                        kc[idx] = kc[idx][..., :max_length, :]
                    if isinstance(vc[idx], torch.Tensor) and vc[idx].ndim >= 2 and vc[idx].shape[-2] > max_length:
                        vc[idx] = vc[idx][..., :max_length, :]

        # Handle tuple past_key_values
        if isinstance(self.past_key_values, (tuple, list)):
            new_layers = []
            for layer in self.past_key_values:
                if isinstance(layer, (tuple, list)):
                    new_tup = []
                    for t in layer:
                        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[-2] > max_length:
                            new_tup.append(t[..., :max_length, :])
                        else:
                            new_tup.append(t)
                    new_layers.append(tuple(new_tup))
                else:
                    new_layers.append(layer)
            self.past_key_values = tuple(new_layers)

        self._seen_tokens = min(self._seen_tokens, max_length)

    def stats(self) -> KVCacheStats:
        return KVCacheStats(
            allocated_bytes=self.allocated_bytes,
            sequence_length=self.sequence_length,
            max_context_length=self.max_context_length,
            num_cached_layers=self.num_cached_layers,
            dtype=self.dtype,
        )

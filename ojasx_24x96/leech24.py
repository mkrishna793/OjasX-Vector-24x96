"""Exact finite-shell Leech-lattice mathematics and compact indexing.

The coordinate construction is the extended-binary-Golay formulation of the
Leech lattice.  In the integer embedding ``sqrt(8) * Lambda_24`` a point has
uniform parity, a Golay mod-4 pattern, and the global sum congruence checked by
:func:`leech_contains`.  The implemented quantizer searches only the origin and
the 196,560-point minimum shell; it is not represented as a general Leech CVP
solver.

The three minimum-shell classes admit exact searches without materializing a
codebook.  Indices are bijective over ``0 .. 196560`` inclusive: zero, octad
points, two-axis points, and odd-coset points.  Eighteen bits therefore identify
every representable point.  Search and decode are independent implementations
of the same public class decomposition, which lets tests compare them by brute
force.

Mathematical provenance:
  * Conway & Sloane, Sphere Packings, Lattices and Groups (Golay/Leech construction).
  * van der Ouderaa et al., arXiv:2603.11021 (2026), sections 2.3--2.4.
  * pjmalandrino/llvq@ddecad3196f4f7fc872e03ca5c0499deb12c47e8,
    llvq-core/src/{golay,leech,shells}.rs, MIT OR Apache-2.0.

This is an independent PyTorch implementation.  No claim of novelty, optimal
LLM rate-distortion, or production performance is made here.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import math
from typing import Dict, Tuple

import torch


LEECH_DIM = 24
SHELL2_COUNT = 196_560
INDEX_COUNT = SHELL2_COUNT + 1
INDEX_BITS = 18
OCTAD_CLASS_COUNT = 97_152
TWO_AXIS_CLASS_COUNT = 1_104
ODD_CLASS_COUNT = 98_304
OCTAD_START = 1
TWO_AXIS_START = OCTAD_START + OCTAD_CLASS_COUNT
ODD_START = TWO_AXIS_START + TWO_AXIS_CLASS_COUNT
GEN_POLY = 0xC75


def _clmul(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


@lru_cache(maxsize=1)
def golay_codewords() -> Tuple[int, ...]:
    """Return the extended [24,12,8] Golay code in stable weight/word order."""

    words = []
    for message in range(1 << 12):
        word = _clmul(message, GEN_POLY)
        if word >= 1 << 23:
            raise AssertionError("Golay generator produced an oversized cyclic word")
        word |= (word.bit_count() & 1) << 23
        words.append(word)
    words.sort(key=lambda word: (word.bit_count(), word))
    if len(words) != 4096 or len(set(words)) != 4096:
        raise AssertionError("Golay construction is not bijective")
    return tuple(words)


@lru_cache(maxsize=1)
def octads() -> Tuple[int, ...]:
    return tuple(word for word in golay_codewords() if word.bit_count() == 8)


@lru_cache(maxsize=1)
def pair_indices() -> Tuple[Tuple[int, int], ...]:
    return tuple(itertools.combinations(range(LEECH_DIM), 2))


@lru_cache(maxsize=1)
def _cpu_tables() -> Dict[str, torch.Tensor]:
    shifts = torch.arange(LEECH_DIM, dtype=torch.int64)
    words = torch.tensor(golay_codewords(), dtype=torch.int64)
    octad_words = torch.tensor(octads(), dtype=torch.int64)
    golay_bits = ((words[:, None] >> shifts) & 1).to(torch.int32)
    octad_bits = ((octad_words[:, None] >> shifts) & 1).to(torch.int32)
    octad_positions = torch.tensor(
        [[i for i in range(LEECH_DIM) if (word >> i) & 1] for word in octads()],
        dtype=torch.int64,
    )
    pairs = torch.tensor(pair_indices(), dtype=torch.int64)
    pair_rank = torch.full((LEECH_DIM, LEECH_DIM), -1, dtype=torch.int64)
    for rank, (left, right) in enumerate(pair_indices()):
        pair_rank[left, right] = rank
        pair_rank[right, left] = rank
    return {
        "sorted_words": torch.tensor(sorted(golay_codewords()), dtype=torch.int64),
        "shifts": shifts,
        "golay_bits": golay_bits.to(torch.int8),
        "octad_bits": octad_bits.to(torch.int8),
        "octad_positions": octad_positions,
        "pairs": pairs,
        "pair_rank": pair_rank,
        "sign_shifts": torch.arange(7, dtype=torch.int64),
    }


_DEVICE_TABLES: Dict[str, Dict[str, torch.Tensor]] = {}
_SEARCH_TABLES: Dict[Tuple[str, torch.dtype], Dict[str, torch.Tensor]] = {}


def _tables(device: torch.device) -> Dict[str, torch.Tensor]:
    key = str(device)
    tables = _DEVICE_TABLES.get(key)
    if tables is None:
        tables = {name: tensor.to(device) for name, tensor in _cpu_tables().items()}
        _DEVICE_TABLES[key] = tables
    return tables


def _search_tables(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Encoder-only float tables, never initialized by inference decoding."""

    key = (str(device), dtype)
    tables = _SEARCH_TABLES.get(key)
    if tables is None:
        integer = _tables(device)
        tables = {
            "golay_float": integer["golay_bits"].to(dtype=dtype),
            "octad_float": integer["octad_bits"].to(dtype=dtype),
        }
        _SEARCH_TABLES[key] = tables
    return tables


def _unique_table_bytes(collections) -> int:
    total = 0
    seen = set()
    for tables in collections:
        for tensor in tables.values():
            key = (
                str(tensor.device),
                tensor.dtype,
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            if key in seen:
                continue
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def shared_table_bytes() -> int:
    """Measured unique tensor bytes in process-wide algebraic tables."""

    return _unique_table_bytes([
        _cpu_tables(),
        *_DEVICE_TABLES.values(),
        *_SEARCH_TABLES.values(),
    ])


def initialized_shared_table_bytes() -> int:
    """Measure already-initialized tables without forcing decoder allocation."""

    collections = [*_DEVICE_TABLES.values(), *_SEARCH_TABLES.values()]
    if _cpu_tables.cache_info().currsize:
        collections.insert(0, _cpu_tables())
    return _unique_table_bytes(collections)


def leech_contains(points: torch.Tensor) -> torch.Tensor:
    """Vectorized exact membership test in the integer Leech embedding."""

    if points.shape[-1] != LEECH_DIM or points.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("Leech points need an integer dtype and final dimension 24")
    values = points.to(torch.int64)
    parity = values[..., :1].remainder(2)
    uniform = values.remainder(2).eq(parity).all(dim=-1)
    halved_patterns = ((values - parity) >> 1).remainder(2)
    words = (halved_patterns << _tables(points.device)["shifts"]).sum(dim=-1)
    sorted_words = _tables(points.device)["sorted_words"]
    positions = torch.searchsorted(sorted_words, words.reshape(-1)).clamp_max(
        sorted_words.numel() - 1
    )
    is_golay = sorted_words[positions].eq(words.reshape(-1)).reshape(words.shape)
    sums = values.sum(dim=-1).remainder(8)
    return uniform & is_golay & sums.eq(4 * parity.squeeze(-1))


def shell_index(points: torch.Tensor) -> torch.Tensor:
    """Return ``||x||^2 / 16`` for integer points (membership is separate)."""

    values = points.to(torch.int64)
    norm = values.square().sum(dim=-1)
    if torch.any(norm.remainder(16) != 0).item():
        raise ValueError("An integer point is not on a Leech shell radius")
    return norm // 16


def decode_indices(indices: torch.Tensor) -> torch.Tensor:
    """Decode bijective indices to integer points of ``sqrt(8) Lambda_24``."""

    if indices.dtype not in {torch.uint8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("Leech indices need an integer dtype")
    original_shape = indices.shape
    flat = indices.to(torch.int64).reshape(-1)
    if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= INDEX_COUNT):
        raise ValueError("A Leech shell index is outside 0 .. 196560")
    tables = _tables(indices.device)
    points = torch.zeros(flat.numel(), LEECH_DIM, dtype=torch.int32, device=indices.device)
    rows = torch.arange(flat.numel(), device=indices.device)

    octad_mask = (flat >= OCTAD_START) & (flat < TWO_AXIS_START)
    if octad_mask.any().item():
        selected = flat[octad_mask] - OCTAD_START
        octad_rank = selected // 128
        sign_pattern = selected % 128
        positions = tables["octad_positions"][octad_rank]
        target_rows = rows[octad_mask]
        negative = torch.zeros_like(positions, dtype=torch.bool)
        negative[:, :7] = ((sign_pattern[:, None] >> tables["sign_shifts"]) & 1).bool()
        negative[:, 7] = negative[:, :7].sum(dim=1).remainder(2).bool()
        for column in range(8):
            points[target_rows, positions[:, column]] = torch.where(
                negative[:, column],
                torch.tensor(-2, dtype=torch.int32, device=indices.device),
                torch.tensor(2, dtype=torch.int32, device=indices.device),
            )

    pair_mask = (flat >= TWO_AXIS_START) & (flat < ODD_START)
    if pair_mask.any().item():
        selected = flat[pair_mask] - TWO_AXIS_START
        pair_rank = selected // 4
        signs = selected % 4
        positions = tables["pairs"][pair_rank]
        target_rows = rows[pair_mask]
        for column in range(2):
            minus = signs.remainder(2 ** (column + 1)) // (2**column) != 0
            points[target_rows, positions[:, column]] = torch.where(
                minus,
                torch.tensor(-4, dtype=torch.int32, device=indices.device),
                torch.tensor(4, dtype=torch.int32, device=indices.device),
            )

    odd_mask = flat >= ODD_START
    if odd_mask.any().item():
        selected = flat[odd_mask] - ODD_START
        word_rank = selected // LEECH_DIM
        special = selected % LEECH_DIM
        bits = tables["golay_bits"][word_rank]
        selected_points = (1 - 2 * bits).to(torch.int32)
        local_rows = torch.arange(selected.shape[0], device=indices.device)
        special_bits = bits[local_rows, special]
        selected_points[local_rows, special] = torch.where(
            special_bits.bool(),
            torch.tensor(3, dtype=torch.int32, device=indices.device),
            torch.tensor(-3, dtype=torch.int32, device=indices.device),
        )
        points[odd_mask] = selected_points
    return points.reshape(*original_shape, LEECH_DIM)


def decode_unit_vectors(indices: torch.Tensor, *, dtype=torch.float32) -> torch.Tensor:
    """Decode indices as unit vectors (integer coordinates divided by sqrt(32))."""

    return decode_indices(indices).to(dtype=dtype) / math.sqrt(32.0)


def _nearest_shell2_chunk(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact maximum-dot search over the three minimum-shell classes."""

    if vectors.ndim != 2 or vectors.shape[1] != LEECH_DIM:
        raise ValueError("Shell search expects [vectors,24]")
    if not vectors.dtype.is_floating_point:
        raise TypeError("Shell search needs floating-point vectors")
    device = vectors.device
    tables = _tables(device)
    dtype = vectors.dtype
    search_tables = _search_tables(device, dtype)
    count = vectors.shape[0]
    if count == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=dtype, device=device),
        )
    absolute = vectors.abs()
    negative = vectors < 0

    # Class 1: +/-2 on each Golay octad.  Sign choices align with x except
    # when the required even-minus parity forces the cheapest coordinate flip.
    octad_bits = tables["octad_bits"].bool()
    octad_float = search_tables["octad_float"]
    absolute_sums = absolute @ octad_float.T
    parity = (negative.to(dtype=dtype) @ octad_float.T).round().remainder(2).bool()
    minimum = absolute[:, None, :].masked_fill(~octad_bits, math.inf).amin(dim=2)
    octad_scores = 2 * absolute_sums - torch.where(parity, 4 * minimum, 0)
    octad_score, octad_rank = octad_scores.max(dim=1)
    positions = tables["octad_positions"][octad_rank]
    sign_bits = negative.gather(1, positions)
    flip = sign_bits.sum(dim=1).remainder(2).bool()
    flip_position = absolute.gather(1, positions).argmin(dim=1, keepdim=True)
    sign_bits = sign_bits.scatter(1, flip_position, ~sign_bits.gather(1, flip_position))
    # The scatter above toggles every row.  Even-parity rows must keep their
    # original bit; restore those entries explicitly.
    original = negative.gather(1, positions).gather(1, flip_position)
    sign_bits = sign_bits.scatter(
        1,
        flip_position,
        torch.where(flip[:, None], sign_bits.gather(1, flip_position), original),
    )
    sign7 = (sign_bits[:, :7].to(torch.int64) << tables["sign_shifts"]).sum(dim=1)
    octad_indices = OCTAD_START + octad_rank * 128 + sign7

    # Class 2: +/-4 on two arbitrary axes; every sign pattern is admissible.
    top = absolute.topk(2, dim=1)
    left = top.indices[:, 0]
    right = top.indices[:, 1]
    pair_rank = tables["pair_rank"][left, right]
    pair_signs = negative.gather(1, left[:, None]).to(torch.int64)[:, 0]
    pair_signs += 2 * negative.gather(1, right[:, None]).to(torch.int64)[:, 0]
    pair_scores = 4 * top.values.sum(dim=1)
    pair_indices_ = TWO_AXIS_START + pair_rank * 4 + pair_signs

    # Class 3: one special +/-3 coordinate on a +/-1 Golay background.
    # For codeword c and special position p, the exact score is
    # sum(x) - 2 <x,c> + 4*x_p (p in c), or ... - 4*x_p otherwise.
    golay_bits = tables["golay_bits"].bool()
    golay_float = search_tables["golay_float"]
    base = vectors.sum(dim=1, keepdim=True) - 2 * (vectors @ golay_float.T)
    inside = vectors[:, None, :].masked_fill(~golay_bits, -math.inf)
    outside = (-vectors)[:, None, :].masked_fill(golay_bits, -math.inf)
    inside_value, inside_position = inside.max(dim=2)
    outside_value, outside_position = outside.max(dim=2)
    use_inside = inside_value >= outside_value
    adjustment = torch.where(use_inside, inside_value, outside_value)
    special = torch.where(use_inside, inside_position, outside_position)
    odd_scores = base + 4 * adjustment
    odd_score, word_rank = odd_scores.max(dim=1)
    special_position = special.gather(1, word_rank[:, None])[:, 0]
    odd_indices = ODD_START + word_rank * LEECH_DIM + special_position

    scores = torch.stack((octad_score, pair_scores, odd_score), dim=1)
    indices = torch.stack((octad_indices, pair_indices_, odd_indices), dim=1)
    best_score, best_class = scores.max(dim=1)
    best_index = indices.gather(1, best_class[:, None])[:, 0]
    return best_index, best_score


def nearest_shell2(
    vectors: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the exact nearest-direction index and unnormalised shell dot.

    Every nonzero shell point has integer squared norm 32.  Maximising the dot
    therefore also finds the exact Euclidean nearest point at any common gain.
    The origin is deliberately handled by the residual codec, not hidden here.
    """

    if chunk_size <= 0:
        raise ValueError("Shell-search chunk size must be positive")
    if vectors.ndim != 2 or vectors.shape[1] != LEECH_DIM:
        raise ValueError("Shell search expects [vectors,24]")
    indices = []
    dots = []
    for start in range(0, vectors.shape[0], chunk_size):
        index, dot = _nearest_shell2_chunk(vectors[start : start + chunk_size])
        indices.append(index)
        dots.append(dot)
    if not indices:
        return _nearest_shell2_chunk(vectors)
    return torch.cat(indices), torch.cat(dots)


def pack_indices18(indices: torch.Tensor) -> torch.Tensor:
    """Pack 18-bit unsigned indices, four indices per nine bytes, little-endian."""

    flat = indices.to(torch.int64).reshape(-1)
    if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= 1 << INDEX_BITS):
        raise ValueError("An index exceeds 18 bits")
    expected = (flat.numel() * INDEX_BITS + 7) // 8
    if not flat.numel():
        return torch.empty(0, dtype=torch.uint8, device=indices.device)
    padded = torch.zeros(
        (flat.numel() + 3) // 4 * 4,
        dtype=torch.int64,
        device=flat.device,
    )
    padded[: flat.numel()] = flat
    groups = padded.reshape(-1, 4)
    output = torch.zeros(groups.shape[0], 9, dtype=torch.uint8, device=flat.device)
    for byte in range(9):
        bit = 8 * byte
        code = bit // INDEX_BITS
        shift = bit - code * INDEX_BITS
        value = groups[:, code] >> shift
        if shift and code + 1 < 4:
            value = value | (groups[:, code + 1] << (INDEX_BITS - shift))
        output[:, byte] = value.bitwise_and(0xFF).to(torch.uint8)
    return output.reshape(-1)[:expected].clone()


def unpack_indices18_range(
    packed: torch.Tensor,
    start: int,
    count: int,
    *,
    total_count: int | None = None,
) -> torch.Tensor:
    """Decode a bounded, possibly unaligned range of 18-bit indices."""

    if packed.dtype != torch.uint8 or start < 0 or count < 0:
        raise ValueError("Packed indices must be uint8 and the range nonnegative")
    if total_count is None:
        total_count = packed.numel() * 8 // INDEX_BITS
    if start + count > total_count:
        raise IndexError("Invalid Leech index range")
    expected = (total_count * INDEX_BITS + 7) // 8
    if packed.numel() != expected:
        raise ValueError(f"Expected {expected} packed bytes, received {packed.numel()}")
    if count == 0:
        return torch.empty(0, dtype=torch.int64, device=packed.device)
    first_bit = start * INDEX_BITS
    first_byte = first_bit // 8
    final_byte = ((start + count) * INDEX_BITS + 7) // 8
    window = packed[first_byte:final_byte]
    relative = first_bit - 8 * first_byte
    positions = relative + torch.arange(count, device=packed.device) * INDEX_BITS
    byte_positions = positions // 8
    shifts = positions % 8
    padded = torch.zeros(
        int(byte_positions.max().item()) + 4,
        dtype=torch.uint8,
        device=packed.device,
    )
    padded[: window.numel()] = window
    values = padded.to(torch.int64)
    word = values[byte_positions]
    for offset in (1, 2, 3):
        word = word | (values[byte_positions + offset] << (8 * offset))
    return (word >> shifts) & ((1 << INDEX_BITS) - 1)


def unpack_indices18(packed: torch.Tensor, count: int) -> torch.Tensor:
    return unpack_indices18_range(packed, 0, count, total_count=count)


@lru_cache(maxsize=16)
def rotation_reflections(count: int, seed: int) -> torch.Tensor:
    """Shared Householder reflections; only count/seed are codec metadata."""

    if not 0 <= count <= 8:
        raise ValueError("Rotation reflection count must be between zero and eight")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    vectors = torch.randn(count, LEECH_DIM, generator=generator, dtype=torch.float64)
    if count and torch.any(vectors.square().sum(dim=1) <= 0).item():
        raise AssertionError("Degenerate Householder reflection")
    return vectors.float()


def apply_rotation(
    vectors: torch.Tensor,
    count: int,
    seed: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a norm-preserving product of shared Householder reflections."""

    if vectors.shape[-1] != LEECH_DIM:
        raise ValueError("Rotation expects final dimension 24")
    if count == 0:
        return vectors.clone()
    reflections = rotation_reflections(count, seed).to(
        device=vectors.device,
        dtype=vectors.dtype,
    )
    order = range(count - 1, -1, -1) if inverse else range(count)
    result = vectors.clone()
    for index in order:
        reflection = reflections[index]
        projection = (result * reflection).sum(dim=-1, keepdim=True)
        result = result - (2 / reflection.dot(reflection)) * projection * reflection
    return result

"""Rotation-conditioned, adaptive residual VQ on the Leech minimum shell.

Every stage uses the origin plus the 196,560 minimum-shell points and one to
four shared radial gains. For a fixed gain this is an exact finite-code
Euclidean quantizer: the algebraic shell search maximizes projection, and the
origin is selected whenever the shell point would not reduce squared error. It
is not a nearest-point solver for the infinite Leech lattice.

No per-tensor directional codebook is stored. Payload consists of 18-bit
indices, zero/one/two-bit radial gain codes, at most four float32 gains per
stage, optional adaptive-stage bitmaps, and one float32 RMS scale per matrix
row. Shared Golay tables and Householder reflections are algorithmic codec
metadata. Actual payload bytes, including padding and bitmaps, decide admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

from .codec import pack_codes, unpack_codes
from .leech24 import (
    INDEX_BITS,
    LEECH_DIM,
    apply_rotation,
    decode_unit_vectors,
    nearest_shell2,
    pack_indices18,
    unpack_indices18,
    unpack_indices18_range,
)


@dataclass(frozen=True)
class LeechQuantizationConfig:
    """Finite-shell residual codec configuration.

    ``max_relative_error=None`` generates all stages for planner comparison.
    Adaptive stages store one bitmap per later stage and indices only for active
    vectors.  The gain grid is fitted to reconstruction improvement on a bounded
    deterministic sample, then measured again over the complete matrix.
    """

    max_stages: int = 3
    max_relative_error: Optional[float] = 0.35
    gain_bits: int = 2
    rotation_reflections: int = 2
    adaptive_stages: bool = True
    max_gain_samples: int = 2048
    gain_grid_size: int = 32
    chunk_size: int = 64
    seed: int = 20261104
    min_weight_elements: int = 256
    min_stage_improvement: float = 1e-10
    require_compression: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_stages <= 8:
            raise ValueError("Leech max_stages must be between one and eight")
        if self.max_relative_error is not None and (
            not math.isfinite(self.max_relative_error) or self.max_relative_error < 0
        ):
            raise ValueError("max_relative_error must be finite and nonnegative")
        if self.gain_bits not in {0, 1, 2}:
            raise ValueError("Leech gain_bits must be zero, one, or two")
        if not 0 <= self.rotation_reflections <= 8:
            raise ValueError("rotation_reflections must be between zero and eight")
        if self.max_gain_samples <= 0 or not 4 <= self.gain_grid_size <= 256:
            raise ValueError("Gain samples must be positive and grid size in [4,256]")
        if self.chunk_size <= 0 or self.min_weight_elements <= 0:
            raise ValueError("chunk_size and min_weight_elements must be positive")
        if not math.isfinite(self.min_stage_improvement) or self.min_stage_improvement < 0:
            raise ValueError("min_stage_improvement must be finite and nonnegative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "codec": "ojasx_leech24_shell2_residual_vq",
            "block_size": LEECH_DIM,
            "index_bits": INDEX_BITS,
            "shell": 2,
            "shell_points": 196_560,
            "max_stages": self.max_stages,
            "max_relative_error": self.max_relative_error,
            "gain_bits": self.gain_bits,
            "rotation_reflections": self.rotation_reflections,
            "rotation_family": "shared_seeded_householder",
            "adaptive_stages": self.adaptive_stages,
            "max_gain_samples": self.max_gain_samples,
            "gain_grid_size": self.gain_grid_size,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
            "min_weight_elements": self.min_weight_elements,
            "min_stage_improvement": self.min_stage_improvement,
            "require_compression": self.require_compression,
        }


@dataclass
class LeechQuantizedWeight:
    """Portable finite-shell residual representation of a rank-two matrix."""

    original_shape: Tuple[int, int]
    packed_indices: Tuple[torch.Tensor, ...]
    packed_gains: Tuple[torch.Tensor, ...]
    presence: Tuple[torch.Tensor, ...]
    gain_codebooks: torch.Tensor
    row_scales: torch.Tensor
    gain_bits: int
    rotation_reflections: int
    rotation_seed: int
    adaptive_stages: bool
    metrics: Dict[str, object]

    def __post_init__(self) -> None:
        if len(self.original_shape) != 2:
            raise ValueError("Only rank-two weights are supported")
        rows, columns = map(int, self.original_shape)
        if rows < 0 or columns < 0:
            raise ValueError("Weight dimensions must be nonnegative")
        if not 1 <= len(self.packed_indices) <= 8:
            raise ValueError("An encoded Leech weight requires one to eight stages")
        if len(self.presence) != self.stages or len(self.packed_gains) != self.stages:
            raise ValueError("Each Leech stage needs gain and presence fields")
        if self.gain_bits not in {0, 1, 2}:
            raise ValueError("Invalid Leech gain width")
        if self.gain_codebooks.shape != (self.stages, 1 << self.gain_bits) or (
            self.row_scales.shape != (rows,)
        ):
            raise ValueError("Leech scale shapes are inconsistent with the matrix")
        if self.gain_codebooks.dtype != torch.float32 or self.row_scales.dtype != torch.float32:
            raise TypeError("Leech gains and row scales use float32 storage")
        if not torch.isfinite(self.gain_codebooks).all().item() or torch.any(
            self.gain_codebooks <= 0
        ).item():
            raise ValueError("Leech stage gains must be finite and positive")
        if not torch.isfinite(self.row_scales).all().item() or torch.any(
            self.row_scales < 0
        ).item():
            raise ValueError("Leech row scales must be finite and nonnegative")
        if not 0 <= self.rotation_reflections <= 8:
            raise ValueError("Invalid Leech rotation reflection count")
        device = self.gain_codebooks.device
        tensors = (*self.packed_indices, *self.packed_gains, *self.presence, self.row_scales)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All Leech payload tensors must be on one device")
        if any(
            tensor.dtype != torch.uint8
            for tensor in (*self.packed_indices, *self.packed_gains, *self.presence)
        ):
            raise TypeError("Leech packed fields must be uint8")
        for stage, (packed, gains, presence) in enumerate(
            zip(self.packed_indices, self.packed_gains, self.presence)
        ):
            active = self.active_count(stage)
            expected = (active * INDEX_BITS + 7) // 8
            if packed.shape != (expected,):
                raise ValueError("Leech packed-index shape is inconsistent with active count")
            expected_gains = (active * self.gain_bits + 7) // 8
            if gains.shape != (expected_gains,):
                raise ValueError("Leech packed-gain shape is inconsistent with active count")
            expected_presence = (
                (self.vector_count + 7) // 8
                if self.adaptive_stages and stage > 0
                else 0
            )
            if presence.shape != (expected_presence,):
                raise ValueError("Leech presence shape is inconsistent with the stage")

    @property
    def stages(self) -> int:
        return len(self.packed_indices)

    @property
    def block_size(self) -> int:
        return LEECH_DIM

    @property
    def bits_per_stage(self) -> int:
        return INDEX_BITS + self.gain_bits

    @property
    def blocks_per_row(self) -> int:
        return (self.original_shape[1] + LEECH_DIM - 1) // LEECH_DIM

    @property
    def vector_count(self) -> int:
        return self.original_shape[0] * self.blocks_per_row

    def active_count(self, stage: int) -> int:
        if not 0 <= stage < self.stages:
            raise IndexError("Invalid Leech residual stage")
        if stage == 0 or not self.adaptive_stages:
            return self.vector_count
        # Exact packing has fewer than eight padding bits, so this floors to the
        # logical active count without scanning the complete presence bitmap.
        return self.packed_indices[stage].numel() * 8 // INDEX_BITS

    @property
    def storage_bytes(self) -> int:
        return (
            sum(tensor.numel() for tensor in self.packed_indices)
            + sum(tensor.numel() for tensor in self.packed_gains)
            + sum(tensor.numel() for tensor in self.presence)
            + self.gain_codebooks.numel() * self.gain_codebooks.element_size()
            + self.row_scales.numel() * self.row_scales.element_size()
        )

    @property
    def nominal_bits_per_weight(self) -> float:
        # Upper bound with every later stage active. Measured bytes are decisive.
        bits = (INDEX_BITS + self.gain_bits) * self.stages
        if self.adaptive_stages:
            bits += max(0, self.stages - 1)
        return bits / LEECH_DIM

    @property
    def measured_bits_per_weight(self) -> float:
        elements = self.original_shape[0] * self.original_shape[1]
        return 8.0 * self.storage_bytes / elements if elements else 0.0

    def indices_range(self, stage: int, start: int, count: int) -> torch.Tensor:
        """Decode logical indices for a bounded vector range, restoring zeros."""

        if start < 0 or count < 0 or start + count > self.vector_count:
            raise IndexError("Invalid logical Leech vector range")
        if stage == 0 or not self.adaptive_stages:
            return unpack_indices18_range(
                self.packed_indices[stage],
                start,
                count,
                total_count=self.vector_count,
            )
        presence = _presence_range(self.presence[stage], start, count)
        active_before = _presence_count_before(self.presence[stage], start)
        active = int(presence.sum().item())
        selected = unpack_indices18_range(
            self.packed_indices[stage],
            active_before,
            active,
            total_count=self.active_count(stage),
        )
        result = torch.zeros(count, dtype=torch.int64, device=presence.device)
        if active:
            result[presence.bool()] = selected
        return result

    def gain_codes_range(self, stage: int, start: int, count: int) -> torch.Tensor:
        if self.gain_bits == 0:
            return torch.zeros(count, dtype=torch.uint8, device=self.row_scales.device)
        if stage == 0 or not self.adaptive_stages:
            return _narrow_codes_range(
                self.packed_gains[stage],
                self.gain_bits,
                start,
                count,
            )
        presence = _presence_range(self.presence[stage], start, count)
        active_before = _presence_count_before(self.presence[stage], start)
        active = int(presence.sum().item())
        selected = _narrow_codes_range(
            self.packed_gains[stage],
            self.gain_bits,
            active_before,
            active,
        )
        result = torch.zeros(count, dtype=torch.uint8, device=presence.device)
        if active:
            result[presence.bool()] = selected
        return result


def _narrow_codes_range(
    packed: torch.Tensor,
    bits: int,
    start: int,
    count: int,
) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)
    per_byte = 8 // bits
    first_byte = start // per_byte
    final_byte = (start + count + per_byte - 1) // per_byte
    window = packed[first_byte:final_byte]
    unpacked = unpack_codes(window, bits, window.numel() * per_byte)
    offset = start - first_byte * per_byte
    return unpacked[offset : offset + count]


def _presence_range(presence: torch.Tensor, start: int, count: int) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.uint8, device=presence.device)
    first_byte = start // 8
    final_byte = (start + count + 7) // 8
    window = presence[first_byte:final_byte]
    unpacked = unpack_codes(window, 1, window.numel() * 8)
    offset = start - 8 * first_byte
    return unpacked[offset : offset + count]


def _presence_count_before(presence: torch.Tensor, count: int) -> int:
    if count <= 0:
        return 0
    lookup = torch.tensor(
        [value.bit_count() for value in range(256)],
        dtype=torch.int16,
        device=presence.device,
    )
    complete, remainder = divmod(count, 8)
    total = int(lookup[presence[:complete].to(torch.int64)].sum().item())
    if remainder:
        last = int(presence[complete].item()) & ((1 << remainder) - 1)
        total += last.bit_count()
    return total


def _relative_error(
    residual: torch.Tensor,
    row_scales: torch.Tensor,
    original_norm_squared: torch.Tensor,
    columns: int,
) -> float:
    if original_norm_squared.item() == 0:
        return 0.0 if residual.square().sum().item() == 0 else math.inf
    matrix = residual.reshape(row_scales.numel(), -1)[:, :columns]
    row_errors = matrix.square().sum(dim=1)
    error_squared = (row_errors.double() * row_scales.double().square()).sum()
    return math.sqrt((error_squared / original_norm_squared).clamp_min(0).item())


def _sample_rows(vectors: torch.Tensor, maximum: int) -> torch.Tensor:
    if vectors.shape[0] <= maximum:
        return vectors
    indices = (torch.arange(maximum, dtype=torch.int64) * vectors.shape[0]) // maximum
    return vectors[indices]


def _fit_gain_codebook(
    residual: torch.Tensor,
    config: LeechQuantizationConfig,
) -> Optional[torch.Tensor]:
    sample = _sample_rows(residual, config.max_gain_samples)
    if sample.shape[0] == 0:
        return None
    _, dots = nearest_shell2(sample, chunk_size=config.chunk_size)
    projection = dots / math.sqrt(32.0)
    positive = projection[projection > 0]
    if not positive.numel():
        return None
    if config.gain_bits == 0:
        quantiles = torch.linspace(
            0.02,
            0.98,
            config.gain_grid_size,
            dtype=projection.dtype,
        )
        candidates = 2 * torch.quantile(positive, quantiles)
        candidates = torch.cat((candidates, 2 * positive.mean().reshape(1)))
        best_gain = None
        best_improvement = 0.0
        for candidate in candidates:
            gain = float(candidate.item())
            if not math.isfinite(gain) or gain <= 0:
                continue
            improvement = (2 * gain * projection - gain * gain).clamp_min(0).sum().item()
            if improvement > best_improvement:
                best_improvement = improvement
                best_gain = gain
        return None if best_gain is None else torch.tensor([best_gain], dtype=torch.float32)

    # Small Lloyd--Max fit for shape--gain only. The 24D directional codebook
    # remains algebraic; at most four float32 radial levels are stored per stage.
    size = 1 << config.gain_bits
    quantiles = (torch.arange(size, dtype=projection.dtype) + 0.5) / size
    centroids = torch.quantile(positive, quantiles)
    for _ in range(16):
        assignments = (positive[:, None] - centroids).abs().argmin(dim=1)
        counts = torch.bincount(assignments, minlength=size)
        updated = centroids.clone()
        for index in range(size):
            if counts[index] > 0:
                updated[index] = positive[assignments == index].mean()
        if torch.equal(updated, centroids):
            break
        centroids = updated
    centroids = centroids.sort().values
    if not torch.isfinite(centroids).all().item() or torch.any(centroids <= 0).item():
        return None
    return centroids.float()


def quantize_leech_weight(
    weight: torch.Tensor,
    config: LeechQuantizationConfig = LeechQuantizationConfig(),
) -> Tuple[Optional[LeechQuantizedWeight], Dict[str, object]]:
    """Encode a matrix with exact shell search and measured-byte admission."""

    if weight.ndim != 2 or not weight.dtype.is_floating_point:
        raise ValueError("Leech residual VQ accepts rank-two floating-point weights")
    if weight.device.type == "meta":
        raise ValueError("Cannot quantize a meta tensor without values")
    rows, columns = map(int, weight.shape)
    source = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    report: Dict[str, object] = {
        "shape": [rows, columns],
        "original_bytes": source.numel() * weight.element_size(),
        "selected": False,
        "reason": "pending",
        "codec": "leech24_shell2_residual",
    }
    if source.numel() < config.min_weight_elements:
        report["reason"] = "below_min_weight_elements"
        return None, report
    if not torch.isfinite(source).all().item():
        report["reason"] = "nonfinite_weights"
        return None, report
    original_norm = torch.linalg.vector_norm(source, dtype=torch.float64)
    original_norm_squared = original_norm.square()
    if not torch.isfinite(original_norm_squared).item():
        report["reason"] = "nonfinite_weight_norm"
        return None, report

    blocks_per_row = (columns + LEECH_DIM - 1) // LEECH_DIM
    padded = torch.zeros(rows, blocks_per_row * LEECH_DIM, dtype=torch.float32)
    padded[:, :columns] = source
    row_scales = (
        torch.linalg.vector_norm(source, dim=1, dtype=torch.float64) / math.sqrt(columns)
    ).float()
    safe_scales = torch.where(row_scales > 0, row_scales, torch.ones_like(row_scales))
    normalized = (padded / safe_scales[:, None]).reshape(-1, LEECH_DIM)
    residual = apply_rotation(
        normalized,
        config.rotation_reflections,
        config.seed,
    )
    vector_count = residual.shape[0]
    packed_stages: List[torch.Tensor] = []
    packed_gain_stages: List[torch.Tensor] = []
    presence_stages: List[torch.Tensor] = []
    gain_books: List[torch.Tensor] = []
    stage_errors: List[float] = []
    active_counts: List[int] = []

    for stage in range(config.max_stages):
        gain_book = _fit_gain_codebook(residual, config)
        if gain_book is None:
            break
        indices, dots = nearest_shell2(residual, chunk_size=config.chunk_size)
        projection = dots / math.sqrt(32.0)
        improvements = 2 * projection[:, None] * gain_book - gain_book.square()
        improvement, gain_codes = improvements.max(dim=1)
        # The explicit origin competes with every shape--gain codeword. This is
        # what makes each stage non-expanding even when all radial gains are > 0.
        improving = improvement > 0
        indices = torch.where(improving, indices, 0)
        gain_codes = torch.where(improving, gain_codes, 0)
        if stage == 0 or not config.adaptive_stages:
            active = torch.ones(vector_count, dtype=torch.bool)
        else:
            active = improving & (improvement > config.min_stage_improvement)
        selected = torch.where(active, indices, 0)
        if stage > 0 and config.adaptive_stages and not active.any().item():
            break
        packed_stages.append(pack_indices18(selected[active]))
        packed_gain_stages.append(
            pack_codes(gain_codes[active].to(torch.uint8), config.gain_bits)
            if config.gain_bits
            else torch.empty(0, dtype=torch.uint8)
        )
        presence_stages.append(
            pack_codes(active.to(torch.uint8), 1)
            if stage > 0 and config.adaptive_stages
            else torch.empty(0, dtype=torch.uint8)
        )
        gain_books.append(gain_book)
        active_counts.append(int(active.sum().item()))
        unit = decode_unit_vectors(selected, dtype=torch.float32)
        selected_gains = gain_book[gain_codes]
        contribution = unit * selected_gains[:, None]
        if stage > 0 and config.adaptive_stages:
            contribution = contribution * active[:, None]
        residual = residual - contribution
        stage_errors.append(
            _relative_error(residual, row_scales, original_norm_squared, columns)
        )
        if (
            config.max_relative_error is not None
            and stage_errors[-1] <= config.max_relative_error
        ):
            break

    if not packed_stages:
        report["reason"] = "no_improving_lattice_stage"
        return None, report
    result = LeechQuantizedWeight(
        original_shape=(rows, columns),
        packed_indices=tuple(packed_stages),
        packed_gains=tuple(packed_gain_stages),
        presence=tuple(presence_stages),
        gain_codebooks=torch.stack(gain_books).float(),
        row_scales=row_scales.clone(),
        gain_bits=config.gain_bits,
        rotation_reflections=config.rotation_reflections,
        rotation_seed=config.seed,
        adaptive_stages=config.adaptive_stages,
        metrics={},
    )
    reconstructed = dequantize_leech_weight(result, dtype=torch.float32)
    error_norm = torch.linalg.vector_norm(reconstructed - source, dtype=torch.float64)
    final_error = (
        (error_norm / original_norm).item()
        if original_norm.item() > 0
        else error_norm.item()
    )
    if not torch.isfinite(reconstructed).all().item() or not math.isfinite(final_error):
        report["reason"] = "nonfinite_reconstruction"
        return None, report
    result.metrics = {
        "relative_error": final_error,
        "stage_relative_errors": stage_errors,
        "selected_stages": result.stages,
        "active_counts": active_counts,
        "vector_count": vector_count,
        "nominal_bits_per_weight": result.nominal_bits_per_weight,
        "measured_bits_per_weight": result.measured_bits_per_weight,
        "storage_bytes": result.storage_bytes,
        "original_bytes": source.numel() * weight.element_size(),
        "rotation_reflections": config.rotation_reflections,
        "gain_bits": config.gain_bits,
        "adaptive_stages": config.adaptive_stages,
    }
    report.update(result.metrics)
    if config.max_relative_error is not None and final_error > config.max_relative_error:
        report["reason"] = "error_budget_not_met"
        return None, report
    if config.require_compression and result.storage_bytes >= report["original_bytes"]:
        report["reason"] = "not_smaller_than_original"
        return None, report
    report["selected"] = True
    report["reason"] = "admitted"
    return result, report


def _decode_unrotated_vectors(
    encoded: LeechQuantizedWeight,
    first_vector: int,
    count: int,
) -> torch.Tensor:
    vectors = torch.zeros(
        count,
        LEECH_DIM,
        dtype=torch.float32,
        device=encoded.gain_codebooks.device,
    )
    for stage in range(encoded.stages):
        indices = encoded.indices_range(stage, first_vector, count)
        contribution = decode_unit_vectors(indices, dtype=torch.float32)
        gain_codes = encoded.gain_codes_range(stage, first_vector, count).to(torch.int64)
        gains = encoded.gain_codebooks[stage][gain_codes]
        contribution = contribution * gains[:, None]
        if stage > 0 and encoded.adaptive_stages:
            presence = _presence_range(encoded.presence[stage], first_vector, count)
            contribution = contribution * presence[:, None].to(torch.float32)
        vectors = vectors + contribution
    return vectors


def _decode_rotated_vectors(
    encoded: LeechQuantizedWeight,
    first_vector: int,
    count: int,
) -> torch.Tensor:
    vectors = _decode_unrotated_vectors(encoded, first_vector, count)
    return apply_rotation(
        vectors,
        encoded.rotation_reflections,
        encoded.rotation_seed,
        inverse=True,
    )


def rotate_activation_for_leech(
    inputs: torch.Tensor,
    columns: int,
    rotation_reflections: int,
    rotation_seed: int,
) -> torch.Tensor:
    """Pad and rotate input activations once per forward pass for Leech24 GEMM.

    By orthogonality and symmetry of Householder reflections $H_0, \\dots, H_{k-1}$:
        <u_rot H_{k-1}...H_0, x> = <u_rot, x H_0...H_{k-1}>
    Rotating padded activations in O(B * K) time avoids O(M * K) weight rotation
    during streaming or fused native GEMM.
    """
    if inputs.shape[-1] != columns:
        raise ValueError(f"Expected final dimension {columns}, got {inputs.shape[-1]}")
    blocks_per_row = (columns + LEECH_DIM - 1) // LEECH_DIM
    padded_columns = blocks_per_row * LEECH_DIM
    batch_shape = inputs.shape[:-1]
    flat = inputs.reshape(-1, columns).to(dtype=torch.float32)
    if padded_columns != columns:
        padded = torch.zeros(
            flat.shape[0],
            padded_columns,
            dtype=torch.float32,
            device=inputs.device,
        )
        padded[:, :columns] = flat
    else:
        padded = flat.contiguous()
    if rotation_reflections <= 0:
        return padded.reshape(*batch_shape, padded_columns)
    rotated_blocks = apply_rotation(
        padded.reshape(-1, LEECH_DIM),
        rotation_reflections,
        rotation_seed,
        inverse=False,
    )
    return rotated_blocks.reshape(*batch_shape, padded_columns)


def dequantize_leech_weight_rows_unrotated(
    encoded: LeechQuantizedWeight,
    row_start: int,
    row_stop: int,
    *,
    dtype: Optional[torch.dtype] = None,
    backend: str = "python",
) -> torch.Tensor:
    """Decode scaled, padded 24D weight blocks in rotated Leech coordinates."""
    if backend not in {"python", "native", "auto"}:
        raise ValueError("Leech decode backend must be python, native, or auto")
    rows, _ = encoded.original_shape
    if not 0 <= row_start <= row_stop <= rows:
        raise IndexError("Invalid decoded Leech weight row range")
    target_dtype = dtype or torch.float32
    padded_columns = encoded.blocks_per_row * LEECH_DIM
    if row_start == row_stop:
        return torch.empty(
            0,
            padded_columns,
            dtype=target_dtype,
            device=encoded.gain_codebooks.device,
        )
    if backend != "python" and encoded.gain_codebooks.device.type == "cpu":
        from .leech_native import decode_leech_rows_unrotated_native, native_backend_status

        native = decode_leech_rows_unrotated_native(encoded, row_start, row_stop)
        if native is not None:
            return native.to(dtype=target_dtype)
        if backend == "native":
            raise RuntimeError(
                f"Native Leech backend unavailable: {native_backend_status()['reason']}"
            )
    elif backend == "native":
        raise RuntimeError("Native Leech decoding supports CPU tensors only")
    count = (row_stop - row_start) * encoded.blocks_per_row
    first = row_start * encoded.blocks_per_row
    vectors = _decode_unrotated_vectors(encoded, first, count)
    reconstructed = vectors.reshape(row_stop - row_start, padded_columns)
    reconstructed = reconstructed * encoded.row_scales[row_start:row_stop, None]
    return reconstructed.to(dtype=target_dtype)


def dequantize_leech_weight_rows(
    encoded: LeechQuantizedWeight,
    row_start: int,
    row_stop: int,
    *,
    dtype: Optional[torch.dtype] = None,
    backend: str = "python",
) -> torch.Tensor:
    """Reconstruct a bounded row slice with an optional native CPU backend."""

    if backend not in {"python", "native", "auto"}:
        raise ValueError("Leech decode backend must be python, native, or auto")
    rows, columns = encoded.original_shape
    if not 0 <= row_start <= row_stop <= rows:
        raise IndexError("Invalid decoded Leech weight row range")
    target_dtype = dtype or torch.float32
    if row_start == row_stop:
        return torch.empty(
            0,
            columns,
            dtype=target_dtype,
            device=encoded.gain_codebooks.device,
        )
    if backend != "python" and encoded.gain_codebooks.device.type == "cpu":
        from .leech_native import decode_leech_rows_native, native_backend_status

        native = decode_leech_rows_native(encoded, row_start, row_stop)
        if native is not None:
            return native.to(dtype=target_dtype)
        if backend == "native":
            raise RuntimeError(
                f"Native Leech backend unavailable: {native_backend_status()['reason']}"
            )
    elif backend == "native":
        raise RuntimeError("Native Leech decoding supports CPU tensors only")
    count = (row_stop - row_start) * encoded.blocks_per_row
    first = row_start * encoded.blocks_per_row
    vectors = _decode_rotated_vectors(encoded, first, count)
    reconstructed = vectors.reshape(row_stop - row_start, encoded.blocks_per_row * LEECH_DIM)
    reconstructed = reconstructed * encoded.row_scales[row_start:row_stop, None]
    return reconstructed[:, :columns].to(dtype=target_dtype)


def dequantize_leech_weight(
    encoded: LeechQuantizedWeight,
    *,
    dtype: Optional[torch.dtype] = None,
    backend: str = "python",
) -> torch.Tensor:
    return dequantize_leech_weight_rows(
        encoded,
        0,
        encoded.original_shape[0],
        dtype=dtype,
        backend=backend,
    )

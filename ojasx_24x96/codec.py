"""Adaptive multi-stage residual vector quantization.

The implementation is a new, standalone inference codec; it does not claim that
multi-stage residual VQ itself is new mathematics.  Every quality statement is
an explicit per-tensor measurement.  No model-level accuracy guarantee is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch


_SUPPORTED_WIDTHS = (1, 2, 4, 8)


@dataclass(frozen=True)
class QuantizationConfig:
    """Codec and deterministic codebook-search configuration.

    ``bits_per_stage`` is the width of one code per vector, not bits per scalar.
    The nominal payload rate is ``selected_stages * bits_per_stage / block_size``.
    Codebooks, scales, bias, metadata, and unquantized tensors are additional.
    """

    block_size: int = 8
    bits_per_stage: int = 4
    max_stages: int = 4
    max_relative_error: Optional[float] = 0.20
    max_codebook_samples: int = 4096
    kmeans_iterations: int = 12
    chunk_size: int = 32768
    seed: int = 20261104
    min_weight_elements: int = 256
    require_compression: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.block_size <= 64:
            raise ValueError("block_size must be between 2 and 64")
        if self.bits_per_stage not in _SUPPORTED_WIDTHS:
            raise ValueError("bits_per_stage must be one of 1, 2, 4, or 8")
        if not 1 <= self.max_stages <= 8:
            raise ValueError("max_stages must be between 1 and 8")
        if self.max_relative_error is not None and (
            not math.isfinite(self.max_relative_error) or self.max_relative_error < 0
        ):
            raise ValueError("max_relative_error must be finite and nonnegative")
        if self.max_codebook_samples <= 0 or self.kmeans_iterations <= 0:
            raise ValueError("Codebook samples and iterations must be positive")
        if self.chunk_size <= 0 or self.min_weight_elements <= 0:
            raise ValueError("chunk_size and min_weight_elements must be positive")

    @property
    def codebook_size(self) -> int:
        return 1 << self.bits_per_stage

    def to_dict(self) -> Dict[str, object]:
        return {
            "codec": "ojasx_adaptive_residual_vq",
            "block_size": self.block_size,
            "bits_per_stage": self.bits_per_stage,
            "max_stages": self.max_stages,
            "max_relative_error": self.max_relative_error,
            "max_codebook_samples": self.max_codebook_samples,
            "kmeans_iterations": self.kmeans_iterations,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
            "min_weight_elements": self.min_weight_elements,
            "require_compression": self.require_compression,
        }


@dataclass
class QuantizedWeight:
    """Portable encoded representation of one rank-two weight matrix."""

    original_shape: Tuple[int, int]
    block_size: int
    bits_per_stage: int
    packed_codes: torch.Tensor
    codebooks: torch.Tensor
    row_scales: torch.Tensor
    metrics: Dict[str, object]

    def __post_init__(self) -> None:
        if len(self.original_shape) != 2:
            raise ValueError("Only rank-two weights are supported")
        if self.bits_per_stage not in _SUPPORTED_WIDTHS:
            raise ValueError("Unsupported code width")
        if self.packed_codes.dtype != torch.uint8:
            raise TypeError("packed_codes must be uint8")
        if self.codebooks.dtype != torch.float32 or self.row_scales.dtype != torch.float32:
            raise TypeError("Codebooks and scales use float32 storage")
        if self.packed_codes.device != self.codebooks.device or (
            self.packed_codes.device != self.row_scales.device
        ):
            raise ValueError("All encoded tensors must be on one device")
        rows, columns = self.original_shape
        if rows < 0 or columns < 0:
            raise ValueError("Weight dimensions must be nonnegative")
        blocks_per_row = (columns + self.block_size - 1) // self.block_size
        vectors = rows * blocks_per_row
        if self.codebooks.ndim != 3:
            raise ValueError("Codebooks must be a rank-three tensor")
        stages = self.codebooks.shape[0]
        if stages < 1:
            raise ValueError("An encoded weight requires at least one residual stage")
        expected_packed = (vectors * self.bits_per_stage + 7) // 8
        if self.codebooks.shape[1:] != (
            1 << self.bits_per_stage,
            self.block_size,
        ):
            raise ValueError("Codebook shape is inconsistent with the codec")
        if self.packed_codes.shape != (stages, expected_packed):
            raise ValueError("Packed-code shape is inconsistent with the weight shape")
        if self.row_scales.shape != (rows,):
            raise ValueError("Row-scale shape is inconsistent with the weight shape")

    @property
    def stages(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def vector_count(self) -> int:
        rows, columns = self.original_shape
        return rows * ((columns + self.block_size - 1) // self.block_size)

    @property
    def storage_bytes(self) -> int:
        return (
            self.packed_codes.numel()
            + self.codebooks.numel() * self.codebooks.element_size()
            + self.row_scales.numel() * self.row_scales.element_size()
        )

    @property
    def nominal_bits_per_weight(self) -> float:
        return self.stages * self.bits_per_stage / self.block_size

    @property
    def measured_bits_per_weight(self) -> float:
        elements = self.original_shape[0] * self.original_shape[1]
        return 8.0 * self.storage_bytes / elements if elements else 0.0

    def codes(self, stage: int) -> torch.Tensor:
        if not 0 <= stage < self.stages:
            raise IndexError("Invalid residual stage")
        return unpack_codes(
            self.packed_codes[stage],
            self.bits_per_stage,
            self.vector_count,
        )

    def codes_range(self, stage: int, start: int, count: int) -> torch.Tensor:
        """Unpack a bounded logical code range, including unaligned bit offsets."""

        if not 0 <= stage < self.stages:
            raise IndexError("Invalid residual stage")
        if start < 0 or count < 0 or start + count > self.vector_count:
            raise IndexError("Invalid logical code range")
        if count == 0:
            return torch.empty(0, dtype=torch.uint8, device=self.packed_codes.device)
        bits = self.bits_per_stage
        per_byte = 8 // bits
        first_byte = (start * bits) // 8
        final_byte = ((start + count) * bits + 7) // 8
        window = self.packed_codes[stage, first_byte:final_byte]
        window_codes = unpack_codes(window, bits, window.numel() * per_byte)
        offset = start - first_byte * per_byte
        return window_codes[offset : offset + count]


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes using a deterministic little-endian bit order."""

    if bits not in _SUPPORTED_WIDTHS:
        raise ValueError("Only 1-, 2-, 4-, and 8-bit codes are supported")
    if codes.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Codes must have an integer dtype")
    flat = codes.reshape(-1)
    if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= (1 << bits)):
        raise ValueError("A code exceeds the requested bit width")
    if bits == 8:
        return flat.to(dtype=torch.uint8).clone()
    per_byte = 8 // bits
    padded_length = (flat.numel() + per_byte - 1) // per_byte * per_byte
    padded = torch.zeros(padded_length, dtype=torch.uint8, device=flat.device)
    padded[: flat.numel()] = flat.to(dtype=torch.uint8)
    groups = padded.to(dtype=torch.int16).reshape(-1, per_byte)
    shifts = torch.arange(per_byte, device=flat.device, dtype=torch.int16) * bits
    packed = (groups << shifts).sum(dim=1)
    return packed.to(dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes`, with explicit logical code count."""

    if bits not in _SUPPORTED_WIDTHS:
        raise ValueError("Only 1-, 2-, 4-, and 8-bit codes are supported")
    if packed.dtype != torch.uint8 or count < 0:
        raise ValueError("Packed codes must be uint8 and count nonnegative")
    expected = (count * bits + 7) // 8
    if packed.numel() != expected:
        raise ValueError(f"Expected {expected} packed bytes, received {packed.numel()}")
    if count == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)
    values = packed.reshape(-1, 1).to(dtype=torch.int16)
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int16) * bits
    decoded = ((values >> shifts) & ((1 << bits) - 1)).reshape(-1)
    return decoded[:count].to(dtype=torch.uint8)


def _nearest(vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    # Squared Euclidean distance without a large [N, K, B] intermediate.
    distances = (
        vectors.square().sum(dim=1, keepdim=True)
        - 2 * vectors @ codebook.T
        + codebook.square().sum(dim=1).unsqueeze(0)
    )
    return distances.argmin(dim=1)


def _fit_codebook(
    samples: torch.Tensor,
    size: int,
    iterations: int,
    generator: torch.Generator,
) -> torch.Tensor:
    count = samples.shape[0]
    if count == 0:
        raise ValueError("Cannot fit a codebook to zero vectors")
    indices = torch.randint(count, (size,), generator=generator, device="cpu")
    centers = samples[indices].clone()
    for _ in range(iterations):
        assignments = _nearest(samples, centers)
        counts = torch.bincount(assignments, minlength=size)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples)
        updated = sums / counts.clamp_min(1).unsqueeze(1)
        empty = torch.nonzero(counts == 0, as_tuple=False).reshape(-1)
        if empty.numel():
            # Deterministic farthest-point repair, with distinct sample ranks.
            residual = samples - centers[assignments]
            ranking = residual.square().sum(dim=1).argsort(descending=True)
            repair = ranking[: empty.numel()]
            if repair.numel() < empty.numel():
                repair = repair.repeat((empty.numel() + repair.numel() - 1) // repair.numel())
            updated[empty] = samples[repair[: empty.numel()]]
        if torch.equal(updated, centers):
            break
        centers = updated
    return centers


def _sample_indices(count: int, maximum: int, seed: int) -> torch.Tensor:
    if count <= maximum:
        return torch.arange(count, dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(count, (maximum,), generator=generator, device="cpu")


def _fit_codebooks(
    normalized: torch.Tensor,
    config: QuantizationConfig,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = _sample_indices(normalized.shape[0], config.max_codebook_samples, seed)
    residual = normalized[indices].clone()
    books = []
    for _ in range(config.max_stages):
        book = _fit_codebook(
            residual,
            config.codebook_size,
            config.kmeans_iterations,
            generator,
        )
        # The zero anchor is a correctness/quality guard: nearest-centroid
        # selection can always choose zero, so a residual stage is non-expanding.
        book[0].zero_()
        books.append(book)
        residual = residual - book[_nearest(residual, book)]
    return torch.stack(books)


def _relative_error(
    residual: torch.Tensor,
    row_scales: torch.Tensor,
    original_norm_squared: torch.Tensor,
    columns: int,
) -> float:
    if original_norm_squared.item() == 0:
        return 0.0 if residual.square().sum().item() == 0 else math.inf
    # Padding is a container detail, not part of the source-matrix error.
    matrix = residual.reshape(row_scales.numel(), -1)[:, :columns]
    row_errors = matrix.square().sum(dim=1)
    # Only the row vectors are promoted: this avoids a full float64 matrix while
    # preventing scale-squared overflow/underflow in the measured error.
    error_squared = (row_errors.double() * row_scales.double().square()).sum()
    return math.sqrt((error_squared / original_norm_squared).clamp_min(0).item())


def quantize_weight(
    weight: torch.Tensor,
    config: QuantizationConfig = QuantizationConfig(),
    *,
    seed: Optional[int] = None,
) -> Tuple[Optional[QuantizedWeight], Dict[str, object]]:
    """Encode one matrix and return ``(representation, measurement)``.

    ``representation`` is ``None`` when the error budget is not met or the stored
    representation is not smaller.  The caller can therefore retain the original
    linear module without inventing an accuracy or compression claim.
    """

    if weight.ndim != 2 or not weight.dtype.is_floating_point:
        raise ValueError("Residual VQ accepts rank-two floating-point weights")
    if weight.device.type == "meta":
        raise ValueError("Cannot quantize a meta tensor without values")
    rows, columns = map(int, weight.shape)
    source = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    report: Dict[str, object] = {
        "shape": [rows, columns],
        "original_bytes": source.numel() * weight.element_size(),
        "selected": False,
        "reason": "pending",
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

    actual_seed = config.seed if seed is None else seed
    blocks_per_row = (columns + config.block_size - 1) // config.block_size
    padded_columns = blocks_per_row * config.block_size
    padded = torch.zeros(rows, padded_columns, dtype=torch.float32)
    padded[:, :columns] = source
    row_scales = (
        torch.linalg.vector_norm(source, dim=1, dtype=torch.float64) / math.sqrt(columns)
    ).float()
    safe_scales = torch.where(row_scales > 0, row_scales, torch.ones_like(row_scales))
    normalized = (padded / safe_scales[:, None]).reshape(-1, config.block_size)
    residual = normalized.clone()
    vector_count = normalized.shape[0]
    codebooks = _fit_codebooks(normalized, config, actual_seed)
    codes = torch.empty(
        config.max_stages,
        vector_count,
        dtype=torch.uint8,
    )
    stage_errors = []
    for stage, book in enumerate(codebooks):
        for start in range(0, vector_count, config.chunk_size):
            stop = min(start + config.chunk_size, vector_count)
            indices = _nearest(residual[start:stop], book)
            codes[stage, start:stop] = indices.to(dtype=torch.uint8)
            residual[start:stop].sub_(book[indices])
        stage_errors.append(_relative_error(residual, row_scales, original_norm_squared, columns))

    selected_stages = config.max_stages
    if config.max_relative_error is not None:
        selected_stages = next(
            (
                stage + 1
                for stage, error in enumerate(stage_errors)
                if error <= config.max_relative_error
            ),
            0,
        )
    report["stage_relative_errors"] = stage_errors
    if selected_stages == 0:
        report["reason"] = "error_budget_not_met"
        report["relative_error"] = stage_errors[-1]
        return None, report

    selected_books = codebooks[:selected_stages].clone()
    packed = torch.stack(
        [pack_codes(codes[stage], config.bits_per_stage) for stage in range(selected_stages)]
    )
    result = QuantizedWeight(
        original_shape=(rows, columns),
        block_size=config.block_size,
        bits_per_stage=config.bits_per_stage,
        packed_codes=packed,
        codebooks=selected_books,
        row_scales=row_scales.clone(),
        metrics={},
    )
    reconstructed = dequantize_weight(result, dtype=torch.float32)
    error_norm = torch.linalg.vector_norm(reconstructed - source, dtype=torch.float64)
    final_error = (
        (error_norm / original_norm).item()
        if original_norm.item() > 0
        else error_norm.item()
    )
    if not torch.isfinite(reconstructed).all().item() or not math.isfinite(final_error):
        report["reason"] = "nonfinite_reconstruction"
        report["stage_relative_errors"] = stage_errors
        return None, report
    result.metrics = {
        "relative_error": final_error,
        "stage_relative_errors": stage_errors,
        "selected_stages": selected_stages,
        "nominal_bits_per_weight": result.nominal_bits_per_weight,
        "measured_bits_per_weight": result.measured_bits_per_weight,
        "storage_bytes": result.storage_bytes,
        "original_bytes": source.numel() * weight.element_size(),
        "seed": actual_seed,
    }
    report.update(result.metrics)
    if config.max_relative_error is not None and final_error > config.max_relative_error:
        report["reason"] = "stored_representation_error_budget_not_met"
        return None, report
    if config.require_compression and result.storage_bytes >= report["original_bytes"]:
        report["reason"] = "not_smaller_than_original"
        return None, report
    report["selected"] = True
    report["reason"] = "admitted"
    return result, report


def dequantize_weight(
    encoded: QuantizedWeight,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reconstruct a matrix from packed codes, residual books, and row scales."""

    rows, columns = encoded.original_shape
    vectors = torch.zeros(
        encoded.vector_count,
        encoded.block_size,
        dtype=torch.float32,
        device=encoded.codebooks.device,
    )
    for stage in range(encoded.stages):
        indices = encoded.codes(stage).to(dtype=torch.int64)
        vectors.add_(encoded.codebooks[stage][indices])
    blocks_per_row = (columns + encoded.block_size - 1) // encoded.block_size
    reconstructed = vectors.reshape(rows, blocks_per_row * encoded.block_size)
    reconstructed = reconstructed * encoded.row_scales[:, None]
    reconstructed = reconstructed[:, :columns]
    return reconstructed.to(dtype=dtype) if dtype is not None else reconstructed


def dequantize_weight_rows(
    encoded: QuantizedWeight,
    row_start: int,
    row_stop: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reconstruct a row slice without materializing the complete dense matrix."""

    rows, columns = encoded.original_shape
    if not 0 <= row_start <= row_stop <= rows:
        raise IndexError("Invalid decoded weight row range")
    target_dtype = dtype or torch.float32
    if row_start == row_stop:
        return torch.empty(
            0,
            columns,
            dtype=target_dtype,
            device=encoded.codebooks.device,
        )
    blocks_per_row = (columns + encoded.block_size - 1) // encoded.block_size
    first_vector = row_start * blocks_per_row
    vector_count = (row_stop - row_start) * blocks_per_row
    vectors = torch.zeros(
        vector_count,
        encoded.block_size,
        dtype=torch.float32,
        device=encoded.codebooks.device,
    )
    for stage in range(encoded.stages):
        indices = encoded.codes_range(stage, first_vector, vector_count).to(dtype=torch.int64)
        vectors.add_(encoded.codebooks[stage][indices])
    reconstructed = vectors.reshape(row_stop - row_start, blocks_per_row * encoded.block_size)
    reconstructed = reconstructed * encoded.row_scales[row_start:row_stop, None]
    reconstructed = reconstructed[:, :columns]
    return reconstructed.to(dtype=target_dtype)

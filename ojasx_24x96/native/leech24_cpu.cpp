// Portable CPU Leech24 minimum-shell row decoder.
// The C ABI uses only C types and needs no Python or PyTorch headers.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::int32_t kDimension = 24;
constexpr std::uint32_t kIndexCount = 196561;
constexpr std::uint32_t kTwoAxisStart = 97153;
constexpr std::uint32_t kOddStart = 98257;
constexpr float kShellScale = 5.6568542494923801952067548968388f;

std::uint32_t multiply_binary_polynomial(std::uint32_t left, std::uint32_t right) {
    std::uint32_t result = 0;
    while (right) {
        if (right & 1U) result ^= left;
        left <<= 1U;
        right >>= 1U;
    }
    return result;
}

struct Tables {
    std::array<std::uint32_t, 4096> words{};
    std::array<std::uint32_t, 759> octads{};
    std::array<std::array<std::int32_t, 2>, 276> pairs{};

    Tables() {
        for (std::uint32_t message = 0; message < 4096; ++message) {
            std::uint32_t word = multiply_binary_polynomial(message, 0xC75U);
            word |= (static_cast<std::uint32_t>(__builtin_popcount(word)) & 1U) << 23U;
            words[message] = word;
        }
        std::sort(words.begin(), words.end(), [](std::uint32_t left, std::uint32_t right) {
            const int left_weight = __builtin_popcount(left);
            const int right_weight = __builtin_popcount(right);
            return left_weight == right_weight ? left < right : left_weight < right_weight;
        });
        std::size_t octad = 0;
        for (const std::uint32_t word : words) {
            if (__builtin_popcount(word) == 8) octads[octad++] = word;
        }
        std::size_t pair = 0;
        for (std::int32_t left = 0; left < kDimension; ++left) {
            for (std::int32_t right = left + 1; right < kDimension; ++right) {
                pairs[pair++] = {left, right};
            }
        }
    }
};

const Tables& tables() {
    static const Tables instance;
    return instance;
}

int decode_point(std::uint32_t index, std::int32_t* output) {
    if (index >= kIndexCount || output == nullptr) return 1;
    std::fill_n(output, kDimension, 0);
    if (index == 0) return 0;

    if (index < kTwoAxisStart) {
        const std::uint32_t selected = index - 1U;
        const std::uint32_t word = tables().octads[selected / 128U];
        const std::uint32_t signs = selected % 128U;
        int position = 0;
        int minus_parity = 0;
        for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
            if (((word >> coordinate) & 1U) == 0) continue;
            const bool negative = position < 7
                ? ((signs >> position) & 1U) != 0
                : minus_parity != 0;
            if (position < 7) minus_parity ^= static_cast<int>(negative);
            output[coordinate] = negative ? -2 : 2;
            ++position;
        }
        return position == 8 ? 0 : 2;
    }

    if (index < kOddStart) {
        const std::uint32_t selected = index - kTwoAxisStart;
        const auto pair = tables().pairs[selected / 4U];
        const std::uint32_t signs = selected % 4U;
        output[pair[0]] = (signs & 1U) ? -4 : 4;
        output[pair[1]] = (signs & 2U) ? -4 : 4;
        return 0;
    }

    const std::uint32_t selected = index - kOddStart;
    const std::uint32_t word = tables().words[selected / 24U];
    const int special = static_cast<int>(selected % 24U);
    for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
        const bool bit = ((word >> coordinate) & 1U) != 0;
        output[coordinate] = coordinate == special ? (bit ? 3 : -3) : (bit ? -1 : 1);
    }
    return 0;
}

std::uint32_t read_index18(const std::uint8_t* packed, std::size_t position) {
    const std::size_t bit = position * 18U;
    const std::size_t byte = bit / 8U;
    const unsigned shift = static_cast<unsigned>(bit % 8U);
    const std::uint32_t word = static_cast<std::uint32_t>(packed[byte])
        | (static_cast<std::uint32_t>(packed[byte + 1]) << 8U)
        | (static_cast<std::uint32_t>(packed[byte + 2]) << 16U);
    return (word >> shift) & 0x3FFFFU;
}

std::uint32_t read_gain(const std::uint8_t* packed, std::size_t position, int bits) {
    if (bits == 0) return 0;
    const std::size_t bit = position * static_cast<std::size_t>(bits);
    return (packed[bit / 8U] >> (bit % 8U)) & ((1U << bits) - 1U);
}

bool read_presence(const std::uint8_t* packed, std::size_t position) {
    return ((packed[position / 8U] >> (position % 8U)) & 1U) != 0;
}

inline int decode_leech_block(
    const std::uint8_t* const* packed_indices,
    const std::uint8_t* const* packed_gains,
    const std::uint8_t* const* presence,
    const float* gain_codebooks,
    int stages,
    int gain_bits,
    int gain_count,
    bool adaptive_stages,
    std::size_t global_vector,
    std::size_t* active_positions,
    float* values) {
    std::fill_n(values, kDimension, 0.0f);
    for (int stage = 0; stage < stages; ++stage) {
        bool active = true;
        std::size_t code_position = global_vector;
        if (adaptive_stages && stage > 0) {
            active = read_presence(presence[stage], global_vector);
            code_position = active_positions[stage];
            if (active) ++active_positions[stage];
        }
        if (!active) continue;
        const std::uint32_t index = read_index18(packed_indices[stage], code_position);
        if (index == 0) continue;
        const std::uint32_t gain_code = read_gain(
            packed_gains[stage], code_position, gain_bits);
        if (gain_code >= static_cast<std::uint32_t>(gain_count)) return -5;
        const float gain = gain_codebooks[stage * gain_count + gain_code];
        std::int32_t point[kDimension];
        const int status = decode_point(index, point);
        if (status != 0) return -6;
        for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
            const float unit = static_cast<float>(point[coordinate]) / kShellScale;
            values[coordinate] += unit * gain;
        }
    }
    return 0;
}

}  // namespace

extern "C" {

int ojasx_leech24_decode_indices(
    const std::uint32_t* indices,
    std::size_t count,
    std::int32_t* output) {
    if ((count && indices == nullptr) || (count && output == nullptr)) return -1;
    for (std::size_t row = 0; row < count; ++row) {
        const int status = decode_point(indices[row], output + row * kDimension);
        if (status != 0) return status;
    }
    return 0;
}

std::size_t ojasx_leech24_table_bytes() {
    return sizeof(Tables);
}

int ojasx_leech24_decode_rows(
    const std::uint8_t* const* packed_indices,
    const std::uint8_t* const* packed_gains,
    const std::uint8_t* const* presence,
    const float* gain_codebooks,
    int stages,
    int gain_bits,
    int adaptive_stages,
    int rows,
    int columns,
    int row_start,
    int row_stop,
    float* output) {
    if (stages <= 0 || stages > 8 || gain_bits < 0 || gain_bits > 2
        || rows <= 0 || columns <= 0 || row_start < 0
        || row_stop <= row_start || row_stop > rows
        || packed_indices == nullptr || packed_gains == nullptr || presence == nullptr
        || gain_codebooks == nullptr || output == nullptr) {
        return -1;
    }

    const int blocks_per_row = (columns + kDimension - 1) / kDimension;
    const int padded_columns = blocks_per_row * kDimension;
    const int row_count = row_stop - row_start;
    const std::size_t vector_count = static_cast<std::size_t>(row_count) * blocks_per_row;
    const std::size_t first_vector = static_cast<std::size_t>(row_start) * blocks_per_row;
    const int gain_count = 1 << gain_bits;
    std::size_t active_positions[8] = {};

    for (int stage = 0; stage < stages; ++stage) {
        if (packed_indices[stage] == nullptr || (gain_bits && packed_gains[stage] == nullptr)) {
            return -2;
        }
        if (adaptive_stages && stage > 0) {
            if (presence[stage] == nullptr) return -3;
            std::size_t active = 0;
            for (std::size_t vector = 0; vector < first_vector; ++vector) {
                active += read_presence(presence[stage], vector) ? 1U : 0U;
            }
            active_positions[stage] = active;
        }
    }

    // The output is padded, unscaled, and still in the rotated coordinate system.
    // PyTorch performs the final Householder operations so its rounding and the
    // reference decoder's rounding remain bitwise identical.
    for (std::size_t local_vector = 0; local_vector < vector_count; ++local_vector) {
        const std::size_t global_vector = first_vector + local_vector;
        float values[kDimension] = {};
        for (int stage = 0; stage < stages; ++stage) {
            bool active = true;
            std::size_t code_position = global_vector;
            if (adaptive_stages && stage > 0) {
                active = read_presence(presence[stage], global_vector);
                code_position = active_positions[stage];
                if (active) ++active_positions[stage];
            }
            if (!active) continue;
            const std::uint32_t index = read_index18(packed_indices[stage], code_position);
            if (index == 0) continue;
            const std::uint32_t gain_code = read_gain(
                packed_gains[stage], code_position, gain_bits);
            if (gain_code >= static_cast<std::uint32_t>(gain_count)) return -5;
            const float gain = gain_codebooks[stage * gain_count + gain_code];
            std::int32_t point[kDimension];
            const int status = decode_point(index, point);
            if (status != 0) return -6;
            for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
                const float unit = static_cast<float>(point[coordinate]) / kShellScale;
                const float contribution = unit * gain;
                values[coordinate] = values[coordinate] + contribution;
            }
        }

        const int local_row = static_cast<int>(local_vector / blocks_per_row);
        const int block = static_cast<int>(local_vector % blocks_per_row);
        for (int coordinate = 0; coordinate < kDimension; ++coordinate) {
            const int column = block * kDimension + coordinate;
            output[local_row * padded_columns + column] = values[coordinate];
        }
    }
    return 0;
}

int ojasx_leech24_fused_gemm(
    const std::uint8_t* const* packed_indices,
    const std::uint8_t* const* packed_gains,
    const std::uint8_t* const* presence,
    const float* gain_codebooks,
    const float* row_scales,
    const float* bias,
    const float* rotated_inputs,
    int batch_size,
    int stages,
    int gain_bits,
    int adaptive_stages,
    int rows,
    int columns,
    int row_start,
    int row_stop,
    float* output) {
    if (stages <= 0 || stages > 8 || gain_bits < 0 || gain_bits > 2
        || rows <= 0 || columns <= 0 || row_start < 0
        || row_stop <= row_start || row_stop > rows
        || batch_size <= 0
        || packed_indices == nullptr || gain_codebooks == nullptr
        || row_scales == nullptr || rotated_inputs == nullptr || output == nullptr) {
        return -1;
    }
    if (adaptive_stages && presence == nullptr) {
        return -1;
    }

    const int blocks_per_row = (columns + kDimension - 1) / kDimension;
    const int padded_columns = blocks_per_row * kDimension;
    const int row_count = row_stop - row_start;
    const std::size_t first_vector = static_cast<std::size_t>(row_start) * blocks_per_row;
    const int gain_count = 1 << gain_bits;
    std::size_t active_positions[8] = {};

    for (int stage = 0; stage < stages; ++stage) {
        if (packed_indices[stage] == nullptr || (gain_bits && packed_gains[stage] == nullptr)) {
            return -2;
        }
        if (adaptive_stages && stage > 0) {
            if (presence[stage] == nullptr) return -3;
            std::size_t active = 0;
            for (std::size_t vector = 0; vector < first_vector; ++vector) {
                active += read_presence(presence[stage], vector) ? 1U : 0U;
            }
            active_positions[stage] = active;
        }
    }

    for (int r = row_start; r < row_stop; ++r) {
        const int local_r = r - row_start;
        const float scale = row_scales[r];
        const float b_val = (bias != nullptr) ? bias[r] : 0.0f;

        if (batch_size == 1) {
            float row_acc = 0.0f;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const std::size_t global_vector = static_cast<std::size_t>(r) * blocks_per_row + blk;
                float values[kDimension];
                const int status = decode_leech_block(
                    packed_indices, packed_gains, presence, gain_codebooks,
                    stages, gain_bits, gain_count, adaptive_stages != 0,
                    global_vector, active_positions, values);
                if (status != 0) return status;

                const float* in_ptr = rotated_inputs + blk * kDimension;
                float dot = 0.0f;
                for (int c = 0; c < kDimension; ++c) {
                    dot += in_ptr[c] * values[c];
                }
                row_acc += dot;
            }
            output[local_r] = row_acc * scale + b_val;
        } else {
            for (int b = 0; b < batch_size; ++b) {
                output[b * row_count + local_r] = 0.0f;
            }
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const std::size_t global_vector = static_cast<std::size_t>(r) * blocks_per_row + blk;
                float values[kDimension];
                const int status = decode_leech_block(
                    packed_indices, packed_gains, presence, gain_codebooks,
                    stages, gain_bits, gain_count, adaptive_stages != 0,
                    global_vector, active_positions, values);
                if (status != 0) return status;

                for (int b = 0; b < batch_size; ++b) {
                    const float* in_ptr = rotated_inputs + b * padded_columns + blk * kDimension;
                    float dot = 0.0f;
                    for (int c = 0; c < kDimension; ++c) {
                        dot += in_ptr[c] * values[c];
                    }
                    output[b * row_count + local_r] += dot;
                }
            }
            for (int b = 0; b < batch_size; ++b) {
                output[b * row_count + local_r] = output[b * row_count + local_r] * scale + b_val;
            }
        }
    }
    return 0;
}

int ojasx_rvq_fused_gemm(
    const std::uint8_t* packed_codes,
    const float* codebooks,
    const float* row_scales,
    const float* bias,
    const float* inputs,
    int batch_size,
    int stages,
    int bits_per_stage,
    int block_size,
    int rows,
    int columns,
    int row_start,
    int row_stop,
    float* output) {
    if (stages <= 0 || bits_per_stage <= 0 || block_size <= 0
        || rows <= 0 || columns <= 0 || row_start < 0
        || row_stop <= row_start || row_stop > rows
        || batch_size <= 0
        || packed_codes == nullptr || codebooks == nullptr
        || row_scales == nullptr || inputs == nullptr || output == nullptr) {
        return -1;
    }
    if (bits_per_stage != 1 && bits_per_stage != 2 && bits_per_stage != 4 && bits_per_stage != 8) {
        return -2;
    }

    const int blocks_per_row = (columns + block_size - 1) / block_size;
    const std::size_t total_vectors = static_cast<std::size_t>(rows) * blocks_per_row;
    const std::size_t stage_stride_bytes = (total_vectors * static_cast<std::size_t>(bits_per_stage) + 7U) / 8U;
    const int codebook_size = 1 << bits_per_stage;
    const int row_count = row_stop - row_start;
    const std::uint32_t mask = (1U << bits_per_stage) - 1U;

    constexpr int kMaxStackBlock = 256;
    float stack_vec[kMaxStackBlock];
    std::vector<float> heap_vec;
    float* vec = stack_vec;
    if (block_size > kMaxStackBlock) {
        heap_vec.resize(block_size);
        vec = heap_vec.data();
    }

    for (int r = row_start; r < row_stop; ++r) {
        const int local_r = r - row_start;
        const float scale = row_scales[r];
        const float b_val = (bias != nullptr) ? bias[r] : 0.0f;

        if (batch_size == 1) {
            float row_acc = 0.0f;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const std::size_t v = static_cast<std::size_t>(r) * blocks_per_row + blk;
                const std::size_t bit_pos = v * static_cast<std::size_t>(bits_per_stage);
                const std::size_t byte_pos = bit_pos / 8U;
                const unsigned shift = static_cast<unsigned>(bit_pos % 8U);

                std::fill_n(vec, block_size, 0.0f);
                for (int s = 0; s < stages; ++s) {
                    const std::uint32_t code = (packed_codes[s * stage_stride_bytes + byte_pos] >> shift) & mask;
                    const float* cb_entry = codebooks + (static_cast<std::size_t>(s) * codebook_size + code) * block_size;
                    for (int c = 0; c < block_size; ++c) {
                        vec[c] += cb_entry[c];
                    }
                }

                const int col_start = blk * block_size;
                const int col_count = std::min(block_size, columns - col_start);
                const float* in_ptr = inputs + col_start;
                float dot = 0.0f;
                for (int c = 0; c < col_count; ++c) {
                    dot += in_ptr[c] * vec[c];
                }
                row_acc += dot;
            }
            output[local_r] = row_acc * scale + b_val;
        } else {
            for (int b = 0; b < batch_size; ++b) {
                output[b * row_count + local_r] = 0.0f;
            }
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const std::size_t v = static_cast<std::size_t>(r) * blocks_per_row + blk;
                const std::size_t bit_pos = v * static_cast<std::size_t>(bits_per_stage);
                const std::size_t byte_pos = bit_pos / 8U;
                const unsigned shift = static_cast<unsigned>(bit_pos % 8U);

                std::fill_n(vec, block_size, 0.0f);
                for (int s = 0; s < stages; ++s) {
                    const std::uint32_t code = (packed_codes[s * stage_stride_bytes + byte_pos] >> shift) & mask;
                    const float* cb_entry = codebooks + (static_cast<std::size_t>(s) * codebook_size + code) * block_size;
                    for (int c = 0; c < block_size; ++c) {
                        vec[c] += cb_entry[c];
                    }
                }

                const int col_start = blk * block_size;
                const int col_count = std::min(block_size, columns - col_start);
                for (int b = 0; b < batch_size; ++b) {
                    const float* in_ptr = inputs + b * columns + col_start;
                    float dot = 0.0f;
                    for (int c = 0; c < col_count; ++c) {
                        dot += in_ptr[c] * vec[c];
                    }
                    output[b * row_count + local_r] += dot;
                }
            }
            for (int b = 0; b < batch_size; ++b) {
                output[b * row_count + local_r] = output[b * row_count + local_r] * scale + b_val;
            }
        }
    }
    return 0;
}

}  // extern "C"

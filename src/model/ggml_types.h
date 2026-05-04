#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace mugen {

/// GGML tensor element types, matching the ggml_type enum values stored in GGUF files.
/// Covers all types used by DeepSeek V3, Qwen3, Mixtral, and other MoE architectures.
enum class GGMLType : uint32_t {
    F32      = 0,
    F16      = 1,
    Q4_0     = 2,
    Q4_1     = 3,
    // 4, 5: deprecated (removed from ggml)
    Q5_0     = 6,
    Q5_1     = 7,
    Q8_0     = 8,
    Q8_1     = 9,
    Q2_K     = 10,
    Q3_K     = 11,
    Q4_K     = 12,
    Q5_K     = 13,
    Q6_K     = 14,
    Q8_K     = 15,
    IQ2_XXS  = 16,
    IQ2_XS   = 17,
    IQ3_XXS  = 18,
    IQ1_S    = 19,
    IQ4_NL   = 20,
    IQ3_S    = 21,
    IQ2_S    = 22,
    IQ4_XS   = 23,
    I8       = 24,
    I16      = 25,
    I32      = 26,
    I64      = 27,
    F64      = 28,
    IQ1_M    = 29,
    BF16     = 30,
};

/// Byte size of one quantization block for the given type.
/// For non-quantized types (F32, F16, I8, etc.), this equals the per-element size.
auto ggml_type_size(GGMLType type) -> size_t;

/// Number of elements per quantization block. Returns 1 for non-quantized types.
auto ggml_type_block_size(GGMLType type) -> size_t;

/// Human-readable name (e.g. "Q4_K", "F16"). Returns "UNKNOWN" for invalid types.
auto ggml_type_name(GGMLType type) -> std::string_view;

/// Whether the raw uint32_t maps to a known, usable GGMLType.
auto ggml_type_is_valid(uint32_t raw) -> bool;

/// Total byte size of a contiguous tensor with the given type and dimensions.
/// Quantization granularity applies along dims[0] (the innermost dimension).
/// Returns 0 if dims[0] is not divisible by block_size or on overflow.
auto ggml_tensor_byte_size(GGMLType type, std::span<const int64_t> dims) -> size_t;

}  // namespace mugen

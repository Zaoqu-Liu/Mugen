#include "model/ggml_types.h"

#include <climits>

namespace mugen {

namespace {

struct TypeTraits {
    size_t           type_size;
    size_t           block_size;
    std::string_view name;
};

// Indexed by the raw enum value.  Entries 4 and 5 are deprecated placeholders.
constexpr TypeTraits kTraits[] = {
    {   4,   1, "F32"      },  //  0
    {   2,   1, "F16"      },  //  1
    {  18,  32, "Q4_0"     },  //  2
    {  20,  32, "Q4_1"     },  //  3
    {   0,   0, "INVALID"  },  //  4  (deprecated)
    {   0,   0, "INVALID"  },  //  5  (deprecated)
    {  22,  32, "Q5_0"     },  //  6
    {  24,  32, "Q5_1"     },  //  7
    {  34,  32, "Q8_0"     },  //  8
    {  40,  32, "Q8_1"     },  //  9
    {  84, 256, "Q2_K"     },  // 10
    { 110, 256, "Q3_K"     },  // 11
    { 144, 256, "Q4_K"     },  // 12
    { 176, 256, "Q5_K"     },  // 13
    { 210, 256, "Q6_K"     },  // 14
    { 292, 256, "Q8_K"     },  // 15
    {  66, 256, "IQ2_XXS"  },  // 16
    {  74, 256, "IQ2_XS"   },  // 17
    {  98, 256, "IQ3_XXS"  },  // 18
    {  50, 256, "IQ1_S"    },  // 19
    {  18,  32, "IQ4_NL"   },  // 20
    { 110, 256, "IQ3_S"    },  // 21
    {  82, 256, "IQ2_S"    },  // 22
    { 136, 256, "IQ4_XS"   },  // 23
    {   1,   1, "I8"       },  // 24
    {   2,   1, "I16"      },  // 25
    {   4,   1, "I32"      },  // 26
    {   8,   1, "I64"      },  // 27
    {   8,   1, "F64"      },  // 28
    {  54, 256, "IQ1_M"    },  // 29
    {   2,   1, "BF16"     },  // 30
};

constexpr size_t kTraitsCount = sizeof(kTraits) / sizeof(kTraits[0]);

}  // namespace

auto ggml_type_size(GGMLType type) -> size_t {
    auto idx = static_cast<uint32_t>(type);
    if (idx >= kTraitsCount) return 0;
    return kTraits[idx].type_size;
}

auto ggml_type_block_size(GGMLType type) -> size_t {
    auto idx = static_cast<uint32_t>(type);
    if (idx >= kTraitsCount) return 0;
    return kTraits[idx].block_size;
}

auto ggml_type_name(GGMLType type) -> std::string_view {
    auto idx = static_cast<uint32_t>(type);
    if (idx >= kTraitsCount) return "UNKNOWN";
    return kTraits[idx].name;
}

auto ggml_type_is_valid(uint32_t raw) -> bool {
    if (raw >= kTraitsCount) return false;
    return kTraits[raw].type_size > 0;
}

auto ggml_tensor_byte_size(GGMLType type, std::span<const int64_t> dims) -> size_t {
    if (dims.empty()) return 0;

    size_t ts = ggml_type_size(type);
    size_t bs = ggml_type_block_size(type);
    if (ts == 0 || bs == 0) return 0;

    auto ne0 = static_cast<uint64_t>(dims[0]);
    if (dims[0] <= 0 || ne0 % bs != 0) return 0;

    size_t total = ts * (ne0 / bs);

    for (size_t i = 1; i < dims.size(); ++i) {
        if (dims[i] <= 0) return 0;
        auto d = static_cast<uint64_t>(dims[i]);
        if (total > SIZE_MAX / d) return 0;  // overflow guard
        total *= d;
    }
    return total;
}

}  // namespace mugen

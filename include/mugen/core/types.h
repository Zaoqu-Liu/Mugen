#pragma once

#include <cstddef>
#include <cstdint>

// Core type aliases and constants shared across the engine.

namespace mugen {

using f16  = _Float16;
using f32  = float;
using i32  = int32_t;
using i64  = int64_t;
using u32  = uint32_t;
using u64  = uint64_t;

struct Dims {
    u64 rows = 0;
    u64 cols = 0;
};

enum class QuantType : u32 {
    F16  = 0,
    Q8_0 = 1,
    Q4_0 = 2,
    Q4_K = 3,
};

}  // namespace mugen

#pragma once

#include <string_view>

#include "mugen/core/types.h"

// Public interface for model loading and metadata access.

namespace mugen {

struct ModelInfo {
    std::string_view name;
    std::string_view architecture;
    u64 parameter_count     = 0;
    u32 num_experts         = 0;
    u32 num_experts_per_tok = 0;
    u32 num_layers          = 0;
};

}  // namespace mugen

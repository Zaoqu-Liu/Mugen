#pragma once

// Mugen -- MoE extreme inference engine for Apple Silicon.
// This is the top-level umbrella header. Downstream code that links
// against mugen-core can include this single header to access the
// public API surface.

#include "mugen/core/types.h"
#include "mugen/model/model.h"
#include "mugen/server/api.h"

namespace mugen {

constexpr const char* kVersion = MUGEN_VERSION;

}  // namespace mugen

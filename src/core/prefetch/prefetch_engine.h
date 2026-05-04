#pragma once

#include <cstddef>
#include <cstdint>

// Asynchronous expert prefetch engine.
// Uses router logit predictions to speculatively stage expert weights from
// SSD into resident memory before the scheduler needs them, targeting full
// overlap of NVMe read latency with GPU compute of the current layer.

namespace mugen {

class PrefetchEngine {
public:
    PrefetchEngine() = default;
    ~PrefetchEngine() = default;

    PrefetchEngine(const PrefetchEngine&) = delete;
    PrefetchEngine& operator=(const PrefetchEngine&) = delete;
    PrefetchEngine(PrefetchEngine&&) noexcept = default;
    PrefetchEngine& operator=(PrefetchEngine&&) noexcept = default;
};

}  // namespace mugen

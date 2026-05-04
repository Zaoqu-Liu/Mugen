#pragma once

// Unified Speculative Prefetch Pipeline (USPP) scheduler.
// Coordinates expert activation prediction, SSD prefetch, and GPU dispatch
// to overlap I/O latency with compute across the MoE forward pass.

namespace mugen {

class Scheduler {
public:
    Scheduler() = default;
    ~Scheduler() = default;

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) noexcept = default;
    Scheduler& operator=(Scheduler&&) noexcept = default;
};

}  // namespace mugen

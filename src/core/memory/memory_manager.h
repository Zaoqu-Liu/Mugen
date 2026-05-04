#pragma once

#include <cstddef>
#include <cstdint>

// Tiered memory manager for MoE expert weights.
// Manages a three-level hierarchy: GPU VRAM (MTLBuffer) -> unified memory
// (mmap resident) -> SSD (mmap non-resident), with page-granular promotion
// and eviction driven by the prefetch engine.

namespace mugen {

class MemoryManager {
public:
    MemoryManager() = default;
    ~MemoryManager() = default;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) noexcept = default;
    MemoryManager& operator=(MemoryManager&&) noexcept = default;
};

}  // namespace mugen

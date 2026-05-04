#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "core/memory/mmap_loader.h"
#include "core/prefetch/expert_index.h"

namespace mugen {

/// Metadata for a single expert's slot within an ExpertBuffer.
struct BufferSlot {
    ExpertKey key;
    size_t offset;      // byte offset in the buffer's backing region
    size_t size;        // total bytes reserved
    bool populated;     // true once all tensor data has been copied in
};

/// Contiguous memory region holding a set of expert weight tensors.
///
/// Uses a bump allocator: allocations advance a pointer, individual
/// deallocations only remove bookkeeping — physical reclamation happens
/// in bulk via clear().  The backing store is MAP_ANONYMOUS so pages
/// are lazily faulted by the kernel.
class ExpertBuffer {
public:
    explicit ExpertBuffer(size_t capacity_bytes);
    ~ExpertBuffer();

    ExpertBuffer(const ExpertBuffer&) = delete;
    ExpertBuffer& operator=(const ExpertBuffer&) = delete;
    ExpertBuffer(ExpertBuffer&&) = delete;
    ExpertBuffer& operator=(ExpertBuffer&&) = delete;

    auto allocate(const ExpertKey& key, size_t bytes) -> BufferSlot*;
    void deallocate(const ExpertKey& key);
    auto find(const ExpertKey& key) const -> const BufferSlot*;

    auto data_ptr(const BufferSlot& slot) const -> const void*;
    auto data_ptr_mut(const BufferSlot& slot) -> void*;

    auto capacity() const -> size_t   { return capacity_; }
    auto used_bytes() const -> size_t { return allocated_; }
    auto free_bytes() const -> size_t { return capacity_ - allocated_; }
    auto slot_count() const -> size_t { return slots_.size(); }

    /// Reset bump pointer and drop all slot records.
    /// Issues MADV_DONTNEED so the kernel can reclaim physical pages.
    void clear();

private:
    size_t capacity_;
    void*  backing_;        // mmap MAP_ANONYMOUS
    size_t allocated_ = 0;
    std::unordered_map<ExpertKey, BufferSlot> slots_;
};

// ---------------------------------------------------------------------------

/// Double-buffered memory manager for MoE expert weights.
///
/// Maintains two equally-sized ExpertBuffers (A/B):
///   • **active** — currently consumed by the GPU dispatch thread (read-only)
///   • **staging** — being filled by the I/O prefetch thread (write-only)
///
/// swap_buffers() atomically promotes staging→active and clears the old
/// active for the next prefetch round.  A separate **pinned** buffer holds
/// frequently-accessed experts that survive swaps.
class BufferManager {
public:
    struct Config {
        size_t buffer_capacity = 15ULL * 1024 * 1024 * 1024;   // per A/B buffer
        size_t pinned_capacity =  5ULL * 1024 * 1024 * 1024;
        size_t system_reserve  =  8ULL * 1024 * 1024 * 1024;   // for OS / KV cache
    };

    explicit BufferManager(Config config);
    ~BufferManager();

    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    // ---- core swap --------------------------------------------------------

    void swap_buffers();

    auto active_buffer() const -> const ExpertBuffer&;
    auto staging_buffer() -> ExpertBuffer&;

    // ---- pinned experts ---------------------------------------------------

    auto pinned_buffer() -> ExpertBuffer&;
    auto pinned_buffer() const -> const ExpertBuffer&;

    // ---- expert I/O -------------------------------------------------------

    auto stage_expert(const ExpertKey& key,
                      const MmapRegion& mmap,
                      const ExpertLocation& location) -> bool;

    auto pin_expert(const ExpertKey& key,
                    const MmapRegion& mmap,
                    const ExpertLocation& location) -> bool;

    /// Lookup: pinned first, then active.  Returns data pointer or nullptr.
    auto find_expert(const ExpertKey& key) const -> const void*;

    void prefetch_expert(const MmapRegion& mmap,
                         const ExpertLocation& location);

    void evict_from_mmap(const MmapRegion& mmap,
                         const ExpertLocation& location);

    // ---- pressure & stats -------------------------------------------------

    /// 0.0 = plenty of headroom, 1.0 = critical.  Uses os_proc_available_memory().
    auto memory_pressure() const -> float;

    struct Stats {
        size_t   active_used;
        size_t   staging_used;
        size_t   pinned_used;
        size_t   total_capacity;
        uint64_t swap_count;
        float    memory_pressure;
    };
    auto stats() const -> Stats;

    auto config() const -> const Config& { return config_; }

private:
    Config config_;

    std::unique_ptr<ExpertBuffer> buffer_a_;
    std::unique_ptr<ExpertBuffer> buffer_b_;
    std::atomic<ExpertBuffer*>    active_;
    std::atomic<ExpertBuffer*>    staging_;

    std::unique_ptr<ExpertBuffer> pinned_;

    std::atomic<uint64_t> swap_count_{0};

    static auto load_expert_to_buffer(ExpertBuffer& buffer,
                                      const ExpertKey& key,
                                      const MmapRegion& mmap,
                                      const ExpertLocation& location) -> bool;
};

}  // namespace mugen

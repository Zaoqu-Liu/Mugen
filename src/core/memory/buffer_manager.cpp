#include "core/memory/buffer_manager.h"

#include <cstring>

#include <sys/mman.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace mugen {

// ===========================================================================
// ExpertBuffer
// ===========================================================================

ExpertBuffer::ExpertBuffer(size_t capacity_bytes)
    : capacity_(capacity_bytes) {
    backing_ = ::mmap(nullptr, capacity_,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1, 0);
    if (backing_ == MAP_FAILED) {
        backing_ = nullptr;
        capacity_ = 0;
    }
}

ExpertBuffer::~ExpertBuffer() {
    if (backing_ && backing_ != MAP_FAILED) {
        ::munmap(backing_, capacity_);
    }
}

auto ExpertBuffer::allocate(const ExpertKey& key, size_t bytes) -> BufferSlot* {
    if (allocated_ + bytes > capacity_) return nullptr;
    if (slots_.contains(key)) return nullptr;

    auto [it, ok] = slots_.emplace(key, BufferSlot{
        .key       = key,
        .offset    = allocated_,
        .size      = bytes,
        .populated = false,
    });

    allocated_ += bytes;
    return &it->second;
}

void ExpertBuffer::deallocate(const ExpertKey& key) {
    slots_.erase(key);
}

auto ExpertBuffer::find(const ExpertKey& key) const -> const BufferSlot* {
    auto it = slots_.find(key);
    return it != slots_.end() ? &it->second : nullptr;
}

auto ExpertBuffer::data_ptr(const BufferSlot& slot) const -> const void* {
    return static_cast<const uint8_t*>(backing_) + slot.offset;
}

auto ExpertBuffer::data_ptr_mut(const BufferSlot& slot) -> void* {
    return static_cast<uint8_t*>(backing_) + slot.offset;
}

void ExpertBuffer::clear() {
    slots_.clear();
    allocated_ = 0;
    if (backing_ && capacity_ > 0) {
        ::madvise(backing_, capacity_, MADV_DONTNEED);
    }
}

// ===========================================================================
// BufferManager
// ===========================================================================

BufferManager::BufferManager(Config config)
    : config_(config)
    , buffer_a_(std::make_unique<ExpertBuffer>(config.buffer_capacity))
    , buffer_b_(std::make_unique<ExpertBuffer>(config.buffer_capacity))
    , pinned_(std::make_unique<ExpertBuffer>(config.pinned_capacity)) {
    active_.store(buffer_a_.get(), std::memory_order_relaxed);
    staging_.store(buffer_b_.get(), std::memory_order_relaxed);
}

BufferManager::~BufferManager() = default;

// ---------------------------------------------------------------------------
// Swap
// ---------------------------------------------------------------------------

void BufferManager::swap_buffers() {
    // Exchange staging pointer with the current active pointer, then store
    // the former active into staging.  This is safe because swap is invoked
    // from a single coordination point after both the GPU dispatch and I/O
    // threads have reached a barrier.
    auto* prev_staging = staging_.load(std::memory_order_acquire);
    auto* prev_active  = active_.exchange(prev_staging,
                                          std::memory_order_acq_rel);
    staging_.store(prev_active, std::memory_order_release);

    prev_active->clear();
    swap_count_.fetch_add(1, std::memory_order_relaxed);
}

auto BufferManager::active_buffer() const -> const ExpertBuffer& {
    return *active_.load(std::memory_order_acquire);
}

auto BufferManager::staging_buffer() -> ExpertBuffer& {
    return *staging_.load(std::memory_order_acquire);
}

auto BufferManager::pinned_buffer() -> ExpertBuffer& { return *pinned_; }
auto BufferManager::pinned_buffer() const -> const ExpertBuffer& { return *pinned_; }

// ---------------------------------------------------------------------------
// Expert loading
// ---------------------------------------------------------------------------

auto BufferManager::load_expert_to_buffer(
        ExpertBuffer& buffer,
        const ExpertKey& key,
        const MmapRegion& mmap,
        const ExpertLocation& location) -> bool {

    auto* slot = buffer.allocate(key, location.total_bytes);
    if (!slot) return false;

    auto* dst = static_cast<uint8_t*>(buffer.data_ptr_mut(*slot));
    size_t write_pos = 0;

    for (const auto& t : location.tensors) {
        if (t.file_offset + t.byte_size > mmap.size()) {
            buffer.deallocate(key);
            return false;
        }
        const auto* src = static_cast<const uint8_t*>(mmap.data())
                        + t.file_offset;
        std::memcpy(dst + write_pos, src, t.byte_size);
        write_pos += t.byte_size;
    }

    slot->populated = true;
    return true;
}

auto BufferManager::stage_expert(
        const ExpertKey& key,
        const MmapRegion& mmap,
        const ExpertLocation& location) -> bool {
    return load_expert_to_buffer(staging_buffer(), key, mmap, location);
}

auto BufferManager::pin_expert(
        const ExpertKey& key,
        const MmapRegion& mmap,
        const ExpertLocation& location) -> bool {
    return load_expert_to_buffer(*pinned_, key, mmap, location);
}

auto BufferManager::find_expert(const ExpertKey& key) const -> const void* {
    if (auto* slot = pinned_->find(key); slot && slot->populated)
        return pinned_->data_ptr(*slot);

    const auto& active = active_buffer();
    if (auto* slot = active.find(key); slot && slot->populated)
        return active.data_ptr(*slot);

    return nullptr;
}

// ---------------------------------------------------------------------------
// mmap advice helpers
// ---------------------------------------------------------------------------

void BufferManager::prefetch_expert(const MmapRegion& mmap,
                                     const ExpertLocation& location) {
    for (const auto& t : location.tensors)
        mmap.advise_willneed(t.file_offset, t.byte_size);
}

void BufferManager::evict_from_mmap(const MmapRegion& mmap,
                                     const ExpertLocation& location) {
    for (const auto& t : location.tensors)
        mmap.advise_dontneed(t.file_offset, t.byte_size);
}

// ---------------------------------------------------------------------------
// Memory pressure
// ---------------------------------------------------------------------------

auto BufferManager::memory_pressure() const -> float {
#if defined(__APPLE__)
    // Query free physical memory via Mach VM statistics
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(host, HOST_VM_INFO64,
                                         reinterpret_cast<host_info64_t>(&vm_stat), &count);
    size_t available = 0;
    if (kr == KERN_SUCCESS) {
        available = static_cast<size_t>(vm_stat.free_count) * vm_page_size;
    }

    // Map [low_mark .. high_mark] → [1.0 .. 0.0]
    //   >= 2 × system_reserve  →  0.0  (Normal)
    //   =  system_reserve      → ~0.5  (Warning zone)
    //   <= system_reserve / 2  →  1.0  (Critical)
    size_t high_mark = config_.system_reserve * 2;
    size_t low_mark  = config_.system_reserve / 2;

    if (available >= high_mark) return 0.0f;
    if (available <= low_mark)  return 1.0f;

    return 1.0f - static_cast<float>(available - low_mark)
                / static_cast<float>(high_mark - low_mark);
#else
    return 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

auto BufferManager::stats() const -> Stats {
    const auto& active  = active_buffer();
    const auto& staging = *staging_.load(std::memory_order_acquire);
    return {
        .active_used     = active.used_bytes(),
        .staging_used    = staging.used_bytes(),
        .pinned_used     = pinned_->used_bytes(),
        .total_capacity  = config_.buffer_capacity * 2 + config_.pinned_capacity,
        .swap_count      = swap_count_.load(std::memory_order_relaxed),
        .memory_pressure = memory_pressure(),
    };
}

}  // namespace mugen

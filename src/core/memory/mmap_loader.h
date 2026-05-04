#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace mugen {

/// RAII handle for a read-only memory-mapped file region.
///
/// Owns both the virtual mapping and the underlying file descriptor.
/// On destruction the mapping is released (munmap) and the fd is closed.
class MmapRegion {
public:
    MmapRegion(MmapRegion&& other) noexcept;
    MmapRegion& operator=(MmapRegion&& other) noexcept;
    ~MmapRegion();

    auto data() const -> const void* { return addr_; }
    auto size() const -> size_t      { return size_; }

    /// Advise the kernel to fault in pages covering [offset, offset+length).
    /// Offsets are internally rounded to page boundaries.
    auto advise_willneed(size_t offset, size_t length) const -> bool;

    /// Advise the kernel that pages covering [offset, offset+length) are no
    /// longer needed and may be evicted from physical memory.
    auto advise_dontneed(size_t offset, size_t length) const -> bool;

    MmapRegion(const MmapRegion&) = delete;
    MmapRegion& operator=(const MmapRegion&) = delete;

private:
    friend class MmapLoader;
    MmapRegion(void* addr, size_t size, int fd);

    void* addr_  = nullptr;
    size_t size_ = 0;
    int    fd_   = -1;
};

/// Utility for creating read-only memory mappings of model files.
class MmapLoader {
public:
    /// Map the entire contents of a file as MAP_PRIVATE | PROT_READ.
    /// A SIGBUS handler is installed (once) to produce a diagnostic if
    /// the underlying file is truncated while mapped.
    static auto map_file(const std::filesystem::path& path)
        -> std::expected<MmapRegion, std::string>;

    /// System virtual-memory page size (16 384 on Apple Silicon).
    static auto page_size() -> size_t;

    /// Round `offset` down to the nearest page boundary.
    static auto align_to_page(size_t offset) -> size_t;

    /// Whether `addr` sits on a page boundary.
    static auto is_page_aligned(const void* addr) -> bool;
};

}  // namespace mugen

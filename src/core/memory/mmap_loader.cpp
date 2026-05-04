#include "core/memory/mmap_loader.h"

#include <cerrno>
#include <cstring>
#include <mutex>
#include <utility>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mugen {

// ---------------------------------------------------------------------------
// SIGBUS handler – installed once on first mmap
// ---------------------------------------------------------------------------
namespace {

void sigbus_handler(int /*sig*/, siginfo_t* /*info*/, void* /*ctx*/) {
    // Only async-signal-safe calls allowed here.
    const char msg[] = "\nmugen: fatal: SIGBUS (file truncated or I/O error)\n";
    (void)::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    ::_exit(99);
}

std::once_flag g_sigbus_once;

void install_sigbus_handler() {
    std::call_once(g_sigbus_once, [] {
        struct sigaction sa{};
        sa.sa_flags     = SA_SIGINFO;
        sa.sa_sigaction = sigbus_handler;
        sigemptyset(&sa.sa_mask);
        ::sigaction(SIGBUS, &sa, nullptr);
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// MmapRegion
// ---------------------------------------------------------------------------

MmapRegion::MmapRegion(void* addr, size_t size, int fd)
    : addr_(addr), size_(size), fd_(fd) {}

MmapRegion::MmapRegion(MmapRegion&& other) noexcept
    : addr_(std::exchange(other.addr_, nullptr))
    , size_(std::exchange(other.size_, 0))
    , fd_(std::exchange(other.fd_, -1)) {}

MmapRegion& MmapRegion::operator=(MmapRegion&& other) noexcept {
    if (this != &other) {
        if (addr_ && addr_ != MAP_FAILED) ::munmap(addr_, size_);
        if (fd_ >= 0) ::close(fd_);

        addr_ = std::exchange(other.addr_, nullptr);
        size_ = std::exchange(other.size_, 0);
        fd_   = std::exchange(other.fd_, -1);
    }
    return *this;
}

MmapRegion::~MmapRegion() {
    if (addr_ && addr_ != MAP_FAILED) ::munmap(addr_, size_);
    if (fd_ >= 0) ::close(fd_);
}

auto MmapRegion::advise_willneed(size_t offset, size_t length) const -> bool {
    if (!addr_ || offset >= size_) return false;
    if (offset + length > size_) length = size_ - offset;

    size_t ps   = MmapLoader::page_size();
    size_t base = (offset / ps) * ps;
    size_t end  = offset + length;
    if (end < offset) end = size_;  // overflow guard
    size_t len  = end - base;

    auto* ptr = static_cast<char*>(addr_) + base;
    return ::madvise(ptr, len, MADV_WILLNEED) == 0;
}

auto MmapRegion::advise_dontneed(size_t offset, size_t length) const -> bool {
    if (!addr_ || offset >= size_) return false;
    if (offset + length > size_) length = size_ - offset;

    size_t ps   = MmapLoader::page_size();
    size_t base = (offset / ps) * ps;
    size_t end  = offset + length;
    if (end < offset) end = size_;
    size_t len  = end - base;

    auto* ptr = static_cast<char*>(addr_) + base;
    return ::madvise(ptr, len, MADV_DONTNEED) == 0;
}

// ---------------------------------------------------------------------------
// MmapLoader
// ---------------------------------------------------------------------------

auto MmapLoader::map_file(const std::filesystem::path& path)
    -> std::expected<MmapRegion, std::string>
{
    install_sigbus_handler();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return std::unexpected("cannot open file: " + path.string()
                               + " (" + std::strerror(errno) + ")");

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected("fstat failed: " + std::string(std::strerror(errno)));
    }

    auto file_size = static_cast<size_t>(st.st_size);
    if (file_size == 0) {
        ::close(fd);
        return std::unexpected("cannot mmap empty file: " + path.string());
    }

    void* addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        return std::unexpected("mmap failed: " + std::string(std::strerror(errno)));
    }

    return MmapRegion{addr, file_size, fd};
}

auto MmapLoader::page_size() -> size_t {
    static const size_t ps = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    return ps;
}

auto MmapLoader::align_to_page(size_t offset) -> size_t {
    size_t ps = page_size();
    return (offset / ps) * ps;
}

auto MmapLoader::is_page_aligned(const void* addr) -> bool {
    return (reinterpret_cast<uintptr_t>(addr) % page_size()) == 0;
}

}  // namespace mugen

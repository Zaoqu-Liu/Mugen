// Mugen SSD Sequential Read Throughput Benchmark
// Measures sustained sequential read on Apple Silicon NVMe via mmap and pread.
// Target: ARM64 macOS, 16KB page size.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr size_t      kFileSize   = 2ULL * 1024 * 1024 * 1024;
constexpr size_t      kARM64Page  = 16384;
constexpr int         kRuns       = 3;
constexpr const char* kPath       = "/tmp/mugen_bench_ssd_2g.bin";

double g_ns_per_tick;

void init_timer() {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = static_cast<double>(tb.numer) / static_cast<double>(tb.denom);
}

double ticks_to_sec(uint64_t t) {
    return static_cast<double>(t) * g_ns_per_tick * 1e-9;
}

double to_gib(size_t b) {
    return static_cast<double>(b) / static_cast<double>(1ULL << 30);
}

double median3(const double (&a)[kRuns]) {
    double tmp[kRuns];
    std::copy(a, a + kRuns, tmp);
    std::sort(tmp, tmp + kRuns);
    return tmp[kRuns / 2];
}

bool ensure_test_file() {
    struct stat st{};
    if (stat(kPath, &st) == 0 &&
        static_cast<size_t>(st.st_size) == kFileSize) {
        std::fprintf(stderr, "[info] Reusing %s\n", kPath);
        return true;
    }

    std::fprintf(stderr, "[info] Writing %.1f GiB to %s ...\n",
                 to_gib(kFileSize), kPath);

    int fd = open(kPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open(%s): %s\n", kPath, std::strerror(errno));
        return false;
    }

    constexpr size_t kChunk = 1U << 20;
    void* buf = std::malloc(kChunk);
    if (!buf) {
        std::fprintf(stderr, "[error] malloc failed\n");
        close(fd);
        return false;
    }
    arc4random_buf(buf, kChunk);

    for (size_t off = 0; off < kFileSize; off += kChunk) {
        ssize_t n = write(fd, buf, kChunk);
        if (n != static_cast<ssize_t>(kChunk)) {
            std::fprintf(stderr, "[error] write at offset %zu: %s\n",
                         off, std::strerror(errno));
            std::free(buf);
            close(fd);
            unlink(kPath);
            return false;
        }
    }

    fsync(fd);
    close(fd);
    std::free(buf);
    std::fprintf(stderr, "[info] File ready.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Method A: mmap + MADV_SEQUENTIAL + MADV_WILLNEED + sequential scan
// ---------------------------------------------------------------------------
void run_mmap() {
    int fd = open(kPath, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open: %s\n", std::strerror(errno));
        return;
    }

    double gbps[kRuns]{};

    for (int r = 0; r < kRuns; ++r) {
        void* m = mmap(nullptr, kFileSize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) {
            std::fprintf(stderr, "[error] mmap: %s\n", std::strerror(errno));
            close(fd);
            return;
        }

        if (madvise(m, kFileSize, MADV_SEQUENTIAL) != 0 && r == 0)
            std::fprintf(stderr, "[warn] madvise(SEQUENTIAL): %s\n", std::strerror(errno));

        auto* p = static_cast<const uint8_t*>(m);
        uint64_t csum = 0;

        uint64_t t0 = mach_absolute_time();
        madvise(m, kFileSize, MADV_WILLNEED);
        for (size_t o = 0; o < kFileSize; o += 64)
            csum ^= *reinterpret_cast<const uint64_t*>(p + o);
        uint64_t t1 = mach_absolute_time();

        double sec = ticks_to_sec(t1 - t0);
        gbps[r] = to_gib(kFileSize) / sec;

        std::fprintf(stderr, "  mmap run %d: %.2f GB/s  %.3fs  csum=%016llx\n",
                     r, gbps[r], sec, static_cast<unsigned long long>(csum));

        munmap(m, kFileSize);
    }

    double med = median3(gbps);
    std::printf("mmap sequential read:   %.2f GB/s  (%.2f GB in %.2fs)\n",
                med, to_gib(kFileSize), to_gib(kFileSize) / med);
    close(fd);
}

// ---------------------------------------------------------------------------
// Method B: pread + F_NOCACHE (bypasses page cache → true SSD throughput)
// ---------------------------------------------------------------------------
void run_pread(size_t blk, const char* label) {
    int fd = open(kPath, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open: %s\n", std::strerror(errno));
        return;
    }

    if (fcntl(fd, F_NOCACHE, 1) == -1)
        std::fprintf(stderr, "[warn] F_NOCACHE failed: %s\n", std::strerror(errno));

    void* raw = nullptr;
    if (posix_memalign(&raw, kARM64Page, blk) != 0) {
        std::fprintf(stderr, "[error] posix_memalign failed\n");
        close(fd);
        return;
    }
    auto* buf = static_cast<uint8_t*>(raw);

    double gbps[kRuns]{};

    for (int r = 0; r < kRuns; ++r) {
        uint64_t csum = 0;
        off_t    pos  = 0;
        auto     end  = static_cast<off_t>(kFileSize);

        uint64_t t0 = mach_absolute_time();
        while (pos < end) {
            ssize_t n = pread(fd, buf, blk, pos);
            if (n <= 0) break;
            csum ^= *reinterpret_cast<const uint64_t*>(buf);
            pos += n;
        }
        uint64_t t1 = mach_absolute_time();

        double sec = ticks_to_sec(t1 - t0);
        gbps[r] = to_gib(static_cast<size_t>(pos)) / sec;

        std::fprintf(stderr, "  pread %s run %d: %.2f GB/s  %.3fs  csum=%016llx\n",
                     label, r, gbps[r], sec, static_cast<unsigned long long>(csum));
    }

    double med = median3(gbps);
    std::printf("pread %-17s %.2f GB/s\n", label, med);

    std::free(buf);
    close(fd);
}

} // namespace

int main() {
    init_timer();

    std::printf("=== Mugen SSD Sequential Read Benchmark ===\n");
    std::printf("file: %s  size: %.1f GiB  page: %zu  runs: %d\n\n",
                kPath, to_gib(kFileSize), kARM64Page, kRuns);

    if (!ensure_test_file()) return 1;

    bool purged = (std::system("sudo -n purge 2>/dev/null") == 0);
    std::fprintf(stderr, "[info] purge: %s\n", purged ? "ok" : "skipped");
    std::printf("cache purge: %s\n\n", purged ? "yes" : "no (pread uses F_NOCACHE)");

    run_mmap();
    run_pread(64  * 1024, "64KB blocks:");
    run_pread(256 * 1024, "256KB blocks:");
    run_pread(1024* 1024, "1MB blocks:");

    std::printf("\n");
    unlink(kPath);
    std::fprintf(stderr, "[info] Removed %s\n", kPath);
    return 0;
}

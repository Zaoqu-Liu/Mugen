// Mugen madvise Syscall Latency Benchmark
// Measures per-call overhead of madvise(MADV_WILLNEED) and madvise(MADV_DONTNEED)
// at various region sizes on ARM64 macOS (16KB page granularity).

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr size_t      kFileSize   = 1ULL * 1024 * 1024 * 1024;
constexpr size_t      kARM64Page  = 16384;
constexpr int         kIterations = 10000;
constexpr int         kWarmup     = 200;
constexpr const char* kPath       = "/tmp/mugen_bench_madvise_1g.bin";

constexpr size_t kRegionSizes[] = {
    16  * 1024,
    64  * 1024,
    256 * 1024,
    1024* 1024,
};

double g_ns_per_tick;

void init_timer() {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = static_cast<double>(tb.numer) / static_cast<double>(tb.denom);
}

double ticks_to_us(uint64_t t) {
    return static_cast<double>(t) * g_ns_per_tick * 1e-3;
}

struct Stats {
    double p50, p95, p99, max;
};

Stats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    auto n = v.size();
    return {
        v[n * 50 / 100],
        v[n * 95 / 100],
        v[n * 99 / 100],
        v[n - 1],
    };
}

bool ensure_test_file() {
    struct stat st{};
    if (stat(kPath, &st) == 0 &&
        static_cast<size_t>(st.st_size) == kFileSize) {
        std::fprintf(stderr, "[info] Reusing %s\n", kPath);
        return true;
    }

    std::fprintf(stderr, "[info] Writing 1 GiB to %s ...\n", kPath);

    int fd = open(kPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open: %s\n", std::strerror(errno));
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
            std::fprintf(stderr, "[error] write: %s\n", std::strerror(errno));
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

void format_size(char* buf, size_t buflen, size_t bytes) {
    if (bytes >= 1024 * 1024)
        std::snprintf(buf, buflen, "%zuMB", bytes / (1024 * 1024));
    else
        std::snprintf(buf, buflen, "%zuKB", bytes / 1024);
}

void bench_advice(void* base, size_t region_size, int advice, const char* advice_name) {
    size_t num_regions = kFileSize / region_size;

    // Warmup: exercise the syscall path and TLB
    for (int i = 0; i < kWarmup; ++i) {
        size_t offset = (static_cast<size_t>(i) % num_regions) * region_size;
        madvise(static_cast<uint8_t*>(base) + offset, region_size, advice);
    }

    std::vector<double> latencies;
    latencies.reserve(kIterations);

    for (int i = 0; i < kIterations; ++i) {
        size_t offset = (static_cast<size_t>(i) % num_regions) * region_size;
        auto*  addr   = static_cast<uint8_t*>(base) + offset;

        uint64_t t0 = mach_absolute_time();
        int ret = madvise(addr, region_size, advice);
        uint64_t t1 = mach_absolute_time();

        if (ret != 0 && i == 0)
            std::fprintf(stderr, "[warn] madvise(%s) failed: %s\n",
                         advice_name, std::strerror(errno));

        latencies.push_back(ticks_to_us(t1 - t0));
    }

    Stats s = compute_stats(latencies);

    char sz_str[16];
    format_size(sz_str, sizeof(sz_str), region_size);

    // Pad label to 28 chars for aligned output
    char label[48];
    std::snprintf(label, sizeof(label), "madvise(%s) %s:", advice_name, sz_str);
    std::printf("%-28s p50=%.1f\xC2\xB5s  p95=%.1f\xC2\xB5s  p99=%.1f\xC2\xB5s  max=%.1f\xC2\xB5s\n",
                label, s.p50, s.p95, s.p99, s.max);
}

} // namespace

int main() {
    init_timer();

    std::printf("=== Mugen madvise Latency Benchmark ===\n");
    std::printf("file: %s  size: 1 GiB  page: %zu\n", kPath, kARM64Page);
    std::printf("iterations: %d  warmup: %d\n\n", kIterations, kWarmup);

    if (!ensure_test_file()) return 1;

    int fd = open(kPath, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open: %s\n", std::strerror(errno));
        return 1;
    }

    void* base = mmap(nullptr, kFileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[error] mmap: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }

    for (size_t sz : kRegionSizes) {
        bench_advice(base, sz, MADV_WILLNEED, "WILLNEED");
        bench_advice(base, sz, MADV_DONTNEED, "DONTNEED");
        std::printf("\n");
    }

    // Risk assessment
    std::printf("--- Risk threshold: p99 > 50\xC2\xB5s is a concern for high-freq prefetch ---\n");

    munmap(base, kFileSize);
    close(fd);
    unlink(kPath);
    std::fprintf(stderr, "[info] Removed %s\n", kPath);
    return 0;
}

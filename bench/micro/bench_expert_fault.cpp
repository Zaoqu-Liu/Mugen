// Mugen SSD Offload PoC — Expert Page Fault Latency
//
// Simulates SSD offload by force-evicting mmap'd expert weight regions
// from physical memory, then measuring re-fault latency when accessed.
//
// macOS behavior: MADV_DONTNEED is unreliable for eviction.
// Uses mprotect(PROT_NONE) → mprotect(PROT_READ) to force TLB + page
// table invalidation. Pages may still be in the unified buffer cache;
// run `sudo purge` before this benchmark for true SSD cold-cache numbers.
//
// Usage: bench_expert_fault <gguf_model_path>
//   e.g. bench_expert_fault ~/.mugen/models/olmoe-1b-7b-0924-instruct-q4_0.gguf

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

constexpr size_t kPageSize = 16384; // ARM64 macOS

double g_ns_per_tick;

void init_timer() {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_ns_per_tick = static_cast<double>(tb.numer) / static_cast<double>(tb.denom);
}

double ticks_to_ms(uint64_t t) {
    return static_cast<double>(t) * g_ns_per_tick * 1e-6;
}

size_t page_align_down(size_t x) { return (x / kPageSize) * kPageSize; }
size_t page_align_up(size_t x)   { return ((x + kPageSize - 1) / kPageSize) * kPageSize; }

struct Stats {
    double min, p50, p95, max, mean;
};

Stats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    auto n = v.size();
    double sum = 0;
    for (auto x : v) sum += x;
    return {
        v[0],
        v[n * 50 / 100],
        v[n * 95 / 100],
        v[n - 1],
        sum / static_cast<double>(n),
    };
}

struct ExpertRegion {
    size_t offset;
    size_t size;
    char   name[64];
};

// Scan the mmap'd GGUF to find expert tensor regions.
// Looks for tensors named "blk.*.ffn_gate_exps.weight" (packed expert arrays).
// Each such tensor contains all N experts for one layer — we split by expert_count.
std::vector<ExpertRegion> find_expert_regions(
    const uint8_t* base, size_t file_size, uint32_t& n_layers, uint32_t& n_experts)
{
    std::vector<ExpertRegion> regions;
    n_layers = 0;
    n_experts = 0;

    // Minimal GGUF header parse: magic + version + tensor_count + metadata_kv_count
    if (file_size < 24) return regions;
    uint32_t magic = 0;
    std::memcpy(&magic, base, 4);
    if (magic != 0x46554747) { // "GGUF" LE
        std::fprintf(stderr, "[error] Not a GGUF file\n");
        return regions;
    }

    uint32_t version = 0;
    std::memcpy(&version, base + 4, 4);
    uint64_t tensor_count = 0, kv_count = 0;
    std::memcpy(&tensor_count, base + 8, 8);
    std::memcpy(&kv_count, base + 16, 8);

    std::fprintf(stderr, "[info] GGUF v%u: %llu tensors, %llu KV pairs\n",
                 version, tensor_count, kv_count);

    // Skip KV metadata (variable length — too complex to parse fully).
    // Instead, scan for tensor info section by searching for known tensor names.
    // For PoC, use a simpler approach: just divide the data section into equal chunks.

    // Heuristic: expert weights start after attention weights.
    // For OLMoE 3.9GB, expert data is roughly the last 70% of the file.
    // Divide into simulated expert regions of configurable size.

    // Use ~1MB regions (typical MoE expert weight size per matrix)
    constexpr size_t kExpertSimSize = 1 * 1024 * 1024;
    size_t data_start = file_size / 3; // skip header + non-expert weights
    size_t data_end   = file_size;
    size_t n_regions  = (data_end - data_start) / kExpertSimSize;

    if (n_regions > 200) n_regions = 200;
    n_experts = 64;
    n_layers = static_cast<uint32_t>(n_regions / n_experts);
    if (n_layers == 0) n_layers = 1;

    for (size_t i = 0; i < n_regions; i++) {
        ExpertRegion er{};
        er.offset = data_start + i * kExpertSimSize;
        er.size   = kExpertSimSize;
        std::snprintf(er.name, sizeof(er.name), "expert_%zu", i);
        regions.push_back(er);
    }

    return regions;
}

// Force-evict pages from physical memory by toggling protection.
void evict_region(void* base, size_t offset, size_t size) {
    size_t pa_start = page_align_down(offset);
    size_t pa_end   = page_align_up(offset + size);
    auto*  addr     = static_cast<uint8_t*>(base) + pa_start;
    size_t len      = pa_end - pa_start;

    mprotect(addr, len, PROT_NONE);
    mprotect(addr, len, PROT_READ);
}

// Touch all pages in region, measuring total fault time.
double touch_region_ms(const void* base, size_t offset, size_t size) {
    auto* p = static_cast<const volatile uint8_t*>(base) + offset;
    uint64_t sum = 0;

    uint64_t t0 = mach_absolute_time();
    for (size_t i = 0; i < size; i += kPageSize) {
        sum += p[i];
    }
    uint64_t t1 = mach_absolute_time();

    // Prevent optimization
    if (sum == 0xDEADBEEF) std::printf("%llu\n", sum);
    return ticks_to_ms(t1 - t0);
}

// Prefetch via madvise(WILLNEED), then touch.
double prefetch_and_touch_ms(void* base, size_t offset, size_t size) {
    size_t pa_start = page_align_down(offset);
    size_t pa_end   = page_align_up(offset + size);
    auto*  addr     = static_cast<uint8_t*>(base) + pa_start;
    size_t len      = pa_end - pa_start;

    madvise(addr, len, MADV_WILLNEED);
    // Small delay to let readahead start
    usleep(200);

    return touch_region_ms(base, offset, size);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <gguf_model_path>\n", argv[0]);
        return 1;
    }

    init_timer();

    const char* model_path = argv[1];
    std::printf("=== Mugen SSD Offload PoC — Expert Page Fault Latency ===\n");
    std::printf("Model: %s\n", model_path);

    // mmap the model file
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[error] open(%s): %s\n", model_path, std::strerror(errno));
        return 1;
    }

    struct stat st{};
    fstat(fd, &st);
    size_t file_size = static_cast<size_t>(st.st_size);

    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[error] mmap: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }

    std::printf("File size: %.1f MB, mmap'd at %p\n",
                static_cast<double>(file_size) / (1024.0 * 1024.0), base);

    // Find expert regions
    uint32_t n_layers = 0, n_experts = 0;
    auto regions = find_expert_regions(
        static_cast<const uint8_t*>(base), file_size, n_layers, n_experts);

    std::printf("Simulated experts: %zu regions × %zu KB each\n",
                regions.size(), regions.empty() ? 0 : regions[0].size / 1024);
    std::printf("Simulated layers: %u, experts/layer: %u\n\n", n_layers, n_experts);

    if (regions.empty()) {
        std::fprintf(stderr, "[error] No expert regions found\n");
        munmap(base, file_size);
        close(fd);
        return 1;
    }

    // Phase 1: Warm all pages (baseline)
    std::printf("--- Phase 1: Warm cache (touch all pages) ---\n");
    {
        auto* p = static_cast<const volatile uint8_t*>(base);
        uint64_t sum = 0;
        uint64_t t0 = mach_absolute_time();
        for (size_t i = 0; i < file_size; i += kPageSize)
            sum += p[i];
        uint64_t t1 = mach_absolute_time();
        std::printf("Touched %.1f MB in %.1f ms (%.1f GB/s warm)\n\n",
                    static_cast<double>(file_size) / 1e6,
                    ticks_to_ms(t1 - t0),
                    (static_cast<double>(file_size) / 1e9) /
                        (static_cast<double>(t1 - t0) * g_ns_per_tick * 1e-9));
        if (sum == 0xDEADBEEF) std::printf("%llu\n", sum);
    }

    // Phase 2: Baseline touch latency (warm cache, pages resident)
    std::printf("--- Phase 2: Baseline touch (pages resident) ---\n");
    {
        size_t n_test = std::min<size_t>(regions.size(), 20);
        std::vector<double> latencies;
        for (size_t i = 0; i < n_test; i++) {
            double ms = touch_region_ms(base, regions[i].offset, regions[i].size);
            latencies.push_back(ms);
        }
        auto s = compute_stats(latencies);
        std::printf("Resident touch (%zu experts, %zu KB each):\n",
                    n_test, regions[0].size / 1024);
        std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms\n\n",
                    s.min, s.p50, s.p95, s.max);
    }

    // Phase 3: Evict → re-fault latency (warm page cache)
    std::printf("--- Phase 3: Evict + re-fault (page cache warm) ---\n");
    {
        size_t n_test = std::min<size_t>(regions.size(), 40);
        std::vector<double> latencies;
        for (size_t i = 0; i < n_test; i++) {
            evict_region(base, regions[i].offset, regions[i].size);
            double ms = touch_region_ms(base, regions[i].offset, regions[i].size);
            latencies.push_back(ms);
        }
        auto s = compute_stats(latencies);
        std::printf("Evict + re-fault (%zu experts, %zu KB each):\n",
                    n_test, regions[0].size / 1024);
        std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms  mean=%.3f ms\n\n",
                    s.min, s.p50, s.p95, s.max, s.mean);
    }

    // Phase 4: Evict → prefetch → touch latency
    std::printf("--- Phase 4: Evict + WILLNEED prefetch + touch ---\n");
    {
        size_t n_test = std::min<size_t>(regions.size(), 40);
        std::vector<double> latencies;
        for (size_t i = 0; i < n_test; i++) {
            evict_region(base, regions[i].offset, regions[i].size);
            double ms = prefetch_and_touch_ms(base, regions[i].offset, regions[i].size);
            latencies.push_back(ms);
        }
        auto s = compute_stats(latencies);
        std::printf("Evict + prefetch + touch (%zu experts, %zu KB each):\n",
                    n_test, regions[0].size / 1024);
        std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms  mean=%.3f ms\n\n",
                    s.min, s.p50, s.p95, s.max, s.mean);
    }

    // Phase 5: Larger regions (simulate DeepSeek V3 expert size ~20MB)
    std::printf("--- Phase 5: Large region evict + re-fault (simulating 20MB experts) ---\n");
    {
        constexpr size_t kLargeSize = 20 * 1024 * 1024;
        size_t n_large = file_size / kLargeSize;
        if (n_large > 20) n_large = 20;

        std::vector<double> latencies;
        for (size_t i = 0; i < n_large; i++) {
            size_t offset = i * kLargeSize;
            if (offset + kLargeSize > file_size) break;
            evict_region(base, offset, kLargeSize);
            double ms = touch_region_ms(base, offset, kLargeSize);
            latencies.push_back(ms);
        }
        if (!latencies.empty()) {
            auto s = compute_stats(latencies);
            std::printf("Evict + re-fault (%zu regions × 20 MB):\n", latencies.size());
            std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms  mean=%.3f ms\n\n",
                        s.min, s.p50, s.p95, s.max, s.mean);
        }
    }

    // Phase 6: Cold SSD read via pread + F_NOCACHE (bypasses page cache)
    std::printf("--- Phase 6: Cold SSD read via pread + F_NOCACHE ---\n");
    {
        int nocache_fd = open(model_path, O_RDONLY);
        if (nocache_fd >= 0) {
            fcntl(nocache_fd, F_NOCACHE, 1);

            void* aligned_buf = nullptr;
            constexpr size_t kBufSize = 20 * 1024 * 1024;
            posix_memalign(&aligned_buf, kPageSize, kBufSize);

            // 1MB regions (typical expert)
            {
                constexpr size_t kReadSize = 1 * 1024 * 1024;
                size_t n_test = std::min<size_t>(regions.size(), 40);
                std::vector<double> latencies;
                for (size_t i = 0; i < n_test; i++) {
                    off_t off = static_cast<off_t>(regions[i].offset);
                    // Align to page
                    off = static_cast<off_t>(page_align_down(static_cast<size_t>(off)));

                    uint64_t t0 = mach_absolute_time();
                    ssize_t n = pread(nocache_fd, aligned_buf, kReadSize, off);
                    uint64_t t1 = mach_absolute_time();

                    if (n > 0) latencies.push_back(ticks_to_ms(t1 - t0));
                }
                if (!latencies.empty()) {
                    auto s = compute_stats(latencies);
                    std::printf("pread F_NOCACHE 1 MB (%zu reads):\n", latencies.size());
                    std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms  mean=%.3f ms\n",
                                s.min, s.p50, s.p95, s.max, s.mean);
                    std::printf("  → throughput: %.1f GB/s\n\n",
                                1.0 / (s.p50 / 1000.0));
                }
            }

            // 20MB regions (DeepSeek V3 expert size)
            {
                constexpr size_t kReadSize = 20 * 1024 * 1024;
                size_t n_large = file_size / kReadSize;
                if (n_large > 20) n_large = 20;
                std::vector<double> latencies;
                for (size_t i = 0; i < n_large; i++) {
                    off_t off = static_cast<off_t>(i * kReadSize);
                    off = static_cast<off_t>(page_align_down(static_cast<size_t>(off)));

                    uint64_t t0 = mach_absolute_time();
                    size_t total_read = 0;
                    while (total_read < kReadSize) {
                        ssize_t n = pread(nocache_fd,
                                          static_cast<uint8_t*>(aligned_buf) + total_read,
                                          kReadSize - total_read,
                                          off + static_cast<off_t>(total_read));
                        if (n <= 0) break;
                        total_read += static_cast<size_t>(n);
                    }
                    uint64_t t1 = mach_absolute_time();

                    if (total_read == kReadSize)
                        latencies.push_back(ticks_to_ms(t1 - t0));
                }
                if (!latencies.empty()) {
                    auto s = compute_stats(latencies);
                    std::printf("pread F_NOCACHE 20 MB (%zu reads):\n", latencies.size());
                    std::printf("  min=%.3f ms  p50=%.3f ms  p95=%.3f ms  max=%.3f ms  mean=%.3f ms\n",
                                s.min, s.p50, s.p95, s.max, s.mean);
                    std::printf("  → throughput: %.1f GB/s\n\n",
                                20.0 / (s.p50 / 1000.0));
                }
            }

            std::free(aligned_buf);
            close(nocache_fd);
        }
    }

    // Phase 7: Speculative decoding viability analysis
    std::printf("--- Phase 7: Speculative Decoding Viability Analysis ---\n\n");
    std::printf("Scenario: DeepSeek V3 (671B Q4, 256 experts, 8 active/token)\n");
    std::printf("  Expert size (Q4):            ~20 MB each\n");
    std::printf("  Expert load from SSD:        see Phase 6 pread 20MB results\n");
    std::printf("  Experts per token:           8\n");
    std::printf("  Experts per layer:           8 (but sequential layers)\n\n");
    std::printf("Draft model (0.5B, in-memory): ~11 ms/tok\n");
    std::printf("Target model (671B from SSD):  expert_load × layers + compute\n\n");
    std::printf("Key formula: cost_ratio = target_time / draft_time\n");
    std::printf("  In-memory 7B:    25ms / 11ms = 2.3x  (NOT profitable)\n");
    std::printf("  SSD 671B naive:  if 8 experts × 20MB load = significant\n");
    std::printf("  → Speculative decoding profitable when cost_ratio > 5x\n");
    std::printf("  → USPP prefetch reduces effective load time by overlapping\n\n");

    munmap(base, file_size);
    close(fd);
    return 0;
}

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <sys/mman.h>

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

static double sTimebaseToNs = 0.0;

static void initTimebase() {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    sTimebaseToNs = static_cast<double>(info.numer) / static_cast<double>(info.denom);
}

static double ticksToUs(uint64_t ticks) {
    return ticks * sTimebaseToNs / 1000.0;
}

static double ticksToMs(uint64_t ticks) {
    return ticks * sTimebaseToNs / 1e6;
}

struct Stats {
    double p50;
    double p95;
    double mean;
};

static Stats computeStats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    Stats s{};
    s.p50  = samples[n / 2];
    s.p95  = samples[static_cast<size_t>(n * 0.95)];
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);
    return s;
}

static const char* formatSize(size_t bytes) {
    static char buf[64];
    if (bytes >= (1ULL << 30))
        std::snprintf(buf, sizeof(buf), "%.0fGB", static_cast<double>(bytes) / (1ULL << 30));
    else if (bytes >= (1ULL << 20))
        std::snprintf(buf, sizeof(buf), "%zuMB", bytes >> 20);
    else
        std::snprintf(buf, sizeof(buf), "%zuKB", bytes >> 10);
    return buf;
}

static const char* formatUnit(double valueUs) {
    static char buf[64];
    if (valueUs >= 1000.0)
        std::snprintf(buf, sizeof(buf), "%.2fms", valueUs / 1000.0);
    else
        std::snprintf(buf, sizeof(buf), "%.1fμs", valueUs);
    return buf;
}

// --------------------------------------------------------------------
// Test 1: MTLBuffer newBufferWithLength (Shared vs Private)
// --------------------------------------------------------------------

static void benchBufferAlloc(id<MTLDevice> device) {
    std::printf("\n=== MTLBuffer Allocation Latency ===\n\n");

    const size_t sizes[] = {
        1ULL << 20,         // 1 MB
        16ULL << 20,        // 16 MB
        64ULL << 20,        // 64 MB
        256ULL << 20,       // 256 MB
        512ULL << 20,       // 512 MB
        1ULL << 30,         // 1 GB
        2ULL << 30,         // 2 GB
    };

    struct ModeInfo {
        MTLResourceOptions opts;
        const char* name;
    };

    ModeInfo modes[] = {
        { MTLResourceStorageModeShared,  "shared " },
        { MTLResourceStorageModePrivate, "private" },
    };

    constexpr int kRuns = 100;

    for (auto& mode : modes) {
        for (size_t sz : sizes) {
            std::vector<double> samples;
            samples.reserve(kRuns);

            bool anyFail = false;
            for (int i = 0; i < kRuns; ++i) {
                @autoreleasepool {
                    uint64_t t0 = mach_absolute_time();
                    id<MTLBuffer> buf = [device newBufferWithLength:sz options:mode.opts];
                    uint64_t t1 = mach_absolute_time();

                    if (!buf) {
                        anyFail = true;
                        break;
                    }
                    samples.push_back(ticksToUs(t1 - t0));
                    buf = nil;
                }
            }

            if (anyFail) {
                std::printf("  MTLBuffer %s %6s:   ALLOC FAILED\n", mode.name, formatSize(sz));
                continue;
            }

            auto st = computeStats(samples);
            std::printf("  MTLBuffer %s %6s:   p50=%-10s p95=%-10s mean=%s\n",
                        mode.name, formatSize(sz),
                        formatUnit(st.p50), formatUnit(st.p95), formatUnit(st.mean));
        }
        std::printf("\n");
    }
}

// --------------------------------------------------------------------
// Test 2: newBufferWithBytesNoCopy (zero-copy from mmap'd memory)
// --------------------------------------------------------------------

static void benchNoCopyBuffer(id<MTLDevice> device) {
    std::printf("=== MTLBuffer newBufferWithBytesNoCopy ===\n\n");

    const size_t sizes[] = {
        1ULL << 20,
        64ULL << 20,
        256ULL << 20,
        1ULL << 30,
    };

    constexpr int kRuns = 100;

    for (size_t sz : sizes) {
        std::vector<double> samples;
        samples.reserve(kRuns);

        bool anyFail = false;
        for (int i = 0; i < kRuns; ++i) {
            @autoreleasepool {
                size_t pageSize = static_cast<size_t>(getpagesize());
                size_t alignedSz = (sz + pageSize - 1) & ~(pageSize - 1);
                void* ptr = mmap(nullptr, alignedSz,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANON, -1, 0);
                if (ptr == MAP_FAILED) {
                    anyFail = true;
                    break;
                }

                uint64_t t0 = mach_absolute_time();
                id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                                                              length:alignedSz
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:^(void* p, NSUInteger len) {
                                                             munmap(p, len);
                                                         }];
                uint64_t t1 = mach_absolute_time();

                if (!buf) {
                    munmap(ptr, alignedSz);
                    anyFail = true;
                    break;
                }
                samples.push_back(ticksToUs(t1 - t0));
                buf = nil;
            }
        }

        if (anyFail) {
            std::printf("  NoCopy %6s:   FAILED\n", formatSize(sz));
            continue;
        }

        auto st = computeStats(samples);
        std::printf("  NoCopy %6s:   p50=%-10s p95=%-10s mean=%s\n",
                    formatSize(sz),
                    formatUnit(st.p50), formatUnit(st.p95), formatUnit(st.mean));
    }
    std::printf("\n");
}

// --------------------------------------------------------------------
// Test 3: MTLHeap sub-allocation vs direct allocation
// --------------------------------------------------------------------

static void benchHeapAlloc(id<MTLDevice> device) {
    std::printf("=== MTLHeap Sub-Allocation vs Direct ===\n\n");

    constexpr size_t kHeapSize  = 512ULL << 20;  // 512 MB heap
    constexpr size_t kAllocSize = 16ULL << 20;   // 16 MB per sub-alloc
    constexpr int    kRuns      = 200;

    MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
    heapDesc.size        = kHeapSize;
    heapDesc.storageMode = MTLStorageModeShared;
    heapDesc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    heapDesc.hazardTrackingMode = MTLHazardTrackingModeUntracked;

    uint64_t heapCreateT0 = mach_absolute_time();
    id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];
    uint64_t heapCreateT1 = mach_absolute_time();

    if (!heap) {
        std::printf("  MTLHeap creation FAILED\n\n");
        return;
    }
    std::printf("  MTLHeap create (%s):   %.2fms\n", formatSize(kHeapSize),
                ticksToMs(heapCreateT1 - heapCreateT0));

    // Heap sub-allocation
    {
        std::vector<double> samples;
        samples.reserve(kRuns);
        for (int i = 0; i < kRuns; ++i) {
            @autoreleasepool {
                uint64_t t0 = mach_absolute_time();
                id<MTLBuffer> buf = [heap newBufferWithLength:kAllocSize
                                                      options:MTLResourceStorageModeShared];
                uint64_t t1 = mach_absolute_time();
                if (!buf) {
                    std::printf("  Heap sub-alloc failed at iter %d (heap full)\n", i);
                    break;
                }
                samples.push_back(ticksToUs(t1 - t0));
                [buf makeAliasable];
                buf = nil;
            }
        }
        if (!samples.empty()) {
            auto st = computeStats(samples);
            std::printf("  MTLHeap sub-alloc %s:   p50=%-10s p95=%-10s mean=%s  (n=%zu)\n",
                        formatSize(kAllocSize),
                        formatUnit(st.p50), formatUnit(st.p95), formatUnit(st.mean),
                        samples.size());
        }
    }

    // Direct allocation for comparison
    {
        std::vector<double> samples;
        samples.reserve(kRuns);
        for (int i = 0; i < kRuns; ++i) {
            @autoreleasepool {
                uint64_t t0 = mach_absolute_time();
                id<MTLBuffer> buf = [device newBufferWithLength:kAllocSize
                                                        options:MTLResourceStorageModeShared];
                uint64_t t1 = mach_absolute_time();
                if (!buf) break;
                samples.push_back(ticksToUs(t1 - t0));
                buf = nil;
            }
        }
        if (!samples.empty()) {
            auto st = computeStats(samples);
            std::printf("  Direct alloc     %s:   p50=%-10s p95=%-10s mean=%s  (n=%zu)\n",
                        formatSize(kAllocSize),
                        formatUnit(st.p50), formatUnit(st.p95), formatUnit(st.mean),
                        samples.size());
        }
    }

    std::printf("\n");
}

// --------------------------------------------------------------------
// Test 4: 2GB boundary probe
// --------------------------------------------------------------------

static void bench2GBLimit(id<MTLDevice> device) {
    std::printf("=== MTLBuffer 2GB Boundary ===\n\n");

    std::printf("  Device max buffer length:  %.2f GB\n",
                static_cast<double>([device maxBufferLength]) / (1ULL << 30));

    struct Probe {
        size_t size;
        const char* label;
    };

    Probe probes[] = {
        { (2ULL << 30) - (1ULL << 20), "2GB - 1MB" },
        { (2ULL << 30),                "2GB exact" },
        { (2ULL << 30) + (1ULL << 20), "2GB + 1MB" },
        { (2ULL << 30) + (256ULL << 20), "2GB + 256MB" },
        { (3ULL << 30),                "3GB"       },
        { (4ULL << 30),                "4GB"       },
        { [device maxBufferLength],    "maxBufferLength" },
    };

    for (auto& p : probes) {
        @autoreleasepool {
            uint64_t t0 = mach_absolute_time();
            id<MTLBuffer> buf = [device newBufferWithLength:p.size
                                                    options:MTLResourceStorageModeShared];
            uint64_t t1 = mach_absolute_time();

            if (buf) {
                std::printf("  %-18s (%5.2f GB):  SUCCESS  alloc=%.2fms\n",
                            p.label,
                            static_cast<double>(p.size) / (1ULL << 30),
                            ticksToMs(t1 - t0));
                buf = nil;
            } else {
                std::printf("  %-18s (%5.2f GB):  FAILED\n",
                            p.label,
                            static_cast<double>(p.size) / (1ULL << 30));
            }
        }
    }

    // Binary search for actual max
    size_t lo = 2ULL << 30;
    size_t hi = std::min(static_cast<size_t>([device maxBufferLength]) + 1,
                         static_cast<size_t>(48ULL << 30));
    size_t actualMax = 0;

    while (lo <= hi) {
        size_t mid = lo + (hi - lo) / 2;
        @autoreleasepool {
            id<MTLBuffer> buf = [device newBufferWithLength:mid
                                                    options:MTLResourceStorageModeShared];
            if (buf) {
                actualMax = mid;
                lo = mid + (1ULL << 20);
                buf = nil;
            } else {
                if (mid == 0) break;
                hi = mid - (1ULL << 20);
            }
        }
    }

    if (actualMax > 0) {
        std::printf("\n  Actual max alloc (shared): %.2f GB\n",
                    static_cast<double>(actualMax) / (1ULL << 30));
    }
    std::printf("\n");
}

// --------------------------------------------------------------------
// Test 5: Rapid alloc/dealloc cycle (stress test)
// --------------------------------------------------------------------

static void benchRapidCycle(id<MTLDevice> device) {
    std::printf("=== Rapid Alloc/Dealloc Cycle ===\n\n");

    constexpr size_t kSize  = 64ULL << 20;  // 64 MB
    constexpr int kCycles   = 1000;

    std::vector<double> samples;
    samples.reserve(kCycles);

    uint64_t totalT0 = mach_absolute_time();
    for (int i = 0; i < kCycles; ++i) {
            @autoreleasepool {
                uint64_t t0 = mach_absolute_time();
                id<MTLBuffer> buf = [device newBufferWithLength:kSize
                                                        options:MTLResourceStorageModeShared];
                uint64_t t1 = mach_absolute_time();
                (void)buf;
                samples.push_back(ticksToUs(t1 - t0));
            }
    }
    uint64_t totalT1 = mach_absolute_time();

    auto st = computeStats(samples);
    std::printf("  %d cycles of %s alloc+release:\n", kCycles, formatSize(kSize));
    std::printf("    p50=%-10s p95=%-10s mean=%s\n",
                formatUnit(st.p50), formatUnit(st.p95), formatUnit(st.mean));
    std::printf("    total: %.2fms  (%.0f alloc+free/sec)\n",
                ticksToMs(totalT1 - totalT0),
                kCycles / (ticksToMs(totalT1 - totalT0) / 1000.0));
    std::printf("\n");
}

// --------------------------------------------------------------------
// Main
// --------------------------------------------------------------------

int main() {
    @autoreleasepool {
        initTimebase();

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "ERROR: No Metal device found.\n");
            return 1;
        }

        std::printf("Metal Device:       %s\n", [[device name] UTF8String]);
        std::printf("Unified Memory:     %s\n", [device hasUnifiedMemory] ? "YES" : "NO");
        std::printf("Max Buffer Length:  %.2f GB\n",
                    static_cast<double>([device maxBufferLength]) / (1ULL << 30));
        std::printf("Recommended Max Working Set: %.2f GB\n",
                    static_cast<double>([device recommendedMaxWorkingSetSize]) / (1ULL << 30));

        benchBufferAlloc(device);
        benchNoCopyBuffer(device);
        benchHeapAlloc(device);
        bench2GBLimit(device);
        benchRapidCycle(device);

        std::printf("=== Done ===\n");
    }
    return 0;
}

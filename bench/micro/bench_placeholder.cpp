#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Measures sequential read throughput on a heap-allocated buffer to establish
// a baseline memory bandwidth number for comparison against mmap and Metal.

static constexpr size_t kBufferSize = 256 * 1024 * 1024;  // 256 MiB
static constexpr int    kIterations = 4;

int main() {
    auto* buf = static_cast<char*>(std::malloc(kBufferSize));
    if (!buf) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }
    std::memset(buf, 0xAB, kBufferSize);

    volatile char sink = 0;

    for (int i = 0; i < kIterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();

        for (size_t j = 0; j < kBufferSize; j += 64) {
            sink = buf[j];
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbps = (static_cast<double>(kBufferSize) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

        std::printf("iter %d: %.2f ms  (%.2f GB/s sequential scan)\n", i, ms, gbps);
    }

    (void)sink;
    std::free(buf);
    return 0;
}

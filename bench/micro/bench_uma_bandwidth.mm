#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

// ====================================================================
// Metal shader source (compiled at runtime via newLibraryWithSource)
// ====================================================================

static NSString* const kShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Pure memory-read bandwidth kernel.
// Each thread reads a float4 (16 bytes). The dummy branch prevents
// the compiler from optimizing the read away.
kernel void bandwidth_read(
    device const float4* input  [[buffer(0)]],
    device atomic_uint*  output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float4 v = input[tid];
    float s = v.x + v.y + v.z + v.w;
    if (s == -999999.0f)
        atomic_fetch_add_explicit(output, 1u, memory_order_relaxed);
}

// Write bandwidth kernel: each thread writes 16 bytes.
kernel void bandwidth_write(
    device float4* output [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = float4(float(tid));
}

// Copy bandwidth kernel: read + write.
kernel void bandwidth_copy(
    device const float4* input  [[buffer(0)]],
    device float4*       output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = input[tid];
}

// FP16 matrix multiply: 128x128 naive (for TFLOPS measurement).
// Each thread computes one element of C = A * B.
kernel void matmul_fp16(
    device const half*  A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device half*        C [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= N) return;

    half acc = 0.0h;
    for (uint k = 0; k < N; ++k) {
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
)";

// ====================================================================
// Helpers
// ====================================================================

static double sTimebaseToNs = 0.0;

static void initTimebase() {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    sTimebaseToNs = static_cast<double>(info.numer) / static_cast<double>(info.denom);
}

static double ticksToMs(uint64_t ticks) {
    return ticks * sTimebaseToNs / 1e6;
}

// ====================================================================
// Compile shaders at runtime
// ====================================================================

static id<MTLLibrary> compileShaders(id<MTLDevice> device) {
    NSError* error = nil;

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;

    id<MTLLibrary> lib = [device newLibraryWithSource:kShaderSource
                                              options:opts
                                                error:&error];
    if (!lib) {
        std::fprintf(stderr, "ERROR: Failed to compile Metal shaders:\n%s\n",
                     [[error localizedDescription] UTF8String]);
        return nil;
    }
    std::printf("  Runtime shader compilation: OK\n\n");
    return lib;
}

// ====================================================================
// Bandwidth test
// ====================================================================

static void benchBandwidth(id<MTLDevice> device, id<MTLLibrary> lib,
                           const char* kernelName, const char* label,
                           bool isRead) {
    NSError* error = nil;

    id<MTLFunction> func = [lib newFunctionWithName:
                            [NSString stringWithUTF8String:kernelName]];
    if (!func) {
        std::fprintf(stderr, "  ERROR: kernel '%s' not found\n", kernelName);
        return;
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                           error:&error];
    if (!pso) {
        std::fprintf(stderr, "  ERROR: PSO creation failed: %s\n",
                     [[error localizedDescription] UTF8String]);
        return;
    }

    const size_t kBufferSize = 1ULL << 30;  // 1 GB
    const size_t kNumFloat4  = kBufferSize / sizeof(float) / 4;

    id<MTLBuffer> inputBuf = [device newBufferWithLength:kBufferSize
                                                 options:MTLResourceStorageModeShared];
    if (!inputBuf) {
        std::fprintf(stderr, "  ERROR: Failed to allocate 1GB buffer\n");
        return;
    }

    // Fill with nonzero data
    auto* ptr = static_cast<float*>([inputBuf contents]);
    for (size_t i = 0; i < kBufferSize / sizeof(float); i += 1024) {
        ptr[i] = 1.0f;
    }

    id<MTLBuffer> outputBuf = nil;
    if (isRead) {
        outputBuf = [device newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
    } else {
        outputBuf = [device newBufferWithLength:kBufferSize
                                        options:MTLResourceStorageModeShared];
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    constexpr int kWarmup = 3;
    constexpr int kRuns   = 10;

    std::printf("  --- %s (1GB, float4) ---\n", label);

    // Threadgroup size sweep
    NSUInteger tgSizes[] = { 64, 128, 256, 512, 1024 };

    for (NSUInteger tg : tgSizes) {
        if (tg > pso.maxTotalThreadsPerThreadgroup) continue;

        std::vector<double> bandwidths;

        for (int i = 0; i < kWarmup + kRuns; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

                [enc setComputePipelineState:pso];
                [enc setBuffer:inputBuf offset:0 atIndex:0];
                if (outputBuf)
                    [enc setBuffer:outputBuf offset:0 atIndex:1];

                MTLSize grid  = MTLSizeMake(kNumFloat4, 1, 1);
                MTLSize group = MTLSizeMake(tg, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:group];
                [enc endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                if (i >= kWarmup) {
                    double gpuTime = [cmdBuf GPUEndTime] - [cmdBuf GPUStartTime];
                    if (gpuTime > 0) {
                        size_t bytes = kBufferSize;
                        // Copy kernel reads + writes
                        if (strcmp(kernelName, "bandwidth_copy") == 0)
                            bytes *= 2;
                        double gbps = (static_cast<double>(bytes) / (1ULL << 30)) / gpuTime;
                        bandwidths.push_back(gbps);
                    }
                }
            }
        }

        if (!bandwidths.empty()) {
            std::sort(bandwidths.begin(), bandwidths.end());
            double best = bandwidths.back();
            double median = bandwidths[bandwidths.size() / 2];
            std::printf("    tg=%4llu:  median=%.1f GB/s  best=%.1f GB/s\n",
                        static_cast<unsigned long long>(tg), median, best);
        }
    }
    std::printf("\n");
}

// ====================================================================
// FP16 matmul TFLOPS test
// ====================================================================

static void benchMatmul(id<MTLDevice> device, id<MTLLibrary> lib) {
    std::printf("  --- FP16 Matrix Multiply ---\n");

    NSError* error = nil;
    id<MTLFunction> func = [lib newFunctionWithName:@"matmul_fp16"];
    if (!func) {
        std::fprintf(stderr, "  ERROR: matmul_fp16 not found\n");
        return;
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                           error:&error];
    if (!pso) {
        std::fprintf(stderr, "  ERROR: PSO failed: %s\n",
                     [[error localizedDescription] UTF8String]);
        return;
    }

    const uint32_t sizes[] = { 128, 256, 512, 1024, 2048, 4096 };

    id<MTLCommandQueue> queue = [device newCommandQueue];

    for (uint32_t N : sizes) {
        size_t matBytes = static_cast<size_t>(N) * N * sizeof(uint16_t);

        id<MTLBuffer> bufA = [device newBufferWithLength:matBytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:matBytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:matBytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [device newBufferWithLength:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];

        if (!bufA || !bufB || !bufC || !bufN) {
            std::fprintf(stderr, "  N=%u: buffer alloc failed\n", N);
            continue;
        }

        // Fill A, B with small values
        auto* pA = static_cast<uint16_t*>([bufA contents]);
        auto* pB = static_cast<uint16_t*>([bufB contents]);
        // half(1.0) = 0x3C00
        for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
            pA[i] = 0x3C00;
            pB[i] = 0x3C00;
        }
        *static_cast<uint32_t*>([bufN contents]) = N;

        constexpr int kWarmup = 2;
        constexpr int kRuns   = 5;
        std::vector<double> tflops_samples;

        NSUInteger tg = std::min(static_cast<NSUInteger>(16),
                                 static_cast<NSUInteger>(std::sqrt(pso.maxTotalThreadsPerThreadgroup)));

        for (int i = 0; i < kWarmup + kRuns; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

                [enc setComputePipelineState:pso];
                [enc setBuffer:bufA offset:0 atIndex:0];
                [enc setBuffer:bufB offset:0 atIndex:1];
                [enc setBuffer:bufC offset:0 atIndex:2];
                [enc setBuffer:bufN offset:0 atIndex:3];

                MTLSize grid  = MTLSizeMake(N, N, 1);
                MTLSize group = MTLSizeMake(tg, tg, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:group];
                [enc endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                if (i >= kWarmup) {
                    double gpuTime = [cmdBuf GPUEndTime] - [cmdBuf GPUStartTime];
                    if (gpuTime > 0) {
                        // FLOPs for NxN matmul: 2*N^3 (multiply + add)
                        double flops = 2.0 * static_cast<double>(N) * N * N;
                        double tf = (flops / 1e12) / gpuTime;
                        tflops_samples.push_back(tf);
                    }
                }
            }
        }

        if (!tflops_samples.empty()) {
            std::sort(tflops_samples.begin(), tflops_samples.end());
            double best   = tflops_samples.back();
            double median  = tflops_samples[tflops_samples.size() / 2];
            std::printf("    N=%4u:  median=%.3f TFLOPS  best=%.3f TFLOPS\n",
                        N, median, best);
        }
    }
    std::printf("\n");
}

// ====================================================================
// CPU memory bandwidth baseline (for comparison)
// ====================================================================

static void benchCPUBandwidth() {
    std::printf("  --- CPU Memory Bandwidth Baseline ---\n");

    constexpr size_t kSize = 1ULL << 30;  // 1 GB
    auto* buf = static_cast<float*>(std::malloc(kSize));
    if (!buf) {
        std::fprintf(stderr, "  malloc failed\n");
        return;
    }
    std::memset(buf, 0x01, kSize);

    constexpr int kRuns = 5;
    std::vector<double> bandwidths;

    for (int i = 0; i < kRuns; ++i) {
        volatile float sink = 0;
        uint64_t t0 = mach_absolute_time();

        const size_t stride = 4;
        for (size_t j = 0; j < kSize / sizeof(float); j += stride) {
            sink = buf[j];
        }

        uint64_t t1 = mach_absolute_time();
        (void)sink;

        double ms = ticksToMs(t1 - t0);
        double gbps = (static_cast<double>(kSize) / (1ULL << 30)) / (ms / 1000.0);
        bandwidths.push_back(gbps);
    }

    std::sort(bandwidths.begin(), bandwidths.end());
    std::printf("    CPU sequential read (1GB): median=%.1f GB/s  best=%.1f GB/s\n",
                bandwidths[bandwidths.size() / 2], bandwidths.back());

    std::free(buf);
    std::printf("\n");
}

// ====================================================================
// Main
// ====================================================================

int main() {
    @autoreleasepool {
        initTimebase();

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "ERROR: No Metal device found.\n");
            return 1;
        }

        std::printf("=== UMA Bandwidth Benchmark ===\n\n");
        std::printf("Metal Device:       %s\n", [[device name] UTF8String]);
        std::printf("Unified Memory:     %s\n", [device hasUnifiedMemory] ? "YES" : "NO");
        std::printf("Max Buffer Length:  %.2f GB\n",
                    static_cast<double>([device maxBufferLength]) / (1ULL << 30));
        std::printf("Theoretical BW:     273 GB/s (M4 Pro spec)\n\n");

        id<MTLLibrary> lib = compileShaders(device);
        if (!lib) return 1;

        benchBandwidth(device, lib, "bandwidth_read",  "GPU Read Bandwidth",  true);
        benchBandwidth(device, lib, "bandwidth_write", "GPU Write Bandwidth", false);
        benchBandwidth(device, lib, "bandwidth_copy",  "GPU Copy Bandwidth",  false);
        benchMatmul(device, lib);
        benchCPUBandwidth();

        std::printf("=== Done ===\n");
    }
    return 0;
}

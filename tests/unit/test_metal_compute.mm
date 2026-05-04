#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "core/compute/metal_compute.h"
#include "metal/kernel_sources.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define CHECK_OK(expr)                                                      \
    do {                                                                    \
        auto _r = (expr);                                                   \
        if (!_r.has_value()) {                                              \
            std::fprintf(stderr, "FAIL: %s => %s  (%s:%d)\n",              \
                         #expr, _r.error().c_str(), __FILE__, __LINE__);    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ─── Half-precision conversion helpers (host side, using _Float16) ───

static uint16_t f32_to_f16_bits(float v) {
    _Float16 h = static_cast<_Float16>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}

static float f16_bits_to_f32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<float>(h);
}

// ─── Q4_0 encoding helper (host side) ───

struct BlockQ4_0 {
    uint16_t scale_f16;
    uint8_t  data[16];
};
static_assert(sizeof(BlockQ4_0) == 18);

static BlockQ4_0 encode_q4_0(const float values[32]) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = std::max(amax, std::abs(values[i]));
    }

    float scale = amax / 7.0f;
    float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    BlockQ4_0 block{};
    block.scale_f16 = f32_to_f16_bits(scale);

    for (int j = 0; j < 16; ++j) {
        int lo = std::clamp(static_cast<int>(std::round(values[j]      * inv_scale)) + 8, 0, 15);
        int hi = std::clamp(static_cast<int>(std::round(values[j + 16] * inv_scale)) + 8, 0, 15);
        block.data[j] = static_cast<uint8_t>(lo | (hi << 4));
    }

    return block;
}

static void dequant_q4_0_ref(const BlockQ4_0& block, float out[32]) {
    float scale = f16_bits_to_f32(block.scale_f16);
    for (int j = 0; j < 16; ++j) {
        int lo = (block.data[j] & 0x0F) - 8;
        int hi = (block.data[j] >> 4)   - 8;
        out[j]      = scale * static_cast<float>(lo);
        out[j + 16] = scale * static_cast<float>(hi);
    }
}

// ─── Q8_0 encoding helper (host side) ───

struct BlockQ8_0 {
    uint16_t scale_f16;
    int8_t   quants[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

static BlockQ8_0 encode_q8_0(const float values[32]) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = std::max(amax, std::abs(values[i]));
    }

    float scale = amax / 127.0f;
    float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    BlockQ8_0 block{};
    block.scale_f16 = f32_to_f16_bits(scale);

    for (int i = 0; i < 32; ++i) {
        block.quants[i] = static_cast<int8_t>(
            std::clamp(static_cast<int>(std::round(values[i] * inv_scale)), -128, 127));
    }

    return block;
}

static void dequant_q8_0_ref(const BlockQ8_0& block, float out[32]) {
    float scale = f16_bits_to_f32(block.scale_f16);
    for (int i = 0; i < 32; ++i) {
        out[i] = scale * static_cast<float>(block.quants[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

static mugen::MetalCompute* g_mc = nullptr;
static void* g_lib = nullptr;

static void setup() {
    std::printf("--- Setup ---\n");
    auto result = mugen::MetalCompute::create();
    CHECK(result.has_value());
    static auto mc = std::move(result.value());
    g_mc = mc.get();
    std::printf("  Device: %s\n", g_mc->device_name().c_str());
    std::printf("  Max buffer: %.2f GB\n",
                static_cast<double>(g_mc->max_buffer_length()) / (1ULL << 30));

    auto lib_result = g_mc->compile_library(mugen::metal::kAllKernelsSource, "mugen_kernels");
    CHECK(lib_result.has_value());
    g_lib = lib_result.value();
    std::printf("  Shader compilation: OK\n\n");
}

// ─── Test: MetalCompute creation ─────────────────────────────────────────

static void test_create() {
    std::printf("[test_create] ");
    auto mc = mugen::MetalCompute::create();
    CHECK(mc.has_value());
    CHECK(!mc.value()->device_name().empty());
    CHECK(mc.value()->max_buffer_length() > 0);
    std::printf("PASS\n");
}

// ─── Test: shader compilation (valid + invalid) ─────────────────────────

static void test_compile() {
    std::printf("[test_compile] ");

    auto good = g_mc->compile_library(R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(uint tid [[thread_position_in_grid]]) {}
    )", "good");
    CHECK(good.has_value());

    auto bad = g_mc->compile_library("this is not valid metal", "bad");
    CHECK(!bad.has_value());

    std::printf("PASS\n");
}

// ─── Test: buffer creation ──────────────────────────────────────────────

static void test_buffers() {
    std::printf("[test_buffers] ");

    void* buf = g_mc->create_buffer(1024);
    CHECK(buf != nullptr);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    void* buf2 = g_mc->create_buffer_from_data(data, sizeof(data));
    CHECK(buf2 != nullptr);

    // Verify data was copied by reading back
    @autoreleasepool {
        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf2;
        auto* ptr = static_cast<float*>([mtl_buf contents]);
        for (int i = 0; i < 4; ++i) {
            CHECK(ptr[i] == data[i]);
        }
    }

    // Clean up
    @autoreleasepool {
        CFRelease(buf);
        CFRelease(buf2);
    }

    std::printf("PASS\n");
}

// ─── Helper: build pipeline from function name ──────────────────────────

static void* make_pipeline(const char* name) {
    auto func = g_mc->get_function(g_lib, name);
    CHECK(func.has_value());
    auto pso = g_mc->create_pipeline(func.value());
    CHECK(pso.has_value());
    // Release the function, we only need the pipeline
    @autoreleasepool { CFRelease(func.value()); }
    return pso.value();
}

// ─── Test: dequantize_q4_0 ──────────────────────────────────────────────

static void test_dequantize_q4_0() {
    std::printf("[test_dequantize_q4_0] ");

    constexpr uint32_t N_BLOCKS = 64;
    constexpr uint32_t N_ELEMS  = N_BLOCKS * 32;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> original(N_ELEMS);
    for (auto& v : original) v = dist(rng);

    // Encode
    std::vector<BlockQ4_0> blocks(N_BLOCKS);
    for (uint32_t b = 0; b < N_BLOCKS; ++b) {
        blocks[b] = encode_q4_0(&original[b * 32]);
    }

    // CPU reference dequant
    std::vector<float> ref(N_ELEMS);
    for (uint32_t b = 0; b < N_BLOCKS; ++b) {
        dequant_q4_0_ref(blocks[b], &ref[b * 32]);
    }

    // GPU dequant
    void* pso = make_pipeline("dequantize_q4_0");
    void* buf_in  = g_mc->create_buffer_from_data(blocks.data(), N_BLOCKS * sizeof(BlockQ4_0));
    void* buf_out = g_mc->create_buffer(N_ELEMS * sizeof(uint16_t));  // half output
    CHECK(buf_in && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&N_BLOCKS, sizeof(uint32_t)}};
    dp.grid_size  = {N_BLOCKS, 1, 1};
    dp.group_size = {std::min(N_BLOCKS, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    // Verify
    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N_ELEMS; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float rel = (std::abs(ref[i]) > 1e-6f) ? (err / std::abs(ref[i])) : err;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.01f);
    }

    @autoreleasepool {
        CFRelease(pso);
        CFRelease(buf_in);
        CFRelease(buf_out);
    }

    std::printf("PASS\n");
}

// ─── Test: matvec_f16 ───────────────────────────────────────────────────

static void test_matvec_f16() {
    std::printf("[test_matvec_f16] ");

    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate matrix and vector (in half precision)
    std::vector<uint16_t> mat_h(M * K);
    std::vector<uint16_t> vec_h(K);

    for (auto& v : mat_h) v = f32_to_f16_bits(dist(rng));
    for (auto& v : vec_h) v = f32_to_f16_bits(dist(rng));

    // CPU reference (in float for accuracy)
    std::vector<float> ref(M, 0.0f);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < K; ++col) {
            ref[row] += f16_bits_to_f32(mat_h[row * K + col]) * f16_bits_to_f32(vec_h[col]);
        }
    }

    void* pso = make_pipeline("matvec_f16");
    void* buf_mat = g_mc->create_buffer_from_data(mat_h.data(), mat_h.size() * 2);
    void* buf_vec = g_mc->create_buffer_from_data(vec_h.data(), vec_h.size() * 2);
    void* buf_out = g_mc->create_buffer(M * 2);
    CHECK(buf_mat && buf_vec && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_mat, buf_vec, buf_out};
    dp.constants = {{&M, sizeof(uint32_t)}, {&K, sizeof(uint32_t)}};
    // One threadgroup (32 threads = 1 simdgroup) per row
    dp.grid_size  = {M * 32, 1, 1};
    dp.group_size = {32, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < M; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            float rel = err / denom;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.02f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_mat); CFRelease(buf_vec); CFRelease(buf_out);
    }

    std::printf("PASS\n");
}

// ─── Test: matvec_q4_0 (fused) ─────────────────────────────────────────

static void test_matvec_q4_0() {
    std::printf("[test_matvec_q4_0] ");

    constexpr uint32_t M = 64;
    constexpr uint32_t K = 256;  // Must be multiple of 32
    constexpr uint32_t BLOCKS_PER_ROW = K / 32;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Original float matrix + vector
    std::vector<float> mat_f(M * K);
    std::vector<float> vec_f(K);
    for (auto& v : mat_f) v = dist(rng);
    for (auto& v : vec_f) v = dist(rng);

    // Quantize matrix to Q4_0
    std::vector<BlockQ4_0> mat_q(M * BLOCKS_PER_ROW);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            mat_q[row * BLOCKS_PER_ROW + b] = encode_q4_0(&mat_f[row * K + b * 32]);
        }
    }

    // CPU reference: dequant then matvec
    std::vector<float> ref(M, 0.0f);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            float deq[32];
            dequant_q4_0_ref(mat_q[row * BLOCKS_PER_ROW + b], deq);
            for (int j = 0; j < 32; ++j) {
                ref[row] += deq[j] * vec_f[b * 32 + j];
            }
        }
    }

    // Prepare half-precision vector for GPU
    std::vector<uint16_t> vec_h(K);
    for (uint32_t i = 0; i < K; ++i) {
        vec_h[i] = f32_to_f16_bits(vec_f[i]);
    }

    void* pso = make_pipeline("matvec_q4_0");
    void* buf_mat = g_mc->create_buffer_from_data(mat_q.data(), mat_q.size() * sizeof(BlockQ4_0));
    void* buf_vec = g_mc->create_buffer_from_data(vec_h.data(), vec_h.size() * 2);
    void* buf_out = g_mc->create_buffer(M * 2);
    CHECK(buf_mat && buf_vec && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_mat, buf_vec, buf_out};
    dp.constants = {{&M, sizeof(uint32_t)}, {&K, sizeof(uint32_t)}};
    // One threadgroup per row, 256 threads per threadgroup
    dp.grid_size  = {M * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < M; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            float rel = err / denom;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.05f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_mat); CFRelease(buf_vec); CFRelease(buf_out);
    }

    std::printf("PASS\n");
}

// ─── Test: softmax ──────────────────────────────────────────────────────

static void test_softmax() {
    std::printf("[test_softmax] ");

    constexpr uint32_t N_ROWS = 4;
    constexpr uint32_t N      = 512;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<uint16_t> input_h(N_ROWS * N);
    std::vector<float> input_f(N_ROWS * N);
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_f[i] = dist(rng);
        input_h[i] = f32_to_f16_bits(input_f[i]);
    }

    // CPU reference softmax per row
    std::vector<float> ref(N_ROWS * N);
    for (uint32_t row = 0; row < N_ROWS; ++row) {
        float max_val = -INFINITY;
        for (uint32_t i = 0; i < N; ++i) {
            max_val = std::max(max_val, input_f[row * N + i]);
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            ref[row * N + i] = std::exp(input_f[row * N + i] - max_val);
            sum += ref[row * N + i];
        }
        for (uint32_t i = 0; i < N; ++i) {
            ref[row * N + i] /= sum;
        }
    }

    void* pso = make_pipeline("softmax");
    void* buf_in  = g_mc->create_buffer_from_data(input_h.data(), input_h.size() * 2);
    void* buf_out = g_mc->create_buffer(N_ROWS * N * 2);
    CHECK(buf_in && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&N, sizeof(uint32_t)}};
    dp.grid_size  = {N_ROWS * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);

        float max_err = 0.0f;
        for (uint32_t row = 0; row < N_ROWS; ++row) {
            float gpu_sum = 0.0f;
            for (uint32_t i = 0; i < N; ++i) {
                float gpu_val = f16_bits_to_f32(out_ptr[row * N + i]);
                gpu_sum += gpu_val;

                float err = std::abs(gpu_val - ref[row * N + i]);
                float denom = std::max(ref[row * N + i], 1e-5f);
                max_err = std::max(max_err, err / denom);
            }
            // Sum should be ~1.0
            CHECK(std::abs(gpu_sum - 1.0f) < 0.02f);
        }
        std::printf("max_rel_err=%.4f%% sum_ok gpu_time=%.3fms ",
                    max_err * 100.0f, result.value() * 1000.0);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_out);
    }

    std::printf("PASS\n");
}

// ─── Test: rms_norm ─────────────────────────────────────────────────────

static void test_rms_norm() {
    std::printf("[test_rms_norm] ");

    constexpr uint32_t N_ROWS = 4;
    constexpr uint32_t N      = 256;
    constexpr float    EPS    = 1e-5f;

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<uint16_t> input_h(N_ROWS * N);
    std::vector<uint16_t> weight_h(N);
    std::vector<float> input_f(N_ROWS * N);
    std::vector<float> weight_f(N);

    for (size_t i = 0; i < input_f.size(); ++i) {
        input_f[i] = dist(rng);
        input_h[i] = f32_to_f16_bits(input_f[i]);
    }
    for (uint32_t i = 0; i < N; ++i) {
        weight_f[i] = dist(rng);
        weight_h[i] = f32_to_f16_bits(weight_f[i]);
    }

    // CPU reference
    std::vector<float> ref(N_ROWS * N);
    for (uint32_t row = 0; row < N_ROWS; ++row) {
        float ss = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            ss += input_f[row * N + i] * input_f[row * N + i];
        }
        float rms_inv = 1.0f / std::sqrt(ss / N + EPS);
        for (uint32_t i = 0; i < N; ++i) {
            ref[row * N + i] = input_f[row * N + i] * rms_inv * weight_f[i];
        }
    }

    void* pso = make_pipeline("rms_norm");
    void* buf_in  = g_mc->create_buffer_from_data(input_h.data(), input_h.size() * 2);
    void* buf_w   = g_mc->create_buffer_from_data(weight_h.data(), weight_h.size() * 2);
    void* buf_out = g_mc->create_buffer(N_ROWS * N * 2);
    CHECK(buf_in && buf_w && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_w, buf_out};
    dp.constants = {{&N, sizeof(uint32_t)}, {&EPS, sizeof(float)}};
    dp.grid_size  = {N_ROWS * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N_ROWS * N; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.02f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_w); CFRelease(buf_out);
    }

    std::printf("PASS\n");
}

// ─── Test: dispatch_batch_sync ──────────────────────────────────────────

static void test_batch_dispatch() {
    std::printf("[test_batch_dispatch] ");

    // Run softmax on two independent rows as a batch
    constexpr uint32_t N = 64;

    std::vector<uint16_t> row0(N), row1(N);
    for (uint32_t i = 0; i < N; ++i) {
        row0[i] = f32_to_f16_bits(static_cast<float>(i) * 0.1f);
        row1[i] = f32_to_f16_bits(static_cast<float>(i) * -0.1f);
    }

    void* pso = make_pipeline("softmax");

    // Concatenate as 2 rows
    std::vector<uint16_t> both(2 * N);
    std::memcpy(both.data(), row0.data(), N * 2);
    std::memcpy(both.data() + N, row1.data(), N * 2);

    void* buf_in  = g_mc->create_buffer_from_data(both.data(), both.size() * 2);
    void* buf_out = g_mc->create_buffer(2 * N * 2);

    uint32_t n_val = N;

    // Single dispatch with 2 threadgroups (one per row)
    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&n_val, sizeof(uint32_t)}};
    dp.grid_size  = {2 * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    std::vector<mugen::MetalCompute::DispatchParams> batch = {dp};
    auto result = g_mc->dispatch_batch_sync(batch);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        for (int r = 0; r < 2; ++r) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < N; ++i) {
                float v = f16_bits_to_f32(out_ptr[r * N + i]);
                CHECK(v >= 0.0f);
                sum += v;
            }
            CHECK(std::abs(sum - 1.0f) < 0.02f);
        }
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_out);
    }

    std::printf("gpu_time=%.3fms PASS\n", result.value() * 1000.0);
}

// ─── Test: silu ─────────────────────────────────────────────────────────

static void test_silu() {
    std::printf("[test_silu] ");

    constexpr uint32_t N = 128;
    std::vector<uint16_t> input_h(N);
    std::vector<float> expected(N);

    for (uint32_t i = 0; i < N; ++i) {
        float x = (static_cast<float>(i) - 64.0f) * 0.1f;
        input_h[i] = f32_to_f16_bits(x);
        float xh = f16_bits_to_f32(input_h[i]);
        expected[i] = xh / (1.0f + std::exp(-xh));
    }

    void* pso = make_pipeline("silu");
    void* buf_in  = g_mc->create_buffer_from_data(input_h.data(), N * 2);
    void* buf_out = g_mc->create_buffer(N * 2);
    CHECK(buf_in && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&N, sizeof(uint32_t)}};
    dp.grid_size  = {N, 1, 1};
    dp.group_size = {std::min(N, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - expected[i]);
            float denom = std::max(std::abs(expected[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.02f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: elementwise_mul ──────────────────────────────────────────────

static void test_elementwise_mul() {
    std::printf("[test_elementwise_mul] ");

    constexpr uint32_t N = 256;
    std::mt19937 rng(100);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<uint16_t> a_h(N), b_h(N);
    std::vector<float> expected(N);

    for (uint32_t i = 0; i < N; ++i) {
        float a = dist(rng), b = dist(rng);
        a_h[i] = f32_to_f16_bits(a);
        b_h[i] = f32_to_f16_bits(b);
        expected[i] = f16_bits_to_f32(a_h[i]) * f16_bits_to_f32(b_h[i]);
    }

    void* pso = make_pipeline("elementwise_mul");
    void* buf_a   = g_mc->create_buffer_from_data(a_h.data(), N * 2);
    void* buf_b   = g_mc->create_buffer_from_data(b_h.data(), N * 2);
    void* buf_out = g_mc->create_buffer(N * 2);
    CHECK(buf_a && buf_b && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_a, buf_b, buf_out};
    dp.constants = {{&N, sizeof(uint32_t)}};
    dp.grid_size  = {N, 1, 1};
    dp.group_size = {std::min(N, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - expected[i]);
            float denom = std::max(std::abs(expected[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.01f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_a); CFRelease(buf_b); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: elementwise_add ──────────────────────────────────────────────

static void test_elementwise_add() {
    std::printf("[test_elementwise_add] ");

    constexpr uint32_t N = 256;
    std::mt19937 rng(200);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<uint16_t> a_h(N), b_h(N);
    std::vector<float> expected(N);

    for (uint32_t i = 0; i < N; ++i) {
        float a = dist(rng), b = dist(rng);
        a_h[i] = f32_to_f16_bits(a);
        b_h[i] = f32_to_f16_bits(b);
        expected[i] = f16_bits_to_f32(a_h[i]) + f16_bits_to_f32(b_h[i]);
    }

    void* pso = make_pipeline("elementwise_add");
    void* buf_a   = g_mc->create_buffer_from_data(a_h.data(), N * 2);
    void* buf_b   = g_mc->create_buffer_from_data(b_h.data(), N * 2);
    void* buf_out = g_mc->create_buffer(N * 2);
    CHECK(buf_a && buf_b && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_a, buf_b, buf_out};
    dp.constants = {{&N, sizeof(uint32_t)}};
    dp.grid_size  = {N, 1, 1};
    dp.group_size = {std::min(N, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - expected[i]);
            float denom = std::max(std::abs(expected[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.01f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_a); CFRelease(buf_b); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: embedding_lookup ─────────────────────────────────────────────

static void test_embedding_lookup() {
    std::printf("[test_embedding_lookup] ");

    constexpr uint32_t VOCAB_SIZE = 64;
    constexpr uint32_t EMBED_DIM  = 128;
    constexpr uint32_t N_TOKENS   = 8;

    std::mt19937 rng(300);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> table_h(VOCAB_SIZE * EMBED_DIM);
    for (auto& v : table_h) v = f32_to_f16_bits(dist(rng));

    std::vector<uint32_t> token_ids(N_TOKENS);
    std::uniform_int_distribution<uint32_t> id_dist(0, VOCAB_SIZE - 1);
    for (auto& id : token_ids) id = id_dist(rng);

    uint32_t total_elems = N_TOKENS * EMBED_DIM;

    void* pso = make_pipeline("embedding_lookup");
    void* buf_table = g_mc->create_buffer_from_data(table_h.data(), table_h.size() * 2);
    void* buf_ids   = g_mc->create_buffer_from_data(token_ids.data(), token_ids.size() * 4);
    void* buf_out   = g_mc->create_buffer(total_elems * 2);
    CHECK(buf_table && buf_ids && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_table, buf_ids, buf_out};
    dp.constants = {{&EMBED_DIM, sizeof(uint32_t)}, {&N_TOKENS, sizeof(uint32_t)}};
    dp.grid_size  = {total_elems, 1, 1};
    dp.group_size = {std::min(total_elems, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        for (uint32_t t = 0; t < N_TOKENS; ++t) {
            uint32_t tok_id = token_ids[t];
            for (uint32_t d = 0; d < EMBED_DIM; ++d) {
                uint16_t exp_val = table_h[tok_id * EMBED_DIM + d];
                uint16_t actual  = out_ptr[t * EMBED_DIM + d];
                CHECK(exp_val == actual);
            }
        }
        std::printf("exact_match gpu_time=%.3fms ", result.value() * 1000.0);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_table); CFRelease(buf_ids); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: dequantize_q8_0 ──────────────────────────────────────────────

static void test_dequantize_q8_0() {
    std::printf("[test_dequantize_q8_0] ");

    constexpr uint32_t N_BLOCKS = 64;
    constexpr uint32_t N_ELEMS  = N_BLOCKS * 32;

    std::mt19937 rng(500);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> original(N_ELEMS);
    for (auto& v : original) v = dist(rng);

    std::vector<BlockQ8_0> blocks(N_BLOCKS);
    for (uint32_t b = 0; b < N_BLOCKS; ++b) {
        blocks[b] = encode_q8_0(&original[b * 32]);
    }

    std::vector<float> ref(N_ELEMS);
    for (uint32_t b = 0; b < N_BLOCKS; ++b) {
        dequant_q8_0_ref(blocks[b], &ref[b * 32]);
    }

    void* pso = make_pipeline("dequantize_q8_0");
    void* buf_in  = g_mc->create_buffer_from_data(blocks.data(), N_BLOCKS * sizeof(BlockQ8_0));
    void* buf_out = g_mc->create_buffer(N_ELEMS * sizeof(uint16_t));
    CHECK(buf_in && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&N_BLOCKS, sizeof(uint32_t)}};
    dp.grid_size  = {N_BLOCKS, 1, 1};
    dp.group_size = {std::min(N_BLOCKS, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < N_ELEMS; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float rel = (std::abs(ref[i]) > 1e-6f) ? (err / std::abs(ref[i])) : err;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.01f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: matvec_q8_0 ─────────────────────────────────────────────────

static void test_matvec_q8_0() {
    std::printf("[test_matvec_q8_0] ");

    constexpr uint32_t M = 64;
    constexpr uint32_t K = 256;
    constexpr uint32_t BLOCKS_PER_ROW = K / 32;

    std::mt19937 rng(600);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> mat_f(M * K);
    std::vector<float> vec_f(K);
    for (auto& v : mat_f) v = dist(rng);
    for (auto& v : vec_f) v = dist(rng);

    std::vector<BlockQ8_0> mat_q(M * BLOCKS_PER_ROW);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            mat_q[row * BLOCKS_PER_ROW + b] = encode_q8_0(&mat_f[row * K + b * 32]);
        }
    }

    std::vector<float> ref(M, 0.0f);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            float deq[32];
            dequant_q8_0_ref(mat_q[row * BLOCKS_PER_ROW + b], deq);
            for (int j = 0; j < 32; ++j) {
                ref[row] += deq[j] * vec_f[b * 32 + j];
            }
        }
    }

    std::vector<uint16_t> vec_h(K);
    for (uint32_t i = 0; i < K; ++i) {
        vec_h[i] = f32_to_f16_bits(vec_f[i]);
    }

    void* pso = make_pipeline("matvec_q8_0");
    void* buf_mat = g_mc->create_buffer_from_data(mat_q.data(), mat_q.size() * sizeof(BlockQ8_0));
    void* buf_vec = g_mc->create_buffer_from_data(vec_h.data(), vec_h.size() * 2);
    void* buf_out = g_mc->create_buffer(M * 2);
    CHECK(buf_mat && buf_vec && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_mat, buf_vec, buf_out};
    dp.constants = {{&M, sizeof(uint32_t)}, {&K, sizeof(uint32_t)}};
    dp.grid_size  = {M * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < M; ++i) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            float rel = err / denom;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.05f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_mat); CFRelease(buf_vec); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Q4_K encoding/dequant helpers (host side) ──────────────────────────

static constexpr uint32_t Q4_K_BLOCK_SIZE = 144;
static constexpr uint32_t Q4_K_QK = 256;

struct BlockQ4_K {
    uint16_t d_f16;
    uint16_t dmin_f16;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == Q4_K_BLOCK_SIZE);

static BlockQ4_K encode_q4_k_simple(float d, float dmin,
                                     const uint8_t sc_low[4], const uint8_t mn_low[4],
                                     const uint8_t sc_high[4], const uint8_t mn_high[4],
                                     const uint8_t qs[128]) {
    BlockQ4_K block{};
    block.d_f16 = f32_to_f16_bits(d);
    block.dmin_f16 = f32_to_f16_bits(dmin);

    for (int i = 0; i < 4; i++) {
        block.scales[i] = sc_low[i] & 0x3F;
        block.scales[4 + i] = mn_low[i] & 0x3F;
        block.scales[8 + i] = (sc_high[i] & 0x03) | (((mn_high[i] & 0x03) << 2));
    }
    std::memcpy(block.qs, qs, 128);
    return block;
}

static void dequant_q4_k_ref(const BlockQ4_K& block, float out[256]) {
    float d = f16_bits_to_f32(block.d_f16);
    float dmin = f16_bits_to_f32(block.dmin_f16);

    float sc[4], mn[4];
    for (int i = 0; i < 4; i++) {
        uint8_t sl = block.scales[i];
        uint8_t ml = block.scales[4 + i];
        uint8_t sh_byte = block.scales[8 + i];
        sc[i] = d * float((sl & 0x3F) | ((sh_byte & 0x03) << 6));
        mn[i] = dmin * float((ml & 0x3F) | (((sh_byte >> 2) & 0x03) << 6));
    }

    for (int sb = 0; sb < 4; sb++) {
        for (int j = 0; j < 32; j++) {
            uint8_t byte = block.qs[sb * 32 + j];
            out[sb * 64 + j * 2]     = sc[sb] * float(byte & 0xF) - mn[sb];
            out[sb * 64 + j * 2 + 1] = sc[sb] * float(byte >> 4) - mn[sb];
        }
    }
}

// ─── Test: rope ─────────────────────────────────────────────────────────

static void test_rope() {
    std::printf("[test_rope] ");

    constexpr uint32_t N_HEADS  = 2;
    constexpr uint32_t HEAD_DIM = 8;
    constexpr uint32_t POSITION = 5;
    constexpr float THETA_BASE  = 10000.0f;
    constexpr uint32_t TOTAL = N_HEADS * HEAD_DIM;
    constexpr uint32_t HALF_DIM = HEAD_DIM / 2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x_f(TOTAL);
    std::vector<uint16_t> x_h(TOTAL);
    for (uint32_t i = 0; i < TOTAL; i++) {
        x_f[i] = dist(rng);
        x_h[i] = f32_to_f16_bits(x_f[i]);
    }

    // CPU reference (NeoX-style halved RoPE)
    std::vector<float> ref(TOTAL);
    for (uint32_t i = 0; i < TOTAL; i++) ref[i] = f16_bits_to_f32(x_h[i]);
    for (uint32_t h = 0; h < N_HEADS; h++) {
        for (uint32_t p = 0; p < HALF_DIM; p++) {
            float freq = 1.0f / std::pow(THETA_BASE, float(2 * p) / float(HEAD_DIM));
            float angle = float(POSITION) * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            uint32_t idx_re = h * HEAD_DIM + p;
            uint32_t idx_im = idx_re + HALF_DIM;
            float re = ref[idx_re];
            float im = ref[idx_im];
            ref[idx_re] = re * cos_val - im * sin_val;
            ref[idx_im] = im * cos_val + re * sin_val;
        }
    }

    void* pso = make_pipeline("rope");
    void* buf_x = g_mc->create_buffer_from_data(x_h.data(), TOTAL * 2);
    CHECK(buf_x);

    uint32_t head_dim = HEAD_DIM;
    uint32_t n_heads = N_HEADS;
    uint32_t position = POSITION;
    float theta_base = THETA_BASE;

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_x};
    dp.constants = {{&head_dim, sizeof(uint32_t)}, {&n_heads, sizeof(uint32_t)},
                    {&position, sizeof(uint32_t)}, {&theta_base, sizeof(float)}};
    uint32_t grid_x = N_HEADS * HALF_DIM;
    dp.grid_size  = {grid_x, 1, 1};
    dp.group_size = {std::min(grid_x, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_x contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < TOTAL; i++) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.02f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_x);
    }
    std::printf("PASS\n");
}

// ─── Test: moe_gate ─────────────────────────────────────────────────────

static void test_moe_gate() {
    std::printf("[test_moe_gate] ");

    constexpr uint32_t N_EXPERTS = 8;
    constexpr uint32_t TOP_K = 2;

    float logits_f[N_EXPERTS] = {0.1f, 2.5f, -0.3f, 1.8f, 0.5f, -1.0f, 3.0f, 0.2f};
    std::vector<uint16_t> logits_h(N_EXPERTS);
    for (uint32_t i = 0; i < N_EXPERTS; i++) {
        logits_h[i] = f32_to_f16_bits(logits_f[i]);
    }

    // CPU reference: softmax over all 8 experts, then pick top-2
    // top-2 are index 6 (3.0) and index 1 (2.5)
    uint32_t ref_indices[2] = {6, 1};
    float max_all = *std::max_element(logits_f, logits_f + N_EXPERTS);
    float sum_all = 0.0f;
    float probs[N_EXPERTS];
    for (uint32_t i = 0; i < N_EXPERTS; i++) {
        probs[i] = std::exp(logits_f[i] - max_all);
        sum_all += probs[i];
    }
    for (uint32_t i = 0; i < N_EXPERTS; i++) probs[i] /= sum_all;
    float ref_weights[2] = {probs[6], probs[1]};

    void* pso = make_pipeline("moe_gate");
    void* buf_logits = g_mc->create_buffer_from_data(logits_h.data(), N_EXPERTS * 2);
    void* buf_indices = g_mc->create_buffer(TOP_K * sizeof(uint32_t));
    void* buf_weights = g_mc->create_buffer(TOP_K * 2);
    CHECK(buf_logits && buf_indices && buf_weights);

    uint32_t n_experts = N_EXPERTS;
    uint32_t top_k = TOP_K;

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_logits, buf_indices, buf_weights};
    dp.constants = {{&n_experts, sizeof(uint32_t)}, {&top_k, sizeof(uint32_t)}};
    dp.grid_size  = {1, 1, 1};
    dp.group_size = {1, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* idx_ptr = static_cast<uint32_t*>([(__bridge id<MTLBuffer>)buf_indices contents]);
        auto* wt_ptr  = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_weights contents]);

        CHECK(idx_ptr[0] == ref_indices[0]);
        CHECK(idx_ptr[1] == ref_indices[1]);

        for (uint32_t k = 0; k < TOP_K; k++) {
            float gpu_w = f16_bits_to_f32(wt_ptr[k]);
            float err = std::abs(gpu_w - ref_weights[k]);
            CHECK(err < 0.02f);
        }

        float wsum = f16_bits_to_f32(wt_ptr[0]) + f16_bits_to_f32(wt_ptr[1]);
        CHECK(wsum > 0.0f && wsum <= 1.02f);

        std::printf("indices=[%u,%u] weights=[%.3f,%.3f] gpu_time=%.3fms ",
                    idx_ptr[0], idx_ptr[1],
                    f16_bits_to_f32(wt_ptr[0]), f16_bits_to_f32(wt_ptr[1]),
                    result.value() * 1000.0);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_logits); CFRelease(buf_indices); CFRelease(buf_weights);
    }
    std::printf("PASS\n");
}

// ─── Test: moe_reduce ───────────────────────────────────────────────────

static void test_moe_reduce() {
    std::printf("[test_moe_reduce] ");

    constexpr uint32_t DIM = 64;
    constexpr uint32_t K_EXPERTS = 3;

    std::mt19937 rng(700);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> expert_h(K_EXPERTS * DIM);
    std::vector<float> expert_f(K_EXPERTS * DIM);
    for (size_t i = 0; i < expert_f.size(); i++) {
        expert_f[i] = dist(rng);
        expert_h[i] = f32_to_f16_bits(expert_f[i]);
    }

    float weights_f[K_EXPERTS] = {0.5f, 0.3f, 0.2f};
    std::vector<uint16_t> weights_h(K_EXPERTS);
    for (uint32_t i = 0; i < K_EXPERTS; i++) {
        weights_h[i] = f32_to_f16_bits(weights_f[i]);
    }

    // CPU reference
    std::vector<float> ref(DIM, 0.0f);
    for (uint32_t d = 0; d < DIM; d++) {
        for (uint32_t k = 0; k < K_EXPERTS; k++) {
            ref[d] += f16_bits_to_f32(expert_h[k * DIM + d]) *
                      f16_bits_to_f32(weights_h[k]);
        }
    }

    void* pso = make_pipeline("moe_reduce");
    void* buf_exp = g_mc->create_buffer_from_data(expert_h.data(), expert_h.size() * 2);
    void* buf_w   = g_mc->create_buffer_from_data(weights_h.data(), weights_h.size() * 2);
    void* buf_out = g_mc->create_buffer(DIM * 2);
    CHECK(buf_exp && buf_w && buf_out);

    uint32_t dim = DIM;
    uint32_t k_val = K_EXPERTS;

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_exp, buf_w, buf_out};
    dp.constants = {{&dim, sizeof(uint32_t)}, {&k_val, sizeof(uint32_t)}};
    dp.grid_size  = {DIM, 1, 1};
    dp.group_size = {std::min(DIM, 256u), 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < DIM; i++) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.02f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_exp); CFRelease(buf_w); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: dequantize_q4_k ──────────────────────────────────────────────

static void test_dequantize_q4_k() {
    std::printf("[test_dequantize_q4_k] ");

    constexpr uint32_t N_BLOCKS = 1;

    // d=1.0, dmin=0.0, all sub-block scales=1, mins=0
    // scales_low = [1,1,1,1], mins_low = [0,0,0,0], high bits = 0
    uint8_t sc_low[4] = {1, 1, 1, 1};
    uint8_t mn_low[4] = {0, 0, 0, 0};
    uint8_t sc_high[4] = {0, 0, 0, 0};
    uint8_t mn_high[4] = {0, 0, 0, 0};

    uint8_t qs[128];
    for (int i = 0; i < 128; i++) {
        qs[i] = static_cast<uint8_t>(((i % 15) << 4) | ((i + 1) % 15));
    }

    auto block = encode_q4_k_simple(1.0f, 0.0f, sc_low, mn_low, sc_high, mn_high, qs);

    float ref[256];
    dequant_q4_k_ref(block, ref);

    void* pso = make_pipeline("dequantize_q4_k");
    void* buf_in  = g_mc->create_buffer_from_data(&block, sizeof(BlockQ4_K));
    void* buf_out = g_mc->create_buffer(256 * sizeof(uint16_t));
    CHECK(buf_in && buf_out);

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_in, buf_out};
    dp.constants = {{&N_BLOCKS, sizeof(uint32_t)}};
    dp.grid_size  = {N_BLOCKS, 1, 1};
    dp.group_size = {1, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < 256; i++) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            max_err = std::max(max_err, err / denom);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.01f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_in); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ─── Test: matvec_q4_k ─────────────────────────────────────────────────

static void test_matvec_q4_k() {
    std::printf("[test_matvec_q4_k] ");

    constexpr uint32_t M = 4;
    constexpr uint32_t K = 256;
    constexpr uint32_t BLOCKS_PER_ROW = K / Q4_K_QK;  // = 1

    // Build simple blocks: d=1.0, dmin=0.0, sc=1, mn=0
    uint8_t sc_low[4] = {1, 1, 1, 1};
    uint8_t mn_low[4] = {0, 0, 0, 0};
    uint8_t sc_high[4] = {0, 0, 0, 0};
    uint8_t mn_high[4] = {0, 0, 0, 0};

    std::mt19937 rng(800);

    std::vector<BlockQ4_K> mat_q(M * BLOCKS_PER_ROW);
    std::vector<float> mat_deq(M * K);

    for (uint32_t row = 0; row < M; row++) {
        uint8_t qs[128];
        for (int i = 0; i < 128; i++) {
            qs[i] = static_cast<uint8_t>(rng() % 256);
        }
        mat_q[row] = encode_q4_k_simple(1.0f, 0.0f, sc_low, mn_low, sc_high, mn_high, qs);
        dequant_q4_k_ref(mat_q[row], &mat_deq[row * K]);
    }

    // Random vector (half precision)
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec_f(K);
    std::vector<uint16_t> vec_h(K);
    for (uint32_t i = 0; i < K; i++) {
        vec_f[i] = dist(rng);
        vec_h[i] = f32_to_f16_bits(vec_f[i]);
    }

    // CPU reference: use half-precision vec values for accurate comparison
    std::vector<float> ref(M, 0.0f);
    for (uint32_t row = 0; row < M; row++) {
        for (uint32_t col = 0; col < K; col++) {
            ref[row] += mat_deq[row * K + col] * f16_bits_to_f32(vec_h[col]);
        }
    }

    void* pso = make_pipeline("matvec_q4_k");
    void* buf_mat = g_mc->create_buffer_from_data(mat_q.data(), mat_q.size() * sizeof(BlockQ4_K));
    void* buf_vec = g_mc->create_buffer_from_data(vec_h.data(), vec_h.size() * 2);
    void* buf_out = g_mc->create_buffer(M * 2);
    CHECK(buf_mat && buf_vec && buf_out);

    uint32_t m_val = M;
    uint32_t k_val = K;

    mugen::MetalCompute::DispatchParams dp;
    dp.pipeline = pso;
    dp.buffers  = {buf_mat, buf_vec, buf_out};
    dp.constants = {{&m_val, sizeof(uint32_t)}, {&k_val, sizeof(uint32_t)}};
    dp.grid_size  = {M * 256, 1, 1};
    dp.group_size = {256, 1, 1};

    auto result = g_mc->dispatch_sync(dp);
    CHECK(result.has_value());

    @autoreleasepool {
        auto* out_ptr = static_cast<uint16_t*>([(__bridge id<MTLBuffer>)buf_out contents]);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < M; i++) {
            float gpu_val = f16_bits_to_f32(out_ptr[i]);
            float err = std::abs(gpu_val - ref[i]);
            float denom = std::max(std::abs(ref[i]), 1e-3f);
            float rel = err / denom;
            max_err = std::max(max_err, rel);
        }
        std::printf("max_rel_err=%.4f%% gpu_time=%.3fms ", max_err * 100.0f, result.value() * 1000.0);
        CHECK(max_err < 0.05f);
    }

    @autoreleasepool {
        CFRelease(pso); CFRelease(buf_mat); CFRelease(buf_vec); CFRelease(buf_out);
    }
    std::printf("PASS\n");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    @autoreleasepool {
        std::printf("=== Metal Compute Unit Tests ===\n\n");

        setup();

        test_create();
        test_compile();
        test_buffers();
        test_dequantize_q4_0();
        test_matvec_f16();
        test_matvec_q4_0();
        test_softmax();
        test_rms_norm();
        test_batch_dispatch();
        test_silu();
        test_elementwise_mul();
        test_elementwise_add();
        test_embedding_lookup();
        test_dequantize_q8_0();
        test_matvec_q8_0();
        test_rope();
        test_moe_gate();
        test_moe_reduce();
        test_dequantize_q4_k();
        test_matvec_q4_k();

        std::printf("\n=== All tests passed ===\n");
    }
    return 0;
}

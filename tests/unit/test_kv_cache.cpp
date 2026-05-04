#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/memory/kv_cache.h"
#include "mugen/core/types.h"

#define MUGEN_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,     \
                         __LINE__);                                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kLayers   = 2;
constexpr uint32_t kHeads    = 4;
constexpr uint32_t kHeadDim  = 64;
constexpr uint32_t kMaxSeq   = 1024;
constexpr uint32_t kPreserve = 256;
constexpr uint32_t kVPT      = kHeads * kHeadDim;  // 256 values per token

mugen::KVCacheConfig test_config() {
    return {
        .n_layers          = kLayers,
        .n_kv_heads        = kHeads,
        .head_dim          = kHeadDim,
        .max_seq_len       = kMaxSeq,
        .quantize_4bit     = true,
        .fp16_preserve_last = kPreserve,
    };
}

void fill_token(mugen::f16* buf, uint32_t vpt, float base) {
    for (uint32_t i = 0; i < vpt; ++i)
        buf[i] = mugen::f16(base + 0.01f * static_cast<float>(i));
}

float max_abs_error(const mugen::f16* a, const mugen::f16* b, uint32_t n) {
    float worst = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float diff = std::fabs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > worst) worst = diff;
    }
    return worst;
}

float max_relative_error(const mugen::f16* orig, const mugen::f16* got, uint32_t n) {
    float worst = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float o = static_cast<float>(orig[i]);
        float g = static_cast<float>(got[i]);
        float denom = std::fabs(o);
        if (denom < 1e-6f) denom = 1e-6f;
        float rel = std::fabs(o - g) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_create() {
    auto result = mugen::KVCache::create(test_config());
    MUGEN_CHECK(result.has_value());
    MUGEN_CHECK(result.value() != nullptr);
    MUGEN_CHECK(result.value()->seq_len() == 0);
    MUGEN_CHECK(result.value()->memory_bytes() == 0);

    auto bad = mugen::KVCache::create({.n_layers = 0, .n_kv_heads = 4, .head_dim = 64});
    MUGEN_CHECK(!bad.has_value());

    std::printf("  create PASS\n");
}

static void test_append_read_fp16_exact() {
    auto cache = std::move(*mugen::KVCache::create(test_config()));

    std::vector<mugen::f16> k(kVPT), v(kVPT);
    std::vector<mugen::f16> k_out(kVPT), v_out(kVPT);

    for (uint32_t t = 0; t < 10; ++t) {
        fill_token(k.data(), kVPT, 1.0f + 0.1f * static_cast<float>(t));
        fill_token(v.data(), kVPT, 2.0f + 0.1f * static_cast<float>(t));

        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }

    MUGEN_CHECK(cache->seq_len() == 10);

    for (uint32_t t = 0; t < 10; ++t) {
        fill_token(k.data(), kVPT, 1.0f + 0.1f * static_cast<float>(t));
        fill_token(v.data(), kVPT, 2.0f + 0.1f * static_cast<float>(t));

        MUGEN_CHECK(cache->read_k(0, t, 1, k_out.data()));
        MUGEN_CHECK(cache->read_v(0, t, 1, v_out.data()));

        MUGEN_CHECK(max_abs_error(k.data(), k_out.data(), kVPT) == 0.0f);
        MUGEN_CHECK(max_abs_error(v.data(), v_out.data(), kVPT) == 0.0f);
    }

    std::printf("  append_read_fp16_exact PASS\n");
}

static void test_append_read_q4_precision() {
    auto cfg = test_config();
    cfg.fp16_preserve_last = 32;
    auto cache = std::move(*mugen::KVCache::create(cfg));

    constexpr uint32_t kFill = 128;
    std::vector<std::vector<mugen::f16>> original_k(kFill, std::vector<mugen::f16>(kVPT));
    std::vector<std::vector<mugen::f16>> original_v(kFill, std::vector<mugen::f16>(kVPT));

    for (uint32_t t = 0; t < kFill; ++t) {
        fill_token(original_k[t].data(), kVPT, 1.0f + 0.05f * static_cast<float>(t));
        fill_token(original_v[t].data(), kVPT, 2.0f + 0.05f * static_cast<float>(t));

        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, original_k[t].data(), original_v[t].data()));
    }

    MUGEN_CHECK(cache->seq_len() == kFill);

    // Q4 region: tokens 0..(kFill - 32 - 1) were quantized
    uint32_t q4_end = kFill - cfg.fp16_preserve_last;

    std::vector<mugen::f16> k_out(kVPT), v_out(kVPT);

    for (uint32_t t = 0; t < q4_end; ++t) {
        MUGEN_CHECK(cache->read_k(0, t, 1, k_out.data()));
        MUGEN_CHECK(cache->read_v(0, t, 1, v_out.data()));

        float k_err = max_relative_error(original_k[t].data(), k_out.data(), kVPT);
        float v_err = max_relative_error(original_v[t].data(), v_out.data(), kVPT);
        MUGEN_CHECK(k_err < 0.10f);
        MUGEN_CHECK(v_err < 0.10f);
    }

    // FP16 region: should be exact
    for (uint32_t t = q4_end; t < kFill; ++t) {
        MUGEN_CHECK(cache->read_k(0, t, 1, k_out.data()));
        MUGEN_CHECK(cache->read_v(0, t, 1, v_out.data()));

        MUGEN_CHECK(max_abs_error(original_k[t].data(), k_out.data(), kVPT) == 0.0f);
        MUGEN_CHECK(max_abs_error(original_v[t].data(), v_out.data(), kVPT) == 0.0f);
    }

    std::printf("  append_read_q4_precision PASS\n");
}

static void test_compression_ratio() {
    auto cfg = test_config();
    cfg.fp16_preserve_last = 0;
    cfg.max_seq_len = 512;
    auto cache = std::move(*mugen::KVCache::create(cfg));

    std::vector<mugen::f16> k(kVPT), v(kVPT);

    for (uint32_t t = 0; t < 512; ++t) {
        fill_token(k.data(), kVPT, static_cast<float>(t) * 0.01f);
        fill_token(v.data(), kVPT, static_cast<float>(t) * 0.01f + 1.0f);
        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }

    float ratio = cache->compression_ratio();
    std::printf("    pure-Q4 compression ratio = %.3f\n", static_cast<double>(ratio));

    // 64 / 20 = 3.2 exactly for pure Q4
    MUGEN_CHECK(ratio > 3.15f && ratio < 3.25f);

    std::printf("  compression_ratio PASS\n");
}

static void test_compression_ratio_mixed() {
    auto cfg = test_config();
    cfg.fp16_preserve_last = 32;
    cfg.max_seq_len = 1024;
    auto cache = std::move(*mugen::KVCache::create(cfg));

    std::vector<mugen::f16> k(kVPT), v(kVPT);

    for (uint32_t t = 0; t < 1024; ++t) {
        fill_token(k.data(), kVPT, static_cast<float>(t) * 0.01f);
        fill_token(v.data(), kVPT, static_cast<float>(t) * 0.01f + 1.0f);
        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }

    float ratio = cache->compression_ratio();
    std::printf("    mixed compression ratio = %.3f\n", static_cast<double>(ratio));

    // With 32 FP16 tokens out of 1024 total, ratio approaches 3.2 but stays below
    MUGEN_CHECK(ratio > 2.8f);

    std::printf("  compression_ratio_mixed PASS\n");
}

static void test_truncate() {
    auto cfg = test_config();
    cfg.fp16_preserve_last = 8;
    auto cache = std::move(*mugen::KVCache::create(cfg));

    std::vector<mugen::f16> k(kVPT), v(kVPT);

    for (uint32_t t = 0; t < 64; ++t) {
        fill_token(k.data(), kVPT, static_cast<float>(t));
        fill_token(v.data(), kVPT, static_cast<float>(t) + 100.0f);
        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }
    MUGEN_CHECK(cache->seq_len() == 64);

    cache->truncate(32);
    MUGEN_CHECK(cache->seq_len() == 32);

    // Further truncation
    cache->truncate(4);
    MUGEN_CHECK(cache->seq_len() == 4);

    // Truncate to 0
    cache->truncate(0);
    MUGEN_CHECK(cache->seq_len() == 0);
    MUGEN_CHECK(cache->memory_bytes() == 0);

    std::printf("  truncate PASS\n");
}

static void test_clear() {
    auto cache = std::move(*mugen::KVCache::create(test_config()));

    std::vector<mugen::f16> k(kVPT), v(kVPT);
    fill_token(k.data(), kVPT, 1.0f);
    fill_token(v.data(), kVPT, 2.0f);

    for (uint32_t layer = 0; layer < kLayers; ++layer)
        MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    MUGEN_CHECK(cache->seq_len() == 1);

    cache->clear();
    MUGEN_CHECK(cache->seq_len() == 0);
    MUGEN_CHECK(cache->memory_bytes() == 0);

    // Can append again after clear
    for (uint32_t layer = 0; layer < kLayers; ++layer)
        MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    MUGEN_CHECK(cache->seq_len() == 1);

    std::printf("  clear PASS\n");
}

static void test_max_seq_len_overflow() {
    auto cfg = test_config();
    cfg.max_seq_len = 8;
    auto cache = std::move(*mugen::KVCache::create(cfg));

    std::vector<mugen::f16> k(kVPT), v(kVPT);
    fill_token(k.data(), kVPT, 1.0f);
    fill_token(v.data(), kVPT, 2.0f);

    for (uint32_t t = 0; t < 8; ++t) {
        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }
    MUGEN_CHECK(cache->seq_len() == 8);

    // 9th append must fail
    for (uint32_t layer = 0; layer < kLayers; ++layer)
        MUGEN_CHECK(!cache->append(layer, k.data(), v.data()));
    MUGEN_CHECK(cache->seq_len() == 8);

    std::printf("  max_seq_len_overflow PASS\n");
}

static void test_multi_token_read() {
    auto cache = std::move(*mugen::KVCache::create(test_config()));

    std::vector<mugen::f16> k(kVPT), v(kVPT);
    constexpr uint32_t kN = 5;

    for (uint32_t t = 0; t < kN; ++t) {
        fill_token(k.data(), kVPT, static_cast<float>(t));
        fill_token(v.data(), kVPT, static_cast<float>(t) + 10.0f);
        for (uint32_t layer = 0; layer < kLayers; ++layer)
            MUGEN_CHECK(cache->append(layer, k.data(), v.data()));
    }

    // Read all 5 tokens at once
    std::vector<mugen::f16> k_bulk(kVPT * kN), v_bulk(kVPT * kN);
    MUGEN_CHECK(cache->read_k(0, 0, kN, k_bulk.data()));
    MUGEN_CHECK(cache->read_v(0, 0, kN, v_bulk.data()));

    for (uint32_t t = 0; t < kN; ++t) {
        fill_token(k.data(), kVPT, static_cast<float>(t));
        fill_token(v.data(), kVPT, static_cast<float>(t) + 10.0f);
        MUGEN_CHECK(max_abs_error(k.data(), k_bulk.data() + t * kVPT, kVPT) == 0.0f);
        MUGEN_CHECK(max_abs_error(v.data(), v_bulk.data() + t * kVPT, kVPT) == 0.0f);
    }

    // Out-of-bounds read must fail
    MUGEN_CHECK(!cache->read_k(0, 3, 5, k_bulk.data()));

    std::printf("  multi_token_read PASS\n");
}

static void test_config_roundtrip() {
    auto cfg = test_config();
    auto cache = std::move(*mugen::KVCache::create(cfg));
    MUGEN_CHECK(cache->config().n_layers == kLayers);
    MUGEN_CHECK(cache->config().n_kv_heads == kHeads);
    MUGEN_CHECK(cache->config().head_dim == kHeadDim);
    MUGEN_CHECK(cache->config().max_seq_len == kMaxSeq);
    MUGEN_CHECK(cache->config().quantize_4bit == true);
    MUGEN_CHECK(cache->config().fp16_preserve_last == kPreserve);

    std::printf("  config_roundtrip PASS\n");
}

// ===========================================================================
int main() {
    std::printf("=== KVCache tests ===\n");
    test_create();
    test_append_read_fp16_exact();
    test_append_read_q4_precision();
    test_compression_ratio();
    test_compression_ratio_mixed();
    test_truncate();
    test_clear();
    test_max_seq_len_overflow();
    test_multi_token_read();
    test_config_roundtrip();

    std::printf("\nAll KV cache tests passed.\n");
    return 0;
}

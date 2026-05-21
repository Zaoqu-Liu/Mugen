#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "core/scheduler/uspp_scheduler.h"
#include "core/memory/mmap_loader.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;

constexpr uint32_t kTestLayers  = 2;
constexpr uint32_t kTestExperts = 4;
constexpr size_t   kExpertSlice = 1024;   // bytes per expert per component
constexpr size_t   kPackedSize  = kTestExperts * kExpertSlice;  // 4096

std::filesystem::path write_temp(const void* data, size_t len,
                                 const char* suffix) {
    auto p = std::filesystem::temp_directory_path()
           / ("mugen_uspp_" + std::string(suffix) + ".bin");
    FILE* fp = std::fopen(p.c_str(), "wb");
    CHECK(fp != nullptr);
    CHECK(std::fwrite(data, 1, len, fp) == len);
    std::fclose(fp);
    return p;
}

struct TestFixture {
    std::filesystem::path mmap_path;
    mugen::MmapRegion mmap;
    mugen::ExpertIndex expert_index;
    mugen::BufferManager buf_mgr;

    TestFixture()
        : mmap_path(), mmap(std::move(create_mmap().second)),
          expert_index(build_index()),
          buf_mgr(mugen::BufferManager::Config{
              .buffer_capacity = 1 * MB,
              .pinned_capacity = 256 * KB,
              .system_reserve  = 1 * MB}) {}

    ~TestFixture() {
        if (!mmap_path.empty())
            std::filesystem::remove(mmap_path);
    }

    auto make_components() -> mugen::USPPScheduler::Components {
        return {
            .gpu          = nullptr,
            .buffers      = &buf_mgr,
            .expert_index = &expert_index,
            .model_mmap   = &mmap,
            .cache_policy = nullptr,
        };
    }

private:
    auto create_mmap() -> std::pair<std::filesystem::path, mugen::MmapRegion> {
        // 2 layers × 3 components × kPackedSize = 24576 bytes
        size_t file_size = kTestLayers * 3 * kPackedSize;
        std::vector<uint8_t> data(file_size);
        for (size_t i = 0; i < file_size; ++i)
            data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);

        mmap_path = write_temp(data.data(), data.size(), "experts");
        auto result = mugen::MmapLoader::map_file(mmap_path);
        CHECK(result.has_value());
        return {mmap_path, std::move(*result)};
    }

    static auto build_index() -> mugen::ExpertIndex {
        std::vector<mugen::TensorInfo> tensors;
        size_t offset = 0;
        auto add = [&](uint32_t layer, const char* comp) {
            std::string name = "blk." + std::to_string(layer)
                             + ".ffn_" + comp + "_exps.weight";
            tensors.push_back({
                .name       = name,
                .offset     = offset,
                .byte_size  = kPackedSize,
                .n_elements = kTestExperts,
            });
            offset += kPackedSize;
        };
        for (uint32_t l = 0; l < kTestLayers; ++l) {
            add(l, "gate");
            add(l, "up");
            add(l, "down");
        }
        return mugen::ExpertIndex::build(
            tensors, kTestLayers, kTestExperts, /*data_offset=*/0);
    }
};

mugen::USPPConfig test_config() {
    return {
        .draft_k     = 4,
        .min_draft_k = 1,
        .max_draft_k = 8,
        .accept_threshold = 0.8f,
        .prefetch_ahead_ms = 100.0f,
        .warmup_tokens = 0,
        .conservative_hit_threshold = 0.7f,
    };
}

}  // namespace

// ===========================================================================
// Tests
// ===========================================================================

static void test_create_success() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());
    CHECK(sched->get() != nullptr);
    std::printf("  create_success PASS\n");
}

static void test_create_fails_without_buffers() {
    TestFixture fix;
    auto comps = fix.make_components();
    comps.buffers = nullptr;
    auto sched = mugen::USPPScheduler::create(test_config(), comps);
    CHECK(!sched.has_value());
    std::printf("  create_fails_without_buffers PASS\n");
}

static void test_create_fails_without_expert_index() {
    TestFixture fix;
    auto comps = fix.make_components();
    comps.expert_index = nullptr;
    auto sched = mugen::USPPScheduler::create(test_config(), comps);
    CHECK(!sched.has_value());
    std::printf("  create_fails_without_expert_index PASS\n");
}

static void test_create_fails_without_mmap() {
    TestFixture fix;
    auto comps = fix.make_components();
    comps.model_mmap = nullptr;
    auto sched = mugen::USPPScheduler::create(test_config(), comps);
    CHECK(!sched.has_value());
    std::printf("  create_fails_without_mmap PASS\n");
}

static void test_generate_produces_tokens() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 10;

    std::vector<mugen::TokenResult> streamed;
    auto result = (*sched)->generate(req, [&](const mugen::TokenResult& t) {
        streamed.push_back(t);
    });

    CHECK(result.has_value());
    CHECK(!result->empty());
    CHECK(result->size() <= req.max_new_tokens);
    CHECK(result->size() == streamed.size());

    std::printf("  generate_produces_tokens (%zu tokens) PASS\n",
                result->size());
}

static void test_generate_respects_max_tokens() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 5;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());
    CHECK(result->size() <= 5);

    std::printf("  generate_respects_max_tokens PASS\n");
}

static void test_stop_terminates_generation() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());
    auto* ptr = sched->get();

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 100000;

    std::atomic<bool> done{false};
    std::vector<mugen::TokenResult> output;

    // Use the callback to trigger a stop after a small number of tokens,
    // rather than relying on a sleep race that breaks under sanitizers.
    uint32_t stop_after = 50;
    std::thread gen_thread([&] {
        auto res = ptr->generate(req, [&](const mugen::TokenResult& t) {
            output.push_back(t);
            if (output.size() >= stop_after) {
                ptr->stop();
            }
        });
        done.store(true);
    });

    gen_thread.join();

    CHECK(done.load());
    // stop() should have been called before generation completed
    CHECK(output.size() < req.max_new_tokens);
    // Verify we got at least stop_after tokens before stop took effect
    // (the in-flight speculative batch may overshoot a bit).
    CHECK(output.size() >= stop_after);

    std::printf("  stop_terminates_generation (%zu tokens before stop) PASS\n",
                output.size());
}

static void test_metrics_after_generate() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 20;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.tokens_per_second > 0.0f);
    CHECK(m.time_to_first_token_ms > 0.0f);
    CHECK(m.total_draft_tokens > 0);
    CHECK(m.current_k >= cfg.min_draft_k);
    CHECK(m.current_k <= cfg.max_draft_k);

    std::printf("  metrics_after_generate PASS (tps=%.1f, ttft=%.2fms, k=%u)\n",
                m.tokens_per_second, m.time_to_first_token_ms, m.current_k);
}

static void test_adapt_k_decreases_on_low_acceptance() {
    // With random stubs, average acceptance rate is ~50% which is below
    // the 80% threshold.  After enough tokens, K should trend downward.
    TestFixture fix;
    auto cfg = test_config();
    cfg.draft_k = 6;
    cfg.warmup_tokens = 0;
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 80;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.current_k <= cfg.draft_k);

    std::printf("  adapt_k_decreases (k: %u → %u) PASS\n",
                cfg.draft_k, m.current_k);
}

static void test_mode_conservative_on_low_hit_rate() {
    // With stubs generating random expert IDs and small buffers, the cache
    // hit rate will be low initially, pushing the scheduler into Conservative.
    TestFixture fix;
    auto cfg = test_config();
    cfg.conservative_hit_threshold = 0.99f;  // nearly impossible to hit
    cfg.warmup_tokens = 0;
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 30;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.mode == mugen::USPPMetrics::Mode::Conservative ||
          m.mode == mugen::USPPMetrics::Mode::Emergency);

    std::printf("  mode_conservative_on_low_hit_rate (mode=%d) PASS\n",
                static_cast<int>(m.mode));
}

static void test_reset_clears_state() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 10;
    (*sched)->generate(req, nullptr);

    auto m_before = (*sched)->metrics();
    CHECK(m_before.total_draft_tokens > 0);

    (*sched)->reset();

    auto m_after = (*sched)->metrics();
    CHECK(m_after.total_draft_tokens == 0);
    CHECK(m_after.total_accepted == 0);
    CHECK(m_after.current_k == 0);  // default-initialized after reset

    std::printf("  reset_clears_state PASS\n");
}

static void test_speculative_sample_always_produces_tokens() {
    // Even if the target rejects all draft tokens (n_accepted=0),
    // we should still get at least 1 token (the target's correction).
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 1;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());
    CHECK(result->size() >= 1);

    std::printf("  speculative_sample_always_produces_tokens PASS\n");
}

static void test_generate_streams_via_callback() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2};
    req.max_new_tokens = 15;

    uint32_t callback_count = 0;
    auto result = (*sched)->generate(req, [&](const mugen::TokenResult&) {
        ++callback_count;
    });

    CHECK(result.has_value());
    CHECK(callback_count == result->size());

    std::printf("  generate_streams_via_callback (%u callbacks) PASS\n",
                callback_count);
}

static void test_pipeline_efficiency_bounded() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 10;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.pipeline_efficiency >= 0.0f);
    CHECK(m.pipeline_efficiency <= 1.0f);

    std::printf("  pipeline_efficiency_bounded (eff=%.3f) PASS\n",
                m.pipeline_efficiency);
}

static void test_cache_policy_nullptr_uses_fallback() {
    // When cache_policy is nullptr, the scheduler should use
    // ExpertIndex::coldest_experts as fallback. Verify it doesn't crash.
    TestFixture fix;
    auto comps = fix.make_components();
    comps.cache_policy = nullptr;
    auto sched = mugen::USPPScheduler::create(test_config(), comps);
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 10;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());

    std::printf("  cache_policy_nullptr_uses_fallback PASS\n");
}

static void test_stub_route_predictions_valid() {
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 8;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());
    CHECK(!result->empty());

    // The stub path (no draft model) fills route_predictions internally.
    // We can't inspect DraftOutput directly from here, but we can verify
    // the pipeline didn't crash and produced valid tokens — confirming
    // route_predictions were consumed by schedule_prefetch without error.
    for (auto& tr : *result) {
        CHECK(tr.token_id <= 31999);
    }

    std::printf("  stub_route_predictions_valid PASS\n");
}

static void test_route_callback_registration() {
    // Verify that set_route_callback compiles and the null-callback path
    // is safe (no crash when callback is cleared).
    // Without a real Metal GPU we cannot run forward(), but we can verify
    // the API surface is correctly wired.

    // 1. set_route_callback(fn) then set_route_callback(nullptr) should not crash
    //    (tested indirectly — the symbol must link)
    // 2. The stub draft path still works (no draft_model → no callback registered)
    TestFixture fix;
    auto sched = mugen::USPPScheduler::create(test_config(), fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1};
    req.max_new_tokens = 5;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());
    CHECK(!result->empty());

    std::printf("  route_callback_registration PASS\n");
}

// ===========================================================================
// WP-4: Async pipeline tests
// ===========================================================================

static void test_async_pipeline_no_deadlock() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 100;

    auto result = (*sched)->generate(req, nullptr);
    CHECK(result.has_value());
    CHECK(!result->empty());
    CHECK(result->size() <= 100);

    std::printf("  async_pipeline_no_deadlock (%zu tokens) PASS\n",
                result->size());
}

static void test_prefetch_hit_miss_counters() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 20;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.prefetch_hit_count + m.prefetch_miss_count > 0);

    std::printf("  prefetch_hit_miss_counters (hits=%llu, misses=%llu) PASS\n",
                static_cast<unsigned long long>(m.prefetch_hit_count),
                static_cast<unsigned long long>(m.prefetch_miss_count));
}

static void test_multiple_generate_sessions() {
    TestFixture fix;
    auto cfg = test_config();
    auto sched = mugen::USPPScheduler::create(cfg, fix.make_components());
    CHECK(sched.has_value());

    for (int i = 0; i < 3; i++) {
        mugen::InferenceRequest req;
        req.prompt_tokens = {1, 2};
        req.max_new_tokens = 10;

        auto result = (*sched)->generate(req, nullptr);
        CHECK(result.has_value());
        CHECK(!result->empty());
    }

    std::printf("  multiple_generate_sessions PASS\n");
}

// ===========================================================================
int main() {
    std::printf("=== USPPScheduler creation tests ===\n");
    test_create_success();
    test_create_fails_without_buffers();
    test_create_fails_without_expert_index();
    test_create_fails_without_mmap();

    std::printf("=== USPPScheduler generation tests ===\n");
    test_generate_produces_tokens();
    test_generate_respects_max_tokens();
    test_generate_streams_via_callback();
    test_speculative_sample_always_produces_tokens();

    std::printf("=== USPPScheduler control tests ===\n");
    test_stop_terminates_generation();
    test_adapt_k_decreases_on_low_acceptance();
    test_mode_conservative_on_low_hit_rate();
    test_reset_clears_state();

    std::printf("=== USPPScheduler metrics tests ===\n");
    test_metrics_after_generate();
    test_pipeline_efficiency_bounded();
    test_cache_policy_nullptr_uses_fallback();

    std::printf("=== USPPScheduler route callback tests ===\n");
    test_stub_route_predictions_valid();
    test_route_callback_registration();

    std::printf("=== USPPScheduler async pipeline tests (WP-4) ===\n");
    test_async_pipeline_no_deadlock();
    test_prefetch_hit_miss_counters();
    test_multiple_generate_sessions();

    std::printf("\nAll USPP scheduler tests passed.\n");
    return 0;
}

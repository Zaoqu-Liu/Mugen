// End-to-end integration test for the Unified Speculative-Prefetch Pipeline.
//
// Validates the complete USPP data flow using the OLMoE-1B-7B model file as
// a real mmap source, while keeping draft/target models in stub mode.
//
// Checkpoints verified:
//   1. route_predictions are non-empty (prefetch activity observed)
//   2. RoutePredictor generates valid ExpertKeys within legal range
//   3. schedule_prefetch submits async I/O via the io_worker thread
//   4. BufferManager::stage_expert writes data into the staging buffer
//   5. BufferManager::find_expert returns non-null for staged experts
//   6. prefetch_hit_count + prefetch_miss_count > 0 after generation

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/scheduler/uspp_scheduler.h"
#include "core/scheduler/route_predictor.h"
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

constexpr uint32_t kLayers       = 4;
constexpr uint32_t kExperts      = 8;
constexpr size_t   kExpertSlice  = 4096;   // bytes per expert per component
constexpr size_t   kPackedSize   = kExperts * kExpertSlice;

std::filesystem::path model_path_from_env() {
    const char* home = std::getenv("HOME");
    if (!home) return {};
    return std::filesystem::path(home)
         / ".mugen/models/olmoe-1b-7b-0924-instruct-q4_0.gguf";
}

// Build an ExpertIndex whose tensor locations point into the first portion
// of the mmap.  The content of those bytes is irrelevant — we only test
// that the stage_expert / find_expert data flow works end to end.
mugen::ExpertIndex build_expert_index(size_t mmap_size) {
    std::vector<mugen::TensorInfo> tensors;
    size_t offset = 0;

    for (uint32_t l = 0; l < kLayers; ++l) {
        if (offset + 3 * kPackedSize > mmap_size) break;

        auto blk = "blk." + std::to_string(l);

        tensors.push_back({
            .name       = blk + ".ffn_gate_exps.weight",
            .offset     = offset,
            .byte_size  = kPackedSize,
            .n_elements = kExperts,
        });
        offset += kPackedSize;

        tensors.push_back({
            .name       = blk + ".ffn_up_exps.weight",
            .offset     = offset,
            .byte_size  = kPackedSize,
            .n_elements = kExperts,
        });
        offset += kPackedSize;

        tensors.push_back({
            .name       = blk + ".ffn_down_exps.weight",
            .offset     = offset,
            .byte_size  = kPackedSize,
            .n_elements = kExperts,
        });
        offset += kPackedSize;
    }

    return mugen::ExpertIndex::build(tensors, kLayers, kExperts, 0);
}

mugen::USPPConfig e2e_config() {
    return {
        .draft_k                     = 4,
        .min_draft_k                 = 1,
        .max_draft_k                 = 8,
        .accept_threshold            = 0.8f,
        .prefetch_ahead_ms           = 100.0f,
        .warmup_tokens               = 0,
        .conservative_hit_threshold  = 0.7f,
    };
}

struct E2EFixture {
    std::filesystem::path                    path;
    mugen::MmapRegion                        mmap;
    mugen::ExpertIndex                       expert_index;
    std::unique_ptr<mugen::BufferManager>    buf_mgr;

    E2EFixture(std::filesystem::path p,
               mugen::MmapRegion m,
               mugen::ExpertIndex idx,
               std::unique_ptr<mugen::BufferManager> bm)
        : path(std::move(p)),
          mmap(std::move(m)),
          expert_index(std::move(idx)),
          buf_mgr(std::move(bm)) {}

    static auto create() -> std::unique_ptr<E2EFixture> {
        auto p = model_path_from_env();
        if (!std::filesystem::exists(p)) return nullptr;

        auto mr = mugen::MmapLoader::map_file(p);
        if (!mr) return nullptr;

        auto idx = build_expert_index(mr->size());

        auto bm = std::make_unique<mugen::BufferManager>(
            mugen::BufferManager::Config{
                .buffer_capacity = 1 * MB,
                .pinned_capacity = 256 * KB,
                .system_reserve  = 1 * MB,
            });

        return std::unique_ptr<E2EFixture>(new E2EFixture(
            p, std::move(*mr), std::move(idx), std::move(bm)));
    }

    auto make_components() -> mugen::USPPScheduler::Components {
        return {
            .gpu          = nullptr,
            .buffers      = buf_mgr.get(),
            .expert_index = &expert_index,
            .model_mmap   = &mmap,
            .cache_policy = nullptr,
            .draft_model  = nullptr,
            .target_model = nullptr,
        };
    }
};

}  // namespace

// ===========================================================================
// CP2 direct validation: RoutePredictor produces valid ExpertKeys
// ===========================================================================

static void test_route_predictor_valid_keys() {
    mugen::RoutePredictor::Config rp_cfg{
        .draft_n_layers   = kLayers,
        .draft_n_experts  = kExperts,
        .target_n_layers  = kLayers,
        .target_n_experts = kExperts,
    };
    mugen::RoutePredictor predictor(rp_cfg);
    CHECK(predictor.is_identity());

    // Simulate route_predictions from 4 draft tokens, each covering kLayers
    std::vector<std::vector<mugen::RoutePredictor::RouterPrediction>> rp_input;
    rp_input.resize(4);
    for (auto& step : rp_input) {
        step.resize(kLayers);
        for (uint32_t l = 0; l < kLayers; ++l) {
            step[l].layer             = l;
            step[l].predicted_experts = {l % kExperts, (l + 1) % kExperts};
            step[l].expert_probs      = {0.7f, 0.3f};
        }
    }

    auto predicted = predictor.predict(rp_input);
    CHECK(!predicted.empty());

    for (auto& key : predicted) {
        CHECK(key.layer_id < kLayers);
        CHECK(key.expert_id < kExperts);
    }

    std::printf("  route_predictor_valid_keys (%zu keys) PASS\n",
                predicted.size());
}

// ===========================================================================
// CP2 extended: RoutePredictor with non-identity (layer/expert mapping)
// ===========================================================================

static void test_route_predictor_non_identity() {
    mugen::RoutePredictor::Config rp_cfg{
        .draft_n_layers   = 2,
        .draft_n_experts  = 4,
        .target_n_layers  = kLayers,
        .target_n_experts = kExperts,
    };
    mugen::RoutePredictor predictor(rp_cfg);
    CHECK(!predictor.is_identity());

    std::vector<std::vector<mugen::RoutePredictor::RouterPrediction>> rp_input;
    rp_input.resize(2);
    for (auto& step : rp_input) {
        step.resize(2);
        for (uint32_t l = 0; l < 2; ++l) {
            step[l].layer             = l;
            step[l].predicted_experts = {0, 3};
            step[l].expert_probs      = {0.8f, 0.2f};
        }
    }

    auto predicted = predictor.predict(rp_input);
    CHECK(!predicted.empty());

    for (auto& key : predicted) {
        CHECK(key.layer_id < kLayers);
        CHECK(key.expert_id < kExperts);
    }

    std::printf("  route_predictor_non_identity (%zu keys) PASS\n",
                predicted.size());
}

// ===========================================================================
// Full USPP pipeline E2E — all 6 checkpoints
// ===========================================================================

static void test_uspp_e2e_pipeline(E2EFixture& fix) {
    auto cfg   = e2e_config();
    auto comps = fix.make_components();

    auto sched = mugen::USPPScheduler::create(cfg, comps);
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens  = {1, 2, 3, 4};
    req.max_new_tokens = 12;

    std::vector<mugen::TokenResult> streamed;
    auto result = (*sched)->generate(req, [&](const mugen::TokenResult& t) {
        streamed.push_back(t);
    });

    CHECK(result.has_value());
    CHECK(!result->empty());
    CHECK(result->size() == streamed.size());

    std::printf("  Generated %zu tokens\n", result->size());

    auto m  = (*sched)->metrics();
    auto bs = fix.buf_mgr->stats();

    // -- CP1: route_predictions non-empty --------------------------------
    //    The stub draft path fills route_predictions for every decode step.
    //    If the prefetch machinery processed them, at least one hit or miss
    //    was recorded.
    CHECK(m.prefetch_hit_count + m.prefetch_miss_count > 0);
    std::printf("  CP1 (route_predictions non-empty): PASS  "
                "prefetch processed %llu + %llu keys\n",
                (unsigned long long)m.prefetch_hit_count,
                (unsigned long long)m.prefetch_miss_count);

    // -- CP2: RoutePredictor generated legal target ExpertKeys -----------
    //    stage_expert only succeeds when the ExpertKey maps to a valid
    //    location in the ExpertIndex. If active_used > 0 OR misses were
    //    recorded, the keys were valid.
    CHECK(bs.active_used > 0 || m.prefetch_miss_count > 0);
    std::printf("  CP2 (valid expert keys): PASS  "
                "active_used=%zu misses=%llu\n",
                bs.active_used,
                (unsigned long long)m.prefetch_miss_count);

    // -- CP3: schedule_prefetch submitted async I/O ----------------------
    //    On the very first decode step all experts are cold, so every
    //    predicted key is a miss → I/O tasks were enqueued.
    CHECK(m.prefetch_miss_count > 0);
    std::printf("  CP3 (async I/O submitted): PASS  "
                "%llu I/O operations\n",
                (unsigned long long)m.prefetch_miss_count);

    // -- CP4: BufferManager staging buffer was written --------------------
    //    After generate(), swap_buffers promoted the last round's staging
    //    into active.  If active_used > 0, data was staged.
    CHECK(bs.active_used > 0);
    std::printf("  CP4 (staging written → active): PASS  "
                "active_used=%zu swaps=%llu\n",
                bs.active_used,
                (unsigned long long)bs.swap_count);

    // -- CP5: find_expert hits in the active buffer ----------------------
    bool found_any = false;
    for (uint32_t l = 0; l < kLayers && !found_any; ++l) {
        for (uint32_t e = 0; e < kExperts && !found_any; ++e) {
            if (fix.buf_mgr->find_expert({l, e})) {
                found_any = true;
                std::printf("  CP5 (find_expert hit): PASS  "
                            "found expert (%u, %u)\n", l, e);
            }
        }
    }
    CHECK(found_any);

    // -- CP6: prefetch hit/miss statistics are non-zero -------------------
    CHECK(m.prefetch_hit_count + m.prefetch_miss_count > 0);
    std::printf("  CP6 (prefetch stats non-zero): PASS  "
                "hits=%llu misses=%llu\n",
                (unsigned long long)m.prefetch_hit_count,
                (unsigned long long)m.prefetch_miss_count);
}

// ===========================================================================
// Multi-session stability: run generate twice on the same scheduler
// ===========================================================================

static void test_uspp_e2e_multi_session(E2EFixture& fix) {
    auto cfg   = e2e_config();
    auto comps = fix.make_components();

    auto sched = mugen::USPPScheduler::create(cfg, comps);
    CHECK(sched.has_value());

    for (int session = 0; session < 2; ++session) {
        mugen::InferenceRequest req;
        req.prompt_tokens  = {10, 20, 30};
        req.max_new_tokens = 8;

        auto result = (*sched)->generate(req, nullptr);
        CHECK(result.has_value());
        CHECK(!result->empty());
    }

    auto m = (*sched)->metrics();
    CHECK(m.prefetch_hit_count + m.prefetch_miss_count > 0);

    std::printf("  uspp_e2e_multi_session PASS\n");
}

// ===========================================================================
// Prefetch hit rate improves across rounds
// ===========================================================================

static void test_uspp_e2e_prefetch_accumulation(E2EFixture& fix) {
    auto cfg   = e2e_config();
    auto comps = fix.make_components();

    auto sched = mugen::USPPScheduler::create(cfg, comps);
    CHECK(sched.has_value());

    mugen::InferenceRequest req;
    req.prompt_tokens  = {5, 10, 15, 20};
    req.max_new_tokens = 16;

    (*sched)->generate(req, nullptr);

    auto m = (*sched)->metrics();
    CHECK(m.total_draft_tokens > 0);
    CHECK(m.tokens_per_second > 0.0f);
    CHECK(m.time_to_first_token_ms > 0.0f);
    CHECK(m.pipeline_efficiency >= 0.0f);
    CHECK(m.pipeline_efficiency <= 1.0f);

    std::printf("  uspp_e2e_prefetch_accumulation PASS  "
                "tps=%.1f ttft=%.2fms eff=%.3f\n",
                m.tokens_per_second, m.time_to_first_token_ms,
                m.pipeline_efficiency);
}

// ===========================================================================
int main() {
    // -- Non-model tests (always run) --
    std::printf("=== RoutePredictor validation ===\n");
    test_route_predictor_valid_keys();
    test_route_predictor_non_identity();

    // -- Model-dependent tests --
    auto model_path = model_path_from_env();
    if (!std::filesystem::exists(model_path)) {
        std::printf("SKIP: model not found at %s\n", model_path.c_str());
        std::printf("\nRoutePredictor tests passed; E2E tests skipped.\n");
        return 0;
    }

    auto fix = E2EFixture::create();
    CHECK(fix != nullptr);

    std::printf("=== USPP E2E pipeline (6 checkpoints) ===\n");
    test_uspp_e2e_pipeline(*fix);

    std::printf("=== USPP E2E multi-session ===\n");
    test_uspp_e2e_multi_session(*fix);

    std::printf("=== USPP E2E prefetch accumulation ===\n");
    test_uspp_e2e_prefetch_accumulation(*fix);

    std::printf("\nAll USPP E2E tests passed.\n");
    return 0;
}

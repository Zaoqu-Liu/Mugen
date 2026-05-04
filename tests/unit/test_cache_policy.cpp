#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/prefetch/cache_policy.h"
#include "core/prefetch/expert_index.h"

#define MUGEN_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__,      \
                         __LINE__);                                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define MUGEN_CHECK_NEAR(a, b, eps)                                           \
    do {                                                                      \
        if (std::fabs((a) - (b)) > (eps)) {                                   \
            std::fprintf(stderr, "FAIL: |%f - %f| > %f (%s:%d)\n",           \
                         static_cast<double>(a), static_cast<double>(b),      \
                         static_cast<double>(eps), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

namespace {

constexpr uint32_t kLayers = 2;
constexpr uint32_t kExperts = 4;
constexpr uint64_t kDataOffset = 4096;
constexpr size_t kTensorBytes = 1024;

auto make_test_tensors() -> std::vector<mugen::TensorInfo> {
    std::vector<mugen::TensorInfo> tensors;
    uint64_t running_offset = 0;

    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* comp : {"gate", "up", "down"}) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                               comp + "_exps.weight";
            size_t packed_size = kTensorBytes * kExperts;
            tensors.push_back({name, running_offset, packed_size,
                               packed_size / sizeof(float)});
            running_offset += packed_size;
        }
        for (const char* comp : {"gate", "up", "down"}) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                               comp + "_shexp.weight";
            tensors.push_back({name, running_offset, kTensorBytes,
                               kTensorBytes / sizeof(float)});
            running_offset += kTensorBytes;
        }
    }
    return tensors;
}

auto make_index() -> mugen::ExpertIndex {
    return mugen::ExpertIndex::build(make_test_tensors(), kLayers, kExperts,
                                     kDataOffset);
}

// ---------- tests ------------------------------------------------------------

void test_default_config() {
    mugen::CachePolicy policy;
    auto s = policy.stats();
    MUGEN_CHECK(s.total_tokens == 0);
    MUGEN_CHECK(s.total_activations == 0);
    MUGEN_CHECK_NEAR(s.current_alpha, 0.6f, 0.001f);
    MUGEN_CHECK(s.cooccurrence_pairs == 0);
    MUGEN_CHECK(!s.topic_shift_detected);
}

void test_record_activation_cooccurrence() {
    mugen::CachePolicy policy;

    // {0,1,2} in layer 0 → C(3,2) = 3 pairs: (0,1), (0,2), (1,2)
    policy.record_activation(0, {0, 1, 2});
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 3);
    MUGEN_CHECK(policy.stats().total_activations == 3);

    // Same activation again: same pairs, counts go from 1 → 2
    policy.record_activation(0, {0, 1, 2});
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 3);
    MUGEN_CHECK(policy.stats().total_activations == 6);

    // Different layer → 1 new pair (layer 1, (0,1))
    policy.record_activation(1, {0, 1});
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 4);
}

void test_window_trimming() {
    mugen::CachePolicyConfig cfg;
    cfg.window_size = 4;
    mugen::CachePolicy policy(cfg);

    for (uint32_t t = 0; t < 6; ++t) {
        policy.record_activation(0, {t % kExperts});
        policy.advance_token();
    }

    MUGEN_CHECK(policy.stats().total_tokens == 6);
}

void test_score_frequency_component() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    for (int i = 0; i < 10; ++i) idx.record_access(0, 0);
    idx.record_access(0, 1);
    // (0,2) never accessed

    float score_0 = policy.compute_score({0, 0}, idx.status(0, 0));
    float score_1 = policy.compute_score({0, 1}, idx.status(0, 1));
    float score_2 = policy.compute_score({0, 2}, idx.status(0, 2));

    MUGEN_CHECK(score_0 > score_1);
    MUGEN_CHECK(score_1 > score_2);
}

void test_score_with_cooccurrence() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    // Experts 0 and 1 co-occur 5 times in layer 0
    for (int i = 0; i < 5; ++i) {
        policy.record_activation(0, {0, 1});
    }

    // Both accessed once so base frequency/recency are similar.
    idx.record_access(0, 0);
    idx.record_access(0, 2);

    float score_0 = policy.compute_score({0, 0}, idx.status(0, 0));
    float score_2 = policy.compute_score({0, 2}, idx.status(0, 2));

    // Expert 0 gets co-occurrence bonus with expert 1 (in window).
    MUGEN_CHECK(score_0 > score_2);
}

void test_eviction_candidates_sorted() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    for (uint32_t e = 0; e < kExperts; ++e) {
        idx.status_mut(0, e).state = mugen::ExpertStatus::State::InMemory;
    }

    for (int i = 0; i < 20; ++i) idx.record_access(0, 0);
    for (int i = 0; i < 10; ++i) idx.record_access(0, 1);
    for (int i = 0; i < 3; ++i) idx.record_access(0, 2);

    auto candidates = policy.eviction_candidates(idx, 4);
    MUGEN_CHECK(candidates.size() == 4);

    for (size_t i = 1; i < candidates.size(); ++i) {
        MUGEN_CHECK(candidates[i].score >= candidates[i - 1].score);
    }

    // Never-accessed expert should be the top eviction target.
    MUGEN_CHECK(candidates[0].key.layer_id == 0);
    MUGEN_CHECK(candidates[0].key.expert_id == 3);
}

void test_eviction_only_in_memory() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    idx.status_mut(0, 1).state = mugen::ExpertStatus::State::Pinned;

    auto candidates = policy.eviction_candidates(idx, 10);
    MUGEN_CHECK(candidates.size() == 1);
    MUGEN_CHECK(candidates[0].key.expert_id == 0);
}

void test_eviction_bytes() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;

    auto candidates = policy.eviction_candidates(idx, 1);
    MUGEN_CHECK(candidates.size() == 1);
    MUGEN_CHECK(candidates[0].bytes == kTensorBytes * 3);
}

void test_prefetch_suggestions() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    for (int i = 0; i < 10; ++i) {
        policy.record_activation(0, {0, 1});
    }

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    // (0,1) stays OnDisk

    auto suggestions = policy.prefetch_suggestions(0, {0}, idx, 5);
    MUGEN_CHECK(!suggestions.empty());
    MUGEN_CHECK(suggestions[0].layer_id == 0);
    MUGEN_CHECK(suggestions[0].expert_id == 1);
}

void test_prefetch_filters_in_memory() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    for (int i = 0; i < 10; ++i) {
        policy.record_activation(0, {0, 1});
    }

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    idx.status_mut(0, 1).state = mugen::ExpertStatus::State::InMemory;

    auto suggestions = policy.prefetch_suggestions(0, {0}, idx, 5);
    MUGEN_CHECK(suggestions.empty());
}

void test_detect_topic_shift() {
    mugen::CachePolicy policy;

    // 32 tokens with hits → stable.
    for (int i = 0; i < 32; ++i) {
        policy.record_activation(0, {0});
        policy.record_cache_result(true);
        policy.advance_token();
    }
    MUGEN_CHECK(!policy.detect_topic_shift());

    // 16 tokens with misses → hit rate crashes.
    for (int i = 0; i < 16; ++i) {
        policy.record_activation(0, {0});
        policy.record_cache_result(false);
        policy.advance_token();
    }
    MUGEN_CHECK(policy.detect_topic_shift());
}

void test_no_false_topic_shift() {
    mugen::CachePolicy policy;

    for (int i = 0; i < 48; ++i) {
        policy.record_activation(0, {0});
        policy.record_cache_result(true);
        policy.advance_token();
    }
    MUGEN_CHECK(!policy.detect_topic_shift());
}

void test_adapt_alpha_increase() {
    mugen::CachePolicy policy;
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.6f, 0.001f);

    policy.adapt_alpha(0.95f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.65f, 0.001f);

    policy.adapt_alpha(0.95f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.70f, 0.001f);
}

void test_adapt_alpha_decrease() {
    mugen::CachePolicy policy;

    policy.adapt_alpha(0.5f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.55f, 0.001f);

    policy.adapt_alpha(0.5f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.50f, 0.001f);
}

void test_adapt_alpha_clamp() {
    mugen::CachePolicy policy;

    for (int i = 0; i < 20; ++i) policy.adapt_alpha(0.99f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.9f, 0.001f);

    for (int i = 0; i < 20; ++i) policy.adapt_alpha(0.1f);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.3f, 0.001f);
}

void test_reset_for_emergency() {
    mugen::CachePolicy policy;

    // 5 activations: pair (0,1) count = 5
    for (int i = 0; i < 5; ++i) {
        policy.record_activation(0, {0, 1});
        policy.record_cache_result(true);
        policy.advance_token();
    }
    policy.adapt_alpha(0.95f);

    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 1);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.65f, 0.001f);

    policy.reset_for_emergency();

    // floor(5 / 10) = 0 → pair removed
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 0);
    MUGEN_CHECK_NEAR(policy.stats().current_alpha, 0.6f, 0.001f);
    MUGEN_CHECK(!policy.stats().topic_shift_detected);
}

void test_reset_preserves_strong_cooccurrence() {
    mugen::CachePolicyConfig cfg;
    cfg.emergency_decay = 2.0f;
    mugen::CachePolicy policy(cfg);

    // 30 activations → pair count = 30; after /2 → 15 (survives)
    for (int i = 0; i < 30; ++i) {
        policy.record_activation(0, {0, 1});
    }
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 1);

    policy.reset_for_emergency();
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 1);
}

void test_stats_hit_rate() {
    mugen::CachePolicy policy;

    // 4 hits + 1 miss = 80%
    for (int i = 0; i < 4; ++i) {
        policy.record_activation(0, {0});
        policy.record_cache_result(true);
        policy.advance_token();
    }
    policy.record_activation(0, {0});
    policy.record_cache_result(false);
    policy.advance_token();

    MUGEN_CHECK_NEAR(policy.stats().avg_hit_rate, 0.8f, 0.001f);
}

void test_empty_window_score() {
    mugen::CachePolicy policy;
    mugen::ExpertStatus status{};
    status.access_count = 5;
    status.recent_window_hits = 2;

    // recency ≈ 0 (last_access_time = 0 → huge elapsed)
    // frequency = log(6) + 2.0*2 ≈ 5.79
    // score ≈ 0.4 * 5.79 ≈ 2.32
    float score = policy.compute_score({0, 0}, status);
    MUGEN_CHECK(score > 2.0f);
    MUGEN_CHECK(score < 3.0f);
}

void test_single_expert_repeated() {
    mugen::CachePolicy policy;

    for (int i = 0; i < 100; ++i) {
        policy.record_activation(0, {0});
        policy.advance_token();
    }

    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 0);
    MUGEN_CHECK(policy.stats().total_tokens == 100);
    MUGEN_CHECK(policy.stats().total_activations == 100);
}

void test_cooccurrence_pruning() {
    mugen::CachePolicy policy;

    // C(4,2) = 6 pairs per layer; 2 layers → 12 total, each count = 1
    policy.record_activation(0, {0, 1, 2, 3});
    policy.record_activation(1, {0, 1, 2, 3});
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 12);

    // default emergency_decay = 10 → floor(1/10) = 0 → all removed
    policy.reset_for_emergency();
    MUGEN_CHECK(policy.stats().cooccurrence_pairs == 0);
}

void test_eviction_empty() {
    auto idx = make_index();
    mugen::CachePolicy policy;

    // No InMemory experts → empty result
    auto candidates = policy.eviction_candidates(idx, 10);
    MUGEN_CHECK(candidates.empty());

    // n = 0 → empty result regardless
    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    candidates = policy.eviction_candidates(idx, 0);
    MUGEN_CHECK(candidates.empty());
}

}  // namespace

int main() {
    test_default_config();
    test_record_activation_cooccurrence();
    test_window_trimming();
    test_score_frequency_component();
    test_score_with_cooccurrence();
    test_eviction_candidates_sorted();
    test_eviction_only_in_memory();
    test_eviction_bytes();
    test_prefetch_suggestions();
    test_prefetch_filters_in_memory();
    test_detect_topic_shift();
    test_no_false_topic_shift();
    test_adapt_alpha_increase();
    test_adapt_alpha_decrease();
    test_adapt_alpha_clamp();
    test_reset_for_emergency();
    test_reset_preserves_strong_cooccurrence();
    test_stats_hit_rate();
    test_empty_window_score();
    test_single_expert_repeated();
    test_cooccurrence_pruning();
    test_eviction_empty();

    std::printf("All cache_policy tests passed. (22/22)\n");
    return 0;
}

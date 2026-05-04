#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "core/scheduler/route_predictor.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

using RP = mugen::RoutePredictor::RouterPrediction;

// Helper: build a RouterPrediction for a given layer with top-k experts.
static RP make_rp(uint32_t layer,
                  std::vector<uint32_t> experts,
                  std::vector<float> probs) {
    RP rp;
    rp.layer = layer;
    rp.predicted_experts = std::move(experts);
    rp.expert_probs = std::move(probs);
    return rp;
}

// ---------------------------------------------------------------------------
// Test 1: Identity mapping (draft == target config)
// ---------------------------------------------------------------------------
static void test_identity_mapping() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 4,
        .draft_n_experts = 8,
        .target_n_layers = 4,
        .target_n_experts = 8,
    };
    mugen::RoutePredictor pred(cfg);
    CHECK(pred.is_identity());

    // 2 draft tokens × 4 layers × top-2 experts
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0, {1, 3}, {0.7f, 0.3f}),
          make_rp(1, {2, 5}, {0.6f, 0.4f}),
          make_rp(2, {0, 7}, {0.8f, 0.2f}),
          make_rp(3, {4, 6}, {0.5f, 0.5f}) },
        { make_rp(0, {1, 2}, {0.9f, 0.1f}),
          make_rp(1, {5, 3}, {0.55f, 0.45f}),
          make_rp(2, {7, 0}, {0.65f, 0.35f}),
          make_rp(3, {6, 4}, {0.52f, 0.48f}) },
    };

    auto result = pred.predict(route_preds);

    // Verify deduplication: there should be no duplicate keys.
    std::unordered_set<mugen::ExpertKey> unique_keys(result.begin(), result.end());
    CHECK(unique_keys.size() == result.size());

    // With identity mapping, layer indices should pass through unchanged.
    for (auto& key : result) {
        CHECK(key.layer_id < 4);
        CHECK(key.expert_id < 8);
    }

    // Verify priority ordering: probabilities should be non-increasing.
    // (We can't directly access the stored probs, but the first element
    //  should be the highest-prob expert across all predictions.)
    // Expert (0,1) has prob 0.9 from token 1, which is the max.
    CHECK(result[0].layer_id == 0);
    CHECK(result[0].expert_id == 1);

    std::printf("  identity_mapping PASS (%zu keys)\n", result.size());
}

// ---------------------------------------------------------------------------
// Test 2: Different layer count mapping (draft 12 → target 32)
// ---------------------------------------------------------------------------
static void test_layer_mapping() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 12,
        .draft_n_experts = 8,
        .target_n_layers = 32,
        .target_n_experts = 8,
    };
    mugen::RoutePredictor pred(cfg);
    CHECK(!pred.is_identity());

    // Single token, 3 draft layers (0, 6, 11)
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0,  {1}, {0.9f}),
          make_rp(6,  {3}, {0.8f}),
          make_rp(11, {5}, {0.7f}) },
    };

    auto result = pred.predict(route_preds);
    CHECK(result.size() == 3);

    // Expected layer mapping: target_layer = draft_layer * 32 / 12
    // draft 0  → target 0  (0 * 32 / 12 = 0)
    // draft 6  → target 16 (6 * 32 / 12 = 16)
    // draft 11 → target 29 (11 * 32 / 12 = 29)
    CHECK(result[0].layer_id == 0);   // prob 0.9
    CHECK(result[0].expert_id == 1);
    CHECK(result[1].layer_id == 16);  // prob 0.8
    CHECK(result[1].expert_id == 3);
    CHECK(result[2].layer_id == 29);  // prob 0.7
    CHECK(result[2].expert_id == 5);

    std::printf("  layer_mapping (12→32) PASS\n");
}

// ---------------------------------------------------------------------------
// Test 3: Dedup + priority sorting
// ---------------------------------------------------------------------------
static void test_dedup_and_priority() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 4,
        .draft_n_experts = 8,
        .target_n_layers = 4,
        .target_n_experts = 8,
    };
    mugen::RoutePredictor pred(cfg);

    // Two tokens predict the same expert (layer=1, expert=3) with different probs.
    // The dedup should keep it once with the higher probability.
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(1, {3, 5}, {0.4f, 0.6f}) },
        { make_rp(1, {3, 7}, {0.8f, 0.2f}) },
    };

    auto result = pred.predict(route_preds);

    // Expected unique keys: (1,3), (1,5), (1,7) — 3 keys
    CHECK(result.size() == 3);

    // Priority: (1,3) has max prob 0.8, (1,5) has 0.6, (1,7) has 0.2
    CHECK(result[0].layer_id == 1 && result[0].expert_id == 3);
    CHECK(result[1].layer_id == 1 && result[1].expert_id == 5);
    CHECK(result[2].layer_id == 1 && result[2].expert_id == 7);

    std::printf("  dedup_and_priority PASS\n");
}

// ---------------------------------------------------------------------------
// Test 4: Empty input
// ---------------------------------------------------------------------------
static void test_empty_input() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 4,
        .draft_n_experts = 8,
        .target_n_layers = 32,
        .target_n_experts = 16,
    };
    mugen::RoutePredictor pred(cfg);

    std::vector<std::vector<RP>> empty;
    auto result = pred.predict(empty);
    CHECK(result.empty());

    // Also test with non-empty outer but empty inner vectors.
    std::vector<std::vector<RP>> hollow = { {}, {} };
    auto result2 = pred.predict(hollow);
    CHECK(result2.empty());

    std::printf("  empty_input PASS\n");
}

// ---------------------------------------------------------------------------
// Test 5: Expert index clamping when target has fewer experts than draft
// ---------------------------------------------------------------------------
static void test_expert_clamping() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 4,
        .draft_n_experts = 16,
        .target_n_layers = 4,
        .target_n_experts = 8,
    };
    mugen::RoutePredictor pred(cfg);

    // Draft expert 15 exceeds target's 8 experts → should be clamped to 7.
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0, {15, 2}, {0.6f, 0.4f}) },
    };

    auto result = pred.predict(route_preds);
    CHECK(result.size() == 2);

    bool found_clamped = false;
    for (auto& key : result) {
        CHECK(key.expert_id < 8);
        if (key.expert_id == 7) found_clamped = true;
    }
    CHECK(found_clamped);

    std::printf("  expert_clamping PASS\n");
}

// ---------------------------------------------------------------------------
// Test 6: Bucket mapping — 64 draft experts → 256 target experts
// ---------------------------------------------------------------------------
static void test_bucket_mapping_64_to_256() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 4,
        .draft_n_experts = 64,
        .target_n_layers = 4,
        .target_n_experts = 256,
    };
    mugen::RoutePredictor pred(cfg);
    CHECK(!pred.is_identity());

    // Feed all 64 draft experts on layer 0, each with equal probability.
    std::vector<uint32_t> all_experts(64);
    std::vector<float> all_probs(64, 0.5f);
    for (uint32_t i = 0; i < 64; ++i) all_experts[i] = i;

    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0, all_experts, all_probs) },
    };

    auto result = pred.predict(route_preds);

    // Every draft expert should expand to 4 target experts (256/64 = 4).
    // Total: 64 * 4 = 256 unique keys covering all target experts.
    CHECK(result.size() == 256);

    std::unordered_set<uint32_t> covered;
    for (auto& key : result) {
        CHECK(key.layer_id == 0);
        CHECK(key.expert_id < 256);
        covered.insert(key.expert_id);
    }
    CHECK(covered.size() == 256);

    std::printf("  bucket_mapping (64→256) PASS — all 256 target experts covered\n");
}

// ---------------------------------------------------------------------------
// Test 7: Bucket mapping — non-divisible (3 draft → 10 target)
// ---------------------------------------------------------------------------
static void test_bucket_mapping_non_divisible() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 1,
        .draft_n_experts = 3,
        .target_n_layers = 1,
        .target_n_experts = 10,
    };
    mugen::RoutePredictor pred(cfg);

    // Draft experts 0, 1, 2 with descending probs.
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0, {0, 1, 2}, {0.9f, 0.6f, 0.3f}) },
    };

    auto result = pred.predict(route_preds);

    // Expected buckets (floor(i*10/3)):
    //   draft 0 → [0, 3)  = {0, 1, 2}     (3 experts)
    //   draft 1 → [3, 6)  = {3, 4, 5}     (3 experts)
    //   draft 2 → [6, 10) = {6, 7, 8, 9}  (4 experts)
    // Total: 10 unique keys
    CHECK(result.size() == 10);

    std::unordered_set<uint32_t> covered;
    for (auto& key : result) {
        CHECK(key.layer_id == 0);
        CHECK(key.expert_id < 10);
        covered.insert(key.expert_id);
    }
    CHECK(covered.size() == 10);

    // Highest-prob experts (from draft 0, prob 0.9) should come first.
    CHECK(result[0].expert_id < 3);

    std::printf("  bucket_mapping_non_divisible (3→10) PASS — all 10 covered\n");
}

// ---------------------------------------------------------------------------
// Test 8: Bucket mapping — single draft expert predicts a full bucket
// ---------------------------------------------------------------------------
static void test_bucket_single_expert_expansion() {
    mugen::RoutePredictor::Config cfg{
        .draft_n_layers = 2,
        .draft_n_experts = 4,
        .target_n_layers = 2,
        .target_n_experts = 16,
    };
    mugen::RoutePredictor pred(cfg);

    // Only predict draft expert 2 on layer 0 → should expand to target [8, 12).
    std::vector<std::vector<RP>> route_preds = {
        { make_rp(0, {2}, {0.95f}) },
    };

    auto result = pred.predict(route_preds);
    CHECK(result.size() == 4);

    for (auto& key : result) {
        CHECK(key.layer_id == 0);
        CHECK(key.expert_id >= 8);
        CHECK(key.expert_id < 12);
    }

    std::printf("  bucket_single_expert_expansion (4→16) PASS\n");
}

// ===========================================================================
int main() {
    std::printf("=== RoutePredictor tests ===\n");
    test_identity_mapping();
    test_layer_mapping();
    test_dedup_and_priority();
    test_empty_input();
    test_expert_clamping();
    test_bucket_mapping_64_to_256();
    test_bucket_mapping_non_divisible();
    test_bucket_single_expert_expansion();

    std::printf("\nAll RoutePredictor tests passed.\n");
    return 0;
}

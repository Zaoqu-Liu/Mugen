#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/scheduler/sampling.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_temperature_scale() {
    // temp=0.5 doubles the logits
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    mugen::Sampler::temperature_scale(logits, 0.5f);
    CHECK(std::abs(logits[0] - 2.0f) < 1e-6f);
    CHECK(std::abs(logits[1] - 4.0f) < 1e-6f);
    CHECK(std::abs(logits[2] - 6.0f) < 1e-6f);

    // temp=1.0 is identity
    logits = {1.0f, 2.0f, 3.0f};
    mugen::Sampler::temperature_scale(logits, 1.0f);
    CHECK(std::abs(logits[0] - 1.0f) < 1e-6f);
    CHECK(std::abs(logits[1] - 2.0f) < 1e-6f);
    CHECK(std::abs(logits[2] - 3.0f) < 1e-6f);

    // temp=0 → greedy: only argmax survives
    logits = {1.0f, 3.0f, 2.0f};
    mugen::Sampler::temperature_scale(logits, 0.0f);
    CHECK(logits[1] == 0.0f);
    CHECK(std::isinf(logits[0]) && logits[0] < 0.0f);
    CHECK(std::isinf(logits[2]) && logits[2] < 0.0f);

    std::printf("  temperature_scale PASS\n");
}

static void test_softmax() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    mugen::Sampler::softmax(logits);

    // All probabilities non-negative and sum to 1
    float sum = 0.0f;
    for (float p : logits) {
        CHECK(p >= 0.0f);
        sum += p;
    }
    CHECK(std::abs(sum - 1.0f) < 1e-6f);

    // Higher logit → higher probability
    CHECK(logits[2] > logits[1]);
    CHECK(logits[1] > logits[0]);

    // Hand-computed: softmax([1,2,3])
    // e^1/(e^1+e^2+e^3) ≈ 0.09003, e^2/... ≈ 0.24473, e^3/... ≈ 0.66524
    CHECK(std::abs(logits[0] - 0.09003f) < 1e-4f);
    CHECK(std::abs(logits[1] - 0.24473f) < 1e-4f);
    CHECK(std::abs(logits[2] - 0.66524f) < 1e-4f);

    // Numerical stability: large logits should not overflow
    std::vector<float> big = {1000.0f, 1001.0f, 1002.0f};
    mugen::Sampler::softmax(big);
    float big_sum = 0.0f;
    for (float p : big) big_sum += p;
    CHECK(std::abs(big_sum - 1.0f) < 1e-5f);

    std::printf("  softmax PASS\n");
}

static void test_top_p_filter() {
    // softmax([3,2,1,0,-1]) ≈ [0.636, 0.234, 0.086, 0.032, 0.012]
    // cumulative: 0.636, 0.870, ...
    // With p=0.8, top-2 (indices 0,1) capture ~87% > p → cutoff=2
    std::vector<float> logits = {3.0f, 2.0f, 1.0f, 0.0f, -1.0f};
    mugen::Sampler::top_p_filter(logits, 0.8f);

    CHECK(!std::isinf(logits[0]));
    CHECK(!std::isinf(logits[1]));
    CHECK(std::isinf(logits[2]) && logits[2] < 0.0f);
    CHECK(std::isinf(logits[3]) && logits[3] < 0.0f);
    CHECK(std::isinf(logits[4]) && logits[4] < 0.0f);

    // Surviving logits keep original values
    CHECK(std::abs(logits[0] - 3.0f) < 1e-6f);
    CHECK(std::abs(logits[1] - 2.0f) < 1e-6f);

    // p=1.0 disables filtering: all logits survive
    logits = {3.0f, 2.0f, 1.0f, 0.0f, -1.0f};
    mugen::Sampler::top_p_filter(logits, 1.0f);
    for (float l : logits) CHECK(!std::isinf(l));

    std::printf("  top_p_filter PASS\n");
}

static void test_sample_token() {
    // Determinism: same seed → same output
    mugen::Sampler s1({.seed = 42});
    mugen::Sampler s2({.seed = 42});
    std::vector<float> probs = {0.1f, 0.2f, 0.3f, 0.4f};
    auto t1 = s1.sample_token(probs);
    auto t2 = s2.sample_token(probs);
    CHECK(t1 == t2);

    // Statistical: highest-prob token most frequent over many samples
    mugen::Sampler s3({.seed = 123});
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < 10000; ++i) {
        auto tok = s3.sample_token(probs);
        CHECK(tok < 4);
        ++counts[tok];
    }
    CHECK(counts[3] > counts[0]);
    CHECK(counts[3] > counts[1]);
    CHECK(counts[3] > counts[2]);

    // Degenerate: single-token distribution
    std::vector<float> single = {0.0f, 0.0f, 1.0f, 0.0f};
    mugen::Sampler s4({.seed = 99});
    for (int i = 0; i < 100; ++i) {
        CHECK(s4.sample_token(single) == 2);
    }

    std::printf("  sample_token PASS (counts: %d %d %d %d)\n",
                counts[0], counts[1], counts[2], counts[3]);
}

static void test_speculative_accept() {
    // p_target >= p_draft → ratio >= 1 → always accept
    for (float r = 0.0f; r < 1.0f; r += 0.01f) {
        CHECK(mugen::Sampler::speculative_accept(0.3f, 0.6f, r));
        CHECK(mugen::Sampler::speculative_accept(0.5f, 0.5f, r));
    }

    // p_target < p_draft → partial rejection
    // ratio = 0.2/0.8 = 0.25
    CHECK(mugen::Sampler::speculative_accept(0.8f, 0.2f, 0.1f));   // 0.1 < 0.25
    CHECK(!mugen::Sampler::speculative_accept(0.8f, 0.2f, 0.5f));  // 0.5 > 0.25
    CHECK(!mugen::Sampler::speculative_accept(0.8f, 0.2f, 0.9f));  // 0.9 > 0.25

    // Boundary: exactly at threshold
    CHECK(!mugen::Sampler::speculative_accept(0.8f, 0.2f, 0.25f)); // not strictly less

    // Edge: p_draft=0, p_target>0 → always accept
    CHECK(mugen::Sampler::speculative_accept(0.0f, 0.5f, 0.999f));

    // Edge: p_draft=0, p_target=0 → reject
    CHECK(!mugen::Sampler::speculative_accept(0.0f, 0.0f, 0.5f));

    std::printf("  speculative_accept PASS\n");
}

static void test_correction_distribution() {
    std::vector<float> p_draft  = {0.5f, 0.3f, 0.2f};
    std::vector<float> p_target = {0.3f, 0.5f, 0.2f};

    auto corrected = mugen::Sampler::correction_distribution(p_draft, p_target);

    // max(0, 0.3-0.5)=0, max(0, 0.5-0.3)=0.2, max(0, 0.2-0.2)=0
    // normalized: [0, 1.0, 0]
    CHECK(corrected.size() == 3);
    CHECK(std::abs(corrected[0] - 0.0f) < 1e-6f);
    CHECK(std::abs(corrected[1] - 1.0f) < 1e-6f);
    CHECK(std::abs(corrected[2] - 0.0f) < 1e-6f);

    // Sum is 1
    float sum = 0.0f;
    for (float c : corrected) sum += c;
    CHECK(std::abs(sum - 1.0f) < 1e-6f);

    // More spread case: p_t everywhere > p_d
    std::vector<float> pd2 = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> pt2 = {0.2f, 0.3f, 0.3f, 0.2f};
    auto c2 = mugen::Sampler::correction_distribution(pd2, pt2);
    // max(0, 0.2-0.1)=0.1, max(0, 0.3-0.2)=0.1, max(0, 0.3-0.3)=0, max(0, 0.2-0.4)=0
    // sum=0.2, normalized: [0.5, 0.5, 0, 0]
    CHECK(std::abs(c2[0] - 0.5f) < 1e-6f);
    CHECK(std::abs(c2[1] - 0.5f) < 1e-6f);
    CHECK(std::abs(c2[2] - 0.0f) < 1e-6f);
    CHECK(std::abs(c2[3] - 0.0f) < 1e-6f);

    std::printf("  correction_distribution PASS\n");
}

static void test_speculative_sample_pipeline() {
    constexpr uint32_t vocab = 8;
    constexpr uint32_t k = 4;
    std::vector<uint32_t> draft_tokens = {0, 1, 2, 3};

    // --- Case 1: p_target/p_draft > 1 for each draft token → all accepted ---
    {
        mugen::Sampler sampler({.seed = 42});
        std::vector<std::vector<float>> draft_probs(k, std::vector<float>(vocab));
        std::vector<std::vector<float>> target_probs(k + 1, std::vector<float>(vocab));

        for (uint32_t i = 0; i < k; ++i) {
            float rest_d = 0.7f / static_cast<float>(vocab - 1);
            float rest_t = 0.5f / static_cast<float>(vocab - 1);
            for (uint32_t j = 0; j < vocab; ++j) {
                draft_probs[i][j] = (j == draft_tokens[i]) ? 0.3f : rest_d;
                target_probs[i][j] = (j == draft_tokens[i]) ? 0.5f : rest_t;
            }
        }
        float bonus_rest = 1.0f / static_cast<float>(vocab);
        for (uint32_t j = 0; j < vocab; ++j)
            target_probs[k][j] = bonus_rest;

        auto res = sampler.speculative_sample(draft_tokens, draft_probs,
                                              target_probs);
        CHECK(res.n_accepted == k);
        CHECK(res.accepted_tokens.size() == k);
        CHECK(res.accepted_logprobs.size() == k);
        for (uint32_t i = 0; i < k; ++i) {
            CHECK(res.accepted_tokens[i] == draft_tokens[i]);
            CHECK(std::abs(res.accepted_logprobs[i] - std::log(0.5f)) < 1e-5f);
        }
        CHECK(res.bonus_token < vocab);

        std::printf("  speculative_sample all-accepted (bonus=%u) PASS\n",
                    res.bonus_token);
    }

    // --- Case 2: p_target=0 for draft token → immediate rejection ---
    {
        mugen::Sampler sampler({.seed = 42});
        std::vector<std::vector<float>> draft_probs(k, std::vector<float>(vocab));
        std::vector<std::vector<float>> target_probs(k + 1, std::vector<float>(vocab));

        for (uint32_t i = 0; i < k; ++i) {
            float rest_d = 0.2f / static_cast<float>(vocab - 1);
            float rest_t = 1.0f / static_cast<float>(vocab - 1);
            for (uint32_t j = 0; j < vocab; ++j) {
                draft_probs[i][j] = (j == draft_tokens[i]) ? 0.8f : rest_d;
                target_probs[i][j] = (j == draft_tokens[i]) ? 0.0f : rest_t;
            }
        }
        float bonus_rest = 1.0f / static_cast<float>(vocab);
        for (uint32_t j = 0; j < vocab; ++j)
            target_probs[k][j] = bonus_rest;

        auto res = sampler.speculative_sample(draft_tokens, draft_probs,
                                              target_probs);
        CHECK(res.n_accepted == 0);
        CHECK(res.accepted_tokens.empty());
        CHECK(res.bonus_token < vocab);
        CHECK(res.bonus_token != draft_tokens[0]);

        std::printf("  speculative_sample all-rejected (bonus=%u) PASS\n",
                    res.bonus_token);
    }

    // --- Case 3: mixed acceptance (statistical, many runs) ---
    {
        uint32_t total_runs = 500;
        uint32_t full_accept_count = 0;
        uint32_t partial_reject_count = 0;

        for (uint32_t run = 0; run < total_runs; ++run) {
            mugen::Sampler sampler({.seed = run + 1000});
            std::vector<std::vector<float>> dp(k, std::vector<float>(vocab));
            std::vector<std::vector<float>> tp(k + 1, std::vector<float>(vocab));

            for (uint32_t i = 0; i < k; ++i) {
                float rest = 0.5f / static_cast<float>(vocab - 1);
                for (uint32_t j = 0; j < vocab; ++j) {
                    dp[i][j] = (j == draft_tokens[i]) ? 0.5f : rest;
                    tp[i][j] = (j == draft_tokens[i]) ? 0.4f : rest;
                }
            }
            float br = 1.0f / static_cast<float>(vocab);
            for (uint32_t j = 0; j < vocab; ++j) tp[k][j] = br;

            auto res = sampler.speculative_sample(draft_tokens, dp, tp);
            if (res.n_accepted == k) ++full_accept_count;
            if (res.n_accepted < k) ++partial_reject_count;

            CHECK(res.accepted_tokens.size() == res.n_accepted);
            CHECK(res.bonus_token < vocab);
        }

        // p_t/p_d = 0.4/0.5 = 0.8 per step. P(all 4 accepted) ≈ 0.8^4 ≈ 0.41
        CHECK(full_accept_count > 0);
        CHECK(partial_reject_count > 0);

        std::printf("  speculative_sample mixed (full=%u partial=%u / %u) PASS\n",
                    full_accept_count, partial_reject_count, total_runs);
    }
}

// ===========================================================================
int main() {
    std::printf("=== Sampler basic tests ===\n");
    test_temperature_scale();
    test_softmax();
    test_top_p_filter();
    test_sample_token();

    std::printf("=== Speculative sampling tests ===\n");
    test_speculative_accept();
    test_correction_distribution();
    test_speculative_sample_pipeline();

    std::printf("\nAll sampling tests passed.\n");
    return 0;
}

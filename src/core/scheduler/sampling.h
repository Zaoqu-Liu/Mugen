#pragma once

#include <cstdint>
#include <vector>
#include <random>

namespace mugen {

class Sampler {
public:
    struct SamplingParams {
        float temperature = 1.0f;     // 0 = greedy
        float top_p = 1.0f;           // 1.0 = disabled
        uint64_t seed = 0;            // 0 = random
    };

    explicit Sampler(SamplingParams params);
    Sampler() : Sampler(SamplingParams{}) {}

    // Temperature scaling: logits / T
    static void temperature_scale(std::vector<float>& logits, float temperature);

    // Top-p (nucleus) filtering: zero out tokens below cumulative p
    static void top_p_filter(std::vector<float>& logits, float p);

    // Softmax: logits → probabilities
    static void softmax(std::vector<float>& logits);

    // Sample one token from probability distribution
    auto sample_token(const std::vector<float>& probs) -> uint32_t;

    // Standard speculative sampling acceptance check (Leviathan et al. 2023)
    // Accept if random_uniform < min(1, p_target / p_draft)
    static auto speculative_accept(float p_draft, float p_target,
                                   float random_uniform) -> bool;

    // Correction distribution: max(0, p_target - p_draft), normalized
    static auto correction_distribution(const std::vector<float>& p_draft,
                                        const std::vector<float>& p_target)
        -> std::vector<float>;

    // Full speculative sampling pipeline:
    // Given K draft tokens with their probabilities and target model probabilities,
    // return accepted tokens + one extra sampled token
    struct SpeculativeResult {
        std::vector<uint32_t> accepted_tokens;
        std::vector<float> accepted_logprobs;
        uint32_t bonus_token{0};
        float bonus_logprob{0.0f};
        uint32_t n_accepted{0};
    };

    auto speculative_sample(
        const std::vector<uint32_t>& draft_tokens,
        const std::vector<std::vector<float>>& draft_probs,  // K × vocab_size
        const std::vector<std::vector<float>>& target_probs  // (K+1) × vocab_size
    ) -> SpeculativeResult;

    auto params() const -> const SamplingParams&;

private:
    SamplingParams params_;
    std::mt19937 rng_;
};

} // namespace mugen

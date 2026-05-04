#include "core/scheduler/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mugen {

Sampler::Sampler(SamplingParams params)
    : params_(params) {
    if (params_.seed != 0) {
        rng_.seed(static_cast<std::mt19937::result_type>(params_.seed));
    } else {
        rng_.seed(std::random_device{}());
    }
}

void Sampler::temperature_scale(std::vector<float>& logits, float temperature) {
    if (temperature == 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        float max_val = *max_it;
        for (auto& l : logits) {
            l = (l == max_val) ? 0.0f
                               : -std::numeric_limits<float>::infinity();
        }
        return;
    }
    for (auto& l : logits) {
        l /= temperature;
    }
}

void Sampler::softmax(std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& l : logits) {
        l = std::exp(l - max_val);
        sum += l;
    }
    for (auto& l : logits) {
        l /= sum;
    }
}

void Sampler::top_p_filter(std::vector<float>& logits, float p) {
    if (p >= 1.0f) return;

    size_t n = logits.size();
    std::vector<float> probs = logits;
    softmax(probs);

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::sort(indices.begin(), indices.end(),
              [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    size_t cutoff = n;
    for (size_t i = 0; i < n; ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    for (size_t i = cutoff; i < n; ++i) {
        logits[indices[i]] = -std::numeric_limits<float>::infinity();
    }
}

auto Sampler::sample_token(const std::vector<float>& probs) -> uint32_t {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) return static_cast<uint32_t>(i);
    }
    return static_cast<uint32_t>(probs.size() - 1);
}

auto Sampler::speculative_accept(float p_draft, float p_target,
                                 float random_uniform) -> bool {
    if (p_draft <= 0.0f) return p_target > 0.0f;
    return random_uniform < std::min(1.0f, p_target / p_draft);
}

auto Sampler::correction_distribution(const std::vector<float>& p_draft,
                                      const std::vector<float>& p_target)
    -> std::vector<float> {
    std::vector<float> corrected(p_target.size());
    float sum = 0.0f;
    for (size_t i = 0; i < corrected.size(); ++i) {
        corrected[i] = std::max(0.0f, p_target[i] - p_draft[i]);
        sum += corrected[i];
    }
    if (sum > 0.0f) {
        for (auto& c : corrected) c /= sum;
    } else {
        float u = 1.0f / static_cast<float>(corrected.size());
        for (auto& c : corrected) c = u;
    }
    return corrected;
}

auto Sampler::speculative_sample(
    const std::vector<uint32_t>& draft_tokens,
    const std::vector<std::vector<float>>& draft_probs,
    const std::vector<std::vector<float>>& target_probs) -> SpeculativeResult {

    auto k = static_cast<uint32_t>(draft_tokens.size());
    SpeculativeResult result;
    result.accepted_tokens.reserve(k);
    result.accepted_logprobs.reserve(k);

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (uint32_t i = 0; i < k; ++i) {
        uint32_t token = draft_tokens[i];
        float p_d = draft_probs[i][token];
        float p_t = target_probs[i][token];
        float r = uniform(rng_);

        if (speculative_accept(p_d, p_t, r)) {
            result.accepted_tokens.push_back(token);
            result.accepted_logprobs.push_back(std::log(p_t));
        } else {
            auto corrected = correction_distribution(draft_probs[i],
                                                     target_probs[i]);
            result.bonus_token = sample_token(corrected);
            result.bonus_logprob = std::log(target_probs[i][result.bonus_token]);
            result.n_accepted = i;
            return result;
        }
    }

    result.bonus_token = sample_token(target_probs[k]);
    result.bonus_logprob = std::log(target_probs[k][result.bonus_token]);
    result.n_accepted = k;
    return result;
}

auto Sampler::params() const -> const SamplingParams& {
    return params_;
}

} // namespace mugen

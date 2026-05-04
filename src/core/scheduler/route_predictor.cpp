#include "core/scheduler/route_predictor.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace mugen {

RoutePredictor::RoutePredictor(Config cfg) : cfg_(cfg) {}

bool RoutePredictor::is_identity() const {
    return cfg_.draft_n_layers == cfg_.target_n_layers &&
           cfg_.draft_n_experts == cfg_.target_n_experts;
}

uint32_t RoutePredictor::map_layer(uint32_t draft_layer) const {
    if (cfg_.draft_n_layers == 0) return 0;
    if (cfg_.draft_n_layers == cfg_.target_n_layers) return draft_layer;
    return draft_layer * cfg_.target_n_layers / cfg_.draft_n_layers;
}

std::pair<uint32_t, uint32_t>
RoutePredictor::map_expert_range(uint32_t draft_expert) const {
    if (cfg_.target_n_experts == 0) return {0, 0};

    if (cfg_.target_n_experts <= cfg_.draft_n_experts) {
        uint32_t mapped = (draft_expert >= cfg_.target_n_experts)
            ? cfg_.target_n_experts - 1
            : draft_expert;
        return {mapped, mapped + 1};
    }

    if (cfg_.draft_n_experts == 0) return {0, 0};

    uint64_t begin =
        static_cast<uint64_t>(draft_expert) * cfg_.target_n_experts /
        cfg_.draft_n_experts;
    uint64_t end =
        static_cast<uint64_t>(draft_expert + 1) * cfg_.target_n_experts /
        cfg_.draft_n_experts;

    if (begin >= cfg_.target_n_experts) begin = cfg_.target_n_experts - 1;
    if (end > cfg_.target_n_experts) end = cfg_.target_n_experts;
    if (end <= begin) end = begin + 1;

    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(end)};
}

auto RoutePredictor::predict(
    const std::vector<std::vector<RouterPrediction>>& route_predictions
) const -> std::vector<ExpertKey> {

    if (route_predictions.empty()) return {};

    // Accumulate max probability per unique (target_layer, target_expert) key.
    // When the same expert is predicted from multiple draft tokens/layers,
    // we keep the highest probability to determine priority.
    std::unordered_map<ExpertKey, float> best_prob;

    for (auto& per_token : route_predictions) {
        for (auto& rp : per_token) {
            uint32_t target_layer = map_layer(rp.layer);

            for (size_t i = 0; i < rp.predicted_experts.size(); ++i) {
                auto [expert_begin, expert_end] =
                    map_expert_range(rp.predicted_experts[i]);
                float prob = (i < rp.expert_probs.size())
                    ? rp.expert_probs[i] : 0.0f;

                for (uint32_t te = expert_begin; te < expert_end; ++te) {
                    ExpertKey key{target_layer, te};
                    auto it = best_prob.find(key);
                    if (it == best_prob.end())
                        best_prob.emplace(key, prob);
                    else if (prob > it->second)
                        it->second = prob;
                }
            }
        }
    }

    // Build sorted output: highest probability first.
    std::vector<std::pair<ExpertKey, float>> entries(
        best_prob.begin(), best_prob.end());

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  if (a.first.layer_id != b.first.layer_id)
                      return a.first.layer_id < b.first.layer_id;
                  return a.first.expert_id < b.first.expert_id;
              });

    std::vector<ExpertKey> result;
    result.reserve(entries.size());
    for (auto& [key, prob] : entries)
        result.push_back(key);

    return result;
}

}  // namespace mugen

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/prefetch/expert_index.h"

namespace mugen {

/// Maps draft-model routing predictions to target-model expert keys.
///
/// When draft and target models differ in layer count or expert count, the
/// predictor applies a linear layer mapping and direct expert-index forwarding
/// to translate draft routing decisions into target prefetch requests.
///
/// If draft == target configuration, the mapping degenerates to identity.
///
/// Output is deduplicated and sorted by descending expert probability so the
/// caller can submit I/O in priority order.
class RoutePredictor {
public:
    struct Config {
        uint32_t draft_n_layers = 0;
        uint32_t draft_n_experts = 0;
        uint32_t target_n_layers = 0;
        uint32_t target_n_experts = 0;
    };

    /// Per-layer routing decision from the draft model's MoE router.
    struct RouterPrediction {
        uint32_t layer;
        std::vector<uint32_t> predicted_experts;
        std::vector<float> expert_probs;
    };

    explicit RoutePredictor(Config cfg);

    /// Translate draft route_predictions into a priority-sorted list of target
    /// ExpertKeys.  Returns an empty vector if input is empty.
    ///
    /// @param route_predictions  [step][layer] router predictions from draft.
    /// @return  Deduplicated target ExpertKeys sorted by descending probability.
    auto predict(
        const std::vector<std::vector<RouterPrediction>>& route_predictions
    ) const -> std::vector<ExpertKey>;

    auto config() const -> const Config& { return cfg_; }

    /// True when draft and target have identical layer/expert counts —
    /// the mapping is a no-op identity transform.
    bool is_identity() const;

private:
    Config cfg_;

    uint32_t map_layer(uint32_t draft_layer) const;

    /// Map a draft expert ID to a [begin, end) range of target expert IDs.
    /// When target_n_experts > draft_n_experts, returns a bucket of multiple
    /// target experts; otherwise returns a single-element range for backward
    /// compatibility.
    std::pair<uint32_t, uint32_t> map_expert_range(uint32_t draft_expert) const;
};

}  // namespace mugen

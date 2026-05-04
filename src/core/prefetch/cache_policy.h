#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/prefetch/expert_index.h"

namespace mugen {

struct CachePolicyConfig {
    float alpha = 0.6f;                // recency vs frequency weight
    float decay_rate = 0.1f;           // recency exponential decay rate
    float recent_window_weight = 2.0f; // bonus for recent-window hits
    uint32_t window_size = 64;         // sliding window size (tokens)
    float cooccurrence_bonus = 0.5f;   // co-occurrence score multiplier
    float emergency_decay = 10.0f;     // divisor for emergency co-occurrence decay
};

struct EvictionCandidate {
    ExpertKey key;
    float score;     // lower = more evictable
    size_t bytes;    // reclaimable bytes
};

/// ML-inspired cache replacement policy combining recency, frequency, and
/// expert co-occurrence patterns.  Designed for MoE inference where only a
/// small subset of experts is active per token.
///
/// Thread safety: none -- caller must synchronize all mutating calls.
class CachePolicy {
public:
    explicit CachePolicy(CachePolicyConfig config = {});

    /// Record that experts were activated (selected by the router) for a layer.
    /// Called once per layer per token.  Updates co-occurrence matrix and the
    /// sliding window.
    void record_activation(uint32_t layer,
                           const std::vector<uint32_t>& expert_ids);

    /// Report whether the current token's experts were all found in cache.
    /// Call after all record_activation calls for the token, before
    /// advance_token.
    void record_cache_result(bool all_hit);

    /// Advance the sliding window by one token.  Call once per token after all
    /// layer activations have been recorded.
    void advance_token();

    /// Return the n lowest-scoring InMemory experts, sorted ascending by score.
    auto eviction_candidates(const ExpertIndex& index, size_t n) const
        -> std::vector<EvictionCandidate>;

    /// Suggest experts likely needed next based on co-occurrence with
    /// current_experts.  Only suggests experts not currently in memory.
    auto prefetch_suggestions(uint32_t layer,
                              const std::vector<uint32_t>& current_experts,
                              const ExpertIndex& index,
                              size_t max_suggestions) const
        -> std::vector<ExpertKey>;

    /// Composite retention score for a single expert.
    /// Higher score = more valuable = less likely to be evicted.
    auto compute_score(const ExpertKey& key,
                       const ExpertStatus& status) const -> float;

    /// True if a sudden drop in hit rate suggests a topic / distribution shift.
    auto detect_topic_shift() const -> bool;

    /// Clear window, rapidly decay co-occurrence, and reset hit tracking.
    void reset_for_emergency();

    /// Adjust recency / frequency balance based on observed hit rate.
    void adapt_alpha(float recent_hit_rate);

    struct Stats {
        uint64_t total_tokens;
        uint64_t total_activations;
        float avg_hit_rate;
        float current_alpha;
        bool topic_shift_detected;
        size_t cooccurrence_pairs;
    };
    auto stats() const -> Stats;

private:
    CachePolicyConfig config_;
    float current_alpha_;

    struct TokenRecord {
        uint64_t token_id;
        uint32_t layer;
        std::vector<uint32_t> expert_ids;
    };
    std::deque<TokenRecord> window_;

    struct PairKey {
        uint32_t layer;
        uint32_t expert_a;  // invariant: expert_a < expert_b
        uint32_t expert_b;
        auto operator==(const PairKey&) const -> bool = default;
    };
    struct PairKeyHash {
        auto operator()(const PairKey& k) const noexcept -> size_t;
    };
    std::unordered_map<PairKey, uint32_t, PairKeyHash> cooccurrence_;

    bool last_token_hit_ = false;
    uint32_t recent_hits_ = 0;
    uint32_t recent_total_ = 0;
    uint64_t total_tokens_ = 0;
    uint64_t total_activations_ = 0;

    static constexpr uint32_t kShiftDetectionWindow = 16;
    static constexpr size_t kMaxCooccurrencePairs = 100'000;

    std::deque<float> hit_rate_history_;

    auto current_time_ns() const -> uint64_t;

    using ActiveSet =
        std::unordered_map<uint32_t, std::unordered_set<uint32_t>>;
    auto build_window_active_set() const -> ActiveSet;
    auto compute_score_with_active(const ExpertKey& key,
                                   const ExpertStatus& status,
                                   const ActiveSet& active) const -> float;
    void prune_cooccurrence();
};

}  // namespace mugen

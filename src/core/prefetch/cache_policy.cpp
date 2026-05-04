#include "core/prefetch/cache_policy.h"

#include <algorithm>
#include <cmath>

#include <mach/mach_time.h>

namespace mugen {

namespace {

auto mach_time_to_ns(uint64_t ticks) -> uint64_t {
    static mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t i;
        mach_timebase_info(&i);
        return i;
    }();
    return ticks * info.numer / info.denom;
}

}  // namespace

// --- construction ------------------------------------------------------------

CachePolicy::CachePolicy(CachePolicyConfig config)
    : config_(config), current_alpha_(config.alpha) {}

// --- recording ---------------------------------------------------------------

void CachePolicy::record_activation(
    uint32_t layer, const std::vector<uint32_t>& expert_ids) {
    if (expert_ids.empty()) return;

    for (size_t i = 0; i < expert_ids.size(); ++i) {
        for (size_t j = i + 1; j < expert_ids.size(); ++j) {
            auto [a, b] = std::minmax(expert_ids[i], expert_ids[j]);
            ++cooccurrence_[PairKey{layer, a, b}];
        }
    }

    if (cooccurrence_.size() > kMaxCooccurrencePairs) {
        prune_cooccurrence();
    }

    total_activations_ += expert_ids.size();
    window_.push_back(TokenRecord{total_tokens_, layer, expert_ids});
}

void CachePolicy::record_cache_result(bool all_hit) {
    last_token_hit_ = all_hit;
}

void CachePolicy::advance_token() {
    if (last_token_hit_) ++recent_hits_;
    ++recent_total_;
    hit_rate_history_.push_back(last_token_hit_ ? 1.0f : 0.0f);
    last_token_hit_ = false;

    ++total_tokens_;

    uint64_t oldest =
        (total_tokens_ > config_.window_size)
            ? total_tokens_ - config_.window_size
            : 0;
    while (!window_.empty() && window_.front().token_id < oldest) {
        window_.pop_front();
    }

    while (hit_rate_history_.size() > kShiftDetectionWindow * 4) {
        hit_rate_history_.pop_front();
    }
}

// --- scoring -----------------------------------------------------------------

auto CachePolicy::compute_score(const ExpertKey& key,
                                const ExpertStatus& status) const -> float {
    auto active = build_window_active_set();
    return compute_score_with_active(key, status, active);
}

auto CachePolicy::compute_score_with_active(
    const ExpertKey& key,
    const ExpertStatus& status,
    const ActiveSet& active) const -> float {

    uint64_t now = current_time_ns();

    // Recency: exponential decay based on elapsed time.
    float elapsed_ns = (status.last_access_time > 0)
                           ? static_cast<float>(now - status.last_access_time)
                           : 1e12f;
    float recency = std::exp(-config_.decay_rate * elapsed_ns / 1e6f);

    // Frequency: log(1 + count) + recent-window bonus.
    float frequency =
        std::log1pf(static_cast<float>(status.access_count)) +
        config_.recent_window_weight *
            static_cast<float>(status.recent_window_hits);

    // Co-occurrence with experts active in the current window (same layer).
    float cooc = 0.0f;
    auto layer_it = active.find(key.layer_id);
    if (layer_it != active.end()) {
        for (uint32_t other : layer_it->second) {
            if (other == key.expert_id) continue;
            auto [a, b] = std::minmax(key.expert_id, other);
            auto it = cooccurrence_.find(PairKey{key.layer_id, a, b});
            if (it != cooccurrence_.end()) {
                cooc += static_cast<float>(it->second);
            }
        }
    }
    cooc *= config_.cooccurrence_bonus;

    return current_alpha_ * recency +
           (1.0f - current_alpha_) * frequency + cooc;
}

// --- eviction ----------------------------------------------------------------

auto CachePolicy::eviction_candidates(const ExpertIndex& index,
                                      size_t n) const
    -> std::vector<EvictionCandidate> {
    if (n == 0) return {};

    auto active = build_window_active_set();
    auto in_mem = index.experts_in_state(ExpertStatus::State::InMemory);

    std::vector<EvictionCandidate> candidates;
    candidates.reserve(in_mem.size());

    for (const auto& key : in_mem) {
        const auto& st = index.status(key.layer_id, key.expert_id);
        float score = compute_score_with_active(key, st, active);

        size_t bytes = 0;
        if (auto* loc = index.location(key.layer_id, key.expert_id)) {
            bytes = loc->total_bytes;
        }
        candidates.push_back({key, score, bytes});
    }

    auto count = std::min(n, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + static_cast<ptrdiff_t>(count),
        candidates.end(),
        [](const auto& a, const auto& b) { return a.score < b.score; });

    candidates.resize(count);
    return candidates;
}

// --- prefetch ----------------------------------------------------------------

auto CachePolicy::prefetch_suggestions(
    uint32_t layer,
    const std::vector<uint32_t>& current_experts,
    const ExpertIndex& index,
    size_t max_suggestions) const -> std::vector<ExpertKey> {
    if (max_suggestions == 0) return {};

    std::unordered_set<uint32_t> current_set(current_experts.begin(),
                                             current_experts.end());

    std::unordered_map<uint32_t, uint32_t> scores;
    for (const auto& [pair, count] : cooccurrence_) {
        if (pair.layer != layer) continue;

        uint32_t target = UINT32_MAX;
        if (current_set.contains(pair.expert_a) &&
            !current_set.contains(pair.expert_b)) {
            target = pair.expert_b;
        } else if (current_set.contains(pair.expert_b) &&
                   !current_set.contains(pair.expert_a)) {
            target = pair.expert_a;
        }
        if (target == UINT32_MAX) continue;

        scores[target] += count;
    }

    std::vector<std::pair<uint32_t, uint32_t>> sorted;
    sorted.reserve(scores.size());
    for (auto [eid, score] : scores) {
        const auto& st = index.status(layer, eid);
        if (st.state != ExpertStatus::State::InMemory &&
            st.state != ExpertStatus::State::Pinned) {
            sorted.emplace_back(eid, score);
        }
    }

    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    auto count = std::min(max_suggestions, sorted.size());
    std::vector<ExpertKey> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(ExpertKey{layer, sorted[i].first});
    }
    return result;
}

// --- topic shift detection ---------------------------------------------------

auto CachePolicy::detect_topic_shift() const -> bool {
    if (hit_rate_history_.size() < kShiftDetectionWindow * 2) return false;

    float total = 0.0f;
    for (float v : hit_rate_history_) total += v;
    float avg = total / static_cast<float>(hit_rate_history_.size());

    float recent = 0.0f;
    for (size_t i = hit_rate_history_.size() - kShiftDetectionWindow;
         i < hit_rate_history_.size(); ++i) {
        recent += hit_rate_history_[i];
    }
    float recent_avg = recent / static_cast<float>(kShiftDetectionWindow);

    return recent_avg < avg * 0.5f;
}

// --- adaptation --------------------------------------------------------------

void CachePolicy::adapt_alpha(float recent_hit_rate) {
    constexpr float kStep = 0.05f;
    constexpr float kAlphaMin = 0.3f;
    constexpr float kAlphaMax = 0.9f;

    if (recent_hit_rate > 0.9f) {
        current_alpha_ = std::min(kAlphaMax, current_alpha_ + kStep);
    } else if (recent_hit_rate < 0.7f) {
        current_alpha_ = std::max(kAlphaMin, current_alpha_ - kStep);
    }
}

void CachePolicy::reset_for_emergency() {
    window_.clear();

    float inv = 1.0f / config_.emergency_decay;
    for (auto it = cooccurrence_.begin(); it != cooccurrence_.end();) {
        it->second =
            static_cast<uint32_t>(static_cast<float>(it->second) * inv);
        if (it->second == 0) {
            it = cooccurrence_.erase(it);
        } else {
            ++it;
        }
    }

    recent_hits_ = 0;
    recent_total_ = 0;
    hit_rate_history_.clear();
    last_token_hit_ = false;
    current_alpha_ = config_.alpha;
}

// --- statistics --------------------------------------------------------------

auto CachePolicy::stats() const -> Stats {
    float avg_hit = 0.0f;
    if (!hit_rate_history_.empty()) {
        auto n = std::min(static_cast<size_t>(config_.window_size),
                          hit_rate_history_.size());
        float sum = 0.0f;
        for (size_t i = hit_rate_history_.size() - n;
             i < hit_rate_history_.size(); ++i) {
            sum += hit_rate_history_[i];
        }
        avg_hit = sum / static_cast<float>(n);
    }

    return Stats{
        .total_tokens = total_tokens_,
        .total_activations = total_activations_,
        .avg_hit_rate = avg_hit,
        .current_alpha = current_alpha_,
        .topic_shift_detected = detect_topic_shift(),
        .cooccurrence_pairs = cooccurrence_.size(),
    };
}

// --- private helpers ---------------------------------------------------------

auto CachePolicy::PairKeyHash::operator()(const PairKey& k) const noexcept
    -> size_t {
    size_t h = std::hash<uint32_t>{}(k.layer);
    h ^= std::hash<uint32_t>{}(k.expert_a) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>{}(k.expert_b) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

auto CachePolicy::current_time_ns() const -> uint64_t {
    return mach_time_to_ns(mach_absolute_time());
}

auto CachePolicy::build_window_active_set() const -> ActiveSet {
    ActiveSet active;
    for (const auto& rec : window_) {
        auto& s = active[rec.layer];
        s.insert(rec.expert_ids.begin(), rec.expert_ids.end());
    }
    return active;
}

void CachePolicy::prune_cooccurrence() {
    for (auto it = cooccurrence_.begin(); it != cooccurrence_.end();) {
        if (it->second <= 2) {
            it = cooccurrence_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace mugen

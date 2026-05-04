#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "mugen/core/types.h"

namespace mugen {

/// Lightweight tensor metadata extracted from GGUF headers.
/// Represents a single tensor's name, file offset, and byte size.
struct TensorInfo {
    std::string name;
    uint64_t offset;
    size_t byte_size;
    uint64_t n_elements;
};

/// Physical location of a single expert's weight tensors within the model file.
struct ExpertLocation {
    uint32_t layer_id;
    uint32_t expert_id;

    /// Reference to a contiguous tensor region in the model file.
    struct TensorRef {
        std::string name;
        uint64_t file_offset;
        size_t byte_size;
    };

    std::vector<TensorRef> tensors;  // gate + up + down
    size_t total_bytes = 0;
};

/// Runtime status of an expert in the caching hierarchy.
struct ExpertStatus {
    enum class State : uint8_t {
        OnDisk,
        Prefetching,
        InMemory,
        Pinned,
    };

    State state = State::OnDisk;
    float heat_score = 0.0f;
    uint64_t last_access_time = 0;
    uint32_t access_count = 0;
    uint32_t recent_window_hits = 0;
};

/// Composite key identifying a unique expert within the model.
struct ExpertKey {
    uint32_t layer_id;
    uint32_t expert_id;

    auto operator==(const ExpertKey&) const -> bool = default;
};

}  // namespace mugen

template <>
struct std::hash<mugen::ExpertKey> {
    auto operator()(const mugen::ExpertKey& k) const noexcept -> size_t {
        auto h1 = std::hash<uint32_t>{}(k.layer_id);
        auto h2 = std::hash<uint32_t>{}(k.expert_id);
        return h1 ^ (h2 << 16) ^ (h2 >> 16);
    }
};

namespace mugen {

/// Maps every MoE expert to its physical file location and tracks runtime
/// caching state. Built once from GGUF tensor metadata; the location data is
/// immutable while status fields are updated during inference.
///
/// Thread safety: the `location()` accessor is safe for concurrent reads.
/// All `status_mut()` / `record_access()` / `advance_window()` calls mutate
/// internal state — the caller is responsible for external synchronization
/// (e.g., a mutex or running these from a single scheduling thread).
class ExpertIndex {
public:
    /// Heat score tuning knobs.
    struct HeatParams {
        float alpha = 0.6f;
        float decay = 0.1f;
        float time_unit_ns = 1'000'000.0f;  // 1 ms in nanoseconds
        float recent_window_weight = 2.0f;
        float window_decay_factor = 0.9f;
    };

    /// Build the index by scanning GGUF tensor metadata.
    ///
    /// @param tensors      Tensor metadata list from GgufParser.
    /// @param n_layers     Number of MoE decoder layers in the model.
    /// @param n_experts    Number of routed experts per layer.
    /// @param data_offset  Base file offset where tensor data begins in GGUF.
    static auto build(const std::vector<TensorInfo>& tensors,
                      uint32_t n_layers,
                      uint32_t n_experts,
                      uint64_t data_offset) -> ExpertIndex;

    /// Look up the physical location of a routed expert. Returns nullptr if
    /// the (layer, expert) pair is not in the index.
    auto location(uint32_t layer, uint32_t expert) const
        -> const ExpertLocation*;

    /// Read-only view of an expert's runtime status.
    auto status(uint32_t layer, uint32_t expert) const -> const ExpertStatus&;

    /// Mutable reference to an expert's runtime status. Caller must
    /// synchronize concurrent access.
    auto status_mut(uint32_t layer, uint32_t expert) -> ExpertStatus&;

    /// Return every expert currently in the given state.
    auto experts_in_state(ExpertStatus::State state) const
        -> std::vector<ExpertKey>;

    /// Return up to `top_n` experts with the highest heat scores,
    /// sorted descending.
    auto experts_by_heat(size_t top_n) const -> std::vector<ExpertKey>;

    /// Return up to `n` experts with the lowest heat scores,
    /// excluding Pinned experts. Sorted ascending.
    auto coldest_experts(size_t n) const -> std::vector<ExpertKey>;

    auto total_experts() const -> size_t;
    auto n_layers() const -> uint32_t;
    auto n_experts_per_layer() const -> uint32_t;

    /// Total byte footprint of all indexed expert tensors.
    auto total_expert_bytes() const -> size_t;

    /// Bytes occupied by experts in the Pinned state.
    auto pinned_bytes() const -> size_t;

    /// Bytes occupied by experts in InMemory or Pinned states.
    auto in_memory_bytes() const -> size_t;

    /// Record that an expert was accessed at the current timestamp.
    /// Updates access count, recency, recent window hits, and heat score.
    void record_access(uint32_t layer, uint32_t expert);

    /// Decay recent_window_hits for all experts. Call every N tokens
    /// (typically 32–64).
    void advance_window();

    /// Persist accumulated heat statistics to a binary file.
    auto save_heat_stats(const std::filesystem::path& path) const -> bool;

    /// Restore heat statistics from a previously saved binary file.
    /// Entries for experts not in the current index are silently ignored.
    auto load_heat_stats(const std::filesystem::path& path) -> bool;

    /// Pin the hottest experts whose cumulative size fits within
    /// `max_pinned_bytes`. All previously-pinned experts are first
    /// demoted to InMemory before re-evaluation.
    void auto_pin_hot_experts(size_t max_pinned_bytes);

    /// Look up a shared expert's location by layer. Returns nullptr if absent.
    auto shared_expert_location(uint32_t layer) const
        -> const ExpertLocation*;

    HeatParams heat_params;

private:
    uint32_t n_layers_ = 0;
    uint32_t n_experts_ = 0;

    std::unordered_map<ExpertKey, ExpertLocation> locations_;
    std::unordered_map<ExpertKey, ExpertStatus> statuses_;
    std::unordered_map<uint32_t, ExpertLocation> shared_experts_;

    void recompute_heat(ExpertKey key);
    auto current_time_ns() const -> uint64_t;
};

}  // namespace mugen

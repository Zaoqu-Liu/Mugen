#include "core/prefetch/expert_index.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>

#include <mach/mach_time.h>

namespace mugen {
namespace {

constexpr uint32_t kHeatFileMagic = 0x4D475848;  // "MGXH"
constexpr uint32_t kHeatFileVersion = 1;

struct HeatFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_entries;
    uint32_t reserved;
};

struct HeatFileEntry {
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t access_count;
    float heat_score;
};

struct ParsedTensorName {
    uint32_t layer;
    std::string component;  // "gate", "up", "down"
    bool is_shared;
    int expert_idx;         // -1 if packed (all experts in one tensor)
};

auto parse_expert_tensor_name(const std::string& name)
    -> std::optional<ParsedTensorName> {
    // Pattern: blk.{L}.ffn_{gate|up|down}_exps.weight
    static const std::regex packed_re(
        R"(blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight)");
    // Pattern: blk.{L}.ffn_{gate|up|down}_shexp.weight
    static const std::regex shared_re(
        R"(blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight)");

    std::smatch m;
    if (std::regex_match(name, m, packed_re)) {
        return ParsedTensorName{
            .layer = static_cast<uint32_t>(std::stoul(m[1].str())),
            .component = m[2].str(),
            .is_shared = false,
            .expert_idx = -1,
        };
    }
    if (std::regex_match(name, m, shared_re)) {
        return ParsedTensorName{
            .layer = static_cast<uint32_t>(std::stoul(m[1].str())),
            .component = m[2].str(),
            .is_shared = true,
            .expert_idx = -1,
        };
    }
    return std::nullopt;
}

auto mach_time_to_ns(uint64_t mach_ticks) -> uint64_t {
    static mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t i;
        mach_timebase_info(&i);
        return i;
    }();
    return mach_ticks * info.numer / info.denom;
}

}  // namespace

auto ExpertIndex::build(const std::vector<TensorInfo>& tensors,
                        uint32_t n_layers,
                        uint32_t n_experts,
                        uint64_t data_offset) -> ExpertIndex {
    ExpertIndex idx;
    idx.n_layers_ = n_layers;
    idx.n_experts_ = n_experts;

    struct PackedTensor {
        std::string name;
        uint64_t offset;
        size_t byte_size;
        uint64_t n_elements;
    };

    // layer -> component -> packed tensor info
    std::unordered_map<uint32_t,
                       std::unordered_map<std::string, PackedTensor>>
        packed_tensors;

    // layer -> component -> tensor info (shared experts)
    std::unordered_map<uint32_t,
                       std::unordered_map<std::string, PackedTensor>>
        shared_tensors;

    for (const auto& t : tensors) {
        auto parsed = parse_expert_tensor_name(t.name);
        if (!parsed.has_value()) continue;

        PackedTensor pt{t.name, data_offset + t.offset, t.byte_size,
                        t.n_elements};

        if (parsed->is_shared) {
            shared_tensors[parsed->layer][parsed->component] = pt;
        } else {
            packed_tensors[parsed->layer][parsed->component] = pt;
        }
    }

    // Build routed expert locations from packed tensors.
    // Each packed tensor contains all experts concatenated along the first
    // dimension, so expert i occupies [i * slice, (i+1) * slice).
    for (auto& [layer, components] : packed_tensors) {
        auto gate_it = components.find("gate");
        auto up_it = components.find("up");
        auto down_it = components.find("down");
        if (gate_it == components.end() || up_it == components.end() ||
            down_it == components.end())
            continue;

        auto make_ref = [&](const PackedTensor& pt,
                            uint32_t expert_i) -> ExpertLocation::TensorRef {
            size_t slice = pt.byte_size / n_experts;
            return {
                .name = pt.name,
                .file_offset = pt.offset + static_cast<uint64_t>(expert_i) * slice,
                .byte_size = slice,
            };
        };

        for (uint32_t e = 0; e < n_experts; ++e) {
            ExpertKey key{layer, e};
            ExpertLocation loc;
            loc.layer_id = layer;
            loc.expert_id = e;

            auto gate_ref = make_ref(gate_it->second, e);
            auto up_ref = make_ref(up_it->second, e);
            auto down_ref = make_ref(down_it->second, e);

            loc.total_bytes =
                gate_ref.byte_size + up_ref.byte_size + down_ref.byte_size;
            loc.tensors.push_back(std::move(gate_ref));
            loc.tensors.push_back(std::move(up_ref));
            loc.tensors.push_back(std::move(down_ref));

            idx.locations_[key] = std::move(loc);
            idx.statuses_[key] = ExpertStatus{};
        }
    }

    // Build shared expert locations.
    for (auto& [layer, components] : shared_tensors) {
        auto gate_it = components.find("gate");
        auto up_it = components.find("up");
        auto down_it = components.find("down");
        if (gate_it == components.end() || up_it == components.end() ||
            down_it == components.end())
            continue;

        ExpertLocation loc;
        loc.layer_id = layer;
        loc.expert_id = UINT32_MAX;  // sentinel for shared

        auto add_ref = [&](const PackedTensor& pt) {
            loc.tensors.push_back({
                .name = pt.name,
                .file_offset = pt.offset,
                .byte_size = pt.byte_size,
            });
            loc.total_bytes += pt.byte_size;
        };

        add_ref(gate_it->second);
        add_ref(up_it->second);
        add_ref(down_it->second);

        idx.shared_experts_[layer] = std::move(loc);
    }

    return idx;
}

auto ExpertIndex::location(uint32_t layer, uint32_t expert) const
    -> const ExpertLocation* {
    auto it = locations_.find(ExpertKey{layer, expert});
    return it != locations_.end() ? &it->second : nullptr;
}

auto ExpertIndex::status(uint32_t layer, uint32_t expert) const
    -> const ExpertStatus& {
    static const ExpertStatus empty{};
    auto it = statuses_.find(ExpertKey{layer, expert});
    return it != statuses_.end() ? it->second : empty;
}

auto ExpertIndex::status_mut(uint32_t layer, uint32_t expert)
    -> ExpertStatus& {
    return statuses_[ExpertKey{layer, expert}];
}

auto ExpertIndex::experts_in_state(ExpertStatus::State state) const
    -> std::vector<ExpertKey> {
    std::vector<ExpertKey> result;
    for (const auto& [key, s] : statuses_) {
        if (s.state == state) result.push_back(key);
    }
    return result;
}

auto ExpertIndex::experts_by_heat(size_t top_n) const
    -> std::vector<ExpertKey> {
    std::vector<std::pair<ExpertKey, float>> scored;
    scored.reserve(statuses_.size());
    for (const auto& [key, s] : statuses_) {
        scored.emplace_back(key, s.heat_score);
    }

    auto n = std::min(top_n, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + static_cast<ptrdiff_t>(n),
                      scored.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });

    std::vector<ExpertKey> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(scored[i].first);
    }
    return result;
}

auto ExpertIndex::coldest_experts(size_t n) const -> std::vector<ExpertKey> {
    std::vector<std::pair<ExpertKey, float>> scored;
    for (const auto& [key, s] : statuses_) {
        if (s.state != ExpertStatus::State::Pinned) {
            scored.emplace_back(key, s.heat_score);
        }
    }

    auto count = std::min(n, scored.size());
    std::partial_sort(scored.begin(),
                      scored.begin() + static_cast<ptrdiff_t>(count),
                      scored.end(),
                      [](const auto& a, const auto& b) {
                          return a.second < b.second;
                      });

    std::vector<ExpertKey> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(scored[i].first);
    }
    return result;
}

auto ExpertIndex::total_experts() const -> size_t {
    return locations_.size();
}

auto ExpertIndex::n_layers() const -> uint32_t { return n_layers_; }

auto ExpertIndex::n_experts_per_layer() const -> uint32_t {
    return n_experts_;
}

auto ExpertIndex::total_expert_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& [_, loc] : locations_) {
        total += loc.total_bytes;
    }
    return total;
}

auto ExpertIndex::pinned_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& [key, s] : statuses_) {
        if (s.state == ExpertStatus::State::Pinned) {
            auto it = locations_.find(key);
            if (it != locations_.end()) total += it->second.total_bytes;
        }
    }
    return total;
}

auto ExpertIndex::in_memory_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& [key, s] : statuses_) {
        if (s.state == ExpertStatus::State::InMemory ||
            s.state == ExpertStatus::State::Pinned) {
            auto it = locations_.find(key);
            if (it != locations_.end()) total += it->second.total_bytes;
        }
    }
    return total;
}

void ExpertIndex::record_access(uint32_t layer, uint32_t expert) {
    ExpertKey key{layer, expert};
    auto it = statuses_.find(key);
    if (it == statuses_.end()) return;

    auto& s = it->second;
    s.last_access_time = current_time_ns();
    s.access_count++;
    s.recent_window_hits++;
    recompute_heat(key);
}

void ExpertIndex::advance_window() {
    for (auto& [key, s] : statuses_) {
        s.recent_window_hits = static_cast<uint32_t>(
            static_cast<float>(s.recent_window_hits) *
            heat_params.window_decay_factor);
        recompute_heat(key);
    }
}

auto ExpertIndex::save_heat_stats(const std::filesystem::path& path) const
    -> bool {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    HeatFileHeader header{};
    header.magic = kHeatFileMagic;
    header.version = kHeatFileVersion;
    header.n_entries = static_cast<uint32_t>(statuses_.size());
    header.reserved = 0;

    f.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto& [key, s] : statuses_) {
        HeatFileEntry entry{};
        entry.layer_id = key.layer_id;
        entry.expert_id = key.expert_id;
        entry.access_count = s.access_count;
        entry.heat_score = s.heat_score;
        f.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }

    return f.good();
}

auto ExpertIndex::load_heat_stats(const std::filesystem::path& path) -> bool {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    HeatFileHeader header{};
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!f) return false;

    if (header.magic != kHeatFileMagic || header.version != kHeatFileVersion)
        return false;

    for (uint32_t i = 0; i < header.n_entries; ++i) {
        HeatFileEntry entry{};
        f.read(reinterpret_cast<char*>(&entry), sizeof(entry));
        if (!f) return false;

        ExpertKey key{entry.layer_id, entry.expert_id};
        auto it = statuses_.find(key);
        if (it != statuses_.end()) {
            it->second.access_count = entry.access_count;
            it->second.heat_score = entry.heat_score;
        }
    }

    return true;
}

void ExpertIndex::auto_pin_hot_experts(size_t max_pinned_bytes) {
    // Demote all currently-pinned experts to InMemory first.
    for (auto& [_, s] : statuses_) {
        if (s.state == ExpertStatus::State::Pinned) {
            s.state = ExpertStatus::State::InMemory;
        }
    }

    auto ranked = experts_by_heat(statuses_.size());

    size_t budget = 0;
    for (const auto& key : ranked) {
        auto loc_it = locations_.find(key);
        if (loc_it == locations_.end()) continue;

        size_t expert_bytes = loc_it->second.total_bytes;
        if (budget + expert_bytes > max_pinned_bytes) break;

        statuses_[key].state = ExpertStatus::State::Pinned;
        budget += expert_bytes;
    }
}

auto ExpertIndex::shared_expert_location(uint32_t layer) const
    -> const ExpertLocation* {
    auto it = shared_experts_.find(layer);
    return it != shared_experts_.end() ? &it->second : nullptr;
}

void ExpertIndex::recompute_heat(ExpertKey key) {
    auto it = statuses_.find(key);
    if (it == statuses_.end()) return;

    auto& s = it->second;
    uint64_t now = current_time_ns();

    float elapsed_ns =
        (s.last_access_time > 0)
            ? static_cast<float>(now - s.last_access_time)
            : 1e12f;  // never accessed → large elapsed

    float recency =
        std::exp(-heat_params.decay * elapsed_ns / heat_params.time_unit_ns);

    float frequency = std::log1p(static_cast<float>(s.access_count)) +
                      static_cast<float>(s.recent_window_hits) *
                          heat_params.recent_window_weight;

    s.heat_score =
        heat_params.alpha * recency + (1.0f - heat_params.alpha) * frequency;
}

auto ExpertIndex::current_time_ns() const -> uint64_t {
    return mach_time_to_ns(mach_absolute_time());
}

}  // namespace mugen

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/prefetch/expert_index.h"

#define MUGEN_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__,      \
                         __LINE__);                                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define MUGEN_CHECK_NEAR(a, b, eps)                                           \
    do {                                                                      \
        if (std::fabs((a) - (b)) > (eps)) {                                   \
            std::fprintf(stderr, "FAIL: |%f - %f| > %f (%s:%d)\n",           \
                         static_cast<double>(a), static_cast<double>(b),      \
                         static_cast<double>(eps), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

namespace {

constexpr uint32_t kLayers = 2;
constexpr uint32_t kExperts = 4;
constexpr uint64_t kDataOffset = 4096;
constexpr size_t kTensorBytes = 1024;

auto make_test_tensors() -> std::vector<mugen::TensorInfo> {
    std::vector<mugen::TensorInfo> tensors;
    uint64_t running_offset = 0;

    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* comp : {"gate", "up", "down"}) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                               comp + "_exps.weight";
            size_t packed_size = kTensorBytes * kExperts;
            tensors.push_back({name, running_offset, packed_size,
                               packed_size / sizeof(float)});
            running_offset += packed_size;
        }

        for (const char* comp : {"gate", "up", "down"}) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                               comp + "_shexp.weight";
            tensors.push_back({name, running_offset, kTensorBytes,
                               kTensorBytes / sizeof(float)});
            running_offset += kTensorBytes;
        }
    }

    // Non-expert tensors that should be ignored.
    tensors.push_back({"blk.0.attn_q.weight", running_offset, 2048, 512});
    running_offset += 2048;
    tensors.push_back({"blk.0.attn_k.weight", running_offset, 2048, 512});

    return tensors;
}

void test_build_and_dimensions() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    MUGEN_CHECK(idx.n_layers() == kLayers);
    MUGEN_CHECK(idx.n_experts_per_layer() == kExperts);
    MUGEN_CHECK(idx.total_experts() == kLayers * kExperts);
}

void test_location_lookup() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    auto* loc = idx.location(0, 0);
    MUGEN_CHECK(loc != nullptr);
    MUGEN_CHECK(loc->layer_id == 0);
    MUGEN_CHECK(loc->expert_id == 0);
    MUGEN_CHECK(loc->tensors.size() == 3);
    MUGEN_CHECK(loc->total_bytes == kTensorBytes * 3);

    // Each tensor slice should be kTensorBytes for a single expert.
    for (const auto& ref : loc->tensors) {
        MUGEN_CHECK(ref.byte_size == kTensorBytes);
    }

    // Expert 2 in layer 1 should also exist.
    auto* loc2 = idx.location(1, 2);
    MUGEN_CHECK(loc2 != nullptr);
    MUGEN_CHECK(loc2->layer_id == 1);
    MUGEN_CHECK(loc2->expert_id == 2);

    // Out-of-range expert returns nullptr.
    MUGEN_CHECK(idx.location(0, kExperts) == nullptr);
    MUGEN_CHECK(idx.location(kLayers, 0) == nullptr);
}

void test_shared_expert_lookup() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    auto* sh0 = idx.shared_expert_location(0);
    MUGEN_CHECK(sh0 != nullptr);
    MUGEN_CHECK(sh0->tensors.size() == 3);
    MUGEN_CHECK(sh0->total_bytes == kTensorBytes * 3);

    auto* sh1 = idx.shared_expert_location(1);
    MUGEN_CHECK(sh1 != nullptr);

    MUGEN_CHECK(idx.shared_expert_location(99) == nullptr);
}

void test_initial_status() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    const auto& s = idx.status(0, 0);
    MUGEN_CHECK(s.state == mugen::ExpertStatus::State::OnDisk);
    MUGEN_CHECK(s.heat_score == 0.0f);
    MUGEN_CHECK(s.access_count == 0);
    MUGEN_CHECK(s.recent_window_hits == 0);
}

void test_record_access_updates_heat() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    idx.record_access(0, 0);

    const auto& s = idx.status(0, 0);
    MUGEN_CHECK(s.access_count == 1);
    MUGEN_CHECK(s.recent_window_hits == 1);
    MUGEN_CHECK(s.heat_score > 0.0f);
    MUGEN_CHECK(s.last_access_time > 0);

    float first_score = s.heat_score;

    idx.record_access(0, 0);
    MUGEN_CHECK(idx.status(0, 0).access_count == 2);
    MUGEN_CHECK(idx.status(0, 0).heat_score >= first_score);
}

void test_advance_window_decays() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    for (int i = 0; i < 10; ++i) idx.record_access(0, 0);

    uint32_t hits_before = idx.status(0, 0).recent_window_hits;
    MUGEN_CHECK(hits_before == 10);

    idx.advance_window();

    uint32_t hits_after = idx.status(0, 0).recent_window_hits;
    MUGEN_CHECK(hits_after < hits_before);
    // 10 * 0.9 = 9 (truncated to uint32_t)
    MUGEN_CHECK(hits_after == 9);
}

void test_experts_by_heat() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    for (int i = 0; i < 20; ++i) idx.record_access(0, 0);
    for (int i = 0; i < 10; ++i) idx.record_access(0, 1);
    for (int i = 0; i < 5; ++i) idx.record_access(1, 0);

    auto top = idx.experts_by_heat(3);
    MUGEN_CHECK(top.size() == 3);
    // Hottest expert should be (0,0) with 20 accesses.
    MUGEN_CHECK(top[0].layer_id == 0);
    MUGEN_CHECK(top[0].expert_id == 0);
}

void test_coldest_experts() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    idx.record_access(0, 0);
    idx.record_access(0, 1);

    auto cold = idx.coldest_experts(2);
    MUGEN_CHECK(cold.size() == 2);
    // The coldest should have heat_score == 0 (never accessed).
    MUGEN_CHECK(idx.status(cold[0].layer_id, cold[0].expert_id).heat_score ==
                0.0f);
}

void test_experts_in_state() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    auto on_disk = idx.experts_in_state(mugen::ExpertStatus::State::OnDisk);
    MUGEN_CHECK(on_disk.size() == kLayers * kExperts);

    auto in_mem =
        idx.experts_in_state(mugen::ExpertStatus::State::InMemory);
    MUGEN_CHECK(in_mem.empty());

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    in_mem = idx.experts_in_state(mugen::ExpertStatus::State::InMemory);
    MUGEN_CHECK(in_mem.size() == 1);
}

void test_auto_pin_hot_experts() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    for (int i = 0; i < 50; ++i) idx.record_access(0, 0);
    for (int i = 0; i < 30; ++i) idx.record_access(0, 1);
    for (int i = 0; i < 10; ++i) idx.record_access(1, 0);

    // Budget for exactly 2 experts (each 3 * kTensorBytes = 3072 bytes).
    size_t budget = 3 * kTensorBytes * 2;
    idx.auto_pin_hot_experts(budget);

    auto pinned = idx.experts_in_state(mugen::ExpertStatus::State::Pinned);
    MUGEN_CHECK(pinned.size() == 2);
    MUGEN_CHECK(idx.pinned_bytes() == budget);
}

void test_total_expert_bytes() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    size_t expected = kLayers * kExperts * kTensorBytes * 3;
    MUGEN_CHECK(idx.total_expert_bytes() == expected);
}

void test_in_memory_bytes() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    MUGEN_CHECK(idx.in_memory_bytes() == 0);

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    MUGEN_CHECK(idx.in_memory_bytes() == kTensorBytes * 3);

    idx.status_mut(0, 1).state = mugen::ExpertStatus::State::Pinned;
    MUGEN_CHECK(idx.in_memory_bytes() == kTensorBytes * 3 * 2);
}

void test_heat_persistence_roundtrip() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    for (int i = 0; i < 15; ++i) idx.record_access(0, 0);
    for (int i = 0; i < 7; ++i) idx.record_access(1, 2);

    float score_0_0 = idx.status(0, 0).heat_score;
    float score_1_2 = idx.status(1, 2).heat_score;
    uint32_t count_0_0 = idx.status(0, 0).access_count;
    uint32_t count_1_2 = idx.status(1, 2).access_count;

    auto tmp = std::filesystem::temp_directory_path() / "mugen_test_heat.bin";
    MUGEN_CHECK(idx.save_heat_stats(tmp));

    // Build a fresh index and load stats into it.
    auto idx2 = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                          kDataOffset);
    MUGEN_CHECK(idx2.status(0, 0).access_count == 0);

    MUGEN_CHECK(idx2.load_heat_stats(tmp));
    MUGEN_CHECK(idx2.status(0, 0).access_count == count_0_0);
    MUGEN_CHECK(idx2.status(1, 2).access_count == count_1_2);
    MUGEN_CHECK_NEAR(idx2.status(0, 0).heat_score, score_0_0, 0.001f);
    MUGEN_CHECK_NEAR(idx2.status(1, 2).heat_score, score_1_2, 0.001f);

    // Nonexistent file returns false.
    MUGEN_CHECK(!idx2.load_heat_stats("/tmp/nonexistent_file_xyz.bin"));

    std::filesystem::remove(tmp);
}

void test_load_rejects_bad_magic() {
    auto tmp = std::filesystem::temp_directory_path() / "mugen_bad_magic.bin";
    {
        std::ofstream f(tmp, std::ios::binary);
        uint32_t bad_magic = 0xDEADBEEF;
        f.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    }

    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);
    MUGEN_CHECK(!idx.load_heat_stats(tmp));

    std::filesystem::remove(tmp);
}

void test_file_offsets_include_data_offset() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    auto* loc = idx.location(0, 0);
    MUGEN_CHECK(loc != nullptr);
    // First expert in first layer: offset = data_offset + 0 (tensor offset) + 0 (expert slice)
    MUGEN_CHECK(loc->tensors[0].file_offset >= kDataOffset);
}

void test_status_mut_modifies_state() {
    auto tensors = make_test_tensors();
    auto idx = mugen::ExpertIndex::build(tensors, kLayers, kExperts,
                                         kDataOffset);

    MUGEN_CHECK(idx.status(0, 0).state == mugen::ExpertStatus::State::OnDisk);

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::Prefetching;
    MUGEN_CHECK(idx.status(0, 0).state ==
                mugen::ExpertStatus::State::Prefetching);

    idx.status_mut(0, 0).state = mugen::ExpertStatus::State::InMemory;
    MUGEN_CHECK(idx.status(0, 0).state ==
                mugen::ExpertStatus::State::InMemory);
}

}  // namespace

int main() {
    test_build_and_dimensions();
    test_location_lookup();
    test_shared_expert_lookup();
    test_initial_status();
    test_record_access_updates_heat();
    test_advance_window_decays();
    test_experts_by_heat();
    test_coldest_experts();
    test_experts_in_state();
    test_auto_pin_hot_experts();
    test_total_expert_bytes();
    test_in_memory_bytes();
    test_heat_persistence_roundtrip();
    test_load_rejects_bad_magic();
    test_file_offsets_include_data_offset();
    test_status_mut_modifies_state();

    std::printf("All expert_index tests passed. (16/16)\n");
    return 0;
}

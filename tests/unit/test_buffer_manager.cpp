#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "core/memory/buffer_manager.h"
#include "core/memory/mmap_loader.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
constexpr size_t TEST_BUFFER_CAP = 64 * MB;
constexpr size_t TEST_PINNED_CAP = 16 * MB;

std::filesystem::path write_temp(const void* data, size_t len,
                                 const char* suffix) {
    auto p = std::filesystem::temp_directory_path()
           / ("mugen_buf_" + std::string(suffix) + ".bin");
    FILE* fp = std::fopen(p.c_str(), "wb");
    CHECK(fp != nullptr);
    CHECK(std::fwrite(data, 1, len, fp) == len);
    std::fclose(fp);
    return p;
}

struct TestMmap {
    std::filesystem::path path;
    mugen::MmapRegion region;
};

auto make_test_mmap(size_t size) -> TestMmap {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);

    auto p = write_temp(data.data(), data.size(), "experts");
    auto result = mugen::MmapLoader::map_file(p);
    CHECK(result.has_value());
    return {p, std::move(*result)};
}

mugen::ExpertLocation make_location(uint32_t layer, uint32_t expert,
                                     uint64_t offset, size_t total) {
    mugen::ExpertLocation loc;
    loc.layer_id  = layer;
    loc.expert_id = expert;

    size_t third = total / 3;
    loc.tensors.push_back({.name = "gate",
                           .file_offset = offset,
                           .byte_size = third});
    loc.tensors.push_back({.name = "up",
                           .file_offset = offset + third,
                           .byte_size = third});
    loc.tensors.push_back({.name = "down",
                           .file_offset = offset + 2 * third,
                           .byte_size = total - 2 * third});
    loc.total_bytes = total;
    return loc;
}

mugen::BufferManager::Config test_config(
        size_t buf_cap = TEST_BUFFER_CAP,
        size_t pin_cap = TEST_PINNED_CAP) {
    return {
        .buffer_capacity = buf_cap,
        .pinned_capacity = pin_cap,
        .system_reserve  = 1 * MB,
    };
}

}  // namespace

// ===========================================================================
// ExpertBuffer unit tests
// ===========================================================================

static void test_buffer_initial_state() {
    mugen::ExpertBuffer buf(1 * MB);
    CHECK(buf.capacity() == 1 * MB);
    CHECK(buf.used_bytes() == 0);
    CHECK(buf.free_bytes() == 1 * MB);
    CHECK(buf.slot_count() == 0);
    std::printf("  buffer_initial_state PASS\n");
}

static void test_buffer_allocate_and_find() {
    mugen::ExpertBuffer buf(1 * MB);

    mugen::ExpertKey k1{.layer_id = 0, .expert_id = 1};
    auto* slot = buf.allocate(k1, 4096);
    CHECK(slot != nullptr);
    CHECK(slot->key == k1);
    CHECK(slot->offset == 0);
    CHECK(slot->size == 4096);
    CHECK(!slot->populated);
    CHECK(buf.used_bytes() == 4096);
    CHECK(buf.slot_count() == 1);

    auto* found = buf.find(k1);
    CHECK(found != nullptr);
    CHECK(found->key == k1);

    mugen::ExpertKey k_absent{.layer_id = 99, .expert_id = 99};
    CHECK(buf.find(k_absent) == nullptr);

    std::printf("  buffer_allocate_and_find PASS\n");
}

static void test_buffer_duplicate_key_rejected() {
    mugen::ExpertBuffer buf(1 * MB);
    mugen::ExpertKey k{0, 0};
    CHECK(buf.allocate(k, 1024) != nullptr);
    CHECK(buf.allocate(k, 1024) == nullptr);
    std::printf("  buffer_duplicate_key_rejected PASS\n");
}

static void test_buffer_sequential_bump() {
    mugen::ExpertBuffer buf(1 * MB);
    mugen::ExpertKey k1{0, 0}, k2{0, 1}, k3{0, 2};

    auto* s1 = buf.allocate(k1, 1024);
    auto* s2 = buf.allocate(k2, 2048);
    auto* s3 = buf.allocate(k3, 512);

    CHECK(s1 && s2 && s3);
    CHECK(s1->offset == 0);
    CHECK(s2->offset == 1024);
    CHECK(s3->offset == 1024 + 2048);
    CHECK(buf.used_bytes() == 1024 + 2048 + 512);
    CHECK(buf.slot_count() == 3);

    std::printf("  buffer_sequential_bump PASS\n");
}

static void test_buffer_data_readwrite() {
    mugen::ExpertBuffer buf(1 * MB);
    mugen::ExpertKey k{1, 2};
    auto* slot = buf.allocate(k, 256);
    CHECK(slot != nullptr);

    auto* w = static_cast<uint8_t*>(buf.data_ptr_mut(*slot));
    for (int i = 0; i < 256; ++i) w[i] = static_cast<uint8_t>(i);

    auto* r = static_cast<const uint8_t*>(buf.data_ptr(*slot));
    for (int i = 0; i < 256; ++i) CHECK(r[i] == static_cast<uint8_t>(i));

    std::printf("  buffer_data_readwrite PASS\n");
}

static void test_buffer_clear() {
    mugen::ExpertBuffer buf(1 * MB);
    mugen::ExpertKey k{0, 0};
    buf.allocate(k, 4096);

    buf.clear();
    CHECK(buf.used_bytes() == 0);
    CHECK(buf.slot_count() == 0);
    CHECK(buf.free_bytes() == 1 * MB);
    CHECK(buf.find(k) == nullptr);

    auto* slot = buf.allocate(k, 2048);
    CHECK(slot != nullptr);
    CHECK(slot->offset == 0);

    std::printf("  buffer_clear PASS\n");
}

static void test_buffer_full_rejection() {
    mugen::ExpertBuffer buf(4096);
    mugen::ExpertKey k1{0, 0};
    CHECK(buf.allocate(k1, 4096) != nullptr);

    mugen::ExpertKey k2{0, 1};
    CHECK(buf.allocate(k2, 1) == nullptr);

    std::printf("  buffer_full_rejection PASS\n");
}

static void test_buffer_deallocate() {
    mugen::ExpertBuffer buf(1 * MB);
    mugen::ExpertKey k{0, 0};
    buf.allocate(k, 4096);

    buf.deallocate(k);
    CHECK(buf.find(k) == nullptr);
    CHECK(buf.slot_count() == 0);
    CHECK(buf.used_bytes() == 4096);   // bump: space not reclaimed

    std::printf("  buffer_deallocate PASS\n");
}

// ===========================================================================
// BufferManager unit tests
// ===========================================================================

static void test_manager_initial_state() {
    mugen::BufferManager mgr(test_config());
    auto s = mgr.stats();
    CHECK(s.active_used == 0);
    CHECK(s.staging_used == 0);
    CHECK(s.pinned_used == 0);
    CHECK(s.total_capacity == TEST_BUFFER_CAP * 2 + TEST_PINNED_CAP);
    CHECK(s.swap_count == 0);
    std::printf("  manager_initial_state PASS\n");
}

static void test_manager_swap_promotes_staging() {
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k{0, 0};
    mgr.staging_buffer().allocate(k, 4096);
    CHECK(mgr.staging_buffer().find(k) != nullptr);

    mgr.swap_buffers();

    CHECK(mgr.active_buffer().find(k) != nullptr);
    CHECK(mgr.staging_buffer().used_bytes() == 0);
    CHECK(mgr.staging_buffer().find(k) == nullptr);
    CHECK(mgr.stats().swap_count == 1);

    std::printf("  manager_swap_promotes_staging PASS\n");
}

static void test_manager_double_swap() {
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k1{0, 0};
    mgr.staging_buffer().allocate(k1, 4096);
    mgr.swap_buffers();   // k1 → active

    mugen::ExpertKey k2{1, 0};
    mgr.staging_buffer().allocate(k2, 2048);
    mgr.swap_buffers();   // k2 → active, old active (k1) → staging → cleared

    CHECK(mgr.active_buffer().find(k2) != nullptr);
    CHECK(mgr.active_buffer().find(k1) == nullptr);
    CHECK(mgr.staging_buffer().used_bytes() == 0);
    CHECK(mgr.stats().swap_count == 2);

    std::printf("  manager_double_swap PASS\n");
}

static void test_stage_expert() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k{0, 0};
    auto loc = make_location(0, 0, 0, 3000);

    CHECK(mgr.stage_expert(k, mmap, loc));

    auto* slot = mgr.staging_buffer().find(k);
    CHECK(slot != nullptr);
    CHECK(slot->populated);
    CHECK(slot->size == 3000);

    auto* got = static_cast<const uint8_t*>(
        mgr.staging_buffer().data_ptr(*slot));
    auto* expected = static_cast<const uint8_t*>(mmap.data());
    CHECK(std::memcmp(got, expected, 3000) == 0);

    std::filesystem::remove(path);
    std::printf("  stage_expert PASS\n");
}

static void test_stage_expert_buffer_full() {
    auto [path, mmap] = make_test_mmap(1 * MB);

    mugen::BufferManager mgr(test_config(4096, 4096));

    mugen::ExpertKey k{0, 0};
    auto loc = make_location(0, 0, 0, 8192);
    CHECK(!mgr.stage_expert(k, mmap, loc));

    std::filesystem::remove(path);
    std::printf("  stage_expert_buffer_full PASS\n");
}

static void test_pin_expert() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k{0, 0};
    auto loc = make_location(0, 0, 0, 6000);
    CHECK(mgr.pin_expert(k, mmap, loc));

    auto* slot = mgr.pinned_buffer().find(k);
    CHECK(slot != nullptr);
    CHECK(slot->populated);

    std::filesystem::remove(path);
    std::printf("  pin_expert PASS\n");
}

static void test_find_expert_pinned_first() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k_pinned{0, 0};
    mugen::ExpertKey k_active{1, 0};
    mugen::ExpertKey k_absent{99, 99};

    CHECK(mgr.pin_expert(k_pinned, mmap, make_location(0, 0, 0, 3000)));

    CHECK(mgr.stage_expert(k_active, mmap, make_location(1, 0, 3000, 3000)));
    mgr.swap_buffers();

    CHECK(mgr.find_expert(k_pinned) != nullptr);
    CHECK(mgr.find_expert(k_active) != nullptr);
    CHECK(mgr.find_expert(k_absent) == nullptr);

    auto* pin_ptr = mgr.find_expert(k_pinned);
    auto* expected = static_cast<const uint8_t*>(mmap.data());
    CHECK(std::memcmp(pin_ptr, expected, 3000) == 0);

    std::filesystem::remove(path);
    std::printf("  find_expert_pinned_first PASS\n");
}

static void test_find_expert_staging_invisible() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    mugen::ExpertKey k{2, 0};
    CHECK(mgr.stage_expert(k, mmap, make_location(2, 0, 0, 3000)));

    CHECK(mgr.find_expert(k) == nullptr);

    std::filesystem::remove(path);
    std::printf("  find_expert_staging_invisible PASS\n");
}

static void test_prefetch_and_evict() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    auto loc = make_location(0, 0, 0, 3000);
    mgr.prefetch_expert(mmap, loc);
    mgr.evict_from_mmap(mmap, loc);

    std::filesystem::remove(path);
    std::printf("  prefetch_and_evict PASS\n");
}

static void test_config_roundtrip() {
    auto cfg = test_config();
    mugen::BufferManager mgr(cfg);
    CHECK(mgr.config().buffer_capacity == TEST_BUFFER_CAP);
    CHECK(mgr.config().pinned_capacity == TEST_PINNED_CAP);
    CHECK(mgr.config().system_reserve == 1 * MB);
    std::printf("  config_roundtrip PASS\n");
}

static void test_multiple_experts_staged() {
    auto [path, mmap] = make_test_mmap(1 * MB);
    mugen::BufferManager mgr(test_config());

    for (uint32_t i = 0; i < 10; ++i) {
        mugen::ExpertKey k{0, i};
        auto loc = make_location(0, i, i * 1024, 900);
        CHECK(mgr.stage_expert(k, mmap, loc));
    }
    CHECK(mgr.staging_buffer().slot_count() == 10);

    mgr.swap_buffers();

    for (uint32_t i = 0; i < 10; ++i) {
        mugen::ExpertKey k{0, i};
        CHECK(mgr.find_expert(k) != nullptr);
    }

    std::filesystem::remove(path);
    std::printf("  multiple_experts_staged PASS\n");
}

// ===========================================================================
int main() {
    std::printf("=== ExpertBuffer tests ===\n");
    test_buffer_initial_state();
    test_buffer_allocate_and_find();
    test_buffer_duplicate_key_rejected();
    test_buffer_sequential_bump();
    test_buffer_data_readwrite();
    test_buffer_clear();
    test_buffer_full_rejection();
    test_buffer_deallocate();

    std::printf("=== BufferManager tests ===\n");
    test_manager_initial_state();
    test_manager_swap_promotes_staging();
    test_manager_double_swap();
    test_stage_expert();
    test_stage_expert_buffer_full();
    test_pin_expert();
    test_find_expert_pinned_first();
    test_find_expert_staging_invisible();
    test_prefetch_and_evict();
    test_config_roundtrip();
    test_multiple_experts_staged();

    std::printf("\nAll buffer manager tests passed.\n");
    return 0;
}

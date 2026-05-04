#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "core/memory/mmap_loader.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

namespace {

std::filesystem::path write_temp(const void* data, size_t len,
                                 const char* suffix) {
    auto p = std::filesystem::temp_directory_path()
           / ("mugen_mmap_" + std::string(suffix) + ".bin");
    FILE* fp = std::fopen(p.c_str(), "wb");
    CHECK(fp != nullptr);
    CHECK(std::fwrite(data, 1, len, fp) == len);
    std::fclose(fp);
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
static void test_page_size() {
    size_t ps = mugen::MmapLoader::page_size();
    CHECK(ps > 0);
    CHECK((ps & (ps - 1)) == 0);  // power of two
#if defined(__aarch64__) || defined(__arm64__)
    CHECK(ps == 16384);
#endif
    std::printf("  page_size = %zu\n", ps);
}

static void test_align_to_page() {
    size_t ps = mugen::MmapLoader::page_size();
    CHECK(mugen::MmapLoader::align_to_page(0) == 0);
    CHECK(mugen::MmapLoader::align_to_page(1) == 0);
    CHECK(mugen::MmapLoader::align_to_page(ps - 1) == 0);
    CHECK(mugen::MmapLoader::align_to_page(ps) == ps);
    CHECK(mugen::MmapLoader::align_to_page(ps + 1) == ps);
    CHECK(mugen::MmapLoader::align_to_page(2 * ps) == 2 * ps);
}

static void test_is_page_aligned() {
    size_t ps = mugen::MmapLoader::page_size();
    CHECK(mugen::MmapLoader::is_page_aligned(reinterpret_cast<const void*>(0)));
    CHECK(mugen::MmapLoader::is_page_aligned(reinterpret_cast<const void*>(ps)));
    CHECK(mugen::MmapLoader::is_page_aligned(reinterpret_cast<const void*>(2 * ps)));
    CHECK(!mugen::MmapLoader::is_page_aligned(reinterpret_cast<const void*>(1)));
    CHECK(!mugen::MmapLoader::is_page_aligned(reinterpret_cast<const void*>(ps + 1)));
}

static void test_map_and_read() {
    // Write a file with known content, map it, verify via the mapping.
    std::vector<uint8_t> data(4096);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<uint8_t>(i & 0xff);

    auto path = write_temp(data.data(), data.size(), "readback");
    auto result = mugen::MmapLoader::map_file(path);
    CHECK(result.has_value());

    auto& region = *result;
    CHECK(region.size() == data.size());
    CHECK(region.data() != nullptr);
    CHECK(mugen::MmapLoader::is_page_aligned(region.data()));
    CHECK(std::memcmp(region.data(), data.data(), data.size()) == 0);

    std::filesystem::remove(path);
}

static void test_advise_operations() {
    std::vector<uint8_t> data(65536, 0xAB);
    auto path = write_temp(data.data(), data.size(), "advise");

    auto result = mugen::MmapLoader::map_file(path);
    CHECK(result.has_value());

    auto& region = *result;

    CHECK(region.advise_willneed(0, region.size()));
    CHECK(region.advise_dontneed(0, region.size()));

    // Partial ranges
    CHECK(region.advise_willneed(100, 200));
    CHECK(region.advise_dontneed(4096, 8192));

    // Beyond-end clamping (should not crash)
    CHECK(region.advise_willneed(0, region.size() + 1000));

    // Offset beyond size
    CHECK(!region.advise_willneed(region.size() + 1, 100));

    std::filesystem::remove(path);
}

static void test_map_nonexistent() {
    auto result = mugen::MmapLoader::map_file("/tmp/mugen_nonexistent_mmap_7123.bin");
    CHECK(!result.has_value());
}

static void test_map_empty_file() {
    auto path = std::filesystem::temp_directory_path() / "mugen_mmap_empty.bin";
    FILE* fp = std::fopen(path.c_str(), "wb");
    CHECK(fp != nullptr);
    std::fclose(fp);

    auto result = mugen::MmapLoader::map_file(path);
    CHECK(!result.has_value());

    std::filesystem::remove(path);
}

static void test_move_semantics() {
    std::vector<uint8_t> data(1024, 0x42);
    auto path = write_temp(data.data(), data.size(), "move");

    auto result = mugen::MmapLoader::map_file(path);
    CHECK(result.has_value());

    // Move-construct
    mugen::MmapRegion moved{std::move(*result)};
    CHECK(moved.size() == 1024);
    CHECK(moved.data() != nullptr);

    // Verify the moved-from region is in a safe state (data() may be null)
    // (We can't easily test this without accessing private members, but
    // the destructor of result should be a no-op.)

    // Move-assign
    auto result2 = mugen::MmapLoader::map_file(path);
    CHECK(result2.has_value());
    moved = std::move(*result2);
    CHECK(moved.size() == 1024);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
int main() {
    test_page_size();
    test_align_to_page();
    test_is_page_aligned();
    test_map_and_read();
    test_advise_operations();
    test_map_nonexistent();
    test_map_empty_file();
    test_move_semantics();

    std::printf("All mmap loader tests passed.\n");
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "model/ggml_types.h"
#include "model/gguf_parser.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers for constructing binary GGUF data
// ---------------------------------------------------------------------------
namespace {

using Buf = std::vector<uint8_t>;

void put_u8(Buf& b, uint8_t v)   { b.push_back(v); }
[[maybe_unused]]
void put_u16(Buf& b, uint16_t v) { for (int i = 0; i < 2; ++i) { b.push_back(v & 0xff); v >>= 8; } }
void put_u32(Buf& b, uint32_t v) { for (int i = 0; i < 4; ++i) { b.push_back(v & 0xff); v >>= 8; } }
void put_u64(Buf& b, uint64_t v) { for (int i = 0; i < 8; ++i) { b.push_back(v & 0xff); v >>= 8; } }
void put_f32(Buf& b, float v)    { uint32_t u; std::memcpy(&u, &v, 4); put_u32(b, u); }

void put_str(Buf& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

void put_kv_string(Buf& b, const std::string& key, const std::string& val) {
    put_str(b, key);
    put_u32(b, 8);  // GGUF_TYPE_STRING
    put_str(b, val);
}

void put_kv_u32(Buf& b, const std::string& key, uint32_t val) {
    put_str(b, key);
    put_u32(b, 4);  // GGUF_TYPE_UINT32
    put_u32(b, val);
}

void put_kv_bool(Buf& b, const std::string& key, bool val) {
    put_str(b, key);
    put_u32(b, 7);  // GGUF_TYPE_BOOL
    put_u8(b, val ? 1 : 0);
}

void put_kv_f32(Buf& b, const std::string& key, float val) {
    put_str(b, key);
    put_u32(b, 6);  // GGUF_TYPE_FLOAT32
    put_f32(b, val);
}

// Array of uint32_t values.
void put_kv_array_u32(Buf& b, const std::string& key,
                      const std::vector<uint32_t>& vals) {
    put_str(b, key);
    put_u32(b, 9);   // GGUF_TYPE_ARRAY
    put_u32(b, 4);   // element type: UINT32
    put_u64(b, vals.size());
    for (auto v : vals) put_u32(b, v);
}

struct TensorDesc {
    std::string          name;
    std::vector<int64_t> dims;
    uint32_t             type;   // GGMLType raw value
    uint64_t             offset; // within data blob
};

void put_tensor_info(Buf& b, const TensorDesc& t) {
    put_str(b, t.name);
    put_u32(b, static_cast<uint32_t>(t.dims.size()));
    for (auto d : t.dims) put_u64(b, static_cast<uint64_t>(d));
    put_u32(b, t.type);
    put_u64(b, t.offset);
}

void pad_to(Buf& b, uint64_t alignment) {
    uint64_t r = b.size() % alignment;
    if (r != 0) b.insert(b.end(), static_cast<size_t>(alignment - r), 0);
}

// Build a complete minimal GGUF v3 file in memory.
Buf build_test_gguf() {
    // Two tensors:
    //   "output.weight"       — F32  [64, 32]  => 64*32*4 = 8192 bytes
    //   "blk.0.attn_q.weight" — F16  [64, 64]  => 64*64*2 = 8192 bytes
    //
    // Five KV pairs including a float, bool, and array to exercise parsing.

    constexpr uint32_t kVersion   = 3;
    constexpr uint64_t kNTensors  = 2;
    constexpr uint64_t kNKV       = 7;
    constexpr uint64_t kAlignment = 32;

    const size_t t0_bytes = 64 * 32 * 4;  // F32
    const size_t t1_bytes = 64 * 64 * 2;  // F16

    Buf b;
    b.reserve(4096);

    // --- header ---
    put_u32(b, 0x46554747);  // magic "GGUF"
    put_u32(b, kVersion);
    put_u64(b, kNTensors);
    put_u64(b, kNKV);

    // --- KV pairs ---
    put_kv_string(b, "general.architecture", "llama");
    put_kv_string(b, "general.name",         "test-model");
    put_kv_u32   (b, "llama.block_count",    2);
    put_kv_u32   (b, "llama.expert_count",   8);
    put_kv_u32   (b, "llama.expert_used_count", 2);
    put_kv_f32   (b, "llama.rope.freq_base", 10000.0f);
    put_kv_bool  (b, "llama.attention.causal", true);

    // --- tensor descriptors ---
    put_tensor_info(b, {"output.weight",       {64, 32}, 0, 0});         // F32
    put_tensor_info(b, {"blk.0.attn_q.weight", {64, 64}, 1, t0_bytes}); // F16

    // --- align to data section ---
    pad_to(b, kAlignment);

    size_t data_start = b.size();

    // --- tensor data (zeroed) ---
    b.resize(b.size() + t0_bytes + t1_bytes, 0);

    (void)data_start;
    return b;
}

// Build a GGUF file with MoE expert tensors.
Buf build_moe_gguf() {
    constexpr uint32_t kVersion  = 3;
    constexpr uint64_t kNExperts = 4;
    constexpr uint64_t kNLayers  = 1;

    // Expert tensors: gate/up/down for 4 experts in layer 0
    // Each is F16 [32, 32] = 32*32*2 = 2048 bytes
    constexpr size_t kExpertBytes = 32 * 32 * 2;
    uint64_t n_tensors = kNExperts * 3;  // gate + up + down per expert
    uint64_t n_kv      = 4;

    Buf b;
    b.reserve(8192);

    put_u32(b, 0x46554747);
    put_u32(b, kVersion);
    put_u64(b, n_tensors);
    put_u64(b, n_kv);

    put_kv_string(b, "general.architecture",   "llama");
    put_kv_string(b, "general.name",           "moe-test");
    put_kv_u32   (b, "llama.block_count",      static_cast<uint32_t>(kNLayers));
    put_kv_u32   (b, "llama.expert_count",     static_cast<uint32_t>(kNExperts));

    uint64_t offset = 0;
    for (uint32_t e = 0; e < kNExperts; ++e) {
        for (const char* kind : {"gate", "up", "down"}) {
            std::string name = "blk.0.ffn_" + std::string(kind)
                             + "_exps." + std::to_string(e) + ".weight";
            put_tensor_info(b, {name, {32, 32}, 1, offset});
            offset += kExpertBytes;
        }
    }

    pad_to(b, 32);
    b.resize(b.size() + static_cast<size_t>(offset), 0);
    return b;
}

std::filesystem::path write_temp(const Buf& data, const char* suffix) {
    auto p = std::filesystem::temp_directory_path() / ("mugen_test_" + std::string(suffix) + ".gguf");
    FILE* fp = std::fopen(p.c_str(), "wb");
    CHECK(fp != nullptr);
    CHECK(std::fwrite(data.data(), 1, data.size(), fp) == data.size());
    std::fclose(fp);
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests: ggml_types helpers
// ---------------------------------------------------------------------------
static void test_ggml_type_sizes() {
    using mugen::GGMLType;

    CHECK(mugen::ggml_type_size(GGMLType::F32)  == 4);
    CHECK(mugen::ggml_type_size(GGMLType::F16)  == 2);
    CHECK(mugen::ggml_type_size(GGMLType::Q4_0) == 18);
    CHECK(mugen::ggml_type_size(GGMLType::Q4_K) == 144);
    CHECK(mugen::ggml_type_size(GGMLType::Q6_K) == 210);
    CHECK(mugen::ggml_type_size(GGMLType::BF16) == 2);
    CHECK(mugen::ggml_type_size(GGMLType::I32)  == 4);

    CHECK(mugen::ggml_type_block_size(GGMLType::F32)  == 1);
    CHECK(mugen::ggml_type_block_size(GGMLType::Q4_0) == 32);
    CHECK(mugen::ggml_type_block_size(GGMLType::Q4_K) == 256);

    CHECK(mugen::ggml_type_name(GGMLType::Q8_0) == "Q8_0");
    CHECK(mugen::ggml_type_name(GGMLType::BF16) == "BF16");

    CHECK(mugen::ggml_type_is_valid(0));
    CHECK(mugen::ggml_type_is_valid(30));
    CHECK(!mugen::ggml_type_is_valid(4));   // deprecated
    CHECK(!mugen::ggml_type_is_valid(999));
}

static void test_tensor_byte_size() {
    using mugen::GGMLType;

    int64_t dims_2d[] = {64, 32};
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::F32,
          std::span<const int64_t>(dims_2d, 2)) == 64 * 32 * 4);
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::F16,
          std::span<const int64_t>(dims_2d, 2)) == 64 * 32 * 2);

    // Q4_0: 18 bytes per 32 elements. 64/32 = 2 blocks per row, 32 rows.
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::Q4_0,
          std::span<const int64_t>(dims_2d, 2)) == 18 * 2 * 32);

    // Q4_K: 144 bytes per 256 elements. dims = [256, 10]
    int64_t dims_k[] = {256, 10};
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::Q4_K,
          std::span<const int64_t>(dims_k, 2)) == 144 * 1 * 10);

    // dims[0] not divisible by block_size -> 0
    int64_t bad_dims[] = {33, 10};
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::Q4_0,
          std::span<const int64_t>(bad_dims, 2)) == 0);

    // Empty dims -> 0
    CHECK(mugen::ggml_tensor_byte_size(GGMLType::F32,
          std::span<const int64_t>{}) == 0);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – basic file
// ---------------------------------------------------------------------------
static void test_parse_valid_gguf() {
    auto data = build_test_gguf();
    auto path = write_temp(data, "basic");

    auto result = mugen::GGUFParser::parse(path);
    CHECK(result.has_value());

    auto& parser = *result;
    auto& meta   = parser.metadata();

    CHECK(meta.version == 3);
    CHECK(meta.arch    == "llama");
    CHECK(meta.name    == "test-model");
    CHECK(meta.n_layers       == 2);
    CHECK(meta.n_experts      == 8);
    CHECK(meta.n_experts_used == 2);

    // Float KV
    auto fit = meta.raw_float_kv.find("llama.rope.freq_base");
    CHECK(fit != meta.raw_float_kv.end());
    CHECK(fit->second > 9999.0 && fit->second < 10001.0);

    // Bool KV (stored as int)
    auto bit = meta.raw_int_kv.find("llama.attention.causal");
    CHECK(bit != meta.raw_int_kv.end());
    CHECK(bit->second == 1);

    CHECK(parser.tensors().size() == 2);

    auto* t0 = parser.tensor_by_name("output.weight");
    CHECK(t0 != nullptr);
    CHECK(t0->dimensions.size() == 2);
    CHECK(t0->dimensions[0] == 64);
    CHECK(t0->dimensions[1] == 32);
    CHECK(t0->type == mugen::GGMLType::F32);
    CHECK(t0->byte_size == 64 * 32 * 4);
    CHECK(t0->file_offset == parser.data_offset());

    auto* t1 = parser.tensor_by_name("blk.0.attn_q.weight");
    CHECK(t1 != nullptr);
    CHECK(t1->type == mugen::GGMLType::F16);
    CHECK(t1->byte_size == 64 * 64 * 2);
    CHECK(t1->file_offset == parser.data_offset() + 64 * 32 * 4);

    CHECK(parser.is_moe());
    CHECK(parser.expert_count() == 8);
    CHECK(parser.file_size() == data.size());

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – MoE expert tensor lookup
// ---------------------------------------------------------------------------
static void test_moe_expert_tensors() {
    auto data = build_moe_gguf();
    auto path = write_temp(data, "moe");

    auto result = mugen::GGUFParser::parse(path);
    CHECK(result.has_value());

    auto& parser = *result;
    CHECK(parser.is_moe());
    CHECK(parser.expert_count() == 4);
    CHECK(parser.tensors().size() == 12);

    auto names = parser.expert_tensor_names(0, 2);
    CHECK(names.size() == 3);
    CHECK(names[0] == "blk.0.ffn_down_exps.2.weight");
    CHECK(names[1] == "blk.0.ffn_gate_exps.2.weight");
    CHECK(names[2] == "blk.0.ffn_up_exps.2.weight");

    auto empty = parser.expert_tensor_names(1, 0);
    CHECK(empty.empty());

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – error cases
// ---------------------------------------------------------------------------
static void test_parse_errors() {
    // Non-existent file
    auto r1 = mugen::GGUFParser::parse("/tmp/mugen_nonexistent_4729.gguf");
    CHECK(!r1.has_value());

    // Bad magic
    Buf bad_magic = {0x00, 0x00, 0x00, 0x00};
    bad_magic.resize(64, 0);
    auto p2 = write_temp(bad_magic, "bad_magic");
    auto r2 = mugen::GGUFParser::parse(p2);
    CHECK(!r2.has_value());
    std::filesystem::remove(p2);

    // Unsupported version (v1)
    Buf bad_ver;
    put_u32(bad_ver, 0x46554747);
    put_u32(bad_ver, 1);  // v1
    bad_ver.resize(64, 0);
    auto p3 = write_temp(bad_ver, "bad_ver");
    auto r3 = mugen::GGUFParser::parse(p3);
    CHECK(!r3.has_value());
    std::filesystem::remove(p3);

    // Truncated file (only magic + version, no counts)
    Buf truncated;
    put_u32(truncated, 0x46554747);
    put_u32(truncated, 3);
    auto p4 = write_temp(truncated, "truncated");
    auto r4 = mugen::GGUFParser::parse(p4);
    CHECK(!r4.has_value());
    std::filesystem::remove(p4);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser with array KV
// ---------------------------------------------------------------------------
static void test_parse_with_array_kv() {
    Buf b;
    put_u32(b, 0x46554747);
    put_u32(b, 3);
    put_u64(b, 1);   // 1 tensor
    put_u64(b, 3);   // 3 KV pairs

    put_kv_string  (b, "general.architecture", "llama");
    put_kv_string  (b, "general.name",         "arr-test");
    put_kv_array_u32(b, "tokenizer.scores",    {100, 200, 300});

    put_tensor_info(b, {"tok.weight", {32, 1}, 0, 0});  // F32 [32,1] = 128 bytes

    pad_to(b, 32);
    b.resize(b.size() + 128, 0);

    auto path = write_temp(b, "array_kv");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(result.has_value());
    CHECK(result->tensors().size() == 1);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – empty file
// ---------------------------------------------------------------------------
static void test_parse_empty_file() {
    Buf empty;
    auto path = write_temp(empty, "empty");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(!result.has_value());
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – truncated after tensor descriptors (data missing)
// ---------------------------------------------------------------------------
static void test_parse_truncated_data() {
    // Build a valid header+KV+tensor descriptors, but truncate before tensor data.
    Buf b;
    put_u32(b, 0x46554747);  // magic
    put_u32(b, 3);           // version
    put_u64(b, 1);           // 1 tensor
    put_u64(b, 1);           // 1 KV pair

    put_kv_string(b, "general.architecture", "llama");

    // Tensor: F32 [64, 32] = 8192 bytes, offset 0
    put_tensor_info(b, {"big.weight", {64, 32}, 0, 0});

    pad_to(b, 32);
    // Do NOT append actual tensor data — file is truncated.

    auto path = write_temp(b, "trunc_data");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(!result.has_value());  // tensor extends past EOF
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – excessive tensor count
// ---------------------------------------------------------------------------
static void test_parse_huge_tensor_count() {
    Buf b;
    put_u32(b, 0x46554747);
    put_u32(b, 3);
    put_u64(b, 0xFFFFFFFFFFFFFFULL);  // absurdly large tensor count
    put_u64(b, 0);                     // 0 KV pairs
    b.resize(b.size() + 128, 0);       // pad so file isn't too small to read header

    auto path = write_temp(b, "huge_tc");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(!result.has_value());
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – illegal magic number variants
// ---------------------------------------------------------------------------
static void test_parse_almost_magic() {
    // One-byte-off magic: "GGUG" (last byte differs)
    Buf b;
    put_u32(b, 0x47554747);  // "GUGG" LE
    put_u32(b, 3);
    put_u64(b, 0);
    put_u64(b, 0);
    b.resize(64, 0);

    auto path = write_temp(b, "almost_magic");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(!result.has_value());
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – big-endian byte-swapped version detection
// ---------------------------------------------------------------------------
static void test_parse_big_endian_version() {
    Buf b;
    put_u32(b, 0x46554747);  // correct magic
    // Version 3 in big-endian: 0x03000000
    put_u32(b, 0x03000000);
    put_u64(b, 0);
    put_u64(b, 0);
    b.resize(64, 0);

    auto path = write_temp(b, "big_endian");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(!result.has_value());  // "big-endian GGUF files are not supported"
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: GGUF parser – tensor_by_name miss
// ---------------------------------------------------------------------------
static void test_tensor_by_name_miss() {
    auto data = build_test_gguf();
    auto path = write_temp(data, "name_miss");

    auto result = mugen::GGUFParser::parse(path);
    CHECK(result.has_value());

    CHECK(result->tensor_by_name("nonexistent.tensor") == nullptr);
    CHECK(result->tensor_by_name("") == nullptr);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
int main() {
    test_ggml_type_sizes();
    test_tensor_byte_size();
    test_parse_valid_gguf();
    test_moe_expert_tensors();
    test_parse_errors();
    test_parse_with_array_kv();
    test_parse_empty_file();
    test_parse_truncated_data();
    test_parse_huge_tensor_count();
    test_parse_almost_magic();
    test_parse_big_endian_version();
    test_tensor_by_name_miss();

    std::printf("All GGUF parser tests passed.\n");
    return 0;
}

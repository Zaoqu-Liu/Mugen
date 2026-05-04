#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "model/gguf_parser.h"
#include "model/tokenizer.h"

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Binary helpers — same format as the existing GGUF test, self-contained
// ---------------------------------------------------------------------------
namespace {

using Buf = std::vector<uint8_t>;

void put_u32(Buf& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) { b.push_back(v & 0xff); v >>= 8; }
}
void put_u64(Buf& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) { b.push_back(v & 0xff); v >>= 8; }
}
void put_f32(Buf& b, float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    put_u32(b, u);
}
void put_str(Buf& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

void put_kv_string(Buf& b, const std::string& key, const std::string& val) {
    put_str(b, key);
    put_u32(b, 8);  // STRING
    put_str(b, val);
}

void put_kv_u32(Buf& b, const std::string& key, uint32_t val) {
    put_str(b, key);
    put_u32(b, 4);  // UINT32
    put_u32(b, val);
}

void put_kv_array_string(Buf& b, const std::string& key,
                         const std::vector<std::string>& vals) {
    put_str(b, key);
    put_u32(b, 9);  // ARRAY
    put_u32(b, 8);  // element: STRING
    put_u64(b, vals.size());
    for (const auto& v : vals) put_str(b, v);
}

void put_kv_array_f32(Buf& b, const std::string& key,
                      const std::vector<float>& vals) {
    put_str(b, key);
    put_u32(b, 9);  // ARRAY
    put_u32(b, 6);  // element: FLOAT32
    put_u64(b, vals.size());
    for (auto v : vals) put_f32(b, v);
}

struct TensorDesc {
    std::string          name;
    std::vector<int64_t> dims;
    uint32_t             type;
    uint64_t             offset;
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

std::filesystem::path write_temp(const Buf& data, const char* suffix) {
    auto p = std::filesystem::temp_directory_path()
           / ("mugen_tok_" + std::string(suffix) + ".gguf");
    FILE* fp = std::fopen(p.c_str(), "wb");
    CHECK(fp != nullptr);
    CHECK(std::fwrite(data.data(), 1, data.size(), fp) == data.size());
    std::fclose(fp);
    return p;
}

// ---------------------------------------------------------------------------
// Reusable SPM test vocabulary
// ---------------------------------------------------------------------------
// Indices:
//    0  <unk>        9  abc
//    1  <s>         10  ▁a
//    2  </s>        11  ▁ab
//    3  ▁           12  ▁abc
//    4  a           13..268  <0x00>..<0xFF>
//    5  b
//    6  c
//    7  ab
//    8  bc

mugen::GGUFMetadata build_spm_metadata() {
    mugen::GGUFMetadata meta;
    meta.raw_string_kv["tokenizer.ggml.model"] = "llama";

    std::vector<std::string> tokens = {
        "<unk>", "<s>", "</s>",
        "\xE2\x96\x81",
        "a", "b", "c",
        "ab", "bc",
        "abc",
        "\xE2\x96\x81" "a",
        "\xE2\x96\x81" "ab",
        "\xE2\x96\x81" "abc",
    };
    std::vector<float> scores = {
        -1000.0f, 0.0f, 0.0f,
        -4.0f,
        -3.0f, -3.0f, -3.0f,
        -2.5f, -2.5f,
        -1.0f,
        -2.0f,
        -1.5f,
        -0.5f,
    };

    // 256 byte-fallback tokens: <0x00> .. <0xFF>
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.emplace_back(buf);
        scores.push_back(-5.0f);
    }

    meta.raw_string_array_kv["tokenizer.ggml.tokens"] = std::move(tokens);
    meta.raw_float_array_kv["tokenizer.ggml.scores"]  = std::move(scores);

    meta.raw_int_kv["tokenizer.ggml.bos_token_id"]     = 1;
    meta.raw_int_kv["tokenizer.ggml.eos_token_id"]     = 2;
    meta.raw_int_kv["tokenizer.ggml.padding_token_id"] = 0;

    return meta;
}

// ---------------------------------------------------------------------------
// BPE test vocabulary
// ---------------------------------------------------------------------------
// Indices:  0 <unk>, 1 <s>, 2 </s>, 3 a, 4 b, 5 c, 6 d, 7 ab, 8 cd, 9 abcd

mugen::GGUFMetadata build_bpe_metadata() {
    mugen::GGUFMetadata meta;
    meta.raw_string_kv["tokenizer.ggml.model"] = "gpt2";

    meta.raw_string_array_kv["tokenizer.ggml.tokens"] = {
        "<unk>", "<s>", "</s>",
        "a", "b", "c", "d",
        "ab", "cd",
        "abcd",
    };
    meta.raw_string_array_kv["tokenizer.ggml.merges"] = {
        "a b",
        "c d",
        "ab cd",
    };

    meta.raw_int_kv["tokenizer.ggml.bos_token_id"]     = 1;
    meta.raw_int_kv["tokenizer.ggml.eos_token_id"]     = 2;
    meta.raw_int_kv["tokenizer.ggml.padding_token_id"] = 0;

    return meta;
}

}  // namespace

// ===========================================================================
// Test 1: GGUF array KV parsing
// ===========================================================================
static void test_gguf_array_parsing() {
    Buf b;
    put_u32(b, 0x46554747);  // magic
    put_u32(b, 3);           // version
    put_u64(b, 1);           // 1 tensor
    put_u64(b, 4);           // 4 KV pairs

    put_kv_string(b, "general.architecture", "llama");
    put_kv_string(b, "general.name",         "tok-test");
    put_kv_array_string(b, "tokenizer.ggml.tokens", {"hello", "world", "foo"});
    put_kv_array_f32   (b, "tokenizer.ggml.scores", {1.0f, 2.0f, 3.0f});

    put_tensor_info(b, {"dummy.weight", {32, 1}, 0, 0});  // F32 [32,1] = 128 B
    pad_to(b, 32);
    b.resize(b.size() + 128, 0);

    auto path = write_temp(b, "array_kv");
    auto result = mugen::GGUFParser::parse(path);
    CHECK(result.has_value());

    auto& meta = result->metadata();

    // Verify string array
    auto sit = meta.raw_string_array_kv.find("tokenizer.ggml.tokens");
    CHECK(sit != meta.raw_string_array_kv.end());
    CHECK(sit->second.size() == 3);
    CHECK(sit->second[0] == "hello");
    CHECK(sit->second[1] == "world");
    CHECK(sit->second[2] == "foo");

    // Verify float array
    auto fit = meta.raw_float_array_kv.find("tokenizer.ggml.scores");
    CHECK(fit != meta.raw_float_array_kv.end());
    CHECK(fit->second.size() == 3);
    CHECK(fit->second[0] > 0.99f && fit->second[0] < 1.01f);
    CHECK(fit->second[1] > 1.99f && fit->second[1] < 2.01f);
    CHECK(fit->second[2] > 2.99f && fit->second[2] < 3.01f);

    std::filesystem::remove(path);
}

// ===========================================================================
// Test 2: Tokenizer construction
// ===========================================================================
static void test_tokenizer_from_gguf() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());

    auto& tok = *result;
    CHECK(tok.vocab_size() == 13 + 256);
    CHECK(tok.bos_token() == 1);
    CHECK(tok.eos_token() == 2);
    CHECK(tok.token_to_text(0) == "<unk>");
    CHECK(tok.token_to_text(4) == "a");
    CHECK(tok.token_to_text(9999) == "");  // out of range
}

static void test_tokenizer_missing_model() {
    mugen::GGUFMetadata meta;
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(!result.has_value());
}

static void test_tokenizer_missing_tokens() {
    mugen::GGUFMetadata meta;
    meta.raw_string_kv["tokenizer.ggml.model"] = "llama";
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(!result.has_value());
}

// ===========================================================================
// Test 3: SPM encode / decode round-trip
// ===========================================================================
static void test_spm_encode_decode() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // "abc" → normalize "▁abc"
    //   merge sequence: [▁,a,b,c] → [▁a,b,c] → [▁ab,c] → [▁abc]
    //   expected: token 12 ("▁abc")
    {
        auto ids = tok.encode("abc");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 12);
        CHECK(tok.decode(ids) == "abc");
    }

    // "ab c" → normalize "▁ab▁c"
    //   [▁,a,b,▁,c] → [▁a,b,▁,c] → [▁ab,▁,c]
    //   no more merges → tokens 11, 3, 6
    {
        auto ids = tok.encode("ab c");
        CHECK(ids.size() == 3);
        CHECK(ids[0] == 11);  // ▁ab
        CHECK(ids[1] == 3);   // ▁
        CHECK(ids[2] == 6);   // c
        CHECK(tok.decode(ids) == "ab c");
    }
}

// ===========================================================================
// Test 4: SPM byte fallback
// ===========================================================================
static void test_spm_byte_fallback() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // 'd' is not in the base vocab; byte 0x64 → <0x64> at index 13+0x64=113
    {
        auto ids = tok.encode("d");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 3);    // ▁
        CHECK(ids[1] == 113);  // <0x64>
        CHECK(tok.decode(ids) == "d");
    }
}

// ===========================================================================
// Test 5: SPM Chinese UTF-8 (byte fallback)
// ===========================================================================
static void test_spm_chinese_utf8() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // "你" = 0xE4 0xBD 0xA0
    // Indices: 13+0xE4=241, 13+0xBD=202, 13+0xA0=173
    const char ni[] = "\xe4\xbd\xa0";
    {
        auto ids = tok.encode(ni);
        CHECK(ids.size() == 4);
        CHECK(ids[0] == 3);    // ▁
        CHECK(ids[1] == 13 + 0xE4);
        CHECK(ids[2] == 13 + 0xBD);
        CHECK(ids[3] == 13 + 0xA0);
        CHECK(tok.decode(ids) == std::string(ni));
    }
}

// ===========================================================================
// Test 6: BPE encode / decode
// ===========================================================================
static void test_bpe_encode_decode() {
    auto meta = build_bpe_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    CHECK(tok.vocab_size() == 10);

    // "abcd":  [a,b,c,d] → merge "a b" → [ab,c,d] → merge "c d" → [ab,cd]
    //          → merge "ab cd" → [abcd] → token 9
    {
        auto ids = tok.encode("abcd");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 9);
        CHECK(tok.decode(ids) == "abcd");
    }

    // "ab" → [a,b] → merge "a b" → [ab] → token 7
    {
        auto ids = tok.encode("ab");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 7);
        CHECK(tok.decode(ids) == "ab");
    }

    // "a" → [a] → no merge → token 3
    {
        auto ids = tok.encode("a");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 3);
        CHECK(tok.decode(ids) == "a");
    }
}

// ===========================================================================
// Test 7: Special tokens
// ===========================================================================
static void test_special_tokens() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // BOS and EOS are skipped during decode
    CHECK(tok.decode({1}) == "");
    CHECK(tok.decode({2}) == "");
    CHECK(tok.decode({1, 12, 2}) == "abc");

    // Out-of-range token IDs are skipped
    CHECK(tok.decode({99999}) == "");
}

// ===========================================================================
// Test 8: Edge cases
// ===========================================================================
static void test_empty_string() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    auto ids = tok.encode("");
    CHECK(ids.empty());
    CHECK(tok.decode({}) == "");
}

static void test_single_char() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // Single 'a' → "▁a" → token 10
    auto ids = tok.encode("a");
    CHECK(ids.size() == 1);
    CHECK(ids[0] == 10);
    CHECK(tok.decode(ids) == "a");
}

static void test_long_text() {
    auto meta = build_spm_metadata();
    auto result = mugen::Tokenizer::from_gguf(meta);
    CHECK(result.has_value());
    auto& tok = *result;

    // Build a long string of "abc" repeated 200 times with spaces.
    // This exercises the merge loop at scale.
    std::string long_text;
    for (int i = 0; i < 200; ++i) {
        if (i > 0) long_text += ' ';
        long_text += "abc";
    }

    auto ids = tok.encode(long_text);
    CHECK(!ids.empty());
    auto decoded = tok.decode(ids);
    CHECK(decoded == long_text);
}

// ===========================================================================
// Test 9: GGUF end-to-end — build file with tokenizer arrays, parse, build tok
// ===========================================================================
static void test_gguf_to_tokenizer() {
    Buf buf;
    put_u32(buf, 0x46554747);
    put_u32(buf, 3);
    put_u64(buf, 1);  // 1 tensor
    put_u64(buf, 6);  // 6 KV pairs

    put_kv_string(buf, "general.architecture", "llama");
    put_kv_string(buf, "general.name", "e2e-tok-test");
    put_kv_string(buf, "tokenizer.ggml.model", "llama");
    put_kv_u32   (buf, "tokenizer.ggml.bos_token_id", 1);
    put_kv_array_string(buf, "tokenizer.ggml.tokens", {"<unk>", "<s>", "</s>", "x", "y", "xy"});
    put_kv_array_f32   (buf, "tokenizer.ggml.scores", {-999.0f, 0.0f, 0.0f, -2.0f, -2.0f, -1.0f});

    put_tensor_info(buf, {"w", {32, 1}, 0, 0});
    pad_to(buf, 32);
    buf.resize(buf.size() + 128, 0);

    auto path = write_temp(buf, "e2e_tok");
    auto parse_result = mugen::GGUFParser::parse(path);
    CHECK(parse_result.has_value());

    auto tok_result = mugen::Tokenizer::from_gguf(parse_result->metadata());
    CHECK(tok_result.has_value());

    auto& tok = *tok_result;
    CHECK(tok.vocab_size() == 6);
    CHECK(tok.bos_token() == 1);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
int main() {
    test_gguf_array_parsing();
    test_tokenizer_from_gguf();
    test_tokenizer_missing_model();
    test_tokenizer_missing_tokens();
    test_spm_encode_decode();
    test_spm_byte_fallback();
    test_spm_chinese_utf8();
    test_bpe_encode_decode();
    test_special_tokens();
    test_empty_string();
    test_single_char();
    test_long_text();
    test_gguf_to_tokenizer();

    std::printf("All tokenizer tests passed.\n");
    return 0;
}

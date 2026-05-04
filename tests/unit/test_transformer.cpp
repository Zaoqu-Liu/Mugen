#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "core/compute/metal_compute.h"
#include "core/memory/kv_cache.h"
#include "core/memory/mmap_loader.h"
#include "core/model/transformer.h"
#include "model/ggml_types.h"
#include "model/gguf_parser.h"

#define MUGEN_CHECK(cond)                                                      \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #cond, __FILE__,       \
                         __LINE__);                                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GGUF binary builder helpers (same conventions as test_gguf_parser.cpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using Buf = std::vector<uint8_t>;

[[maybe_unused]]
void put_u8(Buf& b, uint8_t v)   { b.push_back(v); }
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

void put_kv_f32(Buf& b, const std::string& key, float val) {
    put_str(b, key);
    put_u32(b, 6);  // GGUF_TYPE_FLOAT32
    put_f32(b, val);
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

// ─────────────────────────────────────────────────────────────────────────────
// Tiny model parameters
// ─────────────────────────────────────────────────────────────────────────────

constexpr uint32_t kVocab    = 256;
constexpr uint32_t kEmbed    = 64;
constexpr uint32_t kHeads    = 4;
constexpr uint32_t kKVHeads  = 2;
constexpr uint32_t kHeadDim  = kEmbed / kHeads;  // 16
constexpr uint32_t kFFN      = 128;
constexpr uint32_t kLayers   = 2;
constexpr float    kTheta    = 10000.0f;
constexpr float    kEps      = 1e-5f;
constexpr uint32_t kCtxLen   = 512;

size_t f16_bytes(size_t n_elem) { return n_elem * 2; }

// F32 bytes for a tensor (used for norm weights)
size_t f32_bytes(size_t n_elem) { return n_elem * 4; }

// Fill a region in the data blob with small random F16 values
void fill_f16_random(Buf& data, size_t offset, size_t n_elem,
                     std::mt19937& rng, float scale = 0.01f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n_elem; i++) {
        float v = dist(rng);
        uint16_t h;
        _Float16 fh = static_cast<_Float16>(v);
        std::memcpy(&h, &fh, 2);
        data[offset + i * 2]     = static_cast<uint8_t>(h & 0xff);
        data[offset + i * 2 + 1] = static_cast<uint8_t>((h >> 8) & 0xff);
    }
}

// Fill norm weights with values near 1.0 (F32)
void fill_f32_norm(Buf& data, size_t offset, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        float v = 1.0f;
        uint32_t u;
        std::memcpy(&u, &v, 4);
        data[offset + i * 4]     = static_cast<uint8_t>(u & 0xff);
        data[offset + i * 4 + 1] = static_cast<uint8_t>((u >> 8) & 0xff);
        data[offset + i * 4 + 2] = static_cast<uint8_t>((u >> 16) & 0xff);
        data[offset + i * 4 + 3] = static_cast<uint8_t>((u >> 24) & 0xff);
    }
}

struct TinyModel {
    std::filesystem::path path;

    ~TinyModel() {
        if (!path.empty()) std::filesystem::remove(path);
    }
};

// Build a complete tiny GGUF file that TransformerModel can load.
// All weight matrices are F16, norm weights are F32.
TinyModel build_tiny_gguf() {
    //
    // Tensors we need:
    //   token_embd.weight         — F16 [embed, vocab]   = [64, 256]
    //   output_norm.weight        — F32 [embed]           = [64]
    //   output.weight             — F16 [embed, vocab]   = [64, 256]
    //   Per-layer (x2):
    //     blk.L.attn_norm.weight  — F32 [embed]
    //     blk.L.attn_q.weight     — F16 [embed, embed]   = [64, 64]
    //     blk.L.attn_k.weight     — F16 [embed, kv*hdim]  = [64, 32]
    //     blk.L.attn_v.weight     — F16 [embed, kv*hdim]  = [64, 32]
    //     blk.L.attn_output.weight— F16 [embed, embed]   = [64, 64]  (n_heads*head_dim → embed)
    //     blk.L.ffn_norm.weight   — F32 [embed]
    //     blk.L.ffn_gate.weight   — F16 [embed, ffn]     = [64, 128]
    //     blk.L.ffn_down.weight   — F16 [ffn, embed]     = [128, 64]
    //     blk.L.ffn_up.weight     — F16 [embed, ffn]     = [64, 128]
    //
    // GGUF tensor dimensions convention:
    //   For a weight matrix of shape (M_out × K_in), stored as dims=[K_in, M_out]
    //   (the innermost/fastest dim is first).

    struct TSpec {
        std::string name;
        std::vector<int64_t> dims;
        uint32_t type;      // 0=F32, 1=F16
        size_t byte_size;
    };

    std::vector<TSpec> tensors;

    // Global
    tensors.push_back({"token_embd.weight", {kEmbed, kVocab}, 1,
                        f16_bytes(size_t(kEmbed) * kVocab)});
    tensors.push_back({"output_norm.weight", {kEmbed}, 0,
                        f32_bytes(kEmbed)});
    tensors.push_back({"output.weight", {kEmbed, kVocab}, 1,
                        f16_bytes(size_t(kEmbed) * kVocab)});

    // Per-layer
    for (uint32_t L = 0; L < kLayers; L++) {
        auto blk = "blk." + std::to_string(L) + ".";
        uint32_t kv_dim = kKVHeads * kHeadDim;

        tensors.push_back({blk + "attn_norm.weight", {kEmbed}, 0,
                            f32_bytes(kEmbed)});
        tensors.push_back({blk + "attn_q.weight", {kEmbed, kEmbed}, 1,
                            f16_bytes(size_t(kEmbed) * kEmbed)});
        tensors.push_back({blk + "attn_k.weight", {kEmbed, kv_dim}, 1,
                            f16_bytes(size_t(kEmbed) * kv_dim)});
        tensors.push_back({blk + "attn_v.weight", {kEmbed, kv_dim}, 1,
                            f16_bytes(size_t(kEmbed) * kv_dim)});
        tensors.push_back({blk + "attn_output.weight", {kEmbed, kEmbed}, 1,
                            f16_bytes(size_t(kEmbed) * kEmbed)});
        tensors.push_back({blk + "ffn_norm.weight", {kEmbed}, 0,
                            f32_bytes(kEmbed)});
        tensors.push_back({blk + "ffn_gate.weight", {kEmbed, kFFN}, 1,
                            f16_bytes(size_t(kEmbed) * kFFN)});
        tensors.push_back({blk + "ffn_down.weight", {kFFN, kEmbed}, 1,
                            f16_bytes(size_t(kFFN) * kEmbed)});
        tensors.push_back({blk + "ffn_up.weight", {kEmbed, kFFN}, 1,
                            f16_bytes(size_t(kEmbed) * kFFN)});
    }

    constexpr uint64_t kNKV = 11;

    Buf header;
    header.reserve(4096);

    put_u32(header, 0x46554747);
    put_u32(header, 3);
    put_u64(header, tensors.size());
    put_u64(header, kNKV);

    put_kv_string(header, "general.architecture", "llama");
    put_kv_string(header, "general.name", "tiny-test");
    put_kv_u32(header, "llama.block_count", kLayers);
    put_kv_u32(header, "llama.vocab_size", kVocab);
    put_kv_u32(header, "llama.embedding_length", kEmbed);
    put_kv_u32(header, "llama.attention.head_count", kHeads);
    put_kv_u32(header, "llama.attention.head_count_kv", kKVHeads);
    put_kv_u32(header, "llama.context_length", kCtxLen);
    put_kv_u32(header, "llama.feed_forward_length", kFFN);
    put_kv_f32(header, "llama.rope.freq_base", kTheta);
    put_kv_f32(header, "llama.attention.layer_norm_rms_epsilon", kEps);

    // Write tensor descriptors
    std::vector<uint64_t> offsets;
    uint64_t cur_offset = 0;
    for (auto& t : tensors) {
        TensorDesc td{t.name, t.dims, t.type, cur_offset};
        put_tensor_info(header, td);
        offsets.push_back(cur_offset);
        cur_offset += t.byte_size;
        // Align each tensor data to 32 bytes
        uint64_t rem = cur_offset % 32;
        if (rem != 0) cur_offset += (32 - rem);
    }

    // Pad header to alignment boundary
    pad_to(header, 32);

    // Build data section
    size_t total_data = static_cast<size_t>(cur_offset);
    Buf data(total_data, 0);

    std::mt19937 rng(42);

    for (size_t i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        size_t off = static_cast<size_t>(offsets[i]);

        if (t.type == 1) { // F16
            size_t n_elem = t.byte_size / 2;
            // Norm-like or weight? Use small random for weights
            fill_f16_random(data, off, n_elem, rng, 0.02f);
        } else { // F32 (norm weights)
            size_t n_elem = t.byte_size / 4;
            fill_f32_norm(data, off, n_elem);
        }
    }

    // Combine header + data
    Buf file;
    file.reserve(header.size() + data.size());
    file.insert(file.end(), header.begin(), header.end());
    file.insert(file.end(), data.begin(), data.end());

    // Write to temp file
    auto path = std::filesystem::temp_directory_path() / "mugen_test_transformer.gguf";
    FILE* fp = std::fopen(path.c_str(), "wb");
    MUGEN_CHECK(fp != nullptr);
    MUGEN_CHECK(std::fwrite(file.data(), 1, file.size(), fp) == file.size());
    std::fclose(fp);

    return TinyModel{path};
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Test: model construction from GGUF
// ─────────────────────────────────────────────────────────────────────────────

static void test_from_gguf() {
    auto tiny = build_tiny_gguf();

    auto gpu_res = mugen::MetalCompute::create();
    MUGEN_CHECK(gpu_res.has_value());
    auto& gpu = *gpu_res;

    auto parser_res = mugen::GGUFParser::parse(tiny.path);
    MUGEN_CHECK(parser_res.has_value());

    auto mmap_res = mugen::MmapLoader::map_file(tiny.path);
    MUGEN_CHECK(mmap_res.has_value());

    auto model_res = mugen::TransformerModel::from_gguf(
        gpu.get(), *parser_res, *mmap_res);
    MUGEN_CHECK(model_res.has_value());

    auto& model = *model_res;
    MUGEN_CHECK(model->config().vocab_size == kVocab);
    MUGEN_CHECK(model->config().embed_dim == kEmbed);
    MUGEN_CHECK(model->config().n_layers == kLayers);
    MUGEN_CHECK(model->config().n_heads == kHeads);
    MUGEN_CHECK(model->config().n_kv_heads == kKVHeads);
    MUGEN_CHECK(model->config().head_dim == kHeadDim);
    MUGEN_CHECK(model->config().ffn_dim == kFFN);
    MUGEN_CHECK(model->kv_cache() != nullptr);

    std::printf("  from_gguf PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: forward pass returns correct shape
// ─────────────────────────────────────────────────────────────────────────────

static void test_forward_shape() {
    auto tiny = build_tiny_gguf();

    auto gpu = std::move(*mugen::MetalCompute::create());
    auto parser = std::move(*mugen::GGUFParser::parse(tiny.path));
    auto mmap = std::move(*mugen::MmapLoader::map_file(tiny.path));
    auto model = std::move(*mugen::TransformerModel::from_gguf(
        gpu.get(), parser, mmap));

    std::vector<uint32_t> tokens = {1};
    auto logits_res = model->forward(tokens, 0);
    MUGEN_CHECK(logits_res.has_value());

    auto& logits = *logits_res;
    MUGEN_CHECK(logits.size() == kVocab);

    // Logits should contain finite values
    for (size_t i = 0; i < logits.size(); i++) {
        MUGEN_CHECK(std::isfinite(logits[i]));
    }

    std::printf("  forward_shape PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: multi-token forward (KV cache grows correctly)
// ─────────────────────────────────────────────────────────────────────────────

static void test_multi_token_forward() {
    auto tiny = build_tiny_gguf();

    auto gpu = std::move(*mugen::MetalCompute::create());
    auto parser = std::move(*mugen::GGUFParser::parse(tiny.path));
    auto mmap = std::move(*mugen::MmapLoader::map_file(tiny.path));
    auto model = std::move(*mugen::TransformerModel::from_gguf(
        gpu.get(), parser, mmap));

    // Token 1 at position 0
    auto r1 = model->forward({10}, 0);
    MUGEN_CHECK(r1.has_value());
    MUGEN_CHECK(r1->size() == kVocab);
    MUGEN_CHECK(model->kv_cache()->seq_len() == 1);

    // Token 2 at position 1
    auto r2 = model->forward({20}, 1);
    MUGEN_CHECK(r2.has_value());
    MUGEN_CHECK(r2->size() == kVocab);
    MUGEN_CHECK(model->kv_cache()->seq_len() == 2);

    // Token 3 at position 2
    auto r3 = model->forward({30}, 2);
    MUGEN_CHECK(r3.has_value());
    MUGEN_CHECK(r3->size() == kVocab);
    MUGEN_CHECK(model->kv_cache()->seq_len() == 3);

    // All logits should be finite
    for (float v : *r3) {
        MUGEN_CHECK(std::isfinite(v));
    }

    std::printf("  multi_token_forward PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: determinism — same input should give same output
// ─────────────────────────────────────────────────────────────────────────────

static void test_determinism() {
    auto tiny = build_tiny_gguf();

    // Run 1
    std::vector<float> logits_a;
    {
        auto gpu = std::move(*mugen::MetalCompute::create());
        auto parser = std::move(*mugen::GGUFParser::parse(tiny.path));
        auto mmap = std::move(*mugen::MmapLoader::map_file(tiny.path));
        auto model = std::move(*mugen::TransformerModel::from_gguf(
            gpu.get(), parser, mmap));

        auto r = model->forward({42}, 0);
        MUGEN_CHECK(r.has_value());
        logits_a = std::move(*r);
    }

    // Run 2
    std::vector<float> logits_b;
    {
        auto gpu = std::move(*mugen::MetalCompute::create());
        auto parser = std::move(*mugen::GGUFParser::parse(tiny.path));
        auto mmap = std::move(*mugen::MmapLoader::map_file(tiny.path));
        auto model = std::move(*mugen::TransformerModel::from_gguf(
            gpu.get(), parser, mmap));

        auto r = model->forward({42}, 0);
        MUGEN_CHECK(r.has_value());
        logits_b = std::move(*r);
    }

    MUGEN_CHECK(logits_a.size() == logits_b.size());
    for (size_t i = 0; i < logits_a.size(); i++) {
        float diff = std::fabs(logits_a[i] - logits_b[i]);
        MUGEN_CHECK(diff < 1e-4f);
    }

    std::printf("  determinism PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: multi-token batch input (prefill-style)
// ─────────────────────────────────────────────────────────────────────────────

static void test_batch_forward() {
    auto tiny = build_tiny_gguf();

    auto gpu = std::move(*mugen::MetalCompute::create());
    auto parser = std::move(*mugen::GGUFParser::parse(tiny.path));
    auto mmap = std::move(*mugen::MmapLoader::map_file(tiny.path));
    auto model = std::move(*mugen::TransformerModel::from_gguf(
        gpu.get(), parser, mmap));

    // Pass 3 tokens at once
    auto r = model->forward({5, 10, 15}, 0);
    MUGEN_CHECK(r.has_value());
    MUGEN_CHECK(r->size() == kVocab);
    MUGEN_CHECK(model->kv_cache()->seq_len() == 3);

    for (float v : *r) {
        MUGEN_CHECK(std::isfinite(v));
    }

    std::printf("  batch_forward PASS\n");
}

// ===========================================================================
int main() {
    std::printf("=== Transformer tests ===\n");

    test_from_gguf();
    test_forward_shape();
    test_multi_token_forward();
    test_determinism();
    test_batch_forward();

    std::printf("\nAll transformer tests passed.\n");
    return 0;
}

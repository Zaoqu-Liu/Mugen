#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <expected>
#include <string>
#include <filesystem>

namespace mugen {

struct KVCacheConfig {
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len = 8192;
    bool quantize_4bit = true;
    uint32_t fp16_preserve_last = 256;
};

class KVCache {
public:
    static auto create(KVCacheConfig config) -> std::expected<std::unique_ptr<KVCache>, std::string>;
    ~KVCache();

    auto append(uint32_t layer, const void* k_data, const void* v_data) -> bool;
    auto read_k(uint32_t layer, uint32_t start_pos, uint32_t len, void* out) const -> bool;
    auto read_v(uint32_t layer, uint32_t start_pos, uint32_t len, void* out) const -> bool;

    auto seq_len() const -> uint32_t;
    auto memory_bytes() const -> size_t;
    auto memory_bytes_fp16_equivalent() const -> size_t;
    auto compression_ratio() const -> float;

    void truncate(uint32_t new_len);
    void clear();
    auto config() const -> const KVCacheConfig&;

    /// Advance seq_len without storing data (lazy CPU KV mode).
    /// GPU scatter_kv handles the actual KV buffer writes; this only
    /// keeps the seq_len counter in sync for downstream consumers.
    void advance_seq_len(uint32_t n = 1);

private:
    KVCache(KVCacheConfig config);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mugen

#include "core/memory/kv_cache.h"
#include "mugen/core/types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace mugen {

// ---------------------------------------------------------------------------
// 4-bit group-wise quantization
//
// Format per group of 32 f16 values:
//   [f16 scale][f16 min_val][16 bytes packed nibbles]  = 20 bytes
// Original FP16: 32 * 2 = 64 bytes  =>  compression 64/20 = 3.2x
// ---------------------------------------------------------------------------

static constexpr uint32_t kGroupSize = 32;
static constexpr uint32_t kQ4GroupBytes = 20;

static void quantize_group(const f16* src, uint8_t* dst, uint32_t count) {
    f16 vals[kGroupSize]{};
    for (uint32_t i = 0; i < count; ++i)
        vals[i] = src[i];

    f16 lo = vals[0], hi = vals[0];
    for (uint32_t i = 1; i < kGroupSize; ++i) {
        if (vals[i] < lo) lo = vals[i];
        if (vals[i] > hi) hi = vals[i];
    }

    float range = static_cast<float>(hi) - static_cast<float>(lo);
    f16 scale = (range == 0.0f) ? f16(0) : f16(range / 15.0f);
    f16 min_val = lo;

    std::memcpy(dst, &scale, 2);
    std::memcpy(dst + 2, &min_val, 2);

    float sf = static_cast<float>(scale);
    float mf = static_cast<float>(min_val);

    for (uint32_t i = 0; i < 16; ++i) {
        uint8_t nib_lo = 0, nib_hi = 0;
        if (sf > 0.0f) {
            float v0 = static_cast<float>(vals[2 * i]);
            float v1 = static_cast<float>(vals[2 * i + 1]);
            nib_lo = static_cast<uint8_t>(
                std::min(15.0f, std::round((v0 - mf) / sf)));
            nib_hi = static_cast<uint8_t>(
                std::min(15.0f, std::round((v1 - mf) / sf)));
        }
        dst[4 + i] = static_cast<uint8_t>((nib_hi << 4) | nib_lo);
    }
}

static void dequantize_group(const uint8_t* src, f16* dst) {
    f16 scale, min_val;
    std::memcpy(&scale, src, 2);
    std::memcpy(&min_val, src + 2, 2);

    float sf = static_cast<float>(scale);
    float mf = static_cast<float>(min_val);

    for (uint32_t i = 0; i < 16; ++i) {
        uint8_t packed = src[4 + i];
        uint8_t nib_lo = packed & 0x0F;
        uint8_t nib_hi = (packed >> 4) & 0x0F;
        dst[2 * i]     = f16(mf + static_cast<float>(nib_lo) * sf);
        dst[2 * i + 1] = f16(mf + static_cast<float>(nib_hi) * sf);
    }
}

static void quantize_token(const f16* src, uint8_t* dst,
                           uint32_t values_per_token,
                           uint32_t groups_per_token) {
    for (uint32_t g = 0; g < groups_per_token; ++g) {
        uint32_t off = g * kGroupSize;
        uint32_t cnt = std::min(kGroupSize, values_per_token - off);
        quantize_group(src + off, dst + g * kQ4GroupBytes, cnt);
    }
}

static void dequantize_token(const uint8_t* src, f16* dst,
                             uint32_t values_per_token,
                             uint32_t groups_per_token) {
    f16 tmp[kGroupSize];
    for (uint32_t g = 0; g < groups_per_token; ++g) {
        uint32_t off = g * kGroupSize;
        uint32_t cnt = std::min(kGroupSize, values_per_token - off);
        dequantize_group(src + g * kQ4GroupBytes, tmp);
        std::memcpy(dst + off, tmp, cnt * sizeof(f16));
    }
}

// ---------------------------------------------------------------------------
// Per-layer storage
// ---------------------------------------------------------------------------

struct LayerCache {
    std::vector<uint8_t> q4_k;
    std::vector<uint8_t> q4_v;
    std::vector<f16>     fp16_k;
    std::vector<f16>     fp16_v;
    uint32_t n_q4   = 0;
    uint32_t n_fp16 = 0;
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct KVCache::Impl {
    KVCacheConfig  cfg;
    std::vector<LayerCache> layers;

    uint32_t values_per_token;
    uint32_t token_fp16_bytes;
    uint32_t groups_per_token;
    uint32_t token_q4_bytes;

    // Lazy CPU KV: phantom entries tracked by seq_len but with no CPU-side data.
    // GPU scatter_kv fills the actual GPU KV buffers; this counter keeps
    // seq_len() accurate without the cost of GPU→CPU read_buffer + quantize.
    uint32_t phantom_tokens = 0;

    explicit Impl(KVCacheConfig c)
        : cfg(c)
        , layers(c.n_layers)
        , values_per_token(c.n_kv_heads * c.head_dim)
        , token_fp16_bytes(values_per_token * sizeof(f16))
        , groups_per_token((values_per_token + kGroupSize - 1) / kGroupSize)
        , token_q4_bytes(groups_per_token * kQ4GroupBytes)
    {}
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

KVCache::KVCache(KVCacheConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

KVCache::~KVCache() = default;

auto KVCache::create(KVCacheConfig config)
    -> std::expected<std::unique_ptr<KVCache>, std::string>
{
    if (config.n_layers == 0)
        return std::unexpected("n_layers must be > 0");
    if (config.n_kv_heads == 0)
        return std::unexpected("n_kv_heads must be > 0");
    if (config.head_dim == 0)
        return std::unexpected("head_dim must be > 0");
    if (config.max_seq_len == 0)
        return std::unexpected("max_seq_len must be > 0");

    return std::unique_ptr<KVCache>(new KVCache(std::move(config)));
}

auto KVCache::append(uint32_t layer,
                     const void* k_data,
                     const void* v_data) -> bool
{
    if (layer >= impl_->cfg.n_layers) return false;

    auto& lc = impl_->layers[layer];
    uint32_t layer_seq = lc.n_q4 + lc.n_fp16;
    if (layer_seq >= impl_->cfg.max_seq_len) return false;

    const uint32_t vpt = impl_->values_per_token;
    const auto* k = static_cast<const f16*>(k_data);
    const auto* v = static_cast<const f16*>(v_data);

    lc.fp16_k.insert(lc.fp16_k.end(), k, k + vpt);
    lc.fp16_v.insert(lc.fp16_v.end(), v, v + vpt);
    lc.n_fp16++;

    if (impl_->cfg.quantize_4bit && lc.n_fp16 > impl_->cfg.fp16_preserve_last) {
        uint32_t to_quant = lc.n_fp16 - impl_->cfg.fp16_preserve_last;
        const uint32_t q4b = impl_->token_q4_bytes;
        const uint32_t gpt = impl_->groups_per_token;

        size_t q4k_off = lc.q4_k.size();
        lc.q4_k.resize(q4k_off + to_quant * q4b);

        size_t q4v_off = lc.q4_v.size();
        lc.q4_v.resize(q4v_off + to_quant * q4b);

        for (uint32_t t = 0; t < to_quant; ++t) {
            quantize_token(lc.fp16_k.data() + t * vpt,
                           lc.q4_k.data() + q4k_off + t * q4b,
                           vpt, gpt);
            quantize_token(lc.fp16_v.data() + t * vpt,
                           lc.q4_v.data() + q4v_off + t * q4b,
                           vpt, gpt);
        }

        lc.fp16_k.erase(lc.fp16_k.begin(),
                         lc.fp16_k.begin() +
                             static_cast<ptrdiff_t>(to_quant * vpt));
        lc.fp16_v.erase(lc.fp16_v.begin(),
                         lc.fp16_v.begin() +
                             static_cast<ptrdiff_t>(to_quant * vpt));

        lc.n_q4  += to_quant;
        lc.n_fp16 -= to_quant;
    }

    return true;
}

auto KVCache::read_k(uint32_t layer,
                     uint32_t start_pos,
                     uint32_t len,
                     void* out) const -> bool
{
    if (layer >= impl_->cfg.n_layers) return false;

    const auto& lc = impl_->layers[layer];
    uint32_t total = lc.n_q4 + lc.n_fp16;
    if (start_pos + len > total) return false;
    if (len == 0) return true;

    auto* dst = static_cast<f16*>(out);
    const uint32_t vpt = impl_->values_per_token;
    const uint32_t q4b = impl_->token_q4_bytes;
    const uint32_t gpt = impl_->groups_per_token;

    for (uint32_t i = 0; i < len; ++i) {
        uint32_t pos = start_pos + i;
        f16* token_dst = dst + i * vpt;

        if (pos < lc.n_q4) {
            dequantize_token(lc.q4_k.data() + pos * q4b,
                             token_dst, vpt, gpt);
        } else {
            uint32_t fp_idx = pos - lc.n_q4;
            std::memcpy(token_dst,
                        lc.fp16_k.data() + fp_idx * vpt,
                        vpt * sizeof(f16));
        }
    }
    return true;
}

auto KVCache::read_v(uint32_t layer,
                     uint32_t start_pos,
                     uint32_t len,
                     void* out) const -> bool
{
    if (layer >= impl_->cfg.n_layers) return false;

    const auto& lc = impl_->layers[layer];
    uint32_t total = lc.n_q4 + lc.n_fp16;
    if (start_pos + len > total) return false;
    if (len == 0) return true;

    auto* dst = static_cast<f16*>(out);
    const uint32_t vpt = impl_->values_per_token;
    const uint32_t q4b = impl_->token_q4_bytes;
    const uint32_t gpt = impl_->groups_per_token;

    for (uint32_t i = 0; i < len; ++i) {
        uint32_t pos = start_pos + i;
        f16* token_dst = dst + i * vpt;

        if (pos < lc.n_q4) {
            dequantize_token(lc.q4_v.data() + pos * q4b,
                             token_dst, vpt, gpt);
        } else {
            uint32_t fp_idx = pos - lc.n_q4;
            std::memcpy(token_dst,
                        lc.fp16_v.data() + fp_idx * vpt,
                        vpt * sizeof(f16));
        }
    }
    return true;
}

auto KVCache::seq_len() const -> uint32_t {
    if (impl_->layers.empty()) return 0;
    const auto& lc = impl_->layers[0];
    return lc.n_q4 + lc.n_fp16 + impl_->phantom_tokens;
}

auto KVCache::memory_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& lc : impl_->layers) {
        total += lc.q4_k.size() + lc.q4_v.size();
        total += lc.fp16_k.size() * sizeof(f16);
        total += lc.fp16_v.size() * sizeof(f16);
    }
    return total;
}

auto KVCache::memory_bytes_fp16_equivalent() const -> size_t {
    size_t total = 0;
    for (const auto& lc : impl_->layers) {
        uint32_t seq = lc.n_q4 + lc.n_fp16;
        total += static_cast<size_t>(seq) * impl_->token_fp16_bytes * 2;
    }
    return total;
}

auto KVCache::compression_ratio() const -> float {
    size_t actual = memory_bytes();
    if (actual == 0) return 1.0f;
    return static_cast<float>(memory_bytes_fp16_equivalent())
         / static_cast<float>(actual);
}

void KVCache::truncate(uint32_t new_len) {
    // Absorb phantoms first: if truncating below current materialized len,
    // phantoms are fully consumed. Otherwise reduce phantoms.
    uint32_t mat_len = impl_->layers.empty() ? 0
        : impl_->layers[0].n_q4 + impl_->layers[0].n_fp16;
    if (new_len >= mat_len) {
        impl_->phantom_tokens = new_len - mat_len;
        return;
    }
    impl_->phantom_tokens = 0;
    for (auto& lc : impl_->layers) {
        uint32_t total = lc.n_q4 + lc.n_fp16;
        if (new_len >= total) continue;

        if (new_len <= lc.n_q4) {
            lc.q4_k.resize(static_cast<size_t>(new_len) * impl_->token_q4_bytes);
            lc.q4_v.resize(static_cast<size_t>(new_len) * impl_->token_q4_bytes);
            lc.fp16_k.clear();
            lc.fp16_v.clear();
            lc.n_q4   = new_len;
            lc.n_fp16 = 0;
        } else {
            uint32_t new_fp16 = new_len - lc.n_q4;
            lc.fp16_k.resize(static_cast<size_t>(new_fp16) * impl_->values_per_token);
            lc.fp16_v.resize(static_cast<size_t>(new_fp16) * impl_->values_per_token);
            lc.n_fp16 = new_fp16;
        }
    }
}

void KVCache::clear() {
    impl_->phantom_tokens = 0;
    for (auto& lc : impl_->layers) {
        lc.q4_k.clear();
        lc.q4_v.clear();
        lc.fp16_k.clear();
        lc.fp16_v.clear();
        lc.n_q4   = 0;
        lc.n_fp16 = 0;
    }
}

void KVCache::advance_seq_len(uint32_t n) {
    impl_->phantom_tokens += n;
}

auto KVCache::config() const -> const KVCacheConfig& {
    return impl_->cfg;
}

} // namespace mugen

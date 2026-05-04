#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mugen {

class MetalCompute;
class GGUFParser;
class KVCache;
class MmapRegion;
class BufferManager;

struct DecodeProfile {
    double gpu_ms       = 0;
    double dispatch_wall_ms = 0;
    double logits_ms    = 0;
    double embed_ms     = 0;
    double patch_ms     = 0;
    double total_wall_ms = 0;
    uint32_t n_tokens   = 0;
    void reset() { *this = {}; }
};

struct PrefillProfile {
    double matmul_ms      = 0;
    double rms_norm_ms    = 0;
    double attention_ms   = 0;
    double rope_ms        = 0;
    double elementwise_ms = 0;
    double silu_mul_ms    = 0;
    double other_ms       = 0;
    double total_gpu_ms   = 0;
    double total_wall_ms  = 0;
    uint32_t n_tokens     = 0;
    void reset() { *this = {}; }
};

enum class LayerKind : uint8_t { Dense, MoE };

struct ModelConfig {
    uint32_t vocab_size;
    uint32_t embed_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t expert_ffn_dim = 0;  // MoE expert intermediate size (may differ from ffn_dim)
    float rope_theta;
    float rms_norm_eps;
    bool is_moe;
    uint32_t n_experts;
    uint32_t n_experts_used;
    uint32_t first_k_dense_replace = 0;

    uint32_t n_group = 0;               // DS V3: 8 (0 = no grouped routing)
    uint32_t topk_group = 0;            // DS V3: 4
    float routed_scaling_factor = 1.0f; // DS V3: 2.5

    // MLA (Multi-head Latent Attention) — DeepSeek V3
    uint32_t kv_lora_rank = 0;        // DS V3: 512 (0 = not MLA)
    uint32_t q_lora_rank = 0;         // DS V3: 1536
    uint32_t qk_rope_head_dim = 0;    // DS V3: 64
    uint32_t qk_nope_head_dim = 0;    // DS V3: 128
    uint32_t v_head_dim = 0;          // DS V3: 128
    uint32_t n_shared_experts = 0;    // DS V3: 1

    // YaRN RoPE scaling (DeepSeek V2-Lite: factor=40, original_ctx=4096)
    float rope_scaling_factor = 0.0f;
    uint32_t rope_original_ctx = 0;
    float rope_yarn_log_mult = 0.0f;

    bool is_mla() const { return kv_lora_rank > 0; }
    uint32_t shared_ffn_dim() const { return n_shared_experts * expert_ffn_dim; }
};

class TransformerModel {
public:
    static auto from_gguf(MetalCompute* gpu,
                          const GGUFParser& parser,
                          const MmapRegion& mmap)
        -> std::expected<std::unique_ptr<TransformerModel>, std::string>;

    static auto from_gguf(MetalCompute* gpu,
                          const GGUFParser& parser,
                          const std::vector<MmapRegion>& mmaps)
        -> std::expected<std::unique_ptr<TransformerModel>, std::string>;

    auto forward(const std::vector<uint32_t>& token_ids,
                 uint32_t start_position)
        -> std::expected<std::vector<float>, std::string>;

    auto forward_argmax(const std::vector<uint32_t>& token_ids,
                        uint32_t start_position)
        -> std::expected<uint32_t, std::string>;

    auto config() const -> const ModelConfig&;
    auto kv_cache() -> KVCache*;

    auto forward_prefill(const std::vector<uint32_t>& token_ids,
                         uint32_t start_position)
        -> std::expected<std::vector<float>, std::string>;

    auto forward_prefill_all_logits(const std::vector<uint32_t>& token_ids,
                                     uint32_t start_position)
        -> std::expected<std::vector<std::vector<float>>, std::string>;

    void set_use_gpu_attention(bool enable);
    auto use_gpu_attention() const -> bool;

    auto decode_profile() const -> const DecodeProfile&;
    void reset_decode_profile();

    auto prefill_profile() const -> const PrefillProfile&;
    void reset_prefill_profile();

    void set_prefill_profile_enabled(bool enable);
    auto prefill_profile_enabled() const -> bool;

    using RouteCallback = std::function<void(uint32_t layer,
                                             const uint32_t* expert_indices,
                                             const float* expert_weights,
                                             uint32_t top_k)>;
    void set_route_callback(RouteCallback cb);
    void set_buffer_manager(BufferManager* mgr);

    auto profile_kernel_dispatch()
        -> std::expected<std::vector<std::pair<std::string, double>>, std::string>;

    ~TransformerModel();
    TransformerModel(TransformerModel&&) noexcept;

private:
    TransformerModel();
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static auto from_gguf_impl(MetalCompute* gpu,
                                const GGUFParser& parser,
                                const std::vector<const void*>& mmap_bases)
        -> std::expected<std::unique_ptr<TransformerModel>, std::string>;
};

} // namespace mugen

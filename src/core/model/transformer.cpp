#include "core/model/transformer.h"

#include "core/compute/metal_compute.h"
#include "core/memory/kv_cache.h"
#include "core/memory/mmap_loader.h"
#include "model/ggml_types.h"
#include "model/gguf_parser.h"
#include "metal/kernel_sources.h"
#include "mugen/core/types.h"

// buffer_manager.h → expert_index.h defines a lightweight mugen::TensorInfo that
// collides with gguf_parser.h's richer version.  We only need ExpertKey and
// BufferManager (never ExpertIndex::build), so shadow the duplicate name.
#define TensorInfo TensorInfo_ExpertIndex_
#include "core/memory/buffer_manager.h"
#undef TensorInfo

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mugen {

#define MUGEN_ROPE_STANDARD 0  // 0 = NeoX pairing (current), 1 = Standard pairing
#define MUGEN_ROPE_NO_YARN  0  // 1 = disable YaRN, use base freq only

// YaRN frequency correction per DeepSeek V2 reference (beta_fast=32, beta_slow=1).
// Computes dimension boundaries from rotation counts, then applies a linear ramp:
//   dim < low  → keep original freq   (high-frequency dims, fine position resolution)
//   dim > high → freq / factor         (low-frequency dims, extend context range)
//   between    → linear interpolation
static float yarn_rope_freq(uint32_t dim_idx, uint32_t rope_dim,
                            float factor, uint32_t original_ctx, float theta) {
    float freq = 1.0f / std::pow(theta,
        2.0f * static_cast<float>(dim_idx) / static_cast<float>(rope_dim));
#if MUGEN_ROPE_NO_YARN
    return freq;
#endif
    if (factor <= 1.0f) return freq;

    constexpr float beta_fast = 32.0f;
    constexpr float beta_slow = 1.0f;
    float log_base = std::log(theta);
    float ctx_f = static_cast<float>(original_ctx);
    float dim_f = static_cast<float>(rope_dim);

    float low = std::floor(dim_f * std::log(ctx_f / (beta_fast * 2.0f *
        static_cast<float>(M_PI))) / (2.0f * log_base));
    float high = std::ceil(dim_f * std::log(ctx_f / (beta_slow * 2.0f *
        static_cast<float>(M_PI))) / (2.0f * log_base));
    low = std::max(low, 0.0f);
    high = std::min(high, static_cast<float>(rope_dim / 2 - 1));

    float idx = static_cast<float>(dim_idx);
    if (idx < low) return freq;
    if (idx > high) return freq / factor;

    float ramp = (idx - low) / std::max(0.001f, high - low);
    ramp = std::max(0.0f, std::min(1.0f, ramp));
    return freq * (1.0f - ramp) + (freq / factor) * ramp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

struct LayerWeights {
    LayerKind kind = LayerKind::Dense;

    void* attn_norm_buf = nullptr;
    void* ffn_norm_buf  = nullptr;
    void* wq_buf = nullptr;
    void* wk_buf = nullptr;
    void* wv_buf = nullptr;
    void* wo_buf = nullptr;
    void* w1_buf = nullptr;  // ffn_gate
    void* w2_buf = nullptr;  // ffn_down
    void* w3_buf = nullptr;  // ffn_up
    GGMLType wq_type{}, wk_type{}, wv_type{}, wo_type{};
    GGMLType w1_type{}, w2_type{}, w3_type{};

    void* bq_buf = nullptr;  // attn_q.bias (f16, may be null)
    void* bk_buf = nullptr;  // attn_k.bias
    void* bv_buf = nullptr;  // attn_v.bias

    void* q_norm_buf = nullptr;  // attn_q_norm.weight (f16, may be null — OLMo QK-Norm)
    void* k_norm_buf = nullptr;  // attn_k_norm.weight

    // MLA weights (only populated when model uses MLA attention)
    void* attn_q_a_buf = nullptr;      // blk.L.attn_q_a — q_lora_rank × embed_dim
    void* attn_q_b_buf = nullptr;      // blk.L.attn_q_b — n_heads*head_dim × q_lora_rank
    void* attn_q_a_norm_buf = nullptr; // blk.L.attn_q_a_norm — q_lora_rank (RMSNorm)
    void* attn_kv_a_mqa_buf = nullptr; // blk.L.attn_kv_a_mqa — (kv_lora_rank + qk_rope_head_dim) × embed_dim
    void* attn_kv_a_norm_buf = nullptr;// blk.L.attn_kv_a_norm — kv_lora_rank (RMSNorm)
    void* attn_k_b_buf = nullptr;      // blk.L.attn_k_b — n_heads*qk_nope_head_dim × kv_lora_rank
    void* attn_v_b_buf = nullptr;      // blk.L.attn_v_b — n_heads*v_head_dim × kv_lora_rank
    GGMLType attn_q_a_type{}, attn_q_b_type{};
    GGMLType attn_kv_a_mqa_type{};
    GGMLType attn_k_b_type{}, attn_v_b_type{};

    // MoE weights (only populated when kind == LayerKind::MoE)
    void* router_buf = nullptr;
    GGMLType router_type{};
    void* router_bias_buf = nullptr;  // exp_probs_b (f16, may be null)
    std::vector<void*> expert_gate_bufs;
    std::vector<void*> expert_up_bufs;
    std::vector<void*> expert_down_bufs;
    GGMLType expert_gate_type{}, expert_up_type{}, expert_down_type{};
    size_t expert_gate_bytes = 0;
    size_t expert_up_bytes   = 0;
    size_t expert_down_bytes = 0;

    // Shared expert weights (populated when n_shared_experts > 0)
    void* w1_shared_buf = nullptr;  // ffn_gate_shexp
    void* w2_shared_buf = nullptr;  // ffn_down_shexp
    void* w3_shared_buf = nullptr;  // ffn_up_shexp
    GGMLType w1_shared_type{}, w2_shared_type{}, w3_shared_type{};
};

struct TransformerModel::Impl {
    MetalCompute* gpu = nullptr;
    ModelConfig cfg{};
    std::unique_ptr<KVCache> kv_cache;

    void* token_embed_buf  = nullptr;
    GGMLType token_embed_type{};
    void* output_norm_buf  = nullptr;
    void* output_weight_buf = nullptr;
    GGMLType output_weight_type{};
    std::vector<LayerWeights> layers;

    void* lib = nullptr;
    void* pipe_rms_norm        = nullptr;
    void* pipe_matvec_f16      = nullptr;
    void* pipe_matvec_q4_0     = nullptr;
    void* pipe_matvec_q4_k     = nullptr;
    void* pipe_matvec_q8_0     = nullptr;
    void* pipe_softmax         = nullptr;
    void* pipe_rope            = nullptr;
    void* pipe_silu            = nullptr;
    void* pipe_elementwise_mul = nullptr;
    void* pipe_elementwise_add = nullptr;
    void* pipe_embedding_lookup = nullptr;
    void* pipe_moe_gate         = nullptr;
    void* pipe_moe_gate_grouped = nullptr;
    void* pipe_moe_reduce       = nullptr;
    void* pipe_attention_decode = nullptr;
    void* pipe_matmul_f16       = nullptr;
    void* pipe_matmul_q4_0      = nullptr;
    void* pipe_matmul_f16_naive  = nullptr;
    void* pipe_matmul_q4_0_naive = nullptr;
    void* pipe_batch_rope       = nullptr;
    void* pipe_bias_broadcast      = nullptr;
    void* pipe_prefill_attention   = nullptr;
    void* pipe_flash_prefill_attention = nullptr;
    void* pipe_flash_attention_decode = nullptr;
    void* pipe_scatter_kv = nullptr;
    void* pipe_silu_mul = nullptr;
    void* pipe_argmax_f16 = nullptr;

    void* buf_hidden   = nullptr;
    void* buf_hidden2  = nullptr;
    void* buf_q        = nullptr;
    void* buf_k        = nullptr;
    void* buf_v        = nullptr;
    void* buf_attn_out = nullptr;
    void* buf_ffn_gate = nullptr;
    void* buf_ffn_up   = nullptr;
    void* buf_logits   = nullptr;
    void* buf_token_id = nullptr;
    void* buf_argmax_result = nullptr;

    // MoE intermediate buffers
    void* buf_router_logits = nullptr;
    void* buf_top_indices   = nullptr;
    void* buf_top_weights   = nullptr;
    void* buf_expert_out    = nullptr;

    // Per-layer persistent GPU KV buffers (FP16, shared mode for CPU+GPU access)
    std::vector<void*> buf_kv_gpu_k;
    std::vector<void*> buf_kv_gpu_v;

    // MLA intermediate buffers (only allocated when model uses MLA attention)
    void* buf_mla_q_a = nullptr;
    void* buf_mla_q = nullptr;
    void* buf_mla_kv_a = nullptr;
    void* buf_mla_c_kv = nullptr;
    void* buf_mla_attn_out = nullptr;
    std::vector<void*> buf_kv_compressed;

    // Decode chain dispatch: GPU vs CPU attention toggle
    bool use_gpu_attention_ = false;

    // GPU argmax mode: when true, forward() dispatches argmax instead of logits readback
    bool argmax_mode_ = false;

    // Scratch buffers for decode (pre-allocated, avoid per-token heap allocation)
    std::vector<f16> scratch_k_cpu_;
    std::vector<f16> scratch_v_cpu_;
    std::vector<f16> scratch_q_cpu_;
    std::vector<f16> scratch_attn_out_;
    std::vector<f16> scratch_kv_k_;
    std::vector<f16> scratch_kv_v_;
    std::vector<float> scratch_scores_;

    // Mega-chain DP cache: avoid rebuilding ~600 DispatchParams per token.
    // First decode builds the template; subsequent decodes patch in-place.
    struct MegaChainPatchLoc {
        size_t group_idx;
        size_t dp_idx;
        size_t const_byte_offset;
    };
    std::vector<std::vector<MetalCompute::DispatchParams>> cached_mega_chain_;
    bool mega_chain_cached_ = false;
    std::vector<MegaChainPatchLoc> mega_rope_pos_locs_;
    std::vector<MegaChainPatchLoc> mega_kv_offset_locs_;
    std::vector<MegaChainPatchLoc> mega_seq_len_locs_;

    // Batch prefill buffers (lazily allocated on first prefill call)
    uint32_t batch_alloc_n = 0;
    void* buf_hidden_batch  = nullptr;
    void* buf_hidden2_batch = nullptr;
    void* buf_q_batch       = nullptr;
    void* buf_k_batch       = nullptr;
    void* buf_v_batch       = nullptr;
    void* buf_attn_batch    = nullptr;
    void* buf_ffn_gate_batch = nullptr;
    void* buf_ffn_up_batch   = nullptr;
    void* buf_token_ids_batch = nullptr;

    // MoE batch prefill buffers (lazily allocated alongside batch buffers)
    void* buf_router_logits_batch = nullptr;
    void* buf_moe_gather          = nullptr;
    void* buf_moe_scatter         = nullptr;

    // MLA batch prefill buffers (lazily allocated alongside batch buffers)
    void* buf_mla_q_a_batch      = nullptr;
    void* buf_mla_q_batch        = nullptr;
    void* buf_mla_kv_a_batch     = nullptr;
    void* buf_mla_attn_out_batch = nullptr;

    // All-position logits buffer (lazily allocated for batch logits output)
    void* buf_logits_batch = nullptr;
    uint32_t logits_batch_alloc_n = 0;

    // Pre-allocated logits readback buffer (avoids per-token 296KB malloc)
    std::vector<f16> scratch_logits_f16_;

    // Decode profiling (accumulated across forward() calls, reset via reset_decode_profile)
    DecodeProfile decode_profile_;

    // Prefill profiling (per-kernel GPU timestamps, reset via reset_prefill_profile)
    PrefillProfile prefill_profile_;

    // Profiling opt-in: when false (default), prefill uses batch dispatch for speed.
    // When true, each dispatch group gets its own command buffer for per-kernel timing.
    bool prefill_profile_enabled_ = false;

    // MoE route callback (optional, zero-overhead when null)
    TransformerModel::RouteCallback route_callback_;

    // BufferManager integration for SSD offload (optional, nullptr = pure mmap)
    BufferManager* buffer_mgr_ = nullptr;

    // Pre-allocated Metal staging buffers for CPU→GPU expert weight transfer.
    // When buffer_mgr_ finds an expert in its staging/pinned buffers, the raw
    // CPU data is memcpy'd into these shared Metal buffers, which the GPU can
    // then consume directly.  Allocated lazily on first use.
    void* buf_staging_gate = nullptr;
    void* buf_staging_up   = nullptr;
    void* buf_staging_down = nullptr;

    void ensure_staging_buffers() {
        if (buf_staging_gate) return;
        if (layers.empty() || !cfg.is_moe) return;
        size_t max_gate = 0, max_up = 0, max_down = 0;
        for (const auto& lw : layers) {
            max_gate = std::max(max_gate, lw.expert_gate_bytes);
            max_up   = std::max(max_up,   lw.expert_up_bytes);
            max_down = std::max(max_down, lw.expert_down_bytes);
        }
        if (max_gate > 0) buf_staging_gate = gpu->create_buffer(max_gate);
        if (max_up > 0)   buf_staging_up   = gpu->create_buffer(max_up);
        if (max_down > 0) buf_staging_down = gpu->create_buffer(max_down);
    }

    struct ResolvedExpert {
        void* gate;
        void* up;
        void* down;
    };

    // Look up an expert in BufferManager; if found, copy its tensor data into
    // pre-allocated Metal staging buffers and return those handles.  Falls back
    // to the original mmap-backed Metal buffers when buffer_mgr_ is null or the
    // expert is not cached.
    auto resolve_expert_bufs(uint32_t layer, uint32_t eid,
                             const LayerWeights& lw) -> ResolvedExpert {
        if (buffer_mgr_) {
            auto* expert_data = buffer_mgr_->find_expert({layer, eid});
            if (expert_data && lw.expert_gate_bytes > 0) {
                ensure_staging_buffers();
                if (buf_staging_gate && buf_staging_up && buf_staging_down) {
                    const auto* base = static_cast<const uint8_t*>(expert_data);
                    std::memcpy(gpu->buffer_contents(buf_staging_gate),
                                base,
                                lw.expert_gate_bytes);
                    std::memcpy(gpu->buffer_contents(buf_staging_up),
                                base + lw.expert_gate_bytes,
                                lw.expert_up_bytes);
                    std::memcpy(gpu->buffer_contents(buf_staging_down),
                                base + lw.expert_gate_bytes + lw.expert_up_bytes,
                                lw.expert_down_bytes);
                    return {buf_staging_gate, buf_staging_up, buf_staging_down};
                }
            }
        }
        return {lw.expert_gate_bufs[eid],
                lw.expert_up_bufs[eid],
                lw.expert_down_bufs[eid]};
    }

    // ─── Pipeline compilation ───

    auto compile_pipelines() -> std::expected<void, std::string> {
        auto lib_res = gpu->compile_library(metal::kAllKernelsSource, "mugen_kernels");
        if (!lib_res) return std::unexpected(lib_res.error());
        lib = *lib_res;

        struct PipeInfo {
            const char* name;
            void** target;
        };
        PipeInfo pipes[] = {
            {"rms_norm",          &pipe_rms_norm},
            {"matvec_f16",        &pipe_matvec_f16},
            {"matvec_q4_0",       &pipe_matvec_q4_0},
            {"matvec_q4_k",       &pipe_matvec_q4_k},
            {"matvec_q8_0",       &pipe_matvec_q8_0},
            {"softmax",           &pipe_softmax},
            {"rope",              &pipe_rope},
            {"silu",              &pipe_silu},
            {"elementwise_mul",   &pipe_elementwise_mul},
            {"elementwise_add",   &pipe_elementwise_add},
            {"embedding_lookup",  &pipe_embedding_lookup},
            {"moe_gate",          &pipe_moe_gate},
            {"moe_gate_grouped",  &pipe_moe_gate_grouped},
            {"moe_reduce",        &pipe_moe_reduce},
            {"attention_decode",  &pipe_attention_decode},
            {"matmul_f16",        &pipe_matmul_f16},
            {"matmul_q4_0",       &pipe_matmul_q4_0},
            {"matmul_f16_naive",  &pipe_matmul_f16_naive},
            {"matmul_q4_0_naive", &pipe_matmul_q4_0_naive},
            {"batch_rope",        &pipe_batch_rope},
            {"bias_broadcast",    &pipe_bias_broadcast},
            {"prefill_attention", &pipe_prefill_attention},
            {"flash_prefill_attention", &pipe_flash_prefill_attention},
            {"flash_attention_decode", &pipe_flash_attention_decode},
            {"scatter_kv", &pipe_scatter_kv},
            {"silu_mul", &pipe_silu_mul},
            {"argmax_f16", &pipe_argmax_f16},
        };

        for (auto& p : pipes) {
            auto fn = gpu->get_function(lib, p.name);
            if (!fn) return std::unexpected(fn.error());
            auto pso = gpu->create_pipeline(*fn);
            if (!pso) return std::unexpected(pso.error());
            *p.target = *pso;
        }
        return {};
    }

    // ─── Intermediate buffer allocation ───

    auto alloc_buffers() -> std::expected<void, std::string> {
        auto alloc = [&](size_t bytes) -> void* {
            return gpu->create_buffer(bytes);
        };

        size_t h_bytes = cfg.embed_dim * sizeof(f16);
        buf_hidden  = alloc(h_bytes);
        buf_hidden2 = alloc(h_bytes);
        buf_q       = alloc(cfg.n_heads * cfg.head_dim * sizeof(f16));
        buf_k       = alloc(cfg.n_kv_heads * cfg.head_dim * sizeof(f16));
        buf_v       = alloc(cfg.n_kv_heads * cfg.head_dim * sizeof(f16));
        buf_attn_out = alloc(cfg.n_heads * cfg.head_dim * sizeof(f16));
        uint32_t ffn_buf_dim = std::max(cfg.ffn_dim, cfg.shared_ffn_dim());
        buf_ffn_gate = alloc(ffn_buf_dim * sizeof(f16));
        buf_ffn_up   = alloc(ffn_buf_dim * sizeof(f16));
        buf_logits   = alloc(cfg.vocab_size * sizeof(f16));
        buf_token_id = alloc(sizeof(uint32_t));
        buf_argmax_result = alloc(sizeof(uint32_t));

        void* bufs[] = {buf_hidden, buf_hidden2, buf_q, buf_k, buf_v,
                        buf_attn_out, buf_ffn_gate, buf_ffn_up, buf_logits,
                        buf_token_id, buf_argmax_result};
        for (auto* b : bufs) {
            if (!b) return std::unexpected("Failed to allocate intermediate Metal buffer");
        }

        scratch_logits_f16_.resize(cfg.vocab_size);

        if (cfg.is_moe && cfg.n_experts > 0 && cfg.n_experts_used > 0) {
            buf_router_logits = alloc(cfg.n_experts * sizeof(f16));
            buf_top_indices   = alloc(cfg.n_experts_used * sizeof(uint32_t));
            buf_top_weights   = alloc(cfg.n_experts_used * sizeof(f16));
            buf_expert_out    = alloc(size_t(cfg.n_experts_used) * cfg.embed_dim * sizeof(f16));
            void* moe_bufs[] = {buf_router_logits, buf_top_indices,
                                buf_top_weights, buf_expert_out};
            for (auto* b : moe_bufs) {
                if (!b) return std::unexpected("Failed to allocate MoE intermediate Metal buffer");
            }
        }

        if (cfg.is_mla()) {
            if (cfg.q_lora_rank > 0) {
                buf_mla_q_a = alloc(cfg.q_lora_rank * sizeof(f16));
                if (!buf_mla_q_a)
                    return std::unexpected("Failed to allocate MLA intermediate Metal buffer");
            }
            uint32_t mla_q_dim = cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
            buf_mla_q = alloc(mla_q_dim * sizeof(f16));
            uint32_t compressed_kv_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
            buf_mla_kv_a = alloc(compressed_kv_dim * sizeof(f16));
            buf_mla_c_kv = alloc(cfg.kv_lora_rank * sizeof(f16));
            buf_mla_attn_out = alloc(size_t(cfg.n_heads) * cfg.v_head_dim * sizeof(f16));
            void* mla_bufs[] = {buf_mla_q, buf_mla_kv_a,
                                buf_mla_c_kv, buf_mla_attn_out};
            for (auto* b : mla_bufs) {
                if (!b) return std::unexpected("Failed to allocate MLA intermediate Metal buffer");
            }
        }

        return {};
    }

    // ─── Dispatch helpers ───

    auto dispatch_rms_norm(void* input, void* weight, void* output,
                           uint32_t dim, float eps)
        -> std::expected<void, std::string>
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_rms_norm;
        dp.buffers  = {input, weight, output};
        dp.constants = {
            {&dim, sizeof(dim)},
            {&eps, sizeof(eps)},
        };
        dp.grid_size  = {256, 1, 1};
        dp.group_size = {256, 1, 1};
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto select_matvec_pipeline(GGMLType type) -> void* {
        switch (type) {
            case GGMLType::F16:  return pipe_matvec_f16;
            case GGMLType::Q4_0: return pipe_matvec_q4_0;
            case GGMLType::Q4_K: return pipe_matvec_q4_k;
            case GGMLType::Q8_0: return pipe_matvec_q8_0;
            default: return nullptr;
        }
    }

    auto matvec_grid_group(GGMLType type, uint32_t M)
        -> std::pair<std::array<size_t,3>, std::array<size_t,3>>
    {
        // Returns {threadgroup_count, threads_per_threadgroup} for dispatchThreadgroups
        switch (type) {
            case GGMLType::F16:
                return {{size_t(M), 1, 1}, {32, 1, 1}};
            case GGMLType::Q4_0:
                // 8 rows/tg (NR0=4 × NSG=2), 2 simdgroups (64 threads)
                return {{size_t((M + 7) / 8), 1, 1}, {64, 1, 1}};
            case GGMLType::Q4_K:
            case GGMLType::Q8_0:
                return {{size_t(M), 1, 1}, {256, 1, 1}};
            default:
                return {{size_t(M), 1, 1}, {1, 1, 1}};
        }
    }

    auto dispatch_matvec(void* weight_buf, GGMLType wtype,
                         void* vec, void* output,
                         uint32_t M, uint32_t K)
        -> std::expected<void, std::string>
    {
        auto* pipe = select_matvec_pipeline(wtype);
        if (!pipe) {
            return std::unexpected(
                std::format("Unsupported weight type {} for matvec",
                            static_cast<uint32_t>(wtype)));
        }

        auto [grid, group] = matvec_grid_group(wtype, M);

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe;
        dp.buffers  = {weight_buf, vec, output};
        dp.constants = {
            {&M, sizeof(M)},
            {&K, sizeof(K)},
        };
        dp.grid_size  = grid;
        dp.group_size = group;
        dp.use_dispatch_threadgroups = true;

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_rope(void* x, uint32_t head_dim, uint32_t n_heads,
                       uint32_t position, float theta)
        -> std::expected<void, std::string>
    {
        uint32_t half_dim = head_dim / 2;
        size_t total_pairs = size_t(n_heads) * half_dim;
        size_t grp = std::min(size_t(half_dim), size_t(256));

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_rope;
        dp.buffers  = {x};
        dp.constants = {
            {&head_dim,  sizeof(head_dim)},
            {&n_heads,   sizeof(n_heads)},
            {&position,  sizeof(position)},
            {&theta,     sizeof(theta)},
        };
        dp.grid_size  = {total_pairs, 1, 1};
        dp.group_size = {grp, 1, 1};

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_silu(void* input, void* output, uint32_t N)
        -> std::expected<void, std::string>
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_silu;
        dp.buffers  = {input, output};
        dp.constants = {{&N, sizeof(N)}};
        dp.grid_size  = {N, 1, 1};
        dp.group_size = {std::min(size_t(N), size_t(256)), 1, 1};

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_elementwise(void* pipe, void* a, void* b, void* output, uint32_t N)
        -> std::expected<void, std::string>
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe;
        dp.buffers  = {a, b, output};
        dp.constants = {{&N, sizeof(N)}};
        dp.grid_size  = {N, 1, 1};
        dp.group_size = {std::min(size_t(N), size_t(256)), 1, 1};

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    // ─── DispatchParams builders (no dispatch, just build params) ───
    // Constants are stored in dp.const_data so pointers survive moves.

    static void store_const(MetalCompute::DispatchParams& dp,
                            const void* val, size_t sz) {
        size_t off = dp.const_data.size();
        dp.const_data.resize(off + sz);
        std::memcpy(dp.const_data.data() + off, val, sz);
    }

    static void finalize_consts(MetalCompute::DispatchParams& dp,
                                std::initializer_list<size_t> sizes) {
        dp.constants.clear();
        size_t off = 0;
        for (auto sz : sizes) {
            dp.constants.push_back({dp.const_data.data() + off, sz});
            off += sz;
        }
    }

    // ─── Prefill profiling helpers ───

    void accumulate_prefill_kernel(void* pipe, double ms) {
        if (pipe == pipe_matmul_q4_0 || pipe == pipe_matmul_f16 ||
            pipe == pipe_matmul_q4_0_naive || pipe == pipe_matmul_f16_naive ||
            pipe == pipe_matvec_q4_0 || pipe == pipe_matvec_f16 ||
            pipe == pipe_matvec_q8_0 || pipe == pipe_matvec_q4_k) {
            prefill_profile_.matmul_ms += ms;
        } else if (pipe == pipe_rms_norm) {
            prefill_profile_.rms_norm_ms += ms;
        } else if (pipe == pipe_prefill_attention ||
                   pipe == pipe_flash_prefill_attention) {
            prefill_profile_.attention_ms += ms;
        } else if (pipe == pipe_batch_rope || pipe == pipe_rope) {
            prefill_profile_.rope_ms += ms;
        } else if (pipe == pipe_elementwise_add ||
                   pipe == pipe_elementwise_mul) {
            prefill_profile_.elementwise_ms += ms;
        } else if (pipe == pipe_silu_mul || pipe == pipe_silu) {
            prefill_profile_.silu_mul_ms += ms;
        } else {
            prefill_profile_.other_ms += ms;
        }
        prefill_profile_.total_gpu_ms += ms;
    }

    auto dispatch_chain_profiled(
        const std::vector<std::vector<MetalCompute::DispatchParams>>& groups)
        -> std::expected<void, std::string>
    {
        for (auto& group : groups) {
            if (group.empty()) continue;
            auto r = gpu->dispatch_chain_sync({group});
            if (!r) return std::unexpected(r.error());
            accumulate_prefill_kernel(group[0].pipeline, *r * 1000.0);
        }
        return {};
    }

    auto dispatch_sync_profiled(const MetalCompute::DispatchParams& dp)
        -> std::expected<void, std::string>
    {
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        accumulate_prefill_kernel(dp.pipeline, *r * 1000.0);
        return {};
    }

    auto dispatch_chain_auto(
        const std::vector<std::vector<MetalCompute::DispatchParams>>& groups)
        -> std::expected<void, std::string>
    {
        if (prefill_profile_enabled_) {
            return dispatch_chain_profiled(groups);
        }
        auto r = gpu->dispatch_chain_sync(groups);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_sync_auto(const MetalCompute::DispatchParams& dp)
        -> std::expected<void, std::string>
    {
        if (prefill_profile_enabled_) {
            return dispatch_sync_profiled(dp);
        }
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto make_rms_norm_dp(void* input, void* weight, void* output,
                          uint32_t dim, float eps) -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_rms_norm;
        dp.buffers  = {input, weight, output};
        store_const(dp, &dim, sizeof(dim));
        store_const(dp, &eps, sizeof(eps));
        finalize_consts(dp, {sizeof(dim), sizeof(eps)});
        dp.grid_size  = {256, 1, 1};
        dp.group_size = {256, 1, 1};
        return dp;
    }

    auto make_matvec_dp(void* weight_buf, GGMLType wtype,
                        void* vec, void* output,
                        uint32_t M, uint32_t K) -> MetalCompute::DispatchParams
    {
        auto [grid, group] = matvec_grid_group(wtype, M);
        MetalCompute::DispatchParams dp;
        dp.pipeline = select_matvec_pipeline(wtype);
        dp.buffers  = {weight_buf, vec, output};
        store_const(dp, &M, sizeof(M));
        store_const(dp, &K, sizeof(K));
        finalize_consts(dp, {sizeof(M), sizeof(K)});
        dp.grid_size  = grid;
        dp.group_size = group;
        dp.use_dispatch_threadgroups = true;
        return dp;
    }

    auto make_elementwise_dp(void* pipe, void* a, void* b, void* out,
                             uint32_t dim) -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe;
        dp.buffers  = {a, b, out};
        store_const(dp, &dim, sizeof(dim));
        finalize_consts(dp, {sizeof(dim)});
        dp.grid_size  = {dim, 1, 1};
        dp.group_size = {std::min(size_t(dim), size_t(256)), 1, 1};
        return dp;
    }

    auto make_silu_dp(void* input, void* output, uint32_t N)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_silu;
        dp.buffers  = {input, output};
        store_const(dp, &N, sizeof(N));
        finalize_consts(dp, {sizeof(N)});
        dp.grid_size  = {N, 1, 1};
        dp.group_size = {std::min(size_t(N), size_t(256)), 1, 1};
        return dp;
    }

    auto make_silu_mul_dp(void* gate, void* up, void* output, uint32_t N)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_silu_mul;
        dp.buffers  = {gate, up, output};
        store_const(dp, &N, sizeof(N));
        finalize_consts(dp, {sizeof(N)});
        dp.grid_size  = {N, 1, 1};
        dp.group_size = {std::min(size_t(N), size_t(256)), 1, 1};
        return dp;
    }

    auto make_rope_dp(void* x, uint32_t head_dim, uint32_t n_heads,
                      uint32_t position, float theta)
        -> MetalCompute::DispatchParams
    {
        uint32_t half_dim = head_dim / 2;
        size_t total_pairs = size_t(n_heads) * half_dim;
        size_t grp = std::min(size_t(half_dim), size_t(256));

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_rope;
        dp.buffers  = {x};
        store_const(dp, &head_dim, sizeof(head_dim));
        store_const(dp, &n_heads, sizeof(n_heads));
        store_const(dp, &position, sizeof(position));
        store_const(dp, &theta, sizeof(theta));
        finalize_consts(dp, {sizeof(head_dim), sizeof(n_heads),
                             sizeof(position), sizeof(theta)});
        dp.grid_size  = {total_pairs, 1, 1};
        dp.group_size = {grp, 1, 1};
        return dp;
    }

    // ─── Batch DP builders (for chain dispatch in forward_prefill) ───

    auto make_matmul_dp(void* weight_buf, GGMLType wtype,
                        void* batch_in, void* batch_out,
                        uint32_t M, uint32_t K, uint32_t N)
        -> MetalCompute::DispatchParams
    {
        bool use_tiled = (K >= kTiledGemmMinK);

        MetalCompute::DispatchParams dp;
        dp.pipeline = select_matmul_pipeline(wtype, use_tiled);
        dp.buffers  = {weight_buf, batch_in, batch_out};
        store_const(dp, &M, sizeof(M));
        store_const(dp, &K, sizeof(K));
        store_const(dp, &N, sizeof(N));
        finalize_consts(dp, {sizeof(M), sizeof(K), sizeof(N)});

        if (use_tiled) {
            size_t tiles_M = (M + 63) / 64;
            size_t tiles_N = (N + 31) / 32;
            dp.grid_size  = {tiles_M * 128, tiles_N, 1};
            dp.group_size = {128, 1, 1};
        } else {
            dp.grid_size  = {((M + 15) / 16) * 16, ((N + 15) / 16) * 16, 1};
            dp.group_size = {16, 16, 1};
        }
        return dp;
    }

    auto make_bias_broadcast_dp(void* batch_buf, void* bias_buf,
                                uint32_t dim_per_tok, uint32_t n_tokens)
        -> MetalCompute::DispatchParams
    {
        uint32_t total = n_tokens * dim_per_tok;
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_bias_broadcast;
        dp.buffers  = {batch_buf, bias_buf};
        store_const(dp, &dim_per_tok, sizeof(dim_per_tok));
        finalize_consts(dp, {sizeof(dim_per_tok)});
        dp.grid_size  = {total, 1, 1};
        dp.group_size = {std::min(size_t(total), size_t(256)), 1, 1};
        return dp;
    }

    auto make_batch_rope_dp(void* buf, uint32_t head_dim_, uint32_t n_heads_,
                            uint32_t start_pos, float theta_, uint32_t n_tokens)
        -> MetalCompute::DispatchParams
    {
        uint32_t half_dim = head_dim_ / 2;
        size_t total = size_t(n_tokens) * n_heads_ * half_dim;
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_batch_rope;
        dp.buffers  = {buf};
        store_const(dp, &head_dim_, sizeof(head_dim_));
        store_const(dp, &n_heads_, sizeof(n_heads_));
        store_const(dp, &start_pos, sizeof(start_pos));
        store_const(dp, &theta_, sizeof(theta_));
        store_const(dp, &n_tokens, sizeof(n_tokens));
        finalize_consts(dp, {sizeof(uint32_t), sizeof(uint32_t),
                             sizeof(uint32_t), sizeof(float), sizeof(uint32_t)});
        dp.grid_size  = {total, 1, 1};
        dp.group_size = {std::min(total, size_t(256)), 1, 1};
        return dp;
    }

    auto make_prefill_attention_dp(void* q_batch, void* k_cache, void* v_cache,
                                   void* o_batch, uint32_t n_heads_,
                                   uint32_t n_kv_heads_, uint32_t head_dim_,
                                   uint32_t kv_len, uint32_t start_pos,
                                   uint32_t n_tokens)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_flash_prefill_attention;
        dp.buffers  = {q_batch, k_cache, v_cache, o_batch};
        store_const(dp, &n_heads_, sizeof(uint32_t));
        store_const(dp, &n_kv_heads_, sizeof(uint32_t));
        store_const(dp, &head_dim_, sizeof(uint32_t));
        store_const(dp, &kv_len, sizeof(uint32_t));
        store_const(dp, &start_pos, sizeof(uint32_t));
        store_const(dp, &n_tokens, sizeof(uint32_t));
        finalize_consts(dp, {sizeof(uint32_t), sizeof(uint32_t),
                             sizeof(uint32_t), sizeof(uint32_t),
                             sizeof(uint32_t), sizeof(uint32_t)});
        dp.grid_size  = {size_t(n_tokens) * size_t(n_heads_) * 256, 1, 1};
        dp.group_size = {256, 1, 1};
        return dp;
    }

    auto make_flash_attention_decode_dp(void* q, void* k_cache, void* v_cache,
                                        void* output, uint32_t n_heads_,
                                        uint32_t n_kv_heads_,
                                        uint32_t head_dim_, uint32_t seq_len)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_flash_attention_decode;
        dp.buffers  = {q, k_cache, v_cache, output};
        store_const(dp, &n_heads_, sizeof(uint32_t));
        store_const(dp, &n_kv_heads_, sizeof(uint32_t));
        store_const(dp, &head_dim_, sizeof(uint32_t));
        store_const(dp, &seq_len, sizeof(uint32_t));
        finalize_consts(dp, {sizeof(uint32_t), sizeof(uint32_t),
                             sizeof(uint32_t), sizeof(uint32_t)});
        dp.grid_size  = {size_t(n_heads_) * 256, 1, 1};
        dp.group_size = {256, 1, 1};
        return dp;
    }

    auto make_scatter_kv_dp(void* src, void* dst,
                            uint32_t offset, uint32_t count)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_scatter_kv;
        dp.buffers  = {src, dst};
        store_const(dp, &offset, sizeof(offset));
        store_const(dp, &count, sizeof(count));
        finalize_consts(dp, {sizeof(offset), sizeof(count)});
        size_t n = (size_t(count) + 255) / 256 * 256;
        dp.grid_size  = {n, 1, 1};
        dp.group_size = {256, 1, 1};
        return dp;
    }

    auto make_argmax_dp(void* input, void* result, uint32_t N)
        -> MetalCompute::DispatchParams
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_argmax_f16;
        dp.buffers  = {input, result};
        store_const(dp, &N, sizeof(N));
        finalize_consts(dp, {sizeof(N)});
        dp.grid_size  = {1024, 1, 1};
        dp.group_size = {1024, 1, 1};
        return dp;
    }

    auto dispatch_embedding_lookup(void* table, void* token_ids, void* output,
                                   uint32_t embed_dim, uint32_t n_tokens)
        -> std::expected<void, std::string>
    {
        size_t total = size_t(n_tokens) * embed_dim;

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_embedding_lookup;
        dp.buffers  = {table, token_ids, output};
        dp.constants = {
            {&embed_dim, sizeof(embed_dim)},
            {&n_tokens,  sizeof(n_tokens)},
        };
        dp.grid_size  = {total, 1, 1};
        dp.group_size = {std::min(size_t(embed_dim), size_t(256)), 1, 1};

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    // ─── Matmul dispatch helper ───

    static constexpr uint32_t kTiledGemmMinK = 2048;

    auto select_matmul_pipeline(GGMLType type, bool use_tiled = true) -> void* {
        if (use_tiled) {
            switch (type) {
                case GGMLType::F16:  return pipe_matmul_f16;
                case GGMLType::Q4_0: return pipe_matmul_q4_0;
                default: return nullptr;
            }
        } else {
            switch (type) {
                case GGMLType::F16:  return pipe_matmul_f16_naive;
                case GGMLType::Q4_0: return pipe_matmul_q4_0_naive;
                default: return nullptr;
            }
        }
    }

    auto dispatch_matmul(void* weight_buf, GGMLType wtype,
                         void* batch_in, void* batch_out,
                         uint32_t M, uint32_t K, uint32_t N)
        -> std::expected<void, std::string>
    {
        auto* pipe = select_matmul_pipeline(wtype);
        if (!pipe) {
            return std::unexpected(
                std::format("Unsupported weight type {} for matmul",
                            static_cast<uint32_t>(wtype)));
        }

        size_t gx = ((M + 15) / 16) * 16;
        size_t gy = ((N + 15) / 16) * 16;

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe;
        dp.buffers  = {weight_buf, batch_in, batch_out};
        dp.constants = {
            {&M, sizeof(M)},
            {&K, sizeof(K)},
            {&N, sizeof(N)},
        };
        dp.grid_size  = {gx, gy, 1};
        dp.group_size = {16, 16, 1};

        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    // ─── Batch buffer allocation (lazily on first prefill) ───

    void ensure_batch_buffers(uint32_t n_tokens) {
        if (batch_alloc_n >= n_tokens) return;

        auto alloc = [&](size_t bytes) { return gpu->create_buffer(bytes); };
        uint32_t n = n_tokens;
        batch_alloc_n = n;

        buf_hidden_batch   = alloc(size_t(n) * cfg.embed_dim * sizeof(f16));
        buf_hidden2_batch  = alloc(size_t(n) * cfg.embed_dim * sizeof(f16));
        buf_q_batch        = alloc(size_t(n) * cfg.n_heads * cfg.head_dim * sizeof(f16));
        buf_k_batch        = alloc(size_t(n) * cfg.n_kv_heads * cfg.head_dim * sizeof(f16));
        buf_v_batch        = alloc(size_t(n) * cfg.n_kv_heads * cfg.head_dim * sizeof(f16));
        buf_attn_batch     = alloc(size_t(n) * cfg.n_heads * cfg.head_dim * sizeof(f16));
        uint32_t ffn_buf_dim = std::max(cfg.ffn_dim, cfg.shared_ffn_dim());
        buf_ffn_gate_batch = alloc(size_t(n) * ffn_buf_dim * sizeof(f16));
        buf_ffn_up_batch   = alloc(size_t(n) * ffn_buf_dim * sizeof(f16));
        buf_token_ids_batch = alloc(size_t(n) * sizeof(uint32_t));

        if (cfg.is_moe && cfg.n_experts > 0) {
            buf_router_logits_batch = alloc(size_t(n) * cfg.n_experts * sizeof(f16));
            buf_moe_gather  = alloc(size_t(n) * cfg.embed_dim * sizeof(f16));
            buf_moe_scatter = alloc(size_t(n) * cfg.embed_dim * sizeof(f16));
        }

        if (cfg.is_mla()) {
            uint32_t mla_q_dim = cfg.n_heads
                * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
            uint32_t compressed_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
            uint32_t wo_input_dim = cfg.n_heads * cfg.v_head_dim;
            if (cfg.q_lora_rank > 0)
                buf_mla_q_a_batch = alloc(size_t(n) * cfg.q_lora_rank * sizeof(f16));
            buf_mla_q_batch = alloc(size_t(n) * mla_q_dim * sizeof(f16));
            buf_mla_kv_a_batch = alloc(size_t(n) * compressed_dim * sizeof(f16));
            buf_mla_attn_out_batch = alloc(size_t(n) * wo_input_dim * sizeof(f16));
        }
    }

    // ─── Logits batch buffer allocation (lazily on first all-logits prefill) ───

    void ensure_logits_batch_buffer(uint32_t n_tokens) {
        if (logits_batch_alloc_n >= n_tokens) return;
        logits_batch_alloc_n = n_tokens;
        buf_logits_batch = gpu->create_buffer(
            size_t(n_tokens) * cfg.vocab_size * sizeof(f16));
    }

    // ─── Shared prefill steps 1-2: embedding + transformer layers ───
    // Leaves result in buf_hidden_batch (N × embed_dim).

    auto forward_prefill_layers(const std::vector<uint32_t>& token_ids,
                                uint32_t start_position)
        -> std::expected<void, std::string>
    {
        const auto& c = cfg;
        uint32_t N = static_cast<uint32_t>(token_ids.size());

        ensure_batch_buffers(N);

        // 1. Batch embedding → buf_hidden_batch (N × embed_dim)
        {
            auto* hidden_ptr = static_cast<f16*>(
                gpu->buffer_contents(buf_hidden_batch));

            if (token_embed_type == GGMLType::F16) {
                auto* ids_ptr = static_cast<uint32_t*>(
                    gpu->buffer_contents(buf_token_ids_batch));
                for (uint32_t i = 0; i < N; i++) ids_ptr[i] = token_ids[i];
                auto r = dispatch_embedding_lookup(
                    token_embed_buf, buf_token_ids_batch, buf_hidden_batch,
                    c.embed_dim, N);
                if (!r) return std::unexpected(r.error());
            } else {
                auto* embed_ptr = static_cast<const uint8_t*>(
                    gpu->buffer_contents(token_embed_buf));

                for (uint32_t t = 0; t < N; t++) {
                    uint32_t tok = token_ids[t];
                    f16* row = hidden_ptr + size_t(t) * c.embed_dim;

                    if (token_embed_type == GGMLType::Q8_0) {
                        constexpr uint32_t Q8_BLOCK = 32;
                        constexpr size_t Q8_BYTES = 34;
                        uint32_t n_blocks = c.embed_dim / Q8_BLOCK;
                        const auto* src = embed_ptr + size_t(tok) * n_blocks * Q8_BYTES;
                        for (uint32_t b = 0; b < n_blocks; b++) {
                            const auto* blk = src + b * Q8_BYTES;
                            f16 scale_f16;
                            std::memcpy(&scale_f16, blk, sizeof(f16));
                            float scale = static_cast<float>(scale_f16);
                            const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
                            for (uint32_t j = 0; j < Q8_BLOCK; j++)
                                row[b * Q8_BLOCK + j] = static_cast<f16>(scale * static_cast<float>(qs[j]));
                        }
                    } else if (token_embed_type == GGMLType::Q4_0) {
                        constexpr uint32_t Q4_BLOCK = 32;
                        constexpr size_t Q4_BYTES = 18;
                        uint32_t n_blocks = c.embed_dim / Q4_BLOCK;
                        const auto* src = embed_ptr + size_t(tok) * n_blocks * Q4_BYTES;
                        for (uint32_t b = 0; b < n_blocks; b++) {
                            const auto* blk = src + b * Q4_BYTES;
                            f16 scale_f16;
                            std::memcpy(&scale_f16, blk, sizeof(f16));
                            float scale = static_cast<float>(scale_f16);
                            const uint8_t* qs = blk + 2;
                            for (uint32_t j = 0; j < 16; j++) {
                                int lo = (qs[j] & 0x0F) - 8;
                                int hi = (qs[j] >> 4) - 8;
                                row[b * Q4_BLOCK + j] = static_cast<f16>(lo * scale);
                                row[b * Q4_BLOCK + j + 16] = static_cast<f16>(hi * scale);
                            }
                        }
                    } else if (token_embed_type == GGMLType::F32) {
                        const auto* src = reinterpret_cast<const float*>(
                            embed_ptr + size_t(tok) * c.embed_dim * sizeof(float));
                        for (uint32_t i = 0; i < c.embed_dim; i++)
                            row[i] = static_cast<f16>(src[i]);
                    } else {
                        return std::unexpected("Unsupported embedding type for prefill");
                    }
                }
            }
        }

        // 2. Transformer layers (chain dispatch: 2 GPU round-trips per layer)
        for (uint32_t L = 0; L < c.n_layers; L++) {
            auto& lw = layers[L];

            if (c.is_mla()) {
                // ═══ MLA Prefill Attention Path ═══
                const uint32_t nope_dim = c.qk_nope_head_dim;
                const uint32_t rope_dim = c.qk_rope_head_dim;
                const uint32_t head_dim_mla = nope_dim + rope_dim;
                const uint32_t kv_rank = c.kv_lora_rank;
                const uint32_t compressed_dim = kv_rank + rope_dim;
                const uint32_t mla_q_dim = c.n_heads * head_dim_mla;
                const uint32_t wo_input_dim = c.n_heads * c.v_head_dim;
                uint32_t n_total = N * c.embed_dim;

                bool can_batch_q = (c.q_lora_rank > 0)
                    ? (select_matmul_pipeline(lw.attn_q_a_type) != nullptr &&
                       select_matmul_pipeline(lw.attn_q_b_type) != nullptr)
                    : (select_matmul_pipeline(lw.wq_type) != nullptr);
                bool can_batch_mla = can_batch_q
                    && select_matmul_pipeline(lw.attn_kv_a_mqa_type)
                    && select_matmul_pipeline(lw.wo_type);

                if (can_batch_mla) {

                // A. GPU chain: attn RMSNorm → Q & KV projections (batch)
                {
                    std::vector<std::vector<MetalCompute::DispatchParams>> chain;

                    auto rms_dp = make_rms_norm_dp(
                        buf_hidden_batch, lw.attn_norm_buf, buf_hidden2_batch,
                        c.embed_dim, c.rms_norm_eps);
                    rms_dp.grid_size = {size_t(N) * 256, 1, 1};
                    chain.push_back({std::move(rms_dp)});

                    if (c.q_lora_rank > 0) {
                        chain.push_back({
                            make_matmul_dp(lw.attn_q_a_buf, lw.attn_q_a_type,
                                buf_hidden2_batch, buf_mla_q_a_batch,
                                c.q_lora_rank, c.embed_dim, N),
                            make_matmul_dp(lw.attn_kv_a_mqa_buf, lw.attn_kv_a_mqa_type,
                                buf_hidden2_batch, buf_mla_kv_a_batch,
                                compressed_dim, c.embed_dim, N),
                        });

                        auto qa_rms = make_rms_norm_dp(
                            buf_mla_q_a_batch, lw.attn_q_a_norm_buf, buf_mla_q_a_batch,
                            c.q_lora_rank, c.rms_norm_eps);
                        qa_rms.grid_size = {size_t(N) * 256, 1, 1};
                        chain.push_back({std::move(qa_rms)});

                        chain.push_back({make_matmul_dp(lw.attn_q_b_buf, lw.attn_q_b_type,
                            buf_mla_q_a_batch, buf_mla_q_batch,
                            mla_q_dim, c.q_lora_rank, N)});
                    } else {
                        chain.push_back({
                            make_matmul_dp(lw.wq_buf, lw.wq_type,
                                buf_hidden2_batch, buf_mla_q_batch,
                                mla_q_dim, c.embed_dim, N),
                            make_matmul_dp(lw.attn_kv_a_mqa_buf, lw.attn_kv_a_mqa_type,
                                buf_hidden2_batch, buf_mla_kv_a_batch,
                                compressed_dim, c.embed_dim, N),
                        });
                    }

                    auto r = dispatch_chain_auto(chain);
                    if (!r) return std::unexpected(r.error());
                }

                } else {
                // A'. Per-token fallback for types without batch matmul (Q8_0)
                auto* h_batch_ptr = static_cast<f16*>(
                    gpu->buffer_contents(buf_hidden_batch));
                auto* q_batch_ptr = static_cast<f16*>(
                    gpu->buffer_contents(buf_mla_q_batch));
                auto* kva_batch_ptr = static_cast<f16*>(
                    gpu->buffer_contents(buf_mla_kv_a_batch));

                for (uint32_t t = 0; t < N; t++) {
                    std::memcpy(gpu->buffer_contents(buf_hidden),
                                h_batch_ptr + size_t(t) * c.embed_dim,
                                c.embed_dim * sizeof(f16));

                    {
                        auto r = dispatch_rms_norm(buf_hidden, lw.attn_norm_buf,
                                                    buf_hidden2, c.embed_dim, c.rms_norm_eps);
                        if (!r) return std::unexpected(r.error());
                    }

                    if (c.q_lora_rank > 0) {
                        auto r = dispatch_matvec(lw.attn_q_a_buf, lw.attn_q_a_type,
                                                  buf_hidden2, buf_mla_q_a,
                                                  c.q_lora_rank, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                        r = dispatch_rms_norm(buf_mla_q_a, lw.attn_q_a_norm_buf,
                                              buf_mla_q_a, c.q_lora_rank, c.rms_norm_eps);
                        if (!r) return std::unexpected(r.error());
                        r = dispatch_matvec(lw.attn_q_b_buf, lw.attn_q_b_type,
                                            buf_mla_q_a, buf_mla_q,
                                            mla_q_dim, c.q_lora_rank);
                        if (!r) return std::unexpected(r.error());
                    } else {
                        auto r = dispatch_matvec(lw.wq_buf, lw.wq_type,
                                                  buf_hidden2, buf_mla_q,
                                                  mla_q_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }

                    {
                        auto r = dispatch_matvec(lw.attn_kv_a_mqa_buf, lw.attn_kv_a_mqa_type,
                                                  buf_hidden2, buf_mla_kv_a,
                                                  compressed_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }

                    gpu->read_buffer(buf_mla_q,
                                     q_batch_ptr + size_t(t) * mla_q_dim,
                                     mla_q_dim * sizeof(f16));
                    gpu->read_buffer(buf_mla_kv_a,
                                     kva_batch_ptr + size_t(t) * compressed_dim,
                                     compressed_dim * sizeof(f16));
                }
                } // end can_batch_mla

                // B. CPU: NeoX RoPE on Q rope segments + KV RMSNorm/RoPE
                {
                    auto* q_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_mla_q_batch));
                    for (uint32_t t = 0; t < N; t++) {
                        f16* q_tok = q_ptr + size_t(t) * mla_q_dim;
                        for (uint32_t h = 0; h < c.n_heads; h++) {
                            f16* rs = q_tok + h * head_dim_mla + nope_dim;
                            uint32_t half = rope_dim / 2;
#if MUGEN_ROPE_STANDARD
                            for (uint32_t i = 0; i < half; i++) {
                                float freq = yarn_rope_freq(i, rope_dim,
                                    c.rope_scaling_factor, c.rope_original_ctx,
                                    c.rope_theta);
                                float angle =
                                    static_cast<float>(start_position + t) * freq;
                                float cv = std::cos(angle), sv = std::sin(angle);
                                float x0 = static_cast<float>(rs[2*i]);
                                float x1 = static_cast<float>(rs[2*i+1]);
                                rs[2*i] = static_cast<f16>(x0 * cv - x1 * sv);
                                rs[2*i+1] =
                                    static_cast<f16>(x1 * cv + x0 * sv);
                            }
#else
                            for (uint32_t i = 0; i < half; i++) {
                                float freq = yarn_rope_freq(i, rope_dim,
                                    c.rope_scaling_factor, c.rope_original_ctx,
                                    c.rope_theta);
                                float angle =
                                    static_cast<float>(start_position + t) * freq;
                                float cv = std::cos(angle), sv = std::sin(angle);
                                float x0 = static_cast<float>(rs[i]);
                                float x1 = static_cast<float>(rs[i + half]);
                                rs[i] = static_cast<f16>(x0 * cv - x1 * sv);
                                rs[i + half] =
                                    static_cast<f16>(x1 * cv + x0 * sv);
                            }
#endif
                        }
                    }

                    auto* kv_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_mla_kv_a_batch));
                    const auto* nw = static_cast<const f16*>(
                        gpu->buffer_contents(lw.attn_kv_a_norm_buf));
                    for (uint32_t t = 0; t < N; t++) {
                        f16* kva = kv_ptr + size_t(t) * compressed_dim;
                        float ss = 0.0f;
                        for (uint32_t i = 0; i < kv_rank; i++) {
                            float v = static_cast<float>(kva[i]);
                            ss += v * v;
                        }
                        ss = 1.0f / std::sqrt(ss / static_cast<float>(kv_rank)
                                              + c.rms_norm_eps);
                        for (uint32_t i = 0; i < kv_rank; i++) {
                            float v = static_cast<float>(kva[i]) * ss;
                            kva[i] = static_cast<f16>(
                                v * static_cast<float>(nw[i]));
                        }
                        f16* kr = kva + kv_rank;
                        uint32_t half = rope_dim / 2;
#if MUGEN_ROPE_STANDARD
                        for (uint32_t i = 0; i < half; i++) {
                            float freq = yarn_rope_freq(i, rope_dim,
                                c.rope_scaling_factor, c.rope_original_ctx,
                                c.rope_theta);
                            float angle =
                                static_cast<float>(start_position + t) * freq;
                            float cv = std::cos(angle), sv = std::sin(angle);
                            float x0 = static_cast<float>(kr[2*i]);
                            float x1 = static_cast<float>(kr[2*i+1]);
                            kr[2*i] = static_cast<f16>(x0 * cv - x1 * sv);
                            kr[2*i+1] =
                                static_cast<f16>(x1 * cv + x0 * sv);
                        }
#else
                        for (uint32_t i = 0; i < half; i++) {
                            float freq = yarn_rope_freq(i, rope_dim,
                                c.rope_scaling_factor, c.rope_original_ctx,
                                c.rope_theta);
                            float angle =
                                static_cast<float>(start_position + t) * freq;
                            float cv = std::cos(angle), sv = std::sin(angle);
                            float x0 = static_cast<float>(kr[i]);
                            float x1 = static_cast<float>(kr[i + half]);
                            kr[i] = static_cast<f16>(x0 * cv - x1 * sv);
                            kr[i + half] =
                                static_cast<f16>(x1 * cv + x0 * sv);
                        }
#endif
                    }
                }

                // C. Scatter compressed KV to per-layer cache
                {
                    uint32_t scatter_count = N * compressed_dim;
                    uint32_t scatter_offset = start_position * compressed_dim;
                    auto r = dispatch_sync_auto(make_scatter_kv_dp(
                        buf_mla_kv_a_batch, buf_kv_compressed[L],
                        scatter_offset, scatter_count));
                    if (!r) return std::unexpected(r.error());
                }

                // D. Per-token CPU MLA attention (absorbed)
                {
                    auto* q_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_mla_q_batch));
                    auto* attn_out_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_mla_attn_out_batch));
                    for (uint32_t t = 0; t < N; t++) {
                        f16* q_tok = q_ptr + size_t(t) * mla_q_dim;
                        f16* out_tok = attn_out_ptr + size_t(t) * wo_input_dim;
                        cpu_mla_attention(q_tok, lw, L,
                                          start_position + t + 1, out_tok);
                    }
                }

                // E. Wo projection + residual add
                if (can_batch_mla) {
                    std::vector<std::vector<MetalCompute::DispatchParams>> chain;
                    chain.push_back({make_matmul_dp(
                        lw.wo_buf, lw.wo_type, buf_mla_attn_out_batch,
                        buf_hidden2_batch, c.embed_dim, wo_input_dim, N)});
                    chain.push_back({make_elementwise_dp(
                        pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                        buf_hidden_batch, n_total)});
                    auto r = dispatch_chain_auto(chain);
                    if (!r) return std::unexpected(r.error());
                } else {
                    auto* ao_batch_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_mla_attn_out_batch));
                    auto* hb_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_hidden_batch));
                    for (uint32_t t = 0; t < N; t++) {
                        std::memcpy(gpu->buffer_contents(buf_mla_attn_out),
                                    ao_batch_ptr + size_t(t) * wo_input_dim,
                                    wo_input_dim * sizeof(f16));
                        auto r = dispatch_matvec(lw.wo_buf, lw.wo_type,
                                                  buf_mla_attn_out, buf_hidden2,
                                                  c.embed_dim, wo_input_dim);
                        if (!r) return std::unexpected(r.error());
                        auto* wo_out = static_cast<const f16*>(
                            gpu->buffer_contents(buf_hidden2));
                        f16* dst = hb_ptr + size_t(t) * c.embed_dim;
                        for (uint32_t i = 0; i < c.embed_dim; i++)
                            dst[i] = static_cast<f16>(
                                static_cast<float>(dst[i]) + static_cast<float>(wo_out[i]));
                    }
                }


                // F. FFN (Dense or MoE)
                bool can_batch_dense_ffn = (lw.kind == LayerKind::Dense)
                    && select_matmul_pipeline(lw.w1_type)
                    && select_matmul_pipeline(lw.w2_type)
                    && select_matmul_pipeline(lw.w3_type);

                if (lw.kind == LayerKind::Dense && can_batch_dense_ffn) {
                    std::vector<std::vector<MetalCompute::DispatchParams>> ffn_chain;
                    uint32_t ffn_total = N * c.ffn_dim;

                    auto ffn_rms = make_rms_norm_dp(
                        buf_hidden_batch, lw.ffn_norm_buf, buf_hidden2_batch,
                        c.embed_dim, c.rms_norm_eps);
                    ffn_rms.grid_size = {size_t(N) * 256, 1, 1};
                    ffn_chain.push_back({std::move(ffn_rms)});

                    ffn_chain.push_back({
                        make_matmul_dp(lw.w1_buf, lw.w1_type,
                            buf_hidden2_batch, buf_ffn_gate_batch,
                            c.ffn_dim, c.embed_dim, N),
                        make_matmul_dp(lw.w3_buf, lw.w3_type,
                            buf_hidden2_batch, buf_ffn_up_batch,
                            c.ffn_dim, c.embed_dim, N),
                    });

                    ffn_chain.push_back({make_silu_mul_dp(
                        buf_ffn_gate_batch, buf_ffn_up_batch,
                        buf_ffn_gate_batch, ffn_total)});

                    ffn_chain.push_back({make_matmul_dp(
                        lw.w2_buf, lw.w2_type, buf_ffn_gate_batch,
                        buf_hidden2_batch, c.embed_dim, c.ffn_dim, N)});

                    ffn_chain.push_back({make_elementwise_dp(
                        pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                        buf_hidden_batch, n_total)});

                    auto r = dispatch_chain_auto(ffn_chain);
                    if (!r) return std::unexpected(r.error());
                } else if (lw.kind == LayerKind::Dense) {
                    auto* hb_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_hidden_batch));
                    for (uint32_t t = 0; t < N; t++) {
                        std::memcpy(gpu->buffer_contents(buf_hidden),
                                    hb_ptr + size_t(t) * c.embed_dim,
                                    c.embed_dim * sizeof(f16));
                        {
                            auto r = dispatch_rms_norm(buf_hidden, lw.ffn_norm_buf,
                                                        buf_hidden2, c.embed_dim, c.rms_norm_eps);
                            if (!r) return std::unexpected(r.error());
                        }                        {
                            auto r = dispatch_matvec(lw.w1_buf, lw.w1_type,
                                                      buf_hidden2, buf_ffn_gate,
                                                      c.ffn_dim, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }                        {
                            auto r = dispatch_matvec(lw.w3_buf, lw.w3_type,
                                                      buf_hidden2, buf_ffn_up,
                                                      c.ffn_dim, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }                        {
                            auto r = gpu->dispatch_sync(make_silu_mul_dp(
                                buf_ffn_gate, buf_ffn_up, buf_ffn_gate, c.ffn_dim));
                            if (!r) return std::unexpected(r.error());
                        }                        {
                            auto r = dispatch_matvec(lw.w2_buf, lw.w2_type,
                                                      buf_ffn_gate, buf_hidden2,
                                                      c.embed_dim, c.ffn_dim);
                            if (!r) return std::unexpected(r.error());
                        }                        auto* h2_out = static_cast<const f16*>(
                            gpu->buffer_contents(buf_hidden2));
                        f16* dst = hb_ptr + size_t(t) * c.embed_dim;
                        for (uint32_t i = 0; i < c.embed_dim; i++)
                            dst[i] = static_cast<f16>(
                                static_cast<float>(dst[i]) + static_cast<float>(h2_out[i]));
                    }
                } else {
                    // MoE FFN: RMSNorm then batched or per-token expert routing
                    {
                        auto ffn_rms = make_rms_norm_dp(
                            buf_hidden_batch, lw.ffn_norm_buf, buf_hidden2_batch,
                            c.embed_dim, c.rms_norm_eps);
                        ffn_rms.grid_size = {size_t(N) * 256, 1, 1};
                        auto r = dispatch_sync_auto(std::move(ffn_rms));
                        if (!r) return std::unexpected(r.error());
                    }

                    bool can_batch_moe = select_matmul_pipeline(lw.router_type)
                        && select_matmul_pipeline(lw.expert_gate_type)
                        && select_matmul_pipeline(lw.expert_up_type)
                        && select_matmul_pipeline(lw.expert_down_type);

                    if (can_batch_moe) {
                        {
                            auto r = dispatch_sync_auto(make_matmul_dp(
                                lw.router_buf, lw.router_type,
                                buf_hidden2_batch, buf_router_logits_batch,
                                c.n_experts, c.embed_dim, N));
                            if (!r) return std::unexpected(r.error());
                        }

                        auto* logits_ptr = static_cast<const f16*>(
                            gpu->buffer_contents(buf_router_logits_batch));

                        const f16* bias_ptr = nullptr;
                        if (c.n_group > 0 && lw.router_bias_buf)
                            bias_ptr = static_cast<const f16*>(
                                gpu->buffer_contents(lw.router_bias_buf));

                        std::vector<std::vector<uint32_t>> expert_tok_idx(c.n_experts);
                        std::vector<std::vector<float>> expert_tok_wt(c.n_experts);

                        for (uint32_t t = 0; t < N; t++) {
                            const f16* tok_logits = logits_ptr + size_t(t) * c.n_experts;

                            uint32_t sel_idx[64];
                            float sel_vals[64];

                            if (c.n_group > 0) {
                                float scores[256];
                                for (uint32_t e = 0; e < c.n_experts; e++) {
                                    float x = static_cast<float>(tok_logits[e]);
                                    scores[e] = 1.0f / (1.0f + std::exp(-x));
                                }

                                uint32_t epg = c.n_experts / c.n_group;
                                float grp_sc[32];
                                for (uint32_t g = 0; g < c.n_group; g++) {
                                    float s1 = -std::numeric_limits<float>::infinity();
                                    float s2 = s1;
                                    for (uint32_t i = 0; i < epg; i++) {
                                        float s = scores[g * epg + i];
                                        if (s > s1) { s2 = s1; s1 = s; }
                                        else if (s > s2) { s2 = s; }
                                    }
                                    grp_sc[g] = s1 + s2;
                                }

                                bool gmask[32] = {};
                                for (uint32_t gt = 0; gt < c.topk_group; gt++) {
                                    float best = -std::numeric_limits<float>::infinity();
                                    uint32_t bg = 0;
                                    for (uint32_t g = 0; g < c.n_group; g++) {
                                        if (!gmask[g] && grp_sc[g] > best) {
                                            best = grp_sc[g]; bg = g;
                                        }
                                    }
                                    gmask[bg] = true;
                                }

                                for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                    float best_val = -std::numeric_limits<float>::infinity();
                                    uint32_t best_eid = 0;
                                    for (uint32_t e = 0; e < c.n_experts; e++) {
                                        if (!gmask[e / epg]) continue;
                                        float val = scores[e];
                                        if (bias_ptr) val += static_cast<float>(bias_ptr[e]);
                                        bool already = false;
                                        for (uint32_t p = 0; p < k; p++) {
                                            if (sel_idx[p] == e) { already = true; break; }
                                        }
                                        if (!already && val > best_val) {
                                            best_val = val; best_eid = e;
                                        }
                                    }
                                    sel_idx[k] = best_eid;
                                }

                                float wsum = 0.0f;
                                for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                    sel_vals[k] = scores[sel_idx[k]];
                                    wsum += sel_vals[k];
                                }
                                for (uint32_t k = 0; k < c.n_experts_used; k++)
                                    sel_vals[k] = sel_vals[k] / wsum * c.routed_scaling_factor;
                            } else {
                                for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                    float best_val = -std::numeric_limits<float>::infinity();
                                    uint32_t best_eid = 0;
                                    for (uint32_t e = 0; e < c.n_experts; e++) {
                                        float val = static_cast<float>(tok_logits[e]);
                                        bool already = false;
                                        for (uint32_t p = 0; p < k; p++) {
                                            if (sel_idx[p] == e) { already = true; break; }
                                        }
                                        if (!already && val > best_val) {
                                            best_val = val; best_eid = e;
                                        }
                                    }
                                    sel_idx[k] = best_eid;
                                    sel_vals[k] = best_val;
                                }

                                float max_v = sel_vals[0];
                                for (uint32_t k = 1; k < c.n_experts_used; k++)
                                    max_v = std::max(max_v, sel_vals[k]);
                                float wsum = 0.0f;
                                for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                    sel_vals[k] = std::exp(sel_vals[k] - max_v);
                                    wsum += sel_vals[k];
                                }
                                for (uint32_t k = 0; k < c.n_experts_used; k++)
                                    sel_vals[k] /= wsum;
                            }

                            if (route_callback_)
                                route_callback_(L, sel_idx, sel_vals, c.n_experts_used);

                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                expert_tok_idx[sel_idx[k]].push_back(t);
                                expert_tok_wt[sel_idx[k]].push_back(sel_vals[k]);
                            }
                        }

                        auto* moe_out_ptr = static_cast<f16*>(
                            gpu->buffer_contents(buf_attn_batch));
                        std::memset(moe_out_ptr, 0, size_t(N) * c.embed_dim * sizeof(f16));

                        auto* h2_src = static_cast<const f16*>(
                            gpu->buffer_contents(buf_hidden2_batch));
                        auto* gather_ptr = static_cast<f16*>(
                            gpu->buffer_contents(buf_moe_gather));
                        auto* scatter_ptr = static_cast<const f16*>(
                            gpu->buffer_contents(buf_moe_scatter));

                        for (uint32_t eid = 0; eid < c.n_experts; eid++) {
                            auto M_exp = static_cast<uint32_t>(expert_tok_idx[eid].size());
                            if (M_exp == 0) continue;

                            for (uint32_t m = 0; m < M_exp; m++) {
                                uint32_t ti = expert_tok_idx[eid][m];
                                std::memcpy(gather_ptr + size_t(m) * c.embed_dim,
                                            h2_src + size_t(ti) * c.embed_dim,
                                            c.embed_dim * sizeof(f16));
                            }

                            auto [ebuf_gate, ebuf_up, ebuf_down] =
                                resolve_expert_bufs(L, eid, lw);

                            uint32_t ffn_total = M_exp * c.expert_ffn_dim;
                            std::vector<std::vector<MetalCompute::DispatchParams>> ech;
                            ech.push_back({
                                make_matmul_dp(ebuf_gate, lw.expert_gate_type,
                                    buf_moe_gather, buf_ffn_gate_batch,
                                    c.expert_ffn_dim, c.embed_dim, M_exp),
                                make_matmul_dp(ebuf_up, lw.expert_up_type,
                                    buf_moe_gather, buf_ffn_up_batch,
                                    c.expert_ffn_dim, c.embed_dim, M_exp),
                            });
                            ech.push_back({make_silu_mul_dp(
                                buf_ffn_gate_batch, buf_ffn_up_batch,
                                buf_ffn_gate_batch, ffn_total)});
                            ech.push_back({make_matmul_dp(
                                ebuf_down, lw.expert_down_type,
                                buf_ffn_gate_batch, buf_moe_scatter,
                                c.embed_dim, c.expert_ffn_dim, M_exp)});

                            auto r = dispatch_chain_auto(ech);
                            if (!r) return std::unexpected(r.error());

                            for (uint32_t m = 0; m < M_exp; m++) {
                                uint32_t ti = expert_tok_idx[eid][m];
                                float w = expert_tok_wt[eid][m];
                                f16* dst = moe_out_ptr + size_t(ti) * c.embed_dim;
                                const f16* src = scatter_ptr + size_t(m) * c.embed_dim;
                                for (uint32_t i = 0; i < c.embed_dim; i++) {
                                    dst[i] = static_cast<f16>(
                                        static_cast<float>(dst[i]) +
                                        w * static_cast<float>(src[i]));
                                }
                            }
                        }

                        if (c.n_shared_experts > 0 && lw.w1_shared_buf) {
                            uint32_t ffn_n = N * c.shared_ffn_dim();
                            std::vector<std::vector<MetalCompute::DispatchParams>> sh;
                            sh.push_back({
                                make_matmul_dp(lw.w1_shared_buf, lw.w1_shared_type,
                                    buf_hidden2_batch, buf_ffn_gate_batch,
                                    c.shared_ffn_dim(), c.embed_dim, N),
                                make_matmul_dp(lw.w3_shared_buf, lw.w3_shared_type,
                                    buf_hidden2_batch, buf_ffn_up_batch,
                                    c.shared_ffn_dim(), c.embed_dim, N),
                            });
                            sh.push_back({make_silu_mul_dp(
                                buf_ffn_gate_batch, buf_ffn_up_batch,
                                buf_ffn_gate_batch, ffn_n)});
                            sh.push_back({make_matmul_dp(
                                lw.w2_shared_buf, lw.w2_shared_type,
                                buf_ffn_gate_batch, buf_moe_scatter,
                                c.embed_dim, c.shared_ffn_dim(), N)});
                            sh.push_back({make_elementwise_dp(
                                pipe_elementwise_add, buf_attn_batch,
                                buf_moe_scatter, buf_attn_batch, n_total)});
                            auto r = dispatch_chain_auto(sh);
                            if (!r) return std::unexpected(r.error());
                        }

                        {
                            auto r = dispatch_sync_auto(make_elementwise_dp(
                                pipe_elementwise_add, buf_hidden_batch, buf_attn_batch,
                                buf_hidden_batch, n_total));
                            if (!r) return std::unexpected(r.error());
                        }
                    } else {
                        auto* hb_ptr = static_cast<f16*>(
                            gpu->buffer_contents(buf_hidden_batch));
                        auto* h2_batch_ptr = static_cast<f16*>(
                            gpu->buffer_contents(buf_hidden2_batch));

                        bool has_shared = c.n_shared_experts > 0 && lw.w1_shared_buf;

                        for (uint32_t t = 0; t < N; t++) {
                            size_t tok_off = size_t(t) * c.embed_dim;
                            std::memcpy(gpu->buffer_contents(buf_hidden),
                                        hb_ptr + tok_off,
                                        c.embed_dim * sizeof(f16));

                            {
                                auto r = dispatch_rms_norm(buf_hidden, lw.ffn_norm_buf,
                                                            buf_hidden2, c.embed_dim, c.rms_norm_eps);
                                if (!r) return std::unexpected(r.error());
                            }

                            if (has_shared) {
                                std::memcpy(gpu->buffer_contents(buf_q),
                                            gpu->buffer_contents(buf_hidden2),
                                            c.embed_dim * sizeof(f16));
                            }

                            {
                                auto r = dispatch_matvec(
                                    lw.router_buf, lw.router_type,
                                    buf_hidden2, buf_router_logits,
                                    c.n_experts, c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                            }
                            if (c.n_group > 0) {
                                auto r = dispatch_moe_gate_grouped(
                                    buf_router_logits, buf_top_indices, buf_top_weights,
                                    lw.router_bias_buf, c.n_experts, c.n_experts_used,
                                    c.n_group, c.topk_group, c.routed_scaling_factor);
                                if (!r) return std::unexpected(r.error());
                            } else {
                                auto r = dispatch_moe_gate(
                                    buf_router_logits, buf_top_indices,
                                    buf_top_weights, c.n_experts, c.n_experts_used);
                                if (!r) return std::unexpected(r.error());
                            }

                            std::vector<uint32_t> top_idx(c.n_experts_used);
                            gpu->read_buffer(buf_top_indices, top_idx.data(),
                                             c.n_experts_used * sizeof(uint32_t));

                            if (route_callback_) {
                                std::vector<f16> wt_f16(c.n_experts_used);
                                gpu->read_buffer(buf_top_weights, wt_f16.data(),
                                                 c.n_experts_used * sizeof(f16));
                                std::vector<float> wt_f32(c.n_experts_used);
                                for (uint32_t ki = 0; ki < c.n_experts_used; ki++)
                                    wt_f32[ki] = static_cast<float>(wt_f16[ki]);
                                route_callback_(L, top_idx.data(), wt_f32.data(),
                                                c.n_experts_used);
                            }

                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                uint32_t eid = top_idx[k];
                                auto [ebuf_gate, ebuf_up, ebuf_down] =
                                    resolve_expert_bufs(L, eid, lw);
                                {
                                    auto r = dispatch_matvec(
                                        ebuf_gate, lw.expert_gate_type,
                                        buf_hidden2, buf_ffn_gate,
                                        c.expert_ffn_dim, c.embed_dim);
                                    if (!r) return std::unexpected(r.error());
                                }
                                {
                                    auto r = dispatch_matvec(
                                        ebuf_up, lw.expert_up_type,
                                        buf_hidden2, buf_ffn_up,
                                        c.expert_ffn_dim, c.embed_dim);
                                    if (!r) return std::unexpected(r.error());
                                }
                                {
                                    auto r = dispatch_silu(
                                        buf_ffn_gate, buf_ffn_gate, c.expert_ffn_dim);
                                    if (!r) return std::unexpected(r.error());
                                }
                                {
                                    auto r = dispatch_elementwise(
                                        pipe_elementwise_mul,
                                        buf_ffn_gate, buf_ffn_up,
                                        buf_ffn_gate, c.expert_ffn_dim);
                                    if (!r) return std::unexpected(r.error());
                                }
                                {
                                    auto r = dispatch_matvec(
                                        ebuf_down, lw.expert_down_type,
                                        buf_ffn_gate, buf_attn_out,
                                        c.embed_dim, c.expert_ffn_dim);
                                    if (!r) return std::unexpected(r.error());
                                }

                                std::vector<f16> expert_result(c.embed_dim);
                                gpu->read_buffer(buf_attn_out, expert_result.data(),
                                                 c.embed_dim * sizeof(f16));
                                auto* edst = static_cast<f16*>(
                                    gpu->buffer_contents(buf_expert_out));
                                std::memcpy(edst + size_t(k) * c.embed_dim,
                                            expert_result.data(),
                                            c.embed_dim * sizeof(f16));
                            }

                            {
                                auto r = dispatch_moe_reduce(
                                    buf_expert_out, buf_top_weights,
                                    buf_hidden2, c.embed_dim, c.n_experts_used);
                                if (!r) return std::unexpected(r.error());
                            }

                            if (has_shared) {
                                auto r = dispatch_matvec(lw.w1_shared_buf, lw.w1_shared_type,
                                    buf_q, buf_ffn_gate, c.shared_ffn_dim(), c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                                r = dispatch_matvec(lw.w3_shared_buf, lw.w3_shared_type,
                                    buf_q, buf_ffn_up, c.shared_ffn_dim(), c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                                r = dispatch_silu(buf_ffn_gate, buf_ffn_gate, c.shared_ffn_dim());
                                if (!r) return std::unexpected(r.error());
                                r = dispatch_elementwise(pipe_elementwise_mul,
                                    buf_ffn_gate, buf_ffn_up, buf_ffn_gate, c.shared_ffn_dim());
                                if (!r) return std::unexpected(r.error());
                                r = dispatch_matvec(lw.w2_shared_buf, lw.w2_shared_type,
                                    buf_ffn_gate, buf_attn_out, c.embed_dim, c.shared_ffn_dim());
                                if (!r) return std::unexpected(r.error());
                                r = dispatch_elementwise(pipe_elementwise_add,
                                    buf_hidden2, buf_attn_out, buf_hidden2, c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                            }

                            gpu->read_buffer(buf_hidden2, h2_batch_ptr + tok_off,
                                             c.embed_dim * sizeof(f16));
                        }

                        {
                            auto r = dispatch_sync_auto(make_elementwise_dp(
                                pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                                buf_hidden_batch, n_total));
                            if (!r) return std::unexpected(r.error());
                        }
                    }
                }


                continue;
            }

            uint32_t q_M  = c.n_heads * c.head_dim;
            uint32_t kv_M = c.n_kv_heads * c.head_dim;

            // ── Pre-attention chain: rms_norm → Q/K/V matmul → bias → batch_rope ──
            {
                std::vector<std::vector<MetalCompute::DispatchParams>> pre_attn;

                auto rms_dp = make_rms_norm_dp(
                    buf_hidden_batch, lw.attn_norm_buf, buf_hidden2_batch,
                    c.embed_dim, c.rms_norm_eps);
                rms_dp.grid_size = {size_t(N) * 256, 1, 1};
                pre_attn.push_back({std::move(rms_dp)});

                pre_attn.push_back({
                    make_matmul_dp(lw.wq_buf, lw.wq_type,
                        buf_hidden2_batch, buf_q_batch, q_M, c.embed_dim, N),
                    make_matmul_dp(lw.wk_buf, lw.wk_type,
                        buf_hidden2_batch, buf_k_batch, kv_M, c.embed_dim, N),
                    make_matmul_dp(lw.wv_buf, lw.wv_type,
                        buf_hidden2_batch, buf_v_batch, kv_M, c.embed_dim, N),
                });

                std::vector<MetalCompute::DispatchParams> bias_group;
                if (lw.bq_buf) bias_group.push_back(
                    make_bias_broadcast_dp(buf_q_batch, lw.bq_buf, q_M, N));
                if (lw.bk_buf) bias_group.push_back(
                    make_bias_broadcast_dp(buf_k_batch, lw.bk_buf, kv_M, N));
                if (lw.bv_buf) bias_group.push_back(
                    make_bias_broadcast_dp(buf_v_batch, lw.bv_buf, kv_M, N));
                if (!bias_group.empty()) pre_attn.push_back(std::move(bias_group));

                std::vector<MetalCompute::DispatchParams> qk_norm_group;
                if (lw.q_norm_buf) {
                    auto dp = make_rms_norm_dp(buf_q_batch, lw.q_norm_buf,
                                               buf_q_batch, q_M, c.rms_norm_eps);
                    dp.grid_size = {size_t(N) * 256, 1, 1};
                    qk_norm_group.push_back(std::move(dp));
                }
                if (lw.k_norm_buf) {
                    auto dp = make_rms_norm_dp(buf_k_batch, lw.k_norm_buf,
                                               buf_k_batch, kv_M, c.rms_norm_eps);
                    dp.grid_size = {size_t(N) * 256, 1, 1};
                    qk_norm_group.push_back(std::move(dp));
                }
                if (!qk_norm_group.empty()) pre_attn.push_back(std::move(qk_norm_group));

                pre_attn.push_back({
                    make_batch_rope_dp(buf_q_batch, c.head_dim, c.n_heads,
                                         start_position, c.rope_theta, N),
                    make_batch_rope_dp(buf_k_batch, c.head_dim, c.n_kv_heads,
                                         start_position, c.rope_theta, N),
                });

                auto r = dispatch_chain_auto(pre_attn);
                if (!r) return std::unexpected(r.error());
            }

            // ── CPU: KV cache append (GPU KV buffer updated via scatter_kv in chain) ──
            size_t kv_elems = size_t(c.n_kv_heads) * c.head_dim;
            {
                auto* k_batch_ptr = static_cast<f16*>(gpu->buffer_contents(buf_k_batch));
                auto* v_batch_ptr = static_cast<f16*>(gpu->buffer_contents(buf_v_batch));

                for (uint32_t t = 0; t < N; t++) {
                    f16* k_tok = k_batch_ptr + size_t(t) * kv_elems;
                    f16* v_tok = v_batch_ptr + size_t(t) * kv_elems;
                    kv_cache->append(L, k_tok, v_tok);
                }
            }

            // ── Post-attention chain: scatter_kv → attention → Wo → residual → FFN → residual ──
            if (lw.kind == LayerKind::Dense) {
                std::vector<std::vector<MetalCompute::DispatchParams>> post_attn;
                uint32_t n_total   = N * c.embed_dim;
                uint32_t ffn_total = N * c.ffn_dim;

                uint32_t scatter_count  = N * static_cast<uint32_t>(kv_elems);
                uint32_t scatter_offset = start_position * static_cast<uint32_t>(kv_elems);
                post_attn.push_back({
                    make_scatter_kv_dp(buf_k_batch, buf_kv_gpu_k[L],
                                       scatter_offset, scatter_count),
                    make_scatter_kv_dp(buf_v_batch, buf_kv_gpu_v[L],
                                       scatter_offset, scatter_count),
                });

                post_attn.push_back({make_prefill_attention_dp(
                    buf_q_batch, buf_kv_gpu_k[L], buf_kv_gpu_v[L],
                    buf_attn_batch, c.n_heads, c.n_kv_heads, c.head_dim,
                    start_position + N, start_position, N)});

                post_attn.push_back({make_matmul_dp(
                    lw.wo_buf, lw.wo_type, buf_attn_batch, buf_hidden2_batch,
                    c.embed_dim, q_M, N)});

                post_attn.push_back({make_elementwise_dp(
                    pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                    buf_hidden_batch, n_total)});

                auto ffn_rms = make_rms_norm_dp(
                    buf_hidden_batch, lw.ffn_norm_buf, buf_hidden2_batch,
                    c.embed_dim, c.rms_norm_eps);
                ffn_rms.grid_size = {size_t(N) * 256, 1, 1};
                post_attn.push_back({std::move(ffn_rms)});

                post_attn.push_back({
                    make_matmul_dp(lw.w1_buf, lw.w1_type,
                        buf_hidden2_batch, buf_ffn_gate_batch,
                        c.ffn_dim, c.embed_dim, N),
                    make_matmul_dp(lw.w3_buf, lw.w3_type,
                        buf_hidden2_batch, buf_ffn_up_batch,
                        c.ffn_dim, c.embed_dim, N),
                });

                post_attn.push_back({make_silu_mul_dp(
                    buf_ffn_gate_batch, buf_ffn_up_batch, buf_ffn_gate_batch, ffn_total)});

                post_attn.push_back({make_matmul_dp(
                    lw.w2_buf, lw.w2_type, buf_ffn_gate_batch, buf_hidden2_batch,
                    c.embed_dim, c.ffn_dim, N)});

                post_attn.push_back({make_elementwise_dp(
                    pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                    buf_hidden_batch, n_total)});

                auto r = dispatch_chain_auto(post_attn);
                if (!r) return std::unexpected(r.error());
            } else {
                // MoE prefill: batch attention/Wo/residual/rms_norm, then per-token MoE FFN
                uint32_t n_total = N * c.embed_dim;

                // Steps 0-3: scatter_kv → batch attention → Wo → residual → FFN rms_norm
                {
                    std::vector<std::vector<MetalCompute::DispatchParams>> pre_moe;

                    uint32_t scatter_count  = N * static_cast<uint32_t>(kv_elems);
                    uint32_t scatter_offset = start_position * static_cast<uint32_t>(kv_elems);
                    pre_moe.push_back({
                        make_scatter_kv_dp(buf_k_batch, buf_kv_gpu_k[L],
                                           scatter_offset, scatter_count),
                        make_scatter_kv_dp(buf_v_batch, buf_kv_gpu_v[L],
                                           scatter_offset, scatter_count),
                    });

                    pre_moe.push_back({make_prefill_attention_dp(
                        buf_q_batch, buf_kv_gpu_k[L], buf_kv_gpu_v[L],
                        buf_attn_batch, c.n_heads, c.n_kv_heads, c.head_dim,
                        start_position + N, start_position, N)});

                    pre_moe.push_back({make_matmul_dp(
                        lw.wo_buf, lw.wo_type, buf_attn_batch, buf_hidden2_batch,
                        c.embed_dim, q_M, N)});

                    pre_moe.push_back({make_elementwise_dp(
                        pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                        buf_hidden_batch, n_total)});

                    auto ffn_rms = make_rms_norm_dp(
                        buf_hidden_batch, lw.ffn_norm_buf, buf_hidden2_batch,
                        c.embed_dim, c.rms_norm_eps);
                    ffn_rms.grid_size = {size_t(N) * 256, 1, 1};
                    pre_moe.push_back({std::move(ffn_rms)});

                    auto r = dispatch_chain_auto(pre_moe);
                    if (!r) return std::unexpected(r.error());
                }

                // Step 4: batch MoE FFN
                bool can_batch_moe = select_matmul_pipeline(lw.router_type)
                    && select_matmul_pipeline(lw.expert_gate_type)
                    && select_matmul_pipeline(lw.expert_up_type)
                    && select_matmul_pipeline(lw.expert_down_type);

                if (can_batch_moe) {
                    // 4a. Batch router matmul: N × embed_dim → N × n_experts
                    {
                        auto r = dispatch_sync_auto(make_matmul_dp(
                            lw.router_buf, lw.router_type,
                            buf_hidden2_batch, buf_router_logits_batch,
                            c.n_experts, c.embed_dim, N));
                        if (!r) return std::unexpected(r.error());
                    }

                    // 4b. CPU top-K + expert grouping
                    auto* logits_ptr = static_cast<const f16*>(
                        gpu->buffer_contents(buf_router_logits_batch));

                    const f16* bias_ptr = nullptr;
                    if (c.n_group > 0 && lw.router_bias_buf)
                        bias_ptr = static_cast<const f16*>(
                            gpu->buffer_contents(lw.router_bias_buf));

                    std::vector<std::vector<uint32_t>> expert_tok_idx(c.n_experts);
                    std::vector<std::vector<float>> expert_tok_wt(c.n_experts);

                    for (uint32_t t = 0; t < N; t++) {
                        const f16* tok_logits = logits_ptr + size_t(t) * c.n_experts;

                        uint32_t sel_idx[64];
                        float sel_vals[64];

                        if (c.n_group > 0) {
                            // Grouped routing: sigmoid → group → mask → top-K → renorm
                            float scores[256];
                            for (uint32_t e = 0; e < c.n_experts; e++) {
                                float x = static_cast<float>(tok_logits[e]);
                                scores[e] = 1.0f / (1.0f + std::exp(-x));
                            }

                            uint32_t epg = c.n_experts / c.n_group;
                            float grp_sc[32];
                            for (uint32_t g = 0; g < c.n_group; g++) {
                                float s1 = -std::numeric_limits<float>::infinity();
                                float s2 = s1;
                                for (uint32_t i = 0; i < epg; i++) {
                                    float s = scores[g * epg + i];
                                    if (s > s1) { s2 = s1; s1 = s; }
                                    else if (s > s2) { s2 = s; }
                                }
                                grp_sc[g] = s1 + s2;
                            }

                            bool gmask[32] = {};
                            for (uint32_t gt = 0; gt < c.topk_group; gt++) {
                                float best = -std::numeric_limits<float>::infinity();
                                uint32_t bg = 0;
                                for (uint32_t g = 0; g < c.n_group; g++) {
                                    if (!gmask[g] && grp_sc[g] > best) {
                                        best = grp_sc[g]; bg = g;
                                    }
                                }
                                gmask[bg] = true;
                            }

                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                float best_val = -std::numeric_limits<float>::infinity();
                                uint32_t best_eid = 0;
                                for (uint32_t e = 0; e < c.n_experts; e++) {
                                    if (!gmask[e / epg]) continue;
                                    float val = scores[e];
                                    if (bias_ptr) val += static_cast<float>(bias_ptr[e]);
                                    bool already = false;
                                    for (uint32_t p = 0; p < k; p++) {
                                        if (sel_idx[p] == e) { already = true; break; }
                                    }
                                    if (!already && val > best_val) {
                                        best_val = val; best_eid = e;
                                    }
                                }
                                sel_idx[k] = best_eid;
                            }

                            float wsum = 0.0f;
                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                sel_vals[k] = scores[sel_idx[k]];
                                wsum += sel_vals[k];
                            }
                            for (uint32_t k = 0; k < c.n_experts_used; k++)
                                sel_vals[k] = sel_vals[k] / wsum * c.routed_scaling_factor;
                        } else {
                            // Original: raw logits → top-K → softmax
                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                float best_val = -std::numeric_limits<float>::infinity();
                                uint32_t best_eid = 0;
                                for (uint32_t e = 0; e < c.n_experts; e++) {
                                    float val = static_cast<float>(tok_logits[e]);
                                    bool already = false;
                                    for (uint32_t p = 0; p < k; p++) {
                                        if (sel_idx[p] == e) { already = true; break; }
                                    }
                                    if (!already && val > best_val) {
                                        best_val = val; best_eid = e;
                                    }
                                }
                                sel_idx[k] = best_eid;
                                sel_vals[k] = best_val;
                            }

                            float max_v = sel_vals[0];
                            for (uint32_t k = 1; k < c.n_experts_used; k++)
                                max_v = std::max(max_v, sel_vals[k]);
                            float wsum = 0.0f;
                            for (uint32_t k = 0; k < c.n_experts_used; k++) {
                                sel_vals[k] = std::exp(sel_vals[k] - max_v);
                                wsum += sel_vals[k];
                            }
                            for (uint32_t k = 0; k < c.n_experts_used; k++)
                                sel_vals[k] /= wsum;
                        }

                        if (route_callback_)
                            route_callback_(L, sel_idx, sel_vals, c.n_experts_used);

                        for (uint32_t k = 0; k < c.n_experts_used; k++) {
                            expert_tok_idx[sel_idx[k]].push_back(t);
                            expert_tok_wt[sel_idx[k]].push_back(sel_vals[k]);
                        }
                    }

                    // 4c. Zero MoE output accumulator (reuse buf_attn_batch = N × embed_dim)
                    auto* moe_out_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_attn_batch));
                    std::memset(moe_out_ptr, 0, size_t(N) * c.embed_dim * sizeof(f16));

                    auto* h2_src = static_cast<const f16*>(
                        gpu->buffer_contents(buf_hidden2_batch));
                    auto* gather_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_moe_gather));
                    auto* scatter_ptr = static_cast<const f16*>(
                        gpu->buffer_contents(buf_moe_scatter));

                    // 4d. Per-expert batch matmul FFN
                    for (uint32_t eid = 0; eid < c.n_experts; eid++) {
                        auto M_exp = static_cast<uint32_t>(expert_tok_idx[eid].size());
                        if (M_exp == 0) continue;

                        for (uint32_t m = 0; m < M_exp; m++) {
                            uint32_t ti = expert_tok_idx[eid][m];
                            std::memcpy(gather_ptr + size_t(m) * c.embed_dim,
                                        h2_src + size_t(ti) * c.embed_dim,
                                        c.embed_dim * sizeof(f16));
                        }

                        auto [ebuf_gate, ebuf_up, ebuf_down] =
                            resolve_expert_bufs(L, eid, lw);

                        uint32_t ffn_total = M_exp * c.expert_ffn_dim;
                        std::vector<std::vector<MetalCompute::DispatchParams>> ech;
                        ech.push_back({
                            make_matmul_dp(ebuf_gate, lw.expert_gate_type,
                                buf_moe_gather, buf_ffn_gate_batch,
                                c.expert_ffn_dim, c.embed_dim, M_exp),
                            make_matmul_dp(ebuf_up, lw.expert_up_type,
                                buf_moe_gather, buf_ffn_up_batch,
                                c.expert_ffn_dim, c.embed_dim, M_exp),
                        });
                        ech.push_back({make_silu_mul_dp(
                            buf_ffn_gate_batch, buf_ffn_up_batch, buf_ffn_gate_batch, ffn_total)});
                        ech.push_back({make_matmul_dp(
                            ebuf_down, lw.expert_down_type,
                            buf_ffn_gate_batch, buf_moe_scatter,
                            c.embed_dim, c.expert_ffn_dim, M_exp)});

                        auto r = dispatch_chain_auto(ech);
                        if (!r) return std::unexpected(r.error());

                        for (uint32_t m = 0; m < M_exp; m++) {
                            uint32_t ti = expert_tok_idx[eid][m];
                            float w = expert_tok_wt[eid][m];
                            f16* dst = moe_out_ptr + size_t(ti) * c.embed_dim;
                            const f16* src = scatter_ptr + size_t(m) * c.embed_dim;
                            for (uint32_t i = 0; i < c.embed_dim; i++) {
                                dst[i] = static_cast<f16>(
                                    static_cast<float>(dst[i]) +
                                    w * static_cast<float>(src[i]));
                            }
                        }
                    }

                    if (c.n_shared_experts > 0 && lw.w1_shared_buf) {
                        uint32_t ffn_n = N * c.shared_ffn_dim();
                        std::vector<std::vector<MetalCompute::DispatchParams>> sh;
                        sh.push_back({
                            make_matmul_dp(lw.w1_shared_buf, lw.w1_shared_type,
                                buf_hidden2_batch, buf_ffn_gate_batch,
                                c.shared_ffn_dim(), c.embed_dim, N),
                            make_matmul_dp(lw.w3_shared_buf, lw.w3_shared_type,
                                buf_hidden2_batch, buf_ffn_up_batch,
                                c.shared_ffn_dim(), c.embed_dim, N),
                        });
                        sh.push_back({make_silu_mul_dp(
                            buf_ffn_gate_batch, buf_ffn_up_batch,
                            buf_ffn_gate_batch, ffn_n)});
                        sh.push_back({make_matmul_dp(
                            lw.w2_shared_buf, lw.w2_shared_type,
                            buf_ffn_gate_batch, buf_moe_scatter,
                            c.embed_dim, c.shared_ffn_dim(), N)});
                        sh.push_back({make_elementwise_dp(
                            pipe_elementwise_add, buf_attn_batch,
                            buf_moe_scatter, buf_attn_batch, n_total)});
                        auto r = dispatch_chain_auto(sh);
                        if (!r) return std::unexpected(r.error());
                    }

                    // Step 5: batch residual add — buf_hidden_batch += moe_output
                    {
                        auto r = dispatch_sync_auto(make_elementwise_dp(
                            pipe_elementwise_add, buf_hidden_batch, buf_attn_batch,
                            buf_hidden_batch, n_total));
                        if (!r) return std::unexpected(r.error());
                    }
                } else {
                    // Fallback: per-token MoE FFN for unsupported matmul weight types
                    auto* h2_batch_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_hidden2_batch));
                    auto* h2_ptr = static_cast<f16*>(
                        gpu->buffer_contents(buf_hidden2));

                    bool has_shared = c.n_shared_experts > 0 && lw.w1_shared_buf;

                    for (uint32_t t = 0; t < N; t++) {
                        size_t tok_off = size_t(t) * c.embed_dim;
                        std::memcpy(h2_ptr, h2_batch_ptr + tok_off,
                                    c.embed_dim * sizeof(f16));

                        if (has_shared) {
                            std::memcpy(gpu->buffer_contents(buf_q),
                                        h2_ptr, c.embed_dim * sizeof(f16));
                        }

                        {
                            auto r = dispatch_matvec(
                                lw.router_buf, lw.router_type,
                                buf_hidden2, buf_router_logits,
                                c.n_experts, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        if (c.n_group > 0) {
                            auto r = dispatch_moe_gate_grouped(
                                buf_router_logits, buf_top_indices, buf_top_weights,
                                lw.router_bias_buf, c.n_experts, c.n_experts_used,
                                c.n_group, c.topk_group, c.routed_scaling_factor);
                            if (!r) return std::unexpected(r.error());
                        } else {
                            auto r = dispatch_moe_gate(
                                buf_router_logits, buf_top_indices,
                                buf_top_weights, c.n_experts, c.n_experts_used);
                            if (!r) return std::unexpected(r.error());
                        }

                        std::vector<uint32_t> top_idx(c.n_experts_used);
                        gpu->read_buffer(buf_top_indices, top_idx.data(),
                                         c.n_experts_used * sizeof(uint32_t));

                        if (route_callback_) {
                            std::vector<f16> wt_f16(c.n_experts_used);
                            gpu->read_buffer(buf_top_weights, wt_f16.data(),
                                             c.n_experts_used * sizeof(f16));
                            std::vector<float> wt_f32(c.n_experts_used);
                            for (uint32_t ki = 0; ki < c.n_experts_used; ki++)
                                wt_f32[ki] = static_cast<float>(wt_f16[ki]);
                            route_callback_(L, top_idx.data(), wt_f32.data(),
                                            c.n_experts_used);
                        }

                        for (uint32_t k = 0; k < c.n_experts_used; k++) {
                            uint32_t eid = top_idx[k];
                            auto [ebuf_gate, ebuf_up, ebuf_down] =
                                resolve_expert_bufs(L, eid, lw);
                            {
                                auto r = dispatch_matvec(
                                    ebuf_gate, lw.expert_gate_type,
                                    buf_hidden2, buf_ffn_gate,
                                    c.expert_ffn_dim, c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                            }
                            {
                                auto r = dispatch_matvec(
                                    ebuf_up, lw.expert_up_type,
                                    buf_hidden2, buf_ffn_up,
                                    c.expert_ffn_dim, c.embed_dim);
                                if (!r) return std::unexpected(r.error());
                            }
                            {
                                auto r = dispatch_silu(
                                    buf_ffn_gate, buf_ffn_gate, c.expert_ffn_dim);
                                if (!r) return std::unexpected(r.error());
                            }
                            {
                                auto r = dispatch_elementwise(
                                    pipe_elementwise_mul,
                                    buf_ffn_gate, buf_ffn_up,
                                    buf_ffn_gate, c.expert_ffn_dim);
                                if (!r) return std::unexpected(r.error());
                            }
                            {
                                auto r = dispatch_matvec(
                                    ebuf_down, lw.expert_down_type,
                                    buf_ffn_gate, buf_attn_out,
                                    c.embed_dim, c.expert_ffn_dim);
                                if (!r) return std::unexpected(r.error());
                            }

                            std::vector<f16> expert_result(c.embed_dim);
                            gpu->read_buffer(buf_attn_out, expert_result.data(),
                                             c.embed_dim * sizeof(f16));
                            auto* edst = static_cast<f16*>(
                                gpu->buffer_contents(buf_expert_out));
                            std::memcpy(edst + size_t(k) * c.embed_dim,
                                        expert_result.data(),
                                        c.embed_dim * sizeof(f16));
                        }

                        {
                            auto r = dispatch_moe_reduce(
                                buf_expert_out, buf_top_weights,
                                buf_hidden2, c.embed_dim, c.n_experts_used);
                            if (!r) return std::unexpected(r.error());
                        }

                        if (has_shared) {
                            auto r = dispatch_matvec(lw.w1_shared_buf, lw.w1_shared_type,
                                buf_q, buf_ffn_gate, c.shared_ffn_dim(), c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                            r = dispatch_matvec(lw.w3_shared_buf, lw.w3_shared_type,
                                buf_q, buf_ffn_up, c.shared_ffn_dim(), c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                            r = dispatch_silu(buf_ffn_gate, buf_ffn_gate, c.shared_ffn_dim());
                            if (!r) return std::unexpected(r.error());
                            r = dispatch_elementwise(pipe_elementwise_mul,
                                buf_ffn_gate, buf_ffn_up, buf_ffn_gate, c.shared_ffn_dim());
                            if (!r) return std::unexpected(r.error());
                            r = dispatch_matvec(lw.w2_shared_buf, lw.w2_shared_type,
                                buf_ffn_gate, buf_attn_out, c.embed_dim, c.shared_ffn_dim());
                            if (!r) return std::unexpected(r.error());
                            r = dispatch_elementwise(pipe_elementwise_add,
                                buf_hidden2, buf_attn_out, buf_hidden2, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }

                        gpu->read_buffer(buf_hidden2, h2_batch_ptr + tok_off,
                                         c.embed_dim * sizeof(f16));
                    }

                    // Step 5: batch residual add (fallback)
                    {
                        auto r = dispatch_sync_auto(make_elementwise_dp(
                            pipe_elementwise_add, buf_hidden_batch, buf_hidden2_batch,
                            buf_hidden_batch, n_total));
                        if (!r) return std::unexpected(r.error());
                    }
                }
            }

        }

        if (c.is_mla()) {
            kv_cache->advance_seq_len(N);
        }

        return {};
    }

    // ─── MoE dispatch helpers ───

    auto dispatch_moe_gate(void* gate_logits_buf, void* top_indices_buf,
                           void* top_weights_buf,
                           uint32_t n_experts, uint32_t top_k)
        -> std::expected<void, std::string>
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_moe_gate;
        dp.buffers  = {gate_logits_buf, top_indices_buf, top_weights_buf};
        dp.constants = {
            {&n_experts, sizeof(n_experts)},
            {&top_k, sizeof(top_k)},
        };
        dp.grid_size  = {1, 1, 1};
        dp.group_size = {1, 1, 1};
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_moe_gate_grouped(void* gate_logits_buf, void* top_indices_buf,
                                   void* top_weights_buf, void* router_bias_buf,
                                   uint32_t n_experts, uint32_t top_k,
                                   uint32_t n_group, uint32_t topk_group,
                                   float scaling_factor)
        -> std::expected<void, std::string>
    {
        uint32_t has_bias = (router_bias_buf != nullptr) ? 1u : 0u;
        void* bias_buf = router_bias_buf ? router_bias_buf : gate_logits_buf;

        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_moe_gate_grouped;
        dp.buffers  = {gate_logits_buf, top_indices_buf, top_weights_buf, bias_buf};
        dp.constants = {
            {&n_experts, sizeof(n_experts)},
            {&top_k, sizeof(top_k)},
            {&n_group, sizeof(n_group)},
            {&topk_group, sizeof(topk_group)},
            {&has_bias, sizeof(has_bias)},
            {&scaling_factor, sizeof(scaling_factor)},
        };
        dp.grid_size  = {1, 1, 1};
        dp.group_size = {1, 1, 1};
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    auto dispatch_moe_reduce(void* expert_outputs_buf, void* weights_buf,
                             void* output_buf, uint32_t dim, uint32_t K)
        -> std::expected<void, std::string>
    {
        MetalCompute::DispatchParams dp;
        dp.pipeline = pipe_moe_reduce;
        dp.buffers  = {expert_outputs_buf, weights_buf, output_buf};
        dp.constants = {
            {&dim, sizeof(dim)},
            {&K, sizeof(K)},
        };
        dp.grid_size  = {dim, 1, 1};
        dp.group_size = {std::min(size_t(dim), size_t(256)), 1, 1};
        auto r = gpu->dispatch_sync(dp);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    // ─── CPU Attention ───

    void cpu_attention(const f16* q_data, uint32_t layer, uint32_t seq_len,
                       f16* output)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
        uint32_t heads_per_kv = cfg.n_heads / cfg.n_kv_heads;
        uint32_t kv_stride = cfg.n_kv_heads * cfg.head_dim;

        kv_cache->read_k(layer, 0, seq_len, scratch_kv_k_.data());
        kv_cache->read_v(layer, 0, seq_len, scratch_kv_v_.data());

        for (uint32_t h = 0; h < cfg.n_heads; h++) {
            uint32_t kv_h = h / heads_per_kv;
            const f16* q_head = q_data + h * cfg.head_dim;
            const f16* k_base = scratch_kv_k_.data() + kv_h * cfg.head_dim;
            const f16* v_base = scratch_kv_v_.data() + kv_h * cfg.head_dim;

            float* scores = scratch_scores_.data();
            for (uint32_t s = 0; s < seq_len; s++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < cfg.head_dim; d++) {
                    dot += static_cast<float>(q_head[d]) *
                           static_cast<float>(k_base[s * kv_stride + d]);
                }
                scores[s] = dot * scale;
            }

            float max_val = *std::max_element(scores, scores + seq_len);
            float sum = 0.0f;
            for (uint32_t s = 0; s < seq_len; s++) {
                scores[s] = std::exp(scores[s] - max_val);
                sum += scores[s];
            }
            for (uint32_t s = 0; s < seq_len; s++) scores[s] /= sum;

            f16* out_head = output + h * cfg.head_dim;
            for (uint32_t d = 0; d < cfg.head_dim; d++) {
                float val = 0.0f;
                for (uint32_t s = 0; s < seq_len; s++) {
                    val += scores[s] *
                           static_cast<float>(v_base[s * kv_stride + d]);
                }
                out_head[d] = static_cast<f16>(val);
            }
        }
    }

    // Dequantize a single row from a weight matrix stored in a Metal buffer.
    // Supports F16, Q8_0, Q4_0, Q4_K; row has K columns.
    static void dequant_weight_row(const void* buf_contents, GGMLType type,
                                   uint32_t row, uint32_t K, float* out)
    {
        if (type == GGMLType::F16) {
            const auto* data = static_cast<const f16*>(buf_contents);
            const f16* rp = data + size_t(row) * K;
            for (uint32_t i = 0; i < K; i++)
                out[i] = static_cast<float>(rp[i]);
        } else if (type == GGMLType::Q8_0) {
            constexpr uint32_t BLK = 32;
            constexpr size_t BLK_BYTES = 34;
            uint32_t nb = K / BLK;
            const auto* base = static_cast<const uint8_t*>(buf_contents)
                               + size_t(row) * nb * BLK_BYTES;
            for (uint32_t b = 0; b < nb; b++) {
                const auto* blk = base + b * BLK_BYTES;
                f16 sf16;
                std::memcpy(&sf16, blk, sizeof(f16));
                float s = static_cast<float>(sf16);
                const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
                for (uint32_t j = 0; j < BLK; j++)
                    out[b * BLK + j] = s * static_cast<float>(qs[j]);
            }
        } else if (type == GGMLType::Q4_0) {
            constexpr uint32_t BLK = 32;
            constexpr size_t BLK_BYTES = 18;
            uint32_t nb = K / BLK;
            const auto* base = static_cast<const uint8_t*>(buf_contents)
                               + size_t(row) * nb * BLK_BYTES;
            for (uint32_t b = 0; b < nb; b++) {
                const auto* blk = base + b * BLK_BYTES;
                f16 sf16;
                std::memcpy(&sf16, blk, sizeof(f16));
                float s = static_cast<float>(sf16);
                const uint8_t* qs = blk + 2;
                for (uint32_t j = 0; j < 16; j++) {
                    out[b * BLK + j]      = static_cast<float>((qs[j] & 0x0F) - 8) * s;
                    out[b * BLK + j + 16] = static_cast<float>((qs[j] >> 4) - 8) * s;
                }
            }
        } else {
            fprintf(stderr, "[WARN] dequant_weight_row: unsupported type %u, zero-filling row %u\n",
                    static_cast<unsigned>(type), row);
            std::memset(out, 0, K * sizeof(float));
        }
    }

    // MLA absorbed attention (decode, CPU path).
    // q_data: n_heads × (qk_nope_head_dim + qk_rope_head_dim) as f16
    // output: n_heads × v_head_dim as f16
    void cpu_mla_attention(const f16* q_data, const LayerWeights& lw,
                           uint32_t layer, uint32_t seq_len,
                           f16* output)
    {
        const auto& c = cfg;
        const uint32_t nope_dim = c.qk_nope_head_dim;
        const uint32_t rope_dim = c.qk_rope_head_dim;
        const uint32_t head_dim_mla = nope_dim + rope_dim;
        const uint32_t kv_rank = c.kv_lora_rank;
        const uint32_t compressed_dim = kv_rank + rope_dim;
        // TODO: re-evaluate mscale when GGUF has explicit rope.scaling.attn_factor
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_mla));

        const auto* kv_cache_data = static_cast<const f16*>(
            gpu->buffer_contents(buf_kv_compressed[layer]));
        const auto* k_b_contents = gpu->buffer_contents(lw.attn_k_b_buf);
        const auto* v_b_contents = gpu->buffer_contents(lw.attn_v_b_buf);

        std::vector<float> row_buf(kv_rank);

        for (uint32_t h = 0; h < c.n_heads; h++) {
            const f16* q_nope = q_data + h * head_dim_mla;
            const f16* q_rope = q_data + h * head_dim_mla + nope_dim;

            // Absorb k_b into query: q_absorbed = q_nope × k_b_h^T
            // k_b shape: (n_heads*nope_dim) rows × kv_rank cols
            std::vector<float> q_absorbed(kv_rank, 0.0f);
            for (uint32_t i = 0; i < nope_dim; i++) {
                float qi = static_cast<float>(q_nope[i]);
                if (qi == 0.0f) continue;
                dequant_weight_row(k_b_contents, lw.attn_k_b_type,
                                   h * nope_dim + i, kv_rank, row_buf.data());
                for (uint32_t d = 0; d < kv_rank; d++)
                    q_absorbed[d] += qi * row_buf[d];
            }


            // Score: dot(q_absorbed, c_kv) + dot(q_rope, k_rope) for each position
            float* scores = scratch_scores_.data();
            for (uint32_t s = 0; s < seq_len; s++) {
                const f16* cs = kv_cache_data + size_t(s) * compressed_dim;
                float sc = 0.0f;
                for (uint32_t d = 0; d < kv_rank; d++)
                    sc += q_absorbed[d] * static_cast<float>(cs[d]);
                for (uint32_t d = 0; d < rope_dim; d++)
                    sc += static_cast<float>(q_rope[d]) *
                          static_cast<float>(cs[kv_rank + d]);
                scores[s] = sc * scale;
            }



            // Softmax
            float max_val = *std::max_element(scores, scores + seq_len);
            float sum = 0.0f;
            for (uint32_t s = 0; s < seq_len; s++) {
                scores[s] = std::exp(scores[s] - max_val);
                sum += scores[s];
            }
            for (uint32_t s = 0; s < seq_len; s++) scores[s] /= sum;


            // Weighted sum over c_kv (first kv_rank dims only)
            std::vector<float> out_comp(kv_rank, 0.0f);
            for (uint32_t s = 0; s < seq_len; s++) {
                const f16* cs = kv_cache_data + size_t(s) * compressed_dim;
                float w = scores[s];
                for (uint32_t d = 0; d < kv_rank; d++)
                    out_comp[d] += w * static_cast<float>(cs[d]);
            }

            // Expand through v_b: out_h = out_comp × v_b_h^T
            // v_b shape: (n_heads*v_head_dim) rows × kv_rank cols
            f16* out_head = output + h * c.v_head_dim;
            for (uint32_t j = 0; j < c.v_head_dim; j++) {
                dequant_weight_row(v_b_contents, lw.attn_v_b_type,
                                   h * c.v_head_dim + j, kv_rank, row_buf.data());
                float val = 0.0f;
                for (uint32_t d = 0; d < kv_rank; d++)
                    val += out_comp[d] * row_buf[d];
                out_head[j] = static_cast<f16>(val);
            }

        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction / move
// ─────────────────────────────────────────────────────────────────────────────

TransformerModel::TransformerModel() : impl_(std::make_unique<Impl>()) {}
TransformerModel::~TransformerModel() = default;
TransformerModel::TransformerModel(TransformerModel&&) noexcept = default;

auto TransformerModel::config() const -> const ModelConfig& { return impl_->cfg; }
auto TransformerModel::kv_cache() -> KVCache* { return impl_->kv_cache.get(); }

void TransformerModel::set_use_gpu_attention(bool enable) { impl_->use_gpu_attention_ = enable; }
auto TransformerModel::use_gpu_attention() const -> bool { return impl_->use_gpu_attention_; }

auto TransformerModel::decode_profile() const -> const DecodeProfile& { return impl_->decode_profile_; }
void TransformerModel::reset_decode_profile() { impl_->decode_profile_.reset(); }

auto TransformerModel::prefill_profile() const -> const PrefillProfile& { return impl_->prefill_profile_; }
void TransformerModel::reset_prefill_profile() { impl_->prefill_profile_.reset(); }

void TransformerModel::set_prefill_profile_enabled(bool enable) { impl_->prefill_profile_enabled_ = enable; }
auto TransformerModel::prefill_profile_enabled() const -> bool { return impl_->prefill_profile_enabled_; }

void TransformerModel::set_route_callback(RouteCallback cb) { impl_->route_callback_ = std::move(cb); }
void TransformerModel::set_buffer_manager(BufferManager* mgr) { impl_->buffer_mgr_ = mgr; }

auto TransformerModel::profile_kernel_dispatch()
    -> std::expected<std::vector<std::pair<std::string, double>>, std::string>
{
    auto& d = *impl_;
    if (!d.mega_chain_cached_)
        return std::unexpected(std::string("mega_chain not cached; run forward() first"));

    struct PipeNameEntry { void* pipe; const char* name; };
    const PipeNameEntry name_table[] = {
        {d.pipe_matvec_q4_0,            "matvec_q4_0"},
        {d.pipe_matvec_f16,             "matvec_f16"},
        {d.pipe_matvec_q8_0,            "matvec_q8_0"},
        {d.pipe_matvec_q4_k,            "matvec_q4_k"},
        {d.pipe_rms_norm,               "rms_norm"},
        {d.pipe_rope,                   "rope"},
        {d.pipe_scatter_kv,             "scatter_kv"},
        {d.pipe_flash_attention_decode, "flash_attn_decode"},
        {d.pipe_elementwise_add,        "elementwise_add"},
        {d.pipe_silu_mul,               "silu_mul"},
        {d.pipe_bias_broadcast,         "bias_broadcast"},
        {d.pipe_silu,                   "silu"},
        {d.pipe_elementwise_mul,        "elementwise_mul"},
        {d.pipe_softmax,                "softmax"},
        {d.pipe_attention_decode,       "attention_decode"},
        {d.pipe_argmax_f16,             "argmax_f16"},
        {d.pipe_moe_gate,               "moe_gate"},
        {d.pipe_moe_gate_grouped,       "moe_gate_grouped"},
        {d.pipe_moe_reduce,             "moe_reduce"},
    };

    auto pipe_to_name = [&](void* p) -> std::string {
        for (auto& e : name_table)
            if (e.pipe == p) return e.name;
        return "unknown";
    };

    std::unordered_map<std::string, double> accum;

    for (auto& group : d.cached_mega_chain_) {
        if (group.empty()) continue;
        void* pipe = group[0].pipeline;
        std::string name = pipe_to_name(pipe);

        std::vector<std::vector<MetalCompute::DispatchParams>> single_group;
        single_group.push_back(group);  // copy

        auto r = d.gpu->dispatch_chain_sync(single_group);
        if (!r) return std::unexpected(r.error());

        accum[name] += *r * 1000.0;  // seconds → ms
    }

    std::vector<std::pair<std::string, double>> result(accum.begin(), accum.end());
    std::sort(result.begin(), result.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight loading helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

auto create_weight_buffer(MetalCompute* gpu, const TensorInfo& ti,
                          const void* mmap_base)
    -> void*
{
    const auto* data = static_cast<const uint8_t*>(mmap_base) + ti.file_offset;
    bool page_aligned = MmapLoader::is_page_aligned(data);

    if (page_aligned) {
        return gpu->create_buffer_nocopy(const_cast<void*>(static_cast<const void*>(data)),
                                         ti.byte_size);
    }
    return gpu->create_buffer_from_data(data, ti.byte_size);
}

// Q6_K block layout (210 bytes → 256 elements):
//   ql[128]    lower 4 bits of each 6-bit quant
//   qh[64]     upper 2 bits of each 6-bit quant
//   scales[16] int8 sub-block scales (16 sub-blocks of 16 elements)
//   d (f16)    super-block scale
static void dequantize_q6_k_block(const uint8_t* block, float* out) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const auto* scales = reinterpret_cast<const int8_t*>(block + 192);
    f16 d_f16;
    std::memcpy(&d_f16, block + 208, sizeof(f16));
    float d = static_cast<float>(d_f16);

    for (uint32_t n = 0; n < 256; n += 128) {
        for (uint32_t l = 0; l < 32; ++l) {
            uint32_t is = n / 16 + l / 16;
            auto q1 = static_cast<int8_t>(
                ((ql[n / 2 + l] & 0xF) | (((qh[n / 4 + l] >> 0) & 3) << 4)) - 32);
            auto q2 = static_cast<int8_t>(
                ((ql[n / 2 + l + 32] & 0xF) | (((qh[n / 4 + l] >> 2) & 3) << 4)) - 32);
            auto q3 = static_cast<int8_t>(
                ((ql[n / 2 + l] >> 4) | (((qh[n / 4 + l] >> 4) & 3) << 4)) - 32);
            auto q4 = static_cast<int8_t>(
                ((ql[n / 2 + l + 32] >> 4) | (((qh[n / 4 + l] >> 6) & 3) << 4)) - 32);
            out[n + l]      = d * scales[is]     * q1;
            out[n + l + 32] = d * scales[is + 2] * q2;
            out[n + l + 64] = d * scales[is + 4] * q3;
            out[n + l + 96] = d * scales[is + 6] * q4;
        }
    }
}

auto create_dequantized_f16_buffer(MetalCompute* gpu, const TensorInfo& ti,
                                   const void* mmap_base)
    -> void*
{
    constexpr size_t kQ6KBlockBytes = 210;
    constexpr size_t kQ6KBlockElems = 256;

    size_t n_blocks = ti.byte_size / kQ6KBlockBytes;
    size_t n_elems = n_blocks * kQ6KBlockElems;

    const auto* src = static_cast<const uint8_t*>(mmap_base) + ti.file_offset;
    std::vector<float> tmp(kQ6KBlockElems);
    std::vector<f16> f16_buf(n_elems);

    for (size_t b = 0; b < n_blocks; ++b) {
        dequantize_q6_k_block(src + b * kQ6KBlockBytes, tmp.data());
        for (size_t i = 0; i < kQ6KBlockElems; ++i)
            f16_buf[b * kQ6KBlockElems + i] = static_cast<f16>(tmp[i]);
    }

    return gpu->create_buffer_from_data(f16_buf.data(), n_elems * sizeof(f16));
}

auto create_dequantized_q4_1_f16_buffer(MetalCompute* gpu, const TensorInfo& ti,
                                        const void* mmap_base)
    -> void*
{
    constexpr uint32_t kBlockElems = 32;
    constexpr size_t kBlockBytes = 20; // 2(delta) + 2(min) + 16(nibbles)

    size_t n_blocks = ti.byte_size / kBlockBytes;
    size_t n_elems = n_blocks * kBlockElems;

    const auto* src = static_cast<const uint8_t*>(mmap_base) + ti.file_offset;
    std::vector<f16> f16_buf(n_elems);

    for (size_t b = 0; b < n_blocks; ++b) {
        const auto* blk = src + b * kBlockBytes;
        f16 delta_f16, min_f16;
        std::memcpy(&delta_f16, blk, sizeof(f16));
        std::memcpy(&min_f16, blk + 2, sizeof(f16));
        float delta = static_cast<float>(delta_f16);
        float vmin  = static_cast<float>(min_f16);
        const uint8_t* qs = blk + 4;
        for (uint32_t j = 0; j < 16; ++j) {
            float lo = static_cast<float>(qs[j] & 0x0F) * delta + vmin;
            float hi = static_cast<float>(qs[j] >> 4)   * delta + vmin;
            f16_buf[b * kBlockElems + j]      = static_cast<f16>(lo);
            f16_buf[b * kBlockElems + j + 16] = static_cast<f16>(hi);
        }
    }

    return gpu->create_buffer_from_data(f16_buf.data(), n_elems * sizeof(f16));
}

// Permute Q/K weight rows from NORM (consecutive pairs) to NeoX (halved pairs).
// For each head: new[i] = old[2*i], new[d/2+i] = old[2*i+1] for i in [0, d/2).
// This lets the NeoX-style RoPE kernel produce correct results for NORM-RoPE models.
auto permute_qk_for_neox_rope(MetalCompute* gpu, void* original_buf,
                               GGMLType type, uint32_t M, uint32_t K,
                               uint32_t head_dim) -> void*
{
    size_t row_bytes;
    switch (type) {
        case GGMLType::Q4_0: row_bytes = size_t(K / 32) * 18; break;
        case GGMLType::F16:  row_bytes = size_t(K) * 2; break;
        default: return nullptr;
    }

    auto* src = static_cast<const uint8_t*>(gpu->buffer_contents(original_buf));
    uint32_t n_heads = M / head_dim;
    uint32_t half = head_dim / 2;
    size_t total_bytes = size_t(M) * row_bytes;
    std::vector<uint8_t> permuted(total_bytes);

    for (uint32_t h = 0; h < n_heads; h++) {
        size_t off = h * head_dim * row_bytes;
        for (uint32_t i = 0; i < half; i++) {
            std::memcpy(permuted.data() + off + i * row_bytes,
                       src + off + 2 * i * row_bytes, row_bytes);
            std::memcpy(permuted.data() + off + (half + i) * row_bytes,
                       src + off + (2 * i + 1) * row_bytes, row_bytes);
        }
    }

    return gpu->create_buffer_from_data(permuted.data(), total_bytes);
}

auto create_f16_norm_buffer(MetalCompute* gpu, const TensorInfo& ti,
                            const void* mmap_base)
    -> void*
{
    const auto* data = static_cast<const uint8_t*>(mmap_base) + ti.file_offset;

    if (ti.type == GGMLType::F16) {
        return gpu->create_buffer_from_data(data, ti.byte_size);
    }

    if (ti.type == GGMLType::F32) {
        size_t n_elem = ti.byte_size / sizeof(float);
        std::vector<f16> converted(n_elem);
        const auto* f32_data = reinterpret_cast<const float*>(data);
        for (size_t i = 0; i < n_elem; i++) {
            converted[i] = static_cast<f16>(f32_data[i]);
        }
        return gpu->create_buffer_from_data(converted.data(),
                                            n_elem * sizeof(f16));
    }

    return nullptr;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// from_gguf: parse metadata + load weights → build model
// ─────────────────────────────────────────────────────────────────────────────

auto TransformerModel::from_gguf(MetalCompute* gpu,
                                 const GGUFParser& parser,
                                 const MmapRegion& mmap)
    -> std::expected<std::unique_ptr<TransformerModel>, std::string>
{
    return from_gguf_impl(gpu, parser, {mmap.data()});
}

auto TransformerModel::from_gguf(MetalCompute* gpu,
                                 const GGUFParser& parser,
                                 const std::vector<MmapRegion>& mmaps)
    -> std::expected<std::unique_ptr<TransformerModel>, std::string>
{
    std::vector<const void*> bases;
    bases.reserve(mmaps.size());
    for (auto& m : mmaps) bases.push_back(m.data());
    return from_gguf_impl(gpu, parser, bases);
}

auto TransformerModel::from_gguf_impl(MetalCompute* gpu,
                                       const GGUFParser& parser,
                                       const std::vector<const void*>& mmap_bases)
    -> std::expected<std::unique_ptr<TransformerModel>, std::string>
{
    auto model = std::unique_ptr<TransformerModel>(new TransformerModel());
    auto& impl = *model->impl_;
    impl.gpu = gpu;

    // ─── Extract model config from GGUF metadata ───

    const auto& meta = parser.metadata();
    auto& c = impl.cfg;

    c.vocab_size   = meta.vocab_size;
    c.embed_dim    = meta.embedding_dim;
    c.n_layers     = meta.n_layers;
    c.n_heads      = meta.n_heads;
    c.n_kv_heads   = meta.n_kv_heads > 0 ? meta.n_kv_heads : meta.n_heads;
    c.head_dim     = c.embed_dim / c.n_heads;
    c.rope_theta   = 10000.0f;
    c.rms_norm_eps = 1e-5f;
    c.is_moe       = parser.is_moe();
    c.n_experts    = meta.n_experts;
    c.n_experts_used = meta.n_experts_used;

    if (auto it = meta.raw_int_kv.find(meta.arch + ".leading_dense_block_count");
        it != meta.raw_int_kv.end())
        c.first_k_dense_replace = static_cast<uint32_t>(it->second);

    if (auto it = meta.raw_int_kv.find(meta.arch + ".expert_feed_forward_length");
        it != meta.raw_int_kv.end()) {
        c.expert_ffn_dim = static_cast<uint32_t>(it->second);
    }
    if (auto it = meta.raw_int_kv.find(meta.arch + ".expert_group_count");
        it != meta.raw_int_kv.end())
        c.n_group = static_cast<uint32_t>(it->second);
    if (auto it = meta.raw_int_kv.find(meta.arch + ".expert_top_group_count");
        it != meta.raw_int_kv.end())
        c.topk_group = static_cast<uint32_t>(it->second);
    if (auto it = meta.raw_float_kv.find(meta.arch + ".expert_weights_scale");
        it != meta.raw_float_kv.end())
        c.routed_scaling_factor = static_cast<float>(it->second);

    // Hardcode DS V3 defaults when architecture is deepseek2 and GGUF lacks keys
    // V2-Lite has 64 experts but does NOT use V3-style grouped routing
    if (meta.arch == "deepseek2" && c.n_group == 0 && c.n_experts > 64) {
        c.n_group = 8;
        c.topk_group = 4;
        c.routed_scaling_factor = 2.5f;
    }

    // MLA parameters (deepseek2 architecture)
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.q_lora_rank");
        it != meta.raw_int_kv.end())
        c.q_lora_rank = static_cast<uint32_t>(it->second);
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.kv_lora_rank");
        it != meta.raw_int_kv.end())
        c.kv_lora_rank = static_cast<uint32_t>(it->second);

    // qk_rope_head_dim: rope.dimension_count (works for V2-Lite and V3)
    if (auto it = meta.raw_int_kv.find(meta.arch + ".rope.dimension_count");
        it != meta.raw_int_kv.end()) {
        if (c.kv_lora_rank > 0)
            c.qk_rope_head_dim = static_cast<uint32_t>(it->second);
    }
    // qk_nope_head_dim: key_length - qk_rope_head_dim
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.key_length");
        it != meta.raw_int_kv.end()) {
        uint32_t key_length = static_cast<uint32_t>(it->second);
        if (c.qk_rope_head_dim > 0 && key_length > c.qk_rope_head_dim)
            c.qk_nope_head_dim = key_length - c.qk_rope_head_dim;
    }
    // v_head_dim: value_length (V2-Lite and V3 both have this)
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.value_length");
        it != meta.raw_int_kv.end()) {
        if (c.kv_lora_rank > 0)
            c.v_head_dim = static_cast<uint32_t>(it->second);
    }
    // Fallback overrides: key_length_mla / value_length_mla (V3 may have these)
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.key_length_mla");
        it != meta.raw_int_kv.end()) {
        uint32_t key_length_mla = static_cast<uint32_t>(it->second);
        if (c.qk_rope_head_dim > 0)
            c.qk_nope_head_dim = key_length_mla - c.qk_rope_head_dim;
    }
    if (auto it = meta.raw_int_kv.find(meta.arch + ".attention.value_length_mla");
        it != meta.raw_int_kv.end())
        c.v_head_dim = static_cast<uint32_t>(it->second);
    if (auto it = meta.raw_int_kv.find(meta.arch + ".expert_shared_count");
        it != meta.raw_int_kv.end())
        c.n_shared_experts = static_cast<uint32_t>(it->second);

    if (auto it = meta.raw_float_kv.find(meta.arch + ".rope.freq_base");
        it != meta.raw_float_kv.end()) {
        c.rope_theta = static_cast<float>(it->second);
    }
    if (auto it = meta.raw_float_kv.find(meta.arch + ".rope.scaling.factor");
        it != meta.raw_float_kv.end()) {
        c.rope_scaling_factor = static_cast<float>(it->second);
    }
    if (auto it = meta.raw_int_kv.find(meta.arch + ".rope.scaling.original_context_length");
        it != meta.raw_int_kv.end()) {
        c.rope_original_ctx = static_cast<uint32_t>(it->second);
    }
    if (auto it = meta.raw_float_kv.find(meta.arch + ".rope.scaling.yarn_log_multiplier");
        it != meta.raw_float_kv.end()) {
        c.rope_yarn_log_mult = static_cast<float>(it->second);
    }
    if (auto it = meta.raw_float_kv.find(meta.arch + ".attention.layer_norm_rms_epsilon");
        it != meta.raw_float_kv.end()) {
        c.rms_norm_eps = static_cast<float>(it->second);
    }

    if (c.rope_scaling_factor > 1.0f) {
        fprintf(stderr, "[DBG YaRN] factor=%.1f original_ctx=%u log_mult=%.4f theta=%.1f\n",
                c.rope_scaling_factor, c.rope_original_ctx, c.rope_yarn_log_mult, c.rope_theta);
    }

    if (c.is_mla()) {
        fprintf(stderr, "[MLA Config] kv_lora_rank=%u q_lora_rank=%u "
                "qk_rope=%u qk_nope=%u v_head=%u n_heads=%u\n",
                c.kv_lora_rank, c.q_lora_rank,
                c.qk_rope_head_dim, c.qk_nope_head_dim, c.v_head_dim, c.n_heads);
        fprintf(stderr, "[MLA Config] n_shared_experts=%u n_experts=%u "
                "n_group=%u topk_group=%u first_k_dense=%u\n",
                c.n_shared_experts, c.n_experts,
                c.n_group, c.topk_group, c.first_k_dense_replace);
    }

    // FFN dimension: try metadata, fall back to weight tensor shape
    c.ffn_dim = 0;
    if (auto it = meta.raw_int_kv.find(meta.arch + ".feed_forward_length");
        it != meta.raw_int_kv.end()) {
        c.ffn_dim = static_cast<uint32_t>(it->second);
    }

    if (c.vocab_size == 0 || c.embed_dim == 0 || c.n_layers == 0 || c.n_heads == 0) {
        return std::unexpected("Invalid model config: zero-valued critical dimension");
    }

    // Architectures using NORM RoPE (consecutive pairs) need Q/K weight permutation
    // so the NeoX-style RoPE kernel produces correct results
    bool rope_norm = (meta.arch == "llama" || meta.arch == "deepseek" ||
                      meta.arch == "deepseek2" || meta.arch == "internlm2" ||
                      meta.arch == "baichuan" || meta.arch == "command-r");

    // ─── Compile Metal pipelines ───

    auto pipe_res = impl.compile_pipelines();
    if (!pipe_res) return std::unexpected(pipe_res.error());

    // ─── Load global weights ───

    auto base_for = [&](const TensorInfo& ti) -> const void* {
        return mmap_bases[ti.shard_index];
    };

    // Token embedding
    if (auto* ti = parser.tensor_by_name("token_embd.weight")) {
        const void* base = base_for(*ti);
        if (ti->type == GGMLType::Q6_K) {
            impl.token_embed_buf = create_dequantized_f16_buffer(gpu, *ti, base);
            impl.token_embed_type = GGMLType::F16;
        } else {
            impl.token_embed_type = ti->type;
            impl.token_embed_buf = create_weight_buffer(gpu, *ti, base);
        }
        if (!impl.token_embed_buf) {
            return std::unexpected("Failed to create token_embd buffer");
        }
    } else {
        return std::unexpected("Missing tensor: token_embd.weight");
    }

    // Output norm
    if (auto* ti = parser.tensor_by_name("output_norm.weight")) {
        impl.output_norm_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
        if (!impl.output_norm_buf) {
            return std::unexpected("Failed to create output_norm buffer");
        }
    } else {
        return std::unexpected("Missing tensor: output_norm.weight");
    }

    // Output weight (may be tied with token_embd)
    if (auto* ti = parser.tensor_by_name("output.weight")) {
        const void* base = base_for(*ti);
        if (ti->type == GGMLType::Q6_K) {
            impl.output_weight_buf = create_dequantized_f16_buffer(gpu, *ti, base);
            impl.output_weight_type = GGMLType::F16;
        } else {
            impl.output_weight_type = ti->type;
            impl.output_weight_buf = create_weight_buffer(gpu, *ti, base);
        }
        if (!impl.output_weight_buf) {
            return std::unexpected("Failed to create output.weight buffer");
        }
    } else {
        impl.output_weight_type = impl.token_embed_type;
        impl.output_weight_buf = impl.token_embed_buf;
    }

    // ─── Load per-layer weights ───

    impl.layers.resize(c.n_layers);

    for (uint32_t L = 0; L < c.n_layers; L++) {
        auto& lw = impl.layers[L];
        auto blk = "blk." + std::to_string(L) + ".";

        auto load_norm = [&](const std::string& name, void** dst)
            -> std::expected<void, std::string>
        {
            auto* ti = parser.tensor_by_name(name);
            if (!ti) return std::unexpected("Missing tensor: " + name);
            *dst = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
            if (!*dst) return std::unexpected("Failed to create buffer for " + name);
            return {};
        };

        auto load_weight = [&](const std::string& name, void** dst, GGMLType* type)
            -> std::expected<void, std::string>
        {
            auto* ti = parser.tensor_by_name(name);
            if (!ti) return std::unexpected("Missing tensor: " + name);

            if (ti->type == GGMLType::F32) {
                *dst = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
                *type = GGMLType::F16;
            } else if (ti->type == GGMLType::Q6_K) {
                *dst = create_dequantized_f16_buffer(gpu, *ti, base_for(*ti));
                *type = GGMLType::F16;
            } else if (ti->type == GGMLType::Q4_1) {
                *dst = create_dequantized_q4_1_f16_buffer(gpu, *ti, base_for(*ti));
                *type = GGMLType::F16;
            } else {
                *type = ti->type;
                *dst = create_weight_buffer(gpu, *ti, base_for(*ti));
            }
            if (!*dst) return std::unexpected("Failed to create buffer for " + name);
            return {};
        };

        if (auto r = load_norm(blk + "attn_norm.weight", &lw.attn_norm_buf); !r)
            return std::unexpected(r.error());
        if (auto r = load_norm(blk + "ffn_norm.weight", &lw.ffn_norm_buf); !r)
            return std::unexpected(r.error());

        if (c.is_mla()) {
            // MLA Q path: LoRA Q (V3: q_a → norm → q_b) or full Q (V2-Lite)
            if (c.q_lora_rank > 0) {
                if (parser.tensor_by_name(blk + "attn_q_a.weight")) {
                    if (auto r = load_weight(blk + "attn_q_a.weight", &lw.attn_q_a_buf, &lw.attn_q_a_type); !r)
                        return std::unexpected(r.error());
                }
                if (parser.tensor_by_name(blk + "attn_q_b.weight")) {
                    if (auto r = load_weight(blk + "attn_q_b.weight", &lw.attn_q_b_buf, &lw.attn_q_b_type); !r)
                        return std::unexpected(r.error());
                }
                if (parser.tensor_by_name(blk + "attn_q_a_norm.weight")) {
                    if (auto r = load_norm(blk + "attn_q_a_norm.weight", &lw.attn_q_a_norm_buf); !r)
                        return std::unexpected(r.error());
                }
            } else {
                if (auto r = load_weight(blk + "attn_q.weight", &lw.wq_buf, &lw.wq_type); !r)
                    return std::unexpected(r.error());
            }

            // KV compression path (shared between V2-Lite and V3)
            if (parser.tensor_by_name(blk + "attn_kv_a_mqa.weight")) {
                if (auto r = load_weight(blk + "attn_kv_a_mqa.weight", &lw.attn_kv_a_mqa_buf, &lw.attn_kv_a_mqa_type); !r)
                    return std::unexpected(r.error());
            }
            if (parser.tensor_by_name(blk + "attn_kv_a_norm.weight")) {
                if (auto r = load_norm(blk + "attn_kv_a_norm.weight", &lw.attn_kv_a_norm_buf); !r)
                    return std::unexpected(r.error());
            }

            // K_B / V_B: separate tensors (V3) or merged attn_kv_b (V2-Lite)
            if (parser.tensor_by_name(blk + "attn_k_b.weight")) {
                if (auto r = load_weight(blk + "attn_k_b.weight", &lw.attn_k_b_buf, &lw.attn_k_b_type); !r)
                    return std::unexpected(r.error());
            }
            if (parser.tensor_by_name(blk + "attn_v_b.weight")) {
                if (auto r = load_weight(blk + "attn_v_b.weight", &lw.attn_v_b_buf, &lw.attn_v_b_type); !r)
                    return std::unexpected(r.error());
            }
            if (!lw.attn_k_b_buf && !lw.attn_v_b_buf) {
                auto* ti = parser.tensor_by_name(blk + "attn_kv_b.weight");
                if (ti) {
                    const auto* raw = static_cast<const uint8_t*>(base_for(*ti)) + ti->file_offset;
                    const uint32_t nope = c.qk_nope_head_dim;
                    const uint32_t vdim = c.v_head_dim;
                    const uint32_t head_rows = nope + vdim;
                    const uint32_t total_rows = c.n_heads * head_rows;
                    const size_t bpr = ti->byte_size / total_rows;
                    const uint32_t k_b_rows = c.n_heads * nope;
                    const uint32_t v_b_rows = c.n_heads * vdim;

                    std::vector<uint8_t> k_b_data(size_t(k_b_rows) * bpr);
                    std::vector<uint8_t> v_b_data(size_t(v_b_rows) * bpr);
                    for (uint32_t h = 0; h < c.n_heads; h++) {
                        std::memcpy(k_b_data.data() + size_t(h * nope) * bpr,
                                    raw + size_t(h * head_rows) * bpr,
                                    size_t(nope) * bpr);
                        std::memcpy(v_b_data.data() + size_t(h * vdim) * bpr,
                                    raw + size_t(h * head_rows + nope) * bpr,
                                    size_t(vdim) * bpr);
                    }
                    lw.attn_k_b_buf = gpu->create_buffer_from_data(
                        k_b_data.data(), k_b_data.size());
                    lw.attn_v_b_buf = gpu->create_buffer_from_data(
                        v_b_data.data(), v_b_data.size());
                    lw.attn_k_b_type = ti->type;
                    lw.attn_v_b_type = ti->type;
                    if (!lw.attn_k_b_buf || !lw.attn_v_b_buf)
                        return std::unexpected("Failed to split attn_kv_b for layer " + std::to_string(L));
                }
            }
            if (!lw.attn_k_b_buf || !lw.attn_v_b_buf)
                return std::unexpected("MLA layer " + std::to_string(L) +
                    " missing k_b/v_b weights (need attn_k_b + attn_v_b or attn_kv_b)");
        } else {
            if (auto r = load_weight(blk + "attn_q.weight", &lw.wq_buf, &lw.wq_type); !r)
                return std::unexpected(r.error());
            if (auto r = load_weight(blk + "attn_k.weight", &lw.wk_buf, &lw.wk_type); !r)
                return std::unexpected(r.error());
            if (auto r = load_weight(blk + "attn_v.weight", &lw.wv_buf, &lw.wv_type); !r)
                return std::unexpected(r.error());
        }

        if (auto r = load_weight(blk + "attn_output.weight", &lw.wo_buf, &lw.wo_type); !r)
            return std::unexpected(r.error());

        if (!c.is_mla()) {
            // For NORM-RoPE models (e.g. Llama): permute Q/K weight rows
            // so NeoX-style RoPE kernel produces the correct result
            if (rope_norm) {
                uint32_t q_M = c.n_heads * c.head_dim;
                uint32_t kv_M = c.n_kv_heads * c.head_dim;
                auto* pq = permute_qk_for_neox_rope(
                    gpu, lw.wq_buf, lw.wq_type, q_M, c.embed_dim, c.head_dim);
                if (pq) lw.wq_buf = pq;
                auto* pk = permute_qk_for_neox_rope(
                    gpu, lw.wk_buf, lw.wk_type, kv_M, c.embed_dim, c.head_dim);
                if (pk) lw.wk_buf = pk;
            }

            // Optional attention biases (e.g. Qwen2)
            if (auto* ti = parser.tensor_by_name(blk + "attn_q.bias"))
                lw.bq_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
            if (auto* ti = parser.tensor_by_name(blk + "attn_k.bias"))
                lw.bk_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
            if (auto* ti = parser.tensor_by_name(blk + "attn_v.bias"))
                lw.bv_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));

            if (auto* ti = parser.tensor_by_name(blk + "attn_q_norm.weight"))
                lw.q_norm_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
            if (auto* ti = parser.tensor_by_name(blk + "attn_k_norm.weight"))
                lw.k_norm_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));
        }

        bool layer_is_moe = c.is_moe && (L >= c.first_k_dense_replace);
        lw.kind = layer_is_moe ? LayerKind::MoE : LayerKind::Dense;

        if (lw.kind == LayerKind::MoE) {
            if (auto r = load_weight(blk + "ffn_gate_inp.weight",
                    &lw.router_buf, &lw.router_type); !r)
                return std::unexpected(r.error());

            auto load_expert_bufs = [&](const std::string& name,
                                        std::vector<void*>& bufs, GGMLType* type,
                                        size_t* out_per_expert_bytes = nullptr)
                -> std::expected<void, std::string>
            {
                auto* ti = parser.tensor_by_name(name);
                if (!ti) return std::unexpected("Missing tensor: " + name);
                *type = ti->type;
                size_t expert_bytes = ti->byte_size / c.n_experts;
                if (out_per_expert_bytes) *out_per_expert_bytes = expert_bytes;
                const auto* tdata =
                    static_cast<const uint8_t*>(base_for(*ti)) + ti->file_offset;
                bufs.resize(c.n_experts);
                for (uint32_t e = 0; e < c.n_experts; e++) {
                    const auto* edata = tdata + e * expert_bytes;
                    if (MmapLoader::is_page_aligned(edata)) {
                        bufs[e] = gpu->create_buffer_nocopy(
                            const_cast<void*>(
                                static_cast<const void*>(edata)),
                            expert_bytes);
                    } else {
                        bufs[e] = gpu->create_buffer_from_data(
                            edata, expert_bytes);
                    }
                    if (!bufs[e])
                        return std::unexpected(
                            "Failed to create expert buffer for " + name);
                }
                return {};
            };

            if (auto r = load_expert_bufs(blk + "ffn_gate_exps.weight",
                    lw.expert_gate_bufs, &lw.expert_gate_type,
                    &lw.expert_gate_bytes); !r)
                return std::unexpected(r.error());
            if (auto r = load_expert_bufs(blk + "ffn_up_exps.weight",
                    lw.expert_up_bufs, &lw.expert_up_type,
                    &lw.expert_up_bytes); !r)
                return std::unexpected(r.error());
            if (auto r = load_expert_bufs(blk + "ffn_down_exps.weight",
                    lw.expert_down_bufs, &lw.expert_down_type,
                    &lw.expert_down_bytes); !r)
                return std::unexpected(r.error());

            if (c.n_shared_experts > 0 &&
                parser.tensor_by_name(blk + "ffn_gate_shexp.weight")) {
                if (auto r = load_weight(blk + "ffn_gate_shexp.weight",
                        &lw.w1_shared_buf, &lw.w1_shared_type); !r)
                    return std::unexpected(r.error());
                if (auto r = load_weight(blk + "ffn_down_shexp.weight",
                        &lw.w2_shared_buf, &lw.w2_shared_type); !r)
                    return std::unexpected(r.error());
                if (auto r = load_weight(blk + "ffn_up_shexp.weight",
                        &lw.w3_shared_buf, &lw.w3_shared_type); !r)
                    return std::unexpected(r.error());
            }

            if (auto* ti = parser.tensor_by_name(blk + "exp_probs_b"))
                lw.router_bias_buf = create_f16_norm_buffer(gpu, *ti, base_for(*ti));

            if (c.expert_ffn_dim == 0) {
                auto* ti = parser.tensor_by_name(blk + "ffn_gate_exps.weight");
                if (ti && ti->dimensions.size() >= 2)
                    c.expert_ffn_dim = static_cast<uint32_t>(ti->dimensions[1]);
            }
        } else {
            if (auto r = load_weight(blk + "ffn_gate.weight", &lw.w1_buf, &lw.w1_type); !r)
                return std::unexpected(r.error());
            if (auto r = load_weight(blk + "ffn_down.weight", &lw.w2_buf, &lw.w2_type); !r)
                return std::unexpected(r.error());
            if (auto r = load_weight(blk + "ffn_up.weight", &lw.w3_buf, &lw.w3_type); !r)
                return std::unexpected(r.error());

            if (c.ffn_dim == 0) {
                auto* gate_ti = parser.tensor_by_name(blk + "ffn_gate.weight");
                if (gate_ti && gate_ti->dimensions.size() >= 2) {
                    c.ffn_dim = static_cast<uint32_t>(gate_ti->dimensions[1]);
                }
            }
        }
    }

    if (c.is_moe && c.expert_ffn_dim == 0)
        c.expert_ffn_dim = c.ffn_dim;
    if (c.ffn_dim == 0)
        c.ffn_dim = c.expert_ffn_dim;
    if (c.ffn_dim == 0 && c.expert_ffn_dim == 0) {
        return std::unexpected("Could not determine FFN dimension");
    }

    // ─── Allocate intermediate buffers ───

    auto buf_res = impl.alloc_buffers();
    if (!buf_res) return std::unexpected(buf_res.error());

    // ─── Create KV cache ───

    // MLA uses compressed KV cache (buf_kv_compressed), not standard KV.
    // Allocate minimal KV cache for MLA to preserve seq_len tracking API.
    KVCacheConfig kv_cfg{
        .n_layers    = c.n_layers,
        .n_kv_heads  = c.is_mla() ? 1u : c.n_kv_heads,
        .head_dim    = c.is_mla() ? 1u : c.head_dim,
        .max_seq_len = meta.context_length > 0 ? meta.context_length : 8192u,
        .quantize_4bit = true,
        .fp16_preserve_last = 256,
    };
    auto kv_res = KVCache::create(kv_cfg);
    if (!kv_res) return std::unexpected("KV cache creation failed: " + kv_res.error());
    impl.kv_cache = std::move(*kv_res);

    // Allocate per-layer persistent GPU KV buffers (skip for MLA — uses compressed KV)
    if (!c.is_mla()) {
        uint32_t max_sl = impl.kv_cache->config().max_seq_len;
        size_t per_layer_kv_bytes = size_t(max_sl) * c.n_kv_heads
                                    * c.head_dim * sizeof(f16);
        impl.buf_kv_gpu_k.resize(c.n_layers);
        impl.buf_kv_gpu_v.resize(c.n_layers);
        for (uint32_t l = 0; l < c.n_layers; l++) {
            impl.buf_kv_gpu_k[l] = gpu->create_buffer(per_layer_kv_bytes);
            impl.buf_kv_gpu_v[l] = gpu->create_buffer(per_layer_kv_bytes);
            if (!impl.buf_kv_gpu_k[l] || !impl.buf_kv_gpu_v[l])
                return std::unexpected("Failed to allocate persistent GPU KV buffer");
        }
    }

    // MLA compressed KV cache (576 dims per position vs full KV)
    if (c.is_mla()) {
        uint32_t compressed_kv_dim = c.kv_lora_rank + c.qk_rope_head_dim;
        uint32_t max_sl = impl.kv_cache->config().max_seq_len;
        size_t per_layer_bytes = size_t(max_sl) * compressed_kv_dim * sizeof(f16);
        impl.buf_kv_compressed.resize(c.n_layers);
        for (uint32_t l = 0; l < c.n_layers; l++) {
            impl.buf_kv_compressed[l] = gpu->create_buffer(per_layer_bytes);
            if (!impl.buf_kv_compressed[l])
                return std::unexpected("Failed to allocate compressed KV buffer");
        }
    }

    // Pre-allocate decode scratch buffers
    {
        uint32_t max_sl = impl.kv_cache->config().max_seq_len;
        impl.scratch_scores_.resize(max_sl);

        if (!c.is_mla()) {
            uint32_t kv_elems = c.n_kv_heads * c.head_dim;
            uint32_t q_elems = c.n_heads * c.head_dim;
            impl.scratch_k_cpu_.resize(kv_elems);
            impl.scratch_v_cpu_.resize(kv_elems);
            impl.scratch_q_cpu_.resize(q_elems);
            impl.scratch_attn_out_.resize(q_elems);
            impl.scratch_kv_k_.resize(size_t(max_sl) * kv_elems);
            impl.scratch_kv_v_.resize(size_t(max_sl) * kv_elems);
        }
    }

    // Auto-enable GPU attention for models with enough KV heads.
    // Small models (≤2 KV heads): CPU attention wins (GPU dispatch overhead dominates).
    // Larger models (≥4 KV heads): GPU parallelism wins.
    // MLA models use CPU absorbed attention (no standard KV expansion on GPU).
    impl.use_gpu_attention_ = !c.is_mla() && (c.n_kv_heads >= 4);

    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_prefill: process multiple tokens by delegating to single-token path
// ─────────────────────────────────────────────────────────────────────────────

auto TransformerModel::forward_prefill(const std::vector<uint32_t>& token_ids,
                                       uint32_t start_position)
    -> std::expected<std::vector<float>, std::string>
{
    auto& d = *impl_;
    const auto& c = d.cfg;
    uint32_t N = static_cast<uint32_t>(token_ids.size());

    using clk = std::chrono::high_resolution_clock;
    auto t_wall0 = clk::now();

    auto layers_res = d.forward_prefill_layers(token_ids, start_position);
    if (!layers_res) return std::unexpected(layers_res.error());

    // 3. Final: last token → rms_norm → output projection → logits
    {
        size_t last_off = size_t(N - 1) * c.embed_dim;
        auto* src = static_cast<f16*>(d.gpu->buffer_contents(d.buf_hidden_batch));
        auto* dst = static_cast<f16*>(d.gpu->buffer_contents(d.buf_hidden));
        std::memcpy(dst, src + last_off, c.embed_dim * sizeof(f16));
    }

    {
        auto r = d.dispatch_sync_auto(d.make_rms_norm_dp(
            d.buf_hidden, d.output_norm_buf,
            d.buf_hidden2, c.embed_dim, c.rms_norm_eps));
        if (!r) return std::unexpected(r.error());
    }
    {
        auto r = d.dispatch_sync_auto(d.make_matvec_dp(
            d.output_weight_buf, d.output_weight_type,
            d.buf_hidden2, d.buf_logits, c.vocab_size, c.embed_dim));
        if (!r) return std::unexpected(r.error());
    }

    std::vector<f16> logits_f16(c.vocab_size);
    d.gpu->read_buffer(d.buf_logits, logits_f16.data(), c.vocab_size * sizeof(f16));

    std::vector<float> logits(c.vocab_size);
    for (uint32_t i = 0; i < c.vocab_size; i++)
        logits[i] = static_cast<float>(logits_f16[i]);

    {
        std::vector<std::pair<float,uint32_t>> top;
        for (uint32_t i = 0; i < c.vocab_size; i++)
            top.push_back({logits[i], i});
        std::partial_sort(top.begin(), top.begin() + 10, top.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
    }

    auto t_wall1 = clk::now();
    d.prefill_profile_.total_wall_ms +=
        std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
    d.prefill_profile_.n_tokens = N;

    return logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_prefill_all_logits: return logits for ALL positions (N × vocab_size)
// Used by USPP verify_with_target for parallel verification.
// ─────────────────────────────────────────────────────────────────────────────

auto TransformerModel::forward_prefill_all_logits(
        const std::vector<uint32_t>& token_ids,
        uint32_t start_position)
    -> std::expected<std::vector<std::vector<float>>, std::string>
{
    auto& d = *impl_;
    const auto& c = d.cfg;
    uint32_t N = static_cast<uint32_t>(token_ids.size());

    using clk = std::chrono::high_resolution_clock;
    auto t_wall0 = clk::now();

    auto layers_res = d.forward_prefill_layers(token_ids, start_position);
    if (!layers_res) return std::unexpected(layers_res.error());

    // 3. Final: ALL tokens → batch rms_norm → batch matmul → all logits
    d.ensure_logits_batch_buffer(N);

    {
        auto rms_dp = d.make_rms_norm_dp(
            d.buf_hidden_batch, d.output_norm_buf, d.buf_hidden2_batch,
            c.embed_dim, c.rms_norm_eps);
        rms_dp.grid_size = {size_t(N) * 256, 1, 1};

        auto matmul_dp = d.make_matmul_dp(
            d.output_weight_buf, d.output_weight_type,
            d.buf_hidden2_batch, d.buf_logits_batch,
            c.vocab_size, c.embed_dim, N);

        auto r = d.dispatch_chain_auto({
            {std::move(rms_dp)},
            {std::move(matmul_dp)},
        });
        if (!r) return std::unexpected(r.error());
    }

    size_t total_logits = size_t(N) * c.vocab_size;
    std::vector<f16> logits_f16(total_logits);
    d.gpu->read_buffer(d.buf_logits_batch, logits_f16.data(),
                       total_logits * sizeof(f16));

    std::vector<std::vector<float>> all_logits(N);
    for (uint32_t t = 0; t < N; t++) {
        all_logits[t].resize(c.vocab_size);
        const f16* src = logits_f16.data() + size_t(t) * c.vocab_size;
        for (uint32_t i = 0; i < c.vocab_size; i++) {
            all_logits[t][i] = static_cast<float>(src[i]);
        }
    }

    auto t_wall1 = clk::now();
    d.prefill_profile_.total_wall_ms +=
        std::chrono::duration<double, std::milli>(t_wall1 - t_wall0).count();
    d.prefill_profile_.n_tokens = N;

    return all_logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward: branch to prefill or decode
// ─────────────────────────────────────────────────────────────────────────────

auto TransformerModel::forward(const std::vector<uint32_t>& token_ids,
                               uint32_t start_position)
    -> std::expected<std::vector<float>, std::string>
{
    auto& d = *impl_;
    const auto& c = d.cfg;

    if (token_ids.empty()) {
        return std::unexpected("Empty token_ids");
    }

    // Batch prefill path for multiple tokens
    if (token_ids.size() > 1) {
        return forward_prefill(token_ids, start_position);
    }

    // Single-token decode path
    using clk = std::chrono::high_resolution_clock;
    auto tok_start = clk::now();
    auto t_embed0 = tok_start;
    auto t_embed1 = tok_start;

    for (size_t t = 0; t < token_ids.size(); t++) {
        uint32_t pos = start_position + static_cast<uint32_t>(t);
        uint32_t tok = token_ids[t];

        // 1. Embedding lookup
        t_embed0 = clk::now();
        {
            auto* tok_ptr = static_cast<uint32_t*>(d.gpu->buffer_contents(d.buf_token_id));
            *tok_ptr = tok;
        }

        if (d.token_embed_type == GGMLType::F16) {
            uint32_t n_tok = 1;
            auto r = d.dispatch_embedding_lookup(
                d.token_embed_buf, d.buf_token_id, d.buf_hidden,
                c.embed_dim, n_tok);
            if (!r) return std::unexpected(r.error());
        } else {
            auto* embed_ptr = static_cast<const uint8_t*>(
                d.gpu->buffer_contents(d.token_embed_buf));
            auto* hidden_ptr = static_cast<f16*>(
                d.gpu->buffer_contents(d.buf_hidden));

            if (d.token_embed_type == GGMLType::F32) {
                const auto* src = reinterpret_cast<const float*>(
                    embed_ptr + size_t(tok) * c.embed_dim * sizeof(float));
                for (uint32_t i = 0; i < c.embed_dim; i++)
                    hidden_ptr[i] = static_cast<f16>(src[i]);
            } else if (d.token_embed_type == GGMLType::Q8_0) {
                constexpr uint32_t Q8_BLOCK = 32;
                constexpr size_t   Q8_BYTES = 34;
                uint32_t n_blocks = c.embed_dim / Q8_BLOCK;
                const auto* row = embed_ptr + size_t(tok) * n_blocks * Q8_BYTES;
                for (uint32_t b = 0; b < n_blocks; b++) {
                    const auto* blk = row + b * Q8_BYTES;
                    f16 scale_f16;
                    std::memcpy(&scale_f16, blk, sizeof(f16));
                    float scale = static_cast<float>(scale_f16);
                    const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
                    for (uint32_t j = 0; j < Q8_BLOCK; j++)
                        hidden_ptr[b * Q8_BLOCK + j] = static_cast<f16>(scale * static_cast<float>(qs[j]));
                }
            } else if (d.token_embed_type == GGMLType::Q4_0) {
                constexpr uint32_t Q4_BLOCK = 32;
                constexpr size_t   Q4_BYTES = 18;
                uint32_t n_blocks = c.embed_dim / Q4_BLOCK;
                const auto* row = embed_ptr + size_t(tok) * n_blocks * Q4_BYTES;
                for (uint32_t b = 0; b < n_blocks; b++) {
                    const auto* blk = row + b * Q4_BYTES;
                    f16 scale_f16;
                    std::memcpy(&scale_f16, blk, sizeof(f16));
                    float scale = static_cast<float>(scale_f16);
                    const uint8_t* qs = blk + 2;
                    for (uint32_t j = 0; j < 16; j++) {
                        int lo = (qs[j] & 0x0F) - 8;
                        int hi = (qs[j] >> 4)    - 8;
                        hidden_ptr[b * Q4_BLOCK + j]      = static_cast<f16>(lo * scale);
                        hidden_ptr[b * Q4_BLOCK + j + 16] = static_cast<f16>(hi * scale);
                    }
                }
            } else {
                return std::unexpected(
                    std::format("Unsupported embedding type: {}",
                                static_cast<uint32_t>(d.token_embed_type)));
            }
        }
        t_embed1 = clk::now();

        // 2. Transformer layers
        uint32_t q_M  = c.n_heads * c.head_dim;
        uint32_t kv_M = c.n_kv_heads * c.head_dim;
        size_t kv_bytes = size_t(kv_M) * sizeof(f16);
        size_t q_bytes  = size_t(q_M) * sizeof(f16);

        auto append_pre_attn_dps = [&](
            std::vector<std::vector<MetalCompute::DispatchParams>>& chain,
            uint32_t layer_idx)
        {
            auto& lw = d.layers[layer_idx];
            chain.push_back({d.make_rms_norm_dp(
                d.buf_hidden, lw.attn_norm_buf, d.buf_hidden2,
                c.embed_dim, c.rms_norm_eps)});
            chain.push_back({
                d.make_matvec_dp(lw.wq_buf, lw.wq_type,
                                 d.buf_hidden2, d.buf_q, q_M, c.embed_dim),
                d.make_matvec_dp(lw.wk_buf, lw.wk_type,
                                 d.buf_hidden2, d.buf_k, kv_M, c.embed_dim),
                d.make_matvec_dp(lw.wv_buf, lw.wv_type,
                                 d.buf_hidden2, d.buf_v, kv_M, c.embed_dim),
            });
            std::vector<MetalCompute::DispatchParams> bias_group;
            if (lw.bq_buf) bias_group.push_back(d.make_elementwise_dp(
                d.pipe_elementwise_add, d.buf_q, lw.bq_buf, d.buf_q, q_M));
            if (lw.bk_buf) bias_group.push_back(d.make_elementwise_dp(
                d.pipe_elementwise_add, d.buf_k, lw.bk_buf, d.buf_k, kv_M));
            if (lw.bv_buf) bias_group.push_back(d.make_elementwise_dp(
                d.pipe_elementwise_add, d.buf_v, lw.bv_buf, d.buf_v, kv_M));
            if (!bias_group.empty()) chain.push_back(std::move(bias_group));
            std::vector<MetalCompute::DispatchParams> qk_norm_group;
            if (lw.q_norm_buf) qk_norm_group.push_back(
                d.make_rms_norm_dp(d.buf_q, lw.q_norm_buf, d.buf_q,
                                   q_M, c.rms_norm_eps));
            if (lw.k_norm_buf) qk_norm_group.push_back(
                d.make_rms_norm_dp(d.buf_k, lw.k_norm_buf, d.buf_k,
                                   kv_M, c.rms_norm_eps));
            if (!qk_norm_group.empty()) chain.push_back(std::move(qk_norm_group));
            chain.push_back({
                d.make_rope_dp(d.buf_q, c.head_dim, c.n_heads, pos, c.rope_theta),
                d.make_rope_dp(d.buf_k, c.head_dim, c.n_kv_heads, pos, c.rope_theta),
            });
        };

        // Lazy CPU KV: when GPU attention is on and scatter_kv handles GPU
        // buffer writes, skip the expensive GPU→CPU read_buffer + CPU-side
        // quantize/append per layer. Only track seq_len via advance_seq_len.
        bool lazy_cpu_kv = d.use_gpu_attention_ && !c.is_moe && !c.is_mla();
        uint32_t decode_seq_len = d.kv_cache->seq_len() + 1;

        auto do_cpu_kv_work = [&](uint32_t L, bool gpu_scatter) -> uint32_t {
            if (lazy_cpu_kv) {
                return pos;
            }
            d.gpu->read_buffer(d.buf_k, d.scratch_k_cpu_.data(), kv_bytes);
            d.gpu->read_buffer(d.buf_v, d.scratch_v_cpu_.data(), kv_bytes);
            d.kv_cache->append(L, d.scratch_k_cpu_.data(),
                                  d.scratch_v_cpu_.data());
            uint32_t kv_pos = d.kv_cache->seq_len() - 1;
            if (!gpu_scatter) {
                size_t offset = size_t(kv_pos) * c.n_kv_heads * c.head_dim;
                auto* k_gpu = static_cast<f16*>(
                    d.gpu->buffer_contents(d.buf_kv_gpu_k[L]));
                auto* v_gpu = static_cast<f16*>(
                    d.gpu->buffer_contents(d.buf_kv_gpu_v[L]));
                std::memcpy(k_gpu + offset, d.scratch_k_cpu_.data(), kv_bytes);
                std::memcpy(v_gpu + offset, d.scratch_v_cpu_.data(), kv_bytes);
            }
            if (!d.use_gpu_attention_) {
                d.gpu->read_buffer(d.buf_q, d.scratch_q_cpu_.data(), q_bytes);
                uint32_t sl = d.kv_cache->seq_len();
                d.cpu_attention(d.scratch_q_cpu_.data(), L, sl,
                               d.scratch_attn_out_.data());
                std::memcpy(
                    static_cast<f16*>(d.gpu->buffer_contents(d.buf_attn_out)),
                    d.scratch_attn_out_.data(), q_bytes);
            }
            return kv_pos;
        };

        if (lazy_cpu_kv) {
            // ── Mega-chain: all dense layers + final → 1 Metal dispatch ──
            // lazy_cpu_kv ⟹ use_gpu_attention_=true, is_moe=false.
            // DP-CACHE: first decode builds template; subsequent decodes
            // patch only pos/kv_offset/decode_seq_len in-place (zero alloc).
            uint32_t kv_count = c.n_kv_heads * c.head_dim;
            uint32_t kv_offset = pos * kv_count;

            if (!d.mega_chain_cached_) {
                d.cached_mega_chain_.clear();
                d.cached_mega_chain_.reserve(c.n_layers * 15 + 2);

                for (uint32_t L = 0; L < c.n_layers; L++) {
                    auto& lw = d.layers[L];
                    append_pre_attn_dps(d.cached_mega_chain_, L);
                    d.cached_mega_chain_.push_back({
                        d.make_scatter_kv_dp(d.buf_k, d.buf_kv_gpu_k[L],
                                             kv_offset, kv_count),
                        d.make_scatter_kv_dp(d.buf_v, d.buf_kv_gpu_v[L],
                                             kv_offset, kv_count),
                    });
                    d.cached_mega_chain_.push_back({d.make_flash_attention_decode_dp(
                        d.buf_q, d.buf_kv_gpu_k[L], d.buf_kv_gpu_v[L],
                        d.buf_attn_out, c.n_heads, c.n_kv_heads,
                        c.head_dim, decode_seq_len)});
                    d.cached_mega_chain_.push_back({d.make_matvec_dp(
                        lw.wo_buf, lw.wo_type, d.buf_attn_out, d.buf_hidden2,
                        c.embed_dim, q_M)});
                    d.cached_mega_chain_.push_back({d.make_elementwise_dp(
                        d.pipe_elementwise_add, d.buf_hidden, d.buf_hidden2,
                        d.buf_hidden, c.embed_dim)});
                    d.cached_mega_chain_.push_back({d.make_rms_norm_dp(
                        d.buf_hidden, lw.ffn_norm_buf, d.buf_hidden2,
                        c.embed_dim, c.rms_norm_eps)});
                    d.cached_mega_chain_.push_back({
                        d.make_matvec_dp(lw.w1_buf, lw.w1_type,
                                         d.buf_hidden2, d.buf_ffn_gate,
                                         c.ffn_dim, c.embed_dim),
                        d.make_matvec_dp(lw.w3_buf, lw.w3_type,
                                         d.buf_hidden2, d.buf_ffn_up,
                                         c.ffn_dim, c.embed_dim),
                    });
                    d.cached_mega_chain_.push_back({d.make_silu_mul_dp(
                        d.buf_ffn_gate, d.buf_ffn_up, d.buf_ffn_gate, c.ffn_dim)});
                    d.cached_mega_chain_.push_back({d.make_matvec_dp(
                        lw.w2_buf, lw.w2_type, d.buf_ffn_gate, d.buf_hidden2,
                        c.embed_dim, c.ffn_dim)});
                    d.cached_mega_chain_.push_back({d.make_elementwise_dp(
                        d.pipe_elementwise_add, d.buf_hidden, d.buf_hidden2,
                        d.buf_hidden, c.embed_dim)});
                }

                d.cached_mega_chain_.push_back({d.make_rms_norm_dp(
                    d.buf_hidden, d.output_norm_buf, d.buf_hidden2,
                    c.embed_dim, c.rms_norm_eps)});
                d.cached_mega_chain_.push_back({d.make_matvec_dp(
                    d.output_weight_buf, d.output_weight_type,
                    d.buf_hidden2, d.buf_logits,
                    c.vocab_size, c.embed_dim)});

                d.mega_rope_pos_locs_.clear();
                d.mega_kv_offset_locs_.clear();
                d.mega_seq_len_locs_.clear();
                for (size_t gi = 0; gi < d.cached_mega_chain_.size(); gi++) {
                    for (size_t di = 0; di < d.cached_mega_chain_[gi].size(); di++) {
                        auto* p = d.cached_mega_chain_[gi][di].pipeline;
                        if (p == d.pipe_rope)
                            d.mega_rope_pos_locs_.push_back({gi, di, 8});
                        else if (p == d.pipe_scatter_kv)
                            d.mega_kv_offset_locs_.push_back({gi, di, 0});
                        else if (p == d.pipe_flash_attention_decode)
                            d.mega_seq_len_locs_.push_back({gi, di, 12});
                    }
                }
                d.mega_chain_cached_ = true;
            } else {
                auto t_patch0 = clk::now();
                for (auto& [gi, di, off] : d.mega_rope_pos_locs_)
                    std::memcpy(d.cached_mega_chain_[gi][di].const_data.data() + off,
                                &pos, sizeof(pos));
                for (auto& [gi, di, off] : d.mega_kv_offset_locs_)
                    std::memcpy(d.cached_mega_chain_[gi][di].const_data.data() + off,
                                &kv_offset, sizeof(kv_offset));
                for (auto& [gi, di, off] : d.mega_seq_len_locs_)
                    std::memcpy(d.cached_mega_chain_[gi][di].const_data.data() + off,
                                &decode_seq_len, sizeof(decode_seq_len));
                auto t_patch1 = clk::now();
                d.decode_profile_.patch_ms +=
                    std::chrono::duration<double, std::milli>(t_patch1 - t_patch0).count();
            }

            auto t_disp0 = clk::now();
            auto mc_r = d.gpu->dispatch_chain_sync(d.cached_mega_chain_);
            if (!mc_r) return std::unexpected(mc_r.error());
            auto t_disp1 = clk::now();
            d.decode_profile_.gpu_ms += *mc_r * 1000.0;
            d.decode_profile_.dispatch_wall_ms +=
                std::chrono::duration<double, std::milli>(t_disp1 - t_disp0).count();

            d.kv_cache->advance_seq_len(1);
        } else {

        for (uint32_t L = 0; L < c.n_layers; L++) {
            auto& lw = d.layers[L];

            if (c.is_mla()) {
                // ═══ MLA Attention Path (absorbed attention, v1 CPU) ═══
                const uint32_t nope_dim = c.qk_nope_head_dim;
                const uint32_t rope_dim = c.qk_rope_head_dim;
                const uint32_t head_dim_mla = nope_dim + rope_dim;
                const uint32_t kv_rank = c.kv_lora_rank;
                const uint32_t compressed_dim = kv_rank + rope_dim;
                const uint32_t mla_q_dim = c.n_heads * head_dim_mla;
                const uint32_t wo_input_dim = c.n_heads * c.v_head_dim;

                // A. Attention RMSNorm
                {
                    auto r = d.dispatch_rms_norm(d.buf_hidden, lw.attn_norm_buf,
                                                 d.buf_hidden2, c.embed_dim,
                                                 c.rms_norm_eps);
                    if (!r) return std::unexpected(r.error());
                }
                // B-D. Q projection: LoRA (V3) or full (V2-Lite)
                if (c.q_lora_rank > 0) {
                    {
                        auto r = d.dispatch_matvec(lw.attn_q_a_buf, lw.attn_q_a_type,
                                                   d.buf_hidden2, d.buf_mla_q_a,
                                                   c.q_lora_rank, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_rms_norm(d.buf_mla_q_a, lw.attn_q_a_norm_buf,
                                                     d.buf_mla_q_a, c.q_lora_rank,
                                                     c.rms_norm_eps);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_matvec(lw.attn_q_b_buf, lw.attn_q_b_type,
                                                   d.buf_mla_q_a, d.buf_mla_q,
                                                   mla_q_dim, c.q_lora_rank);
                        if (!r) return std::unexpected(r.error());
                    }
                } else {
                    auto r = d.dispatch_matvec(lw.wq_buf, lw.wq_type,
                                               d.buf_hidden2, d.buf_mla_q,
                                               mla_q_dim, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // E. CPU NORM RoPE on q rope portions (GGUF stores in interleaved format)
                {
                    std::vector<f16> q_cpu(mla_q_dim);
                    d.gpu->read_buffer(d.buf_mla_q, q_cpu.data(),
                                       mla_q_dim * sizeof(f16));
                    uint32_t half = rope_dim / 2;


                    for (uint32_t h = 0; h < c.n_heads; h++) {
                        f16* rs = q_cpu.data() + h * head_dim_mla + nope_dim;
#if MUGEN_ROPE_STANDARD
                        for (uint32_t i = 0; i < half; i++) {
                            float freq = yarn_rope_freq(i, rope_dim,
                                c.rope_scaling_factor, c.rope_original_ctx,
                                c.rope_theta);
                            float angle = static_cast<float>(pos) * freq;
                            float cv = std::cos(angle), sv = std::sin(angle);
                            float x0 = static_cast<float>(rs[2*i]);
                            float x1 = static_cast<float>(rs[2*i+1]);
                            rs[2*i] = static_cast<f16>(x0 * cv - x1 * sv);
                            rs[2*i+1] = static_cast<f16>(x1 * cv + x0 * sv);
                        }
#else
                        for (uint32_t i = 0; i < half; i++) {
                            float freq = yarn_rope_freq(i, rope_dim,
                                c.rope_scaling_factor, c.rope_original_ctx,
                                c.rope_theta);
                            float angle = static_cast<float>(pos) * freq;
                            float cv = std::cos(angle), sv = std::sin(angle);
                            float x0 = static_cast<float>(rs[i]);
                            float x1 = static_cast<float>(rs[i + half]);
                            rs[i] = static_cast<f16>(x0 * cv - x1 * sv);
                            rs[i + half] =
                                static_cast<f16>(x1 * cv + x0 * sv);
                        }
#endif
                    }
                    std::memcpy(d.gpu->buffer_contents(d.buf_mla_q),
                                q_cpu.data(), mla_q_dim * sizeof(f16));
                }
                // F. kv_a = hidden2 × W_kv_a_mqa
                {
                    auto r = d.dispatch_matvec(lw.attn_kv_a_mqa_buf,
                                               lw.attn_kv_a_mqa_type,
                                               d.buf_hidden2, d.buf_mla_kv_a,
                                               compressed_dim, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // G. CPU: RMSNorm(c_kv) + NeoX RoPE(k_rope)
                {
                    std::vector<f16> kva(compressed_dim);
                    d.gpu->read_buffer(d.buf_mla_kv_a, kva.data(),
                                       compressed_dim * sizeof(f16));
                    float ss = 0.0f;
                    for (uint32_t i = 0; i < kv_rank; i++) {
                        float v = static_cast<float>(kva[i]);
                        ss += v * v;
                    }
                    ss = 1.0f / std::sqrt(ss / static_cast<float>(kv_rank)
                                          + c.rms_norm_eps);
                    const auto* nw = static_cast<const f16*>(
                        d.gpu->buffer_contents(lw.attn_kv_a_norm_buf));
                    for (uint32_t i = 0; i < kv_rank; i++) {
                        float v = static_cast<float>(kva[i]) * ss;
                        kva[i] = static_cast<f16>(
                            v * static_cast<float>(nw[i]));
                    }
                    f16* kr = kva.data() + kv_rank;
                    uint32_t half_kr = rope_dim / 2;
#if MUGEN_ROPE_STANDARD
                    for (uint32_t i = 0; i < half_kr; i++) {
                        float freq = yarn_rope_freq(i, rope_dim,
                            c.rope_scaling_factor, c.rope_original_ctx,
                            c.rope_theta);
                        float angle = static_cast<float>(pos) * freq;
                        float cv = std::cos(angle), sv = std::sin(angle);
                        float x0 = static_cast<float>(kr[2*i]);
                        float x1 = static_cast<float>(kr[2*i+1]);
                        kr[2*i] = static_cast<f16>(x0 * cv - x1 * sv);
                        kr[2*i+1] = static_cast<f16>(x1 * cv + x0 * sv);
                    }
#else
                    for (uint32_t i = 0; i < half_kr; i++) {
                        float freq = yarn_rope_freq(i, rope_dim,
                            c.rope_scaling_factor, c.rope_original_ctx,
                            c.rope_theta);
                        float angle = static_cast<float>(pos) * freq;
                        float cv = std::cos(angle), sv = std::sin(angle);
                        float x0 = static_cast<float>(kr[i]);
                        float x1 = static_cast<float>(kr[i + half_kr]);
                        kr[i] = static_cast<f16>(x0 * cv - x1 * sv);
                        kr[i + half_kr] =
                            static_cast<f16>(x1 * cv + x0 * sv);
                    }
#endif
                    std::memcpy(d.gpu->buffer_contents(d.buf_mla_kv_a),
                                kva.data(), compressed_dim * sizeof(f16));
                }
                // H. Scatter compressed KV to per-layer cache
                {
                    uint32_t kv_off = pos * compressed_dim;
                    auto r = d.gpu->dispatch_sync(d.make_scatter_kv_dp(
                        d.buf_mla_kv_a, d.buf_kv_compressed[L],
                        kv_off, compressed_dim));
                    if (!r) return std::unexpected(r.error());
                }
                // I. CPU MLA attention (absorbed)
                {
                    std::vector<f16> q_cpu(mla_q_dim);
                    d.gpu->read_buffer(d.buf_mla_q, q_cpu.data(),
                                       mla_q_dim * sizeof(f16));
                    std::vector<f16> attn_out(
                        size_t(c.n_heads) * c.v_head_dim);
                    d.cpu_mla_attention(q_cpu.data(), lw, L,
                                        decode_seq_len, attn_out.data());
                    std::memcpy(
                        d.gpu->buffer_contents(d.buf_mla_attn_out),
                        attn_out.data(),
                        size_t(c.n_heads) * c.v_head_dim * sizeof(f16));
                }
                // J. Wo projection
                {
                    auto r = d.dispatch_matvec(lw.wo_buf, lw.wo_type,
                                               d.buf_mla_attn_out, d.buf_hidden2,
                                               c.embed_dim, wo_input_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // K. Residual add
                {
                    auto r = d.dispatch_elementwise(d.pipe_elementwise_add,
                                                    d.buf_hidden, d.buf_hidden2,
                                                    d.buf_hidden, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // L. FFN RMSNorm
                {
                    auto r = d.dispatch_rms_norm(d.buf_hidden, lw.ffn_norm_buf,
                                                 d.buf_hidden2, c.embed_dim,
                                                 c.rms_norm_eps);
                    if (!r) return std::unexpected(r.error());
                }
                // M. FFN (Dense or MoE)
                if (lw.kind == LayerKind::MoE) {
                    bool has_shared = c.n_shared_experts > 0 && lw.w1_shared_buf;
                    if (has_shared) {
                        std::memcpy(d.gpu->buffer_contents(d.buf_q),
                                    d.gpu->buffer_contents(d.buf_hidden2),
                                    c.embed_dim * sizeof(f16));
                    }
                    {
                        auto r = d.dispatch_matvec(lw.router_buf, lw.router_type,
                                                   d.buf_hidden2, d.buf_router_logits,
                                                   c.n_experts, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    if (c.n_group > 0) {
                        auto r = d.dispatch_moe_gate_grouped(
                            d.buf_router_logits, d.buf_top_indices,
                            d.buf_top_weights, lw.router_bias_buf,
                            c.n_experts, c.n_experts_used,
                            c.n_group, c.topk_group,
                            c.routed_scaling_factor);
                        if (!r) return std::unexpected(r.error());
                    } else {
                        auto r = d.dispatch_moe_gate(d.buf_router_logits,
                                                     d.buf_top_indices,
                                                     d.buf_top_weights,
                                                     c.n_experts,
                                                     c.n_experts_used);
                        if (!r) return std::unexpected(r.error());
                    }
                    std::vector<uint32_t> top_idx(c.n_experts_used);
                    d.gpu->read_buffer(d.buf_top_indices, top_idx.data(),
                                       c.n_experts_used * sizeof(uint32_t));
                    if (d.route_callback_) {
                        std::vector<f16> wf(c.n_experts_used);
                        d.gpu->read_buffer(d.buf_top_weights, wf.data(),
                                           c.n_experts_used * sizeof(f16));
                        std::vector<float> wf32(c.n_experts_used);
                        for (uint32_t ki = 0; ki < c.n_experts_used; ki++)
                            wf32[ki] = static_cast<float>(wf[ki]);
                        d.route_callback_(L, top_idx.data(), wf32.data(),
                                          c.n_experts_used);
                    }
                    for (uint32_t k = 0; k < c.n_experts_used; k++) {
                        uint32_t eid = top_idx[k];
                        auto [ebuf_gate, ebuf_up, ebuf_down] =
                            d.resolve_expert_bufs(L, eid, lw);
                        {
                            auto r = d.dispatch_matvec(
                                ebuf_gate, lw.expert_gate_type,
                                d.buf_hidden2, d.buf_ffn_gate,
                                c.expert_ffn_dim, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        {
                            auto r = d.dispatch_matvec(
                                ebuf_up, lw.expert_up_type,
                                d.buf_hidden2, d.buf_ffn_up,
                                c.expert_ffn_dim, c.embed_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        {
                            auto r = d.dispatch_silu(d.buf_ffn_gate,
                                                     d.buf_ffn_gate, c.expert_ffn_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        {
                            auto r = d.dispatch_elementwise(
                                d.pipe_elementwise_mul,
                                d.buf_ffn_gate, d.buf_ffn_up,
                                d.buf_ffn_gate, c.expert_ffn_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        {
                            auto r = d.dispatch_matvec(
                                ebuf_down, lw.expert_down_type,
                                d.buf_ffn_gate, d.buf_attn_out,
                                c.embed_dim, c.expert_ffn_dim);
                            if (!r) return std::unexpected(r.error());
                        }
                        std::vector<f16> eres(c.embed_dim);
                        d.gpu->read_buffer(d.buf_attn_out, eres.data(),
                                           c.embed_dim * sizeof(f16));
                        auto* edst = static_cast<f16*>(
                            d.gpu->buffer_contents(d.buf_expert_out));
                        std::memcpy(edst + size_t(k) * c.embed_dim,
                                    eres.data(),
                                    c.embed_dim * sizeof(f16));
                    }
                    {
                        auto r = d.dispatch_moe_reduce(d.buf_expert_out,
                                                       d.buf_top_weights,
                                                       d.buf_hidden2,
                                                       c.embed_dim,
                                                       c.n_experts_used);
                        if (!r) return std::unexpected(r.error());
                    }
                    if (has_shared) {
                        auto r = d.dispatch_matvec(lw.w1_shared_buf, lw.w1_shared_type,
                            d.buf_q, d.buf_ffn_gate, c.shared_ffn_dim(), c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                        r = d.dispatch_matvec(lw.w3_shared_buf, lw.w3_shared_type,
                            d.buf_q, d.buf_ffn_up, c.shared_ffn_dim(), c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                        r = d.dispatch_silu(d.buf_ffn_gate, d.buf_ffn_gate, c.shared_ffn_dim());
                        if (!r) return std::unexpected(r.error());
                        r = d.dispatch_elementwise(d.pipe_elementwise_mul,
                            d.buf_ffn_gate, d.buf_ffn_up, d.buf_ffn_gate, c.shared_ffn_dim());
                        if (!r) return std::unexpected(r.error());
                        r = d.dispatch_matvec(lw.w2_shared_buf, lw.w2_shared_type,
                            d.buf_ffn_gate, d.buf_attn_out, c.embed_dim, c.shared_ffn_dim());
                        if (!r) return std::unexpected(r.error());
                        r = d.dispatch_elementwise(d.pipe_elementwise_add,
                            d.buf_hidden2, d.buf_attn_out, d.buf_hidden2, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_elementwise(d.pipe_elementwise_add,
                                                        d.buf_hidden, d.buf_hidden2,
                                                        d.buf_hidden, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                } else {
                    // Dense FFN
                    {
                        auto r = d.dispatch_matvec(lw.w1_buf, lw.w1_type,
                                                   d.buf_hidden2, d.buf_ffn_gate,
                                                   c.ffn_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_matvec(lw.w3_buf, lw.w3_type,
                                                   d.buf_hidden2, d.buf_ffn_up,
                                                   c.ffn_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.gpu->dispatch_sync(d.make_silu_mul_dp(
                            d.buf_ffn_gate, d.buf_ffn_up,
                            d.buf_ffn_gate, c.ffn_dim));
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_matvec(lw.w2_buf, lw.w2_type,
                                                   d.buf_ffn_gate, d.buf_hidden2,
                                                   c.embed_dim, c.ffn_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_elementwise(d.pipe_elementwise_add,
                                                        d.buf_hidden, d.buf_hidden2,
                                                        d.buf_hidden, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                }
            } else {
            // ═══ Standard MHA/GQA Attention Path ═══
            bool pre_attn_already_dispatched = (L > 0 && d.layers[L - 1].kind == LayerKind::Dense);
            if (!pre_attn_already_dispatched) {
                std::vector<std::vector<MetalCompute::DispatchParams>> pre;
                append_pre_attn_dps(pre, L);
                auto r = d.gpu->dispatch_chain_sync(pre);
                if (!r) return std::unexpected(r.error());
            }

            uint32_t kv_pos = do_cpu_kv_work(L, /*gpu_scatter=*/lw.kind == LayerKind::Dense);

            // ── Post-attention: dense chain vs MoE individual dispatches ──
            if (lw.kind == LayerKind::MoE) {
                if (d.use_gpu_attention_) {
                    uint32_t sl = d.kv_cache->seq_len();
                    auto r = d.gpu->dispatch_sync(d.make_flash_attention_decode_dp(
                        d.buf_q, d.buf_kv_gpu_k[L], d.buf_kv_gpu_v[L],
                        d.buf_attn_out, c.n_heads, c.n_kv_heads, c.head_dim, sl));
                    if (!r) return std::unexpected(r.error());
                }
                // Wo projection
                {
                    auto r = d.dispatch_matvec(lw.wo_buf, lw.wo_type,
                                               d.buf_attn_out, d.buf_hidden2,
                                               c.embed_dim, q_M);
                    if (!r) return std::unexpected(r.error());
                }
                // Residual add
                {
                    auto r = d.dispatch_elementwise(d.pipe_elementwise_add,
                                                    d.buf_hidden, d.buf_hidden2,
                                                    d.buf_hidden, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // FFN RMSNorm
                {
                    auto r = d.dispatch_rms_norm(d.buf_hidden, lw.ffn_norm_buf,
                                                 d.buf_hidden2, c.embed_dim,
                                                 c.rms_norm_eps);
                    if (!r) return std::unexpected(r.error());
                }
                // MoE FFN (hard sync on top_indices readback)
                bool has_shared = c.n_shared_experts > 0 && lw.w1_shared_buf;
                if (has_shared) {
                    std::memcpy(d.gpu->buffer_contents(d.buf_q),
                                d.gpu->buffer_contents(d.buf_hidden2),
                                c.embed_dim * sizeof(f16));
                }
                {
                    auto r = d.dispatch_matvec(lw.router_buf, lw.router_type,
                                               d.buf_hidden2, d.buf_router_logits,
                                               c.n_experts, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                if (c.n_group > 0) {
                    auto r = d.dispatch_moe_gate_grouped(
                        d.buf_router_logits, d.buf_top_indices, d.buf_top_weights,
                        lw.router_bias_buf, c.n_experts, c.n_experts_used,
                        c.n_group, c.topk_group, c.routed_scaling_factor);
                    if (!r) return std::unexpected(r.error());
                } else {
                    auto r = d.dispatch_moe_gate(d.buf_router_logits,
                                                 d.buf_top_indices,
                                                 d.buf_top_weights,
                                                 c.n_experts, c.n_experts_used);
                    if (!r) return std::unexpected(r.error());
                }

                std::vector<uint32_t> top_idx(c.n_experts_used);
                d.gpu->read_buffer(d.buf_top_indices, top_idx.data(),
                                   c.n_experts_used * sizeof(uint32_t));

                if (d.route_callback_) {
                    std::vector<f16> wt_f16(c.n_experts_used);
                    d.gpu->read_buffer(d.buf_top_weights, wt_f16.data(),
                                       c.n_experts_used * sizeof(f16));
                    std::vector<float> wt_f32(c.n_experts_used);
                    for (uint32_t ki = 0; ki < c.n_experts_used; ki++)
                        wt_f32[ki] = static_cast<float>(wt_f16[ki]);
                    d.route_callback_(L, top_idx.data(), wt_f32.data(),
                                      c.n_experts_used);
                }

                for (uint32_t k = 0; k < c.n_experts_used; k++) {
                    uint32_t eid = top_idx[k];
                    auto [ebuf_gate, ebuf_up, ebuf_down] =
                        d.resolve_expert_bufs(L, eid, lw);
                    {
                        auto r = d.dispatch_matvec(
                            ebuf_gate, lw.expert_gate_type,
                            d.buf_hidden2, d.buf_ffn_gate,
                            c.expert_ffn_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_matvec(
                            ebuf_up, lw.expert_up_type,
                            d.buf_hidden2, d.buf_ffn_up,
                            c.expert_ffn_dim, c.embed_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_silu(d.buf_ffn_gate,
                                                 d.buf_ffn_gate, c.expert_ffn_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_elementwise(d.pipe_elementwise_mul,
                                                        d.buf_ffn_gate,
                                                        d.buf_ffn_up,
                                                        d.buf_ffn_gate, c.expert_ffn_dim);
                        if (!r) return std::unexpected(r.error());
                    }
                    {
                        auto r = d.dispatch_matvec(
                            ebuf_down, lw.expert_down_type,
                            d.buf_ffn_gate, d.buf_attn_out,
                            c.embed_dim, c.expert_ffn_dim);
                        if (!r) return std::unexpected(r.error());
                    }

                    std::vector<f16> expert_result(c.embed_dim);
                    d.gpu->read_buffer(d.buf_attn_out, expert_result.data(),
                                       c.embed_dim * sizeof(f16));
                    auto* dst = static_cast<f16*>(
                        d.gpu->buffer_contents(d.buf_expert_out));
                    std::memcpy(dst + size_t(k) * c.embed_dim,
                                expert_result.data(),
                                c.embed_dim * sizeof(f16));
                }

                {
                    auto r = d.dispatch_moe_reduce(d.buf_expert_out,
                                                   d.buf_top_weights,
                                                   d.buf_hidden2,
                                                   c.embed_dim,
                                                   c.n_experts_used);
                    if (!r) return std::unexpected(r.error());
                }
                if (has_shared) {
                    auto r = d.dispatch_matvec(lw.w1_shared_buf, lw.w1_shared_type,
                        d.buf_q, d.buf_ffn_gate, c.shared_ffn_dim(), c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                    r = d.dispatch_matvec(lw.w3_shared_buf, lw.w3_shared_type,
                        d.buf_q, d.buf_ffn_up, c.shared_ffn_dim(), c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                    r = d.dispatch_silu(d.buf_ffn_gate, d.buf_ffn_gate, c.shared_ffn_dim());
                    if (!r) return std::unexpected(r.error());
                    r = d.dispatch_elementwise(d.pipe_elementwise_mul,
                        d.buf_ffn_gate, d.buf_ffn_up, d.buf_ffn_gate, c.shared_ffn_dim());
                    if (!r) return std::unexpected(r.error());
                    r = d.dispatch_matvec(lw.w2_shared_buf, lw.w2_shared_type,
                        d.buf_ffn_gate, d.buf_attn_out, c.embed_dim, c.shared_ffn_dim());
                    if (!r) return std::unexpected(r.error());
                    r = d.dispatch_elementwise(d.pipe_elementwise_add,
                        d.buf_hidden2, d.buf_attn_out, d.buf_hidden2, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
                // MoE final residual add
                {
                    auto r = d.dispatch_elementwise(d.pipe_elementwise_add,
                                                    d.buf_hidden, d.buf_hidden2,
                                                    d.buf_hidden, c.embed_dim);
                    if (!r) return std::unexpected(r.error());
                }
            } else {
                // Dense: scatter_kv → [attention →] Wo → residual → FFN → residual [→ next pre-attn]
                std::vector<std::vector<MetalCompute::DispatchParams>> post_attn;

                uint32_t kv_count = c.n_kv_heads * c.head_dim;
                uint32_t kv_offset = kv_pos * kv_count;
                post_attn.push_back({
                    d.make_scatter_kv_dp(d.buf_k, d.buf_kv_gpu_k[L],
                                         kv_offset, kv_count),
                    d.make_scatter_kv_dp(d.buf_v, d.buf_kv_gpu_v[L],
                                         kv_offset, kv_count),
                });

                if (d.use_gpu_attention_) {
                    uint32_t sl = lazy_cpu_kv ? decode_seq_len
                                              : d.kv_cache->seq_len();
                    post_attn.push_back({d.make_flash_attention_decode_dp(
                        d.buf_q, d.buf_kv_gpu_k[L], d.buf_kv_gpu_v[L],
                        d.buf_attn_out, c.n_heads, c.n_kv_heads, c.head_dim, sl)});
                }

                post_attn.push_back({d.make_matvec_dp(
                    lw.wo_buf, lw.wo_type, d.buf_attn_out, d.buf_hidden2,
                    c.embed_dim, q_M)});
                post_attn.push_back({d.make_elementwise_dp(
                    d.pipe_elementwise_add, d.buf_hidden, d.buf_hidden2,
                    d.buf_hidden, c.embed_dim)});
                post_attn.push_back({d.make_rms_norm_dp(
                    d.buf_hidden, lw.ffn_norm_buf, d.buf_hidden2,
                    c.embed_dim, c.rms_norm_eps)});
                post_attn.push_back({
                    d.make_matvec_dp(lw.w1_buf, lw.w1_type,
                                     d.buf_hidden2, d.buf_ffn_gate,
                                     c.ffn_dim, c.embed_dim),
                    d.make_matvec_dp(lw.w3_buf, lw.w3_type,
                                     d.buf_hidden2, d.buf_ffn_up,
                                     c.ffn_dim, c.embed_dim),
                });
                post_attn.push_back({d.make_silu_mul_dp(
                    d.buf_ffn_gate, d.buf_ffn_up, d.buf_ffn_gate, c.ffn_dim)});
                post_attn.push_back({d.make_matvec_dp(
                    lw.w2_buf, lw.w2_type, d.buf_ffn_gate, d.buf_hidden2,
                    c.embed_dim, c.ffn_dim)});
                post_attn.push_back({d.make_elementwise_dp(
                    d.pipe_elementwise_add, d.buf_hidden, d.buf_hidden2,
                    d.buf_hidden, c.embed_dim)});

                if (L + 1 < c.n_layers) {
                    append_pre_attn_dps(post_attn, L + 1);
                }

                auto r = d.gpu->dispatch_chain_sync(post_attn);
                if (!r) return std::unexpected(r.error());
            }
            } // end non-MLA else
        }

        if (c.is_mla()) {
            d.kv_cache->advance_seq_len(1);
        }

        } // end per-layer else
    }

    // 3. Final: rms_norm → output projection (per-layer path only;
    //    mega-chain already includes final)
    if (!(d.use_gpu_attention_ && !c.is_moe && !c.is_mla())) {
        auto r = d.gpu->dispatch_chain_sync({
            {d.make_rms_norm_dp(d.buf_hidden, d.output_norm_buf, d.buf_hidden2,
                                c.embed_dim, c.rms_norm_eps)},
            {d.make_matvec_dp(d.output_weight_buf, d.output_weight_type,
                              d.buf_hidden2, d.buf_logits,
                              c.vocab_size, c.embed_dim)},
        });
        if (!r) return std::unexpected(r.error());
    }

    // 4a. GPU argmax fast-path: skip 296KB readback, dispatch argmax kernel
    if (d.argmax_mode_) {
        auto ar = d.gpu->dispatch_sync(
            d.make_argmax_dp(d.buf_logits, d.buf_argmax_result, c.vocab_size));
        if (!ar) return std::unexpected(ar.error());
        return std::vector<float>{};
    }

    // 4b. Read logits (F16) → convert to F32
    auto t_logits0 = clk::now();
    d.gpu->read_buffer(d.buf_logits, d.scratch_logits_f16_.data(),
                       c.vocab_size * sizeof(f16));

    std::vector<float> logits(c.vocab_size);
    const auto* src_f16 = d.scratch_logits_f16_.data();
    auto* dst_f32 = logits.data();
    for (uint32_t i = 0; i < c.vocab_size; i++) {
        dst_f32[i] = static_cast<float>(src_f16[i]);
    }
    auto t_logits1 = clk::now();

    d.decode_profile_.logits_ms +=
        std::chrono::duration<double, std::milli>(t_logits1 - t_logits0).count();
    d.decode_profile_.embed_ms +=
        std::chrono::duration<double, std::milli>(t_embed1 - t_embed0).count();
    auto tok_end = clk::now();
    d.decode_profile_.total_wall_ms +=
        std::chrono::duration<double, std::milli>(tok_end - tok_start).count();
    d.decode_profile_.n_tokens++;

    return logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward_argmax: GPU-side argmax, skip full logits readback (4B vs 296KB)
// ─────────────────────────────────────────────────────────────────────────────

auto TransformerModel::forward_argmax(const std::vector<uint32_t>& token_ids,
                                       uint32_t start_position)
    -> std::expected<uint32_t, std::string>
{
    auto& d = *impl_;
    if (token_ids.empty())
        return std::unexpected("forward_argmax: empty token_ids");

    d.argmax_mode_ = true;
    auto result = forward(token_ids, start_position);
    d.argmax_mode_ = false;

    if (!result) return std::unexpected(result.error());

    uint32_t token_id = 0;
    d.gpu->read_buffer(d.buf_argmax_result, &token_id, sizeof(uint32_t));
    return token_id;
}

} // namespace mugen

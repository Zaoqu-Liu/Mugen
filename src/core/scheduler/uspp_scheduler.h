#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/compute/metal_compute.h"
#include "core/memory/buffer_manager.h"
#include "core/prefetch/cache_policy.h"
#include "core/scheduler/sampling.h"

namespace mugen {

class TransformerModel;

struct USPPConfig {
    uint32_t draft_k = 6;
    uint32_t min_draft_k = 2;
    uint32_t max_draft_k = 8;
    float accept_threshold = 0.8f;
    float prefetch_ahead_ms = 300.0f;
    uint32_t warmup_tokens = 50;
    float conservative_hit_threshold = 0.7f;
};

struct InferenceRequest {
    std::vector<uint32_t> prompt_tokens;
    uint32_t max_new_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool stream = true;
};

struct TokenResult {
    uint32_t token_id;
    float logprob;
    bool from_draft;
};

struct USPPMetrics {
    float tokens_per_second = 0.0f;
    float time_to_first_token_ms = 0.0f;

    uint32_t current_k = 0;
    float accept_rate = 0.0f;
    uint32_t total_draft_tokens = 0;
    uint32_t total_accepted = 0;

    float cache_hit_rate = 0.0f;
    uint32_t prefetch_queue_depth = 0;
    uint64_t prefetch_hit_count = 0;
    uint64_t prefetch_miss_count = 0;

    float memory_pressure = 0.0f;
    size_t active_buffer_used = 0;
    size_t pinned_buffer_used = 0;

    float pipeline_efficiency = 0.0f;

    enum class Mode { Normal, Conservative, Emergency } mode = Mode::Normal;
};

using TokenCallback = std::function<void(const TokenResult&)>;

/// Unified Speculative-Prefetch Pipeline scheduler.
///
/// Orchestrates the USPP decode loop: draft speculation → route prediction →
/// asynchronous expert prefetch → buffer swap → target verification →
/// speculative sampling.  Does not perform inference directly; delegates
/// to MetalCompute, BufferManager, and ExpertIndex.
///
/// Phase 1: run_draft() and verify_with_target() are stubs.  The focus is on
/// correct pipeline orchestration, prefetch scheduling, and adaptive control.
class USPPScheduler {
public:
    struct Components {
        MetalCompute* gpu = nullptr;
        BufferManager* buffers = nullptr;
        ExpertIndex* expert_index = nullptr;
        MmapRegion* model_mmap = nullptr;
        CachePolicy* cache_policy = nullptr;
        TransformerModel* draft_model = nullptr;
        TransformerModel* target_model = nullptr;
    };

    static auto create(USPPConfig config, Components components)
        -> std::expected<std::unique_ptr<USPPScheduler>, std::string>;

    ~USPPScheduler();

    USPPScheduler(const USPPScheduler&) = delete;
    USPPScheduler& operator=(const USPPScheduler&) = delete;

    auto generate(const InferenceRequest& request, TokenCallback callback)
        -> std::expected<std::vector<TokenResult>, std::string>;

    void stop();

    auto metrics() const -> USPPMetrics;

    void reset();

private:
    USPPScheduler(USPPConfig config, Components components);

    // ===== Pipeline stages =====

    auto prefill(const std::vector<uint32_t>& tokens)
        -> std::expected<void, std::string>;

    struct DecodeResult {
        std::vector<TokenResult> accepted_tokens;
        float accept_rate;
        float cache_hit_rate;
        double gpu_time_ms;
        double io_time_ms;
    };
    auto decode_step() -> std::expected<DecodeResult, std::string>;

    struct DraftOutput {
        std::vector<uint32_t> candidate_tokens;
        std::vector<float> logprobs;
        std::vector<std::vector<float>> token_probs;  // K × vocab_size
        struct RouterPrediction {
            uint32_t layer;
            std::vector<uint32_t> predicted_experts;
            std::vector<float> expert_probs;
        };
        std::vector<std::vector<RouterPrediction>> route_predictions;
    };
    auto run_draft(uint32_t k) -> std::expected<DraftOutput, std::string>;

    auto schedule_prefetch(const DraftOutput& draft)
        -> std::expected<void, std::string>;

    struct VerifyOutput {
        std::vector<uint32_t> verified_tokens;
        std::vector<float> logprobs;
        std::vector<std::vector<float>> token_probs;  // (K+1) × vocab_size
        uint32_t n_accepted;
    };
    auto verify_with_target(const DraftOutput& draft)
        -> std::expected<VerifyOutput, std::string>;

    auto speculative_sample(const DraftOutput& draft,
                            const VerifyOutput& target)
        -> std::vector<TokenResult>;

    // ===== Adaptive control =====

    void adapt_k(float accept_rate);
    void check_and_switch_mode();

    // ===== I/O thread =====

    void io_worker();
    uint64_t submit_io(std::function<void()> task);
    void wait_io_complete();

    // ===== State =====

    USPPConfig config_;
    Components components_;

    std::vector<uint32_t> context_tokens_;
    uint32_t current_k_;
    USPPMetrics::Mode mode_ = USPPMetrics::Mode::Normal;

    uint32_t total_draft_ = 0;
    uint32_t total_accepted_ = 0;
    uint64_t total_tokens_generated_ = 0;
    std::chrono::steady_clock::time_point gen_start_;

    float accept_rate_ema_ = 0.5f;
    float cache_hit_rate_ema_ = 1.0f;
    float last_cache_hit_rate_ = 1.0f;
    uint32_t last_prefetch_depth_ = 0;
    static constexpr float kEmaAlpha = 0.3f;

    bool first_decode_step_{true};
    uint64_t prefetch_hits_{0};
    uint64_t prefetch_misses_{0};

    // I/O thread
    std::thread io_thread_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    std::condition_variable io_done_cv_;
    struct IOTask {
        std::function<void()> fn;
        uint64_t generation;
    };
    std::deque<IOTask> io_queue_;
    uint64_t io_next_gen_{1};
    uint64_t io_completed_gen_{0};
    std::atomic<bool> stop_requested_{false};

    // Metrics (read from any thread via metrics())
    mutable std::mutex metrics_mutex_;
    USPPMetrics metrics_{};

    // RNG for Phase 1 stubs
    std::mt19937 rng_;

    // Sampler for speculative decoding (Leviathan et al. 2023)
    std::unique_ptr<Sampler> sampler_;
    float temperature_{0.7f};
    float top_p_{0.9f};
};

}  // namespace mugen

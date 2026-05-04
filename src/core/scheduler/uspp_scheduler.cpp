#include "core/scheduler/uspp_scheduler.h"

#include "core/memory/kv_cache.h"
#include "core/model/transformer.h"
#include "core/scheduler/route_predictor.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace mugen {

// ===========================================================================
// Factory / lifecycle
// ===========================================================================

auto USPPScheduler::create(USPPConfig config, Components components)
    -> std::expected<std::unique_ptr<USPPScheduler>, std::string> {

    if (!components.buffers)
        return std::unexpected(std::string("BufferManager is required"));
    if (!components.expert_index)
        return std::unexpected(std::string("ExpertIndex is required"));
    if (!components.model_mmap)
        return std::unexpected(std::string("MmapRegion is required"));

    config.min_draft_k = std::max(config.min_draft_k, 1u);
    config.draft_k = std::clamp(config.draft_k,
                                config.min_draft_k, config.max_draft_k);

    return std::unique_ptr<USPPScheduler>(
        new USPPScheduler(std::move(config), components));
}

USPPScheduler::USPPScheduler(USPPConfig config, Components components)
    : config_(std::move(config))
    , components_(components)
    , current_k_(config_.draft_k)
    , rng_(std::random_device{}()) {
    io_thread_ = std::thread(&USPPScheduler::io_worker, this);
}

USPPScheduler::~USPPScheduler() {
    stop_requested_.store(true, std::memory_order_release);
    io_cv_.notify_all();
    io_done_cv_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();
}

// ===========================================================================
// Public API
// ===========================================================================

auto USPPScheduler::generate(const InferenceRequest& request,
                             TokenCallback callback)
    -> std::expected<std::vector<TokenResult>, std::string> {

    stop_requested_.store(false, std::memory_order_release);
    gen_start_ = std::chrono::steady_clock::now();
    first_decode_step_ = true;

    // Wire BufferManager into the target model so MoE forward can consume
    // experts that USPP has prefetched into staging/pinned buffers.
    if (components_.target_model && components_.buffers)
        components_.target_model->set_buffer_manager(components_.buffers);

    temperature_ = request.temperature;
    top_p_ = request.top_p;
    sampler_ = std::make_unique<Sampler>(Sampler::SamplingParams{
        .temperature = request.temperature,
        .top_p = request.top_p,
        .seed = 0,
    });

    auto pf = prefill(request.prompt_tokens);
    if (!pf) return std::unexpected(pf.error());

    std::vector<TokenResult> all_tokens;
    bool first_token_emitted = true;

    while (all_tokens.size() < request.max_new_tokens &&
           !stop_requested_.load(std::memory_order_acquire)) {

        auto step = decode_step();
        if (!step) return std::unexpected(step.error());

        for (auto& tok : step->accepted_tokens) {
            if (first_token_emitted) {
                auto ttft = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - gen_start_);
                std::lock_guard lk(metrics_mutex_);
                metrics_.time_to_first_token_ms = ttft.count();
                first_token_emitted = false;
            }

            context_tokens_.push_back(tok.token_id);
            ++total_tokens_generated_;

            if (callback) callback(tok);
            all_tokens.push_back(tok);

            if (all_tokens.size() >= request.max_new_tokens) break;
        }

        auto elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - gen_start_).count();
        if (elapsed > 0.0f) {
            std::lock_guard lk(metrics_mutex_);
            metrics_.tokens_per_second =
                static_cast<float>(all_tokens.size()) / elapsed;
        }
    }

    return all_tokens;
}

void USPPScheduler::stop() {
    stop_requested_.store(true, std::memory_order_release);
    io_done_cv_.notify_all();
}

auto USPPScheduler::metrics() const -> USPPMetrics {
    std::lock_guard lk(metrics_mutex_);
    return metrics_;
}

void USPPScheduler::reset() {
    wait_io_complete();

    context_tokens_.clear();
    current_k_ = config_.draft_k;
    mode_ = USPPMetrics::Mode::Normal;
    total_draft_ = 0;
    total_accepted_ = 0;
    total_tokens_generated_ = 0;
    accept_rate_ema_ = 0.5f;
    cache_hit_rate_ema_ = 1.0f;
    last_cache_hit_rate_ = 1.0f;
    last_prefetch_depth_ = 0;
    first_decode_step_ = true;
    prefetch_hits_ = 0;
    prefetch_misses_ = 0;
    stop_requested_.store(false, std::memory_order_relaxed);

    {
        std::lock_guard lk(io_mutex_);
        io_next_gen_ = 1;
        io_completed_gen_ = 0;
    }

    {
        std::lock_guard lk(metrics_mutex_);
        metrics_ = {};
    }
}

// ===========================================================================
// Pipeline: prefill
// ===========================================================================

auto USPPScheduler::prefill(const std::vector<uint32_t>& tokens)
    -> std::expected<void, std::string> {
    context_tokens_ = tokens;
    if (components_.target_model && !tokens.empty()) {
        auto result = components_.target_model->forward(tokens, 0);
        if (!result) return std::unexpected(result.error());
    }
    return {};
}

// ===========================================================================
// Pipeline: decode_step — the heart of USPP
// ===========================================================================

auto USPPScheduler::decode_step() -> std::expected<DecodeResult, std::string> {
    auto t0 = std::chrono::steady_clock::now();

    // 1. Draft model generates K candidate tokens + route predictions.
    auto draft_res = run_draft(current_k_);
    if (!draft_res) return std::unexpected(draft_res.error());
    auto& draft = *draft_res;

    // 2. Async prefetch: enqueue I/O for predicted experts (returns immediately).
    schedule_prefetch(draft);

    // 3. First decode step has no previous prefetch — sync fallback:
    //    wait for I/O + swap before GPU verify.
    bool is_first = first_decode_step_;
    if (is_first) {
        wait_io_complete();
        components_.buffers->swap_buffers();
        first_decode_step_ = false;
    }

    auto t_before_gpu = std::chrono::steady_clock::now();

    // 4. GPU verify — in pipelined mode (non-first), I/O runs in background.
    //    Experts come from active buffer (staged by previous round's prefetch).
    //    Misses fall back to mmap reads (Deliverable C / WP-3 resolve_expert_bufs).
    auto verify_res = verify_with_target(draft);
    if (!verify_res) return std::unexpected(verify_res.error());
    auto& verify = *verify_res;

    auto t_after_gpu = std::chrono::steady_clock::now();

    // 5. Pipelined steps: wait for this round's I/O, then swap for next round.
    if (!is_first) {
        wait_io_complete();
        components_.buffers->swap_buffers();
    }

    auto t_done = std::chrono::steady_clock::now();

    // 6. Speculative sampling: accept/reject based on draft vs target.
    auto accepted = speculative_sample(draft, verify);

    uint32_t n_draft_accepted = 0;
    for (auto& t : accepted)
        if (t.from_draft) ++n_draft_accepted;

    // 6b. Truncate KV caches: verify_with_target added K+1 entries,
    //     but only n_draft_accepted+1 are valid (re-eval + accepted drafts).
    uint32_t ctx_len = static_cast<uint32_t>(context_tokens_.size());
    uint32_t keep_len = ctx_len + n_draft_accepted + 1;

    if (components_.target_model) {
        auto* kv = components_.target_model->kv_cache();
        if (kv && kv->seq_len() > keep_len)
            kv->truncate(keep_len);
    }
    if (components_.draft_model) {
        auto* kv = components_.draft_model->kv_cache();
        uint32_t draft_keep = ctx_len + n_draft_accepted;
        if (kv && kv->seq_len() > draft_keep)
            kv->truncate(draft_keep);
    }

    // 7. Update statistics and adaptive control.

    float step_accept_rate = draft.candidate_tokens.empty()
        ? 0.0f
        : static_cast<float>(n_draft_accepted)
          / static_cast<float>(draft.candidate_tokens.size());

    total_draft_ += static_cast<uint32_t>(draft.candidate_tokens.size());
    total_accepted_ += n_draft_accepted;

    adapt_k(step_accept_rate);
    cache_hit_rate_ema_ = kEmaAlpha * last_cache_hit_rate_
                        + (1.0f - kEmaAlpha) * cache_hit_rate_ema_;
    check_and_switch_mode();

    double gpu_ms = std::chrono::duration<double, std::milli>(
                        t_after_gpu - t_before_gpu).count();
    double io_ms;
    if (is_first) {
        // Serial: I/O blocked before GPU.
        io_ms = std::chrono::duration<double, std::milli>(
                    t_before_gpu - t0).count();
    } else {
        // Pipelined: only count the non-overlapped I/O wait after GPU.
        io_ms = std::chrono::duration<double, std::milli>(
                    t_done - t_after_gpu).count();
    }

    double total_ms = gpu_ms + io_ms;
    float efficiency = (total_ms > 0.001)
        ? static_cast<float>(gpu_ms / total_ms)
        : 1.0f;

    {
        std::lock_guard lk(metrics_mutex_);
        metrics_.current_k = current_k_;
        metrics_.accept_rate = accept_rate_ema_;
        metrics_.cache_hit_rate = cache_hit_rate_ema_;
        metrics_.total_draft_tokens = total_draft_;
        metrics_.total_accepted = total_accepted_;
        metrics_.prefetch_queue_depth = last_prefetch_depth_;
        metrics_.prefetch_hit_count = prefetch_hits_;
        metrics_.prefetch_miss_count = prefetch_misses_;
        metrics_.pipeline_efficiency = efficiency;
        metrics_.mode = mode_;

        auto bs = components_.buffers->stats();
        metrics_.memory_pressure = bs.memory_pressure;
        metrics_.active_buffer_used = bs.active_used;
        metrics_.pinned_buffer_used = bs.pinned_used;
    }

    return DecodeResult{
        .accepted_tokens = std::move(accepted),
        .accept_rate = step_accept_rate,
        .cache_hit_rate = last_cache_hit_rate_,
        .gpu_time_ms = gpu_ms,
        .io_time_ms = io_ms,
    };
}

// ===========================================================================
// Pipeline: run_draft (STUB — Phase 1)
// ===========================================================================

auto USPPScheduler::run_draft(uint32_t k)
    -> std::expected<DraftOutput, std::string> {

    if (!components_.draft_model) {
        // Stub fallback: random candidate tokens and route predictions
        DraftOutput output;
        std::uniform_int_distribution<uint32_t> vocab(0, 31999);
        std::uniform_real_distribution<float> lp(-5.0f, -0.01f);

        output.candidate_tokens.resize(k);
        output.logprobs.resize(k);
        for (uint32_t i = 0; i < k; ++i) {
            output.candidate_tokens[i] = vocab(rng_);
            output.logprobs[i] = lp(rng_);
        }

        uint32_t n_layers = components_.expert_index->n_layers();
        uint32_t n_experts = components_.expert_index->n_experts_per_layer();

        output.route_predictions.resize(k);
        if (n_layers > 0 && n_experts > 0) {
            std::uniform_int_distribution<uint32_t> edist(0, n_experts - 1);
            std::uniform_real_distribution<float> pdist(0.1f, 0.9f);

            for (uint32_t t = 0; t < k; ++t) {
                output.route_predictions[t].resize(n_layers);
                for (uint32_t l = 0; l < n_layers; ++l) {
                    auto& rp = output.route_predictions[t][l];
                    rp.layer = l;

                    uint32_t e1 = edist(rng_);
                    uint32_t e2 = edist(rng_);
                    while (e2 == e1 && n_experts > 1) e2 = edist(rng_);

                    float p1 = pdist(rng_);
                    rp.predicted_experts = {e1, e2};
                    rp.expert_probs = {p1, 1.0f - p1};
                }
            }
        }

        return output;
    }

    // Real draft model inference
    DraftOutput output;
    uint32_t pos = static_cast<uint32_t>(context_tokens_.size());
    uint32_t last_token = context_tokens_.back();

    auto draft_cfg = components_.draft_model->config();
    std::vector<std::vector<DraftOutput::RouterPrediction>> per_step_routes;

    if (draft_cfg.is_moe) {
        components_.draft_model->set_route_callback(
            [&per_step_routes](uint32_t layer, const uint32_t* expert_indices,
                               const float* expert_weights, uint32_t top_k) {
                if (per_step_routes.empty()) return;
                DraftOutput::RouterPrediction rp;
                rp.layer = layer;
                rp.predicted_experts.assign(expert_indices,
                                            expert_indices + top_k);
                rp.expert_probs.assign(expert_weights,
                                       expert_weights + top_k);
                per_step_routes.back().push_back(std::move(rp));
            });
    }

    for (uint32_t i = 0; i < k; i++) {
        per_step_routes.emplace_back();

        auto logits = components_.draft_model->forward({last_token}, pos + i);
        if (!logits) {
            if (draft_cfg.is_moe)
                components_.draft_model->set_route_callback(nullptr);
            return std::unexpected(logits.error());
        }

        auto& l = *logits;
        Sampler::temperature_scale(l, temperature_);
        Sampler::top_p_filter(l, top_p_);
        Sampler::softmax(l);

        uint32_t token = sampler_->sample_token(l);
        float logprob = std::log(l[token] + 1e-10f);

        output.candidate_tokens.push_back(token);
        output.logprobs.push_back(logprob);
        output.token_probs.push_back(l);
        last_token = token;
    }

    if (draft_cfg.is_moe)
        components_.draft_model->set_route_callback(nullptr);

    output.route_predictions = std::move(per_step_routes);
    return output;
}

// ===========================================================================
// Pipeline: schedule_prefetch
// ===========================================================================

auto USPPScheduler::schedule_prefetch(const DraftOutput& draft)
    -> std::expected<void, std::string> {

    // Build RoutePredictor from draft/target model configs.
    // When models are absent (stub path), fall back to ExpertIndex dimensions.
    RoutePredictor::Config rp_cfg;
    if (components_.draft_model) {
        auto dc = components_.draft_model->config();
        rp_cfg.draft_n_layers = dc.n_layers;
        rp_cfg.draft_n_experts = dc.n_experts;
    } else {
        rp_cfg.draft_n_layers = components_.expert_index->n_layers();
        rp_cfg.draft_n_experts = components_.expert_index->n_experts_per_layer();
    }
    if (components_.target_model) {
        auto tc = components_.target_model->config();
        rp_cfg.target_n_layers = tc.n_layers;
        rp_cfg.target_n_experts = tc.n_experts;
    } else {
        rp_cfg.target_n_layers = components_.expert_index->n_layers();
        rp_cfg.target_n_experts = components_.expert_index->n_experts_per_layer();
    }

    RoutePredictor predictor(rp_cfg);

    // Convert DraftOutput::RouterPrediction → RoutePredictor::RouterPrediction.
    std::vector<std::vector<RoutePredictor::RouterPrediction>> rp_input;
    rp_input.reserve(draft.route_predictions.size());
    for (auto& per_token : draft.route_predictions) {
        auto& step = rp_input.emplace_back();
        step.reserve(per_token.size());
        for (auto& rp : per_token) {
            step.push_back({rp.layer, rp.predicted_experts, rp.expert_probs});
        }
    }

    auto predicted = predictor.predict(rp_input);

    // Partition into cache hits and misses, preserving priority order.
    std::vector<ExpertKey> missing;
    size_t hits = 0;
    for (auto& key : predicted) {
        if (components_.buffers->find_expert(key))
            ++hits;
        else
            missing.push_back(key);
    }

    last_cache_hit_rate_ = predicted.empty()
        ? 1.0f
        : static_cast<float>(hits) / static_cast<float>(predicted.size());
    last_prefetch_depth_ = static_cast<uint32_t>(missing.size());
    prefetch_hits_ += hits;
    prefetch_misses_ += missing.size();

    if (missing.empty()) return {};

    // Submit asynchronous I/O to the prefetch worker thread.
    submit_io([this, missing = std::move(missing)]() {
        for (auto& key : missing) {
            if (stop_requested_.load(std::memory_order_relaxed)) break;

            auto* loc = components_.expert_index->location(
                            key.layer_id, key.expert_id);
            if (!loc) continue;

            // Hint the kernel to fault the pages in from SSD.
            components_.buffers->prefetch_expert(
                *components_.model_mmap, *loc);
            // Copy into the staging buffer.
            components_.buffers->stage_expert(
                key, *components_.model_mmap, *loc);

            components_.expert_index->status_mut(
                key.layer_id, key.expert_id).state =
                    ExpertStatus::State::InMemory;
        }

        // Evict cold experts to relieve memory pressure.
        size_t n_evict = missing.size();
        std::vector<ExpertKey> cold;
        if (components_.cache_policy) {
            auto candidates = components_.cache_policy->eviction_candidates(
                *components_.expert_index, n_evict);
            cold.reserve(candidates.size());
            for (auto& c : candidates) cold.push_back(c.key);
        } else {
            cold = components_.expert_index->coldest_experts(n_evict);
        }

        for (auto& key : cold) {
            if (stop_requested_.load(std::memory_order_relaxed)) break;
            auto* loc = components_.expert_index->location(
                            key.layer_id, key.expert_id);
            if (!loc) continue;
            components_.buffers->evict_from_mmap(
                *components_.model_mmap, *loc);
        }

        if (components_.cache_policy) {
            std::unordered_map<uint32_t, std::vector<uint32_t>> by_layer;
            for (auto& key : missing)
                by_layer[key.layer_id].push_back(key.expert_id);
            for (auto& [layer, experts] : by_layer)
                components_.cache_policy->record_activation(layer, experts);
        }
    });

    return {};
}

// ===========================================================================
// Pipeline: verify_with_target (STUB — Phase 1)
// ===========================================================================

auto USPPScheduler::verify_with_target(const DraftOutput& draft)
    -> std::expected<VerifyOutput, std::string> {

    if (!components_.target_model) {
        // Stub fallback: random prefix acceptance
        uint32_t k = static_cast<uint32_t>(draft.candidate_tokens.size());
        std::uniform_int_distribution<uint32_t> accept_dist(0, k);
        std::uniform_int_distribution<uint32_t> vocab(0, 31999);
        std::uniform_real_distribution<float> lp(-5.0f, -0.01f);

        VerifyOutput output;
        output.n_accepted = accept_dist(rng_);

        output.verified_tokens.resize(k + 1);
        output.logprobs.resize(k + 1);

        for (uint32_t i = 0; i <= k; ++i) {
            if (i < output.n_accepted) {
                output.verified_tokens[i] = draft.candidate_tokens[i];
                output.logprobs[i] = draft.logprobs[i];
            } else {
                output.verified_tokens[i] = vocab(rng_);
                output.logprobs[i] = lp(rng_);
            }
        }

        return output;
    }

    // Parallel target verification: single batch prefill for all K+1 positions.
    // Tokens: [last_context_token, draft[0], ..., draft[K-1]]
    VerifyOutput output;
    uint32_t pos = static_cast<uint32_t>(context_tokens_.size());

    std::vector<uint32_t> verify_tokens;
    verify_tokens.reserve(draft.candidate_tokens.size() + 1);
    verify_tokens.push_back(context_tokens_.back());
    for (auto tok : draft.candidate_tokens)
        verify_tokens.push_back(tok);

    auto all_logits = components_.target_model->forward_prefill_all_logits(
        verify_tokens, pos);
    if (!all_logits) return std::unexpected(all_logits.error());

    for (auto& l : *all_logits) {
        Sampler::temperature_scale(l, temperature_);
        Sampler::top_p_filter(l, top_p_);
        Sampler::softmax(l);

        uint32_t best = static_cast<uint32_t>(
            std::distance(l.begin(), std::max_element(l.begin(), l.end())));
        float logprob = std::log(l[best] + 1e-10f);

        output.verified_tokens.push_back(best);
        output.logprobs.push_back(logprob);
        output.token_probs.push_back(std::move(l));
    }

    output.n_accepted = 0;
    return output;
}

// ===========================================================================
// Pipeline: speculative_sample
// ===========================================================================

auto USPPScheduler::speculative_sample(const DraftOutput& draft,
                                       const VerifyOutput& target)
    -> std::vector<TokenResult> {

    if (sampler_ && !draft.token_probs.empty() && !target.token_probs.empty()) {
        auto spec_result = sampler_->speculative_sample(
            draft.candidate_tokens,
            draft.token_probs,
            target.token_probs
        );

        std::vector<TokenResult> results;
        results.reserve(spec_result.n_accepted + 1);

        for (uint32_t i = 0; i < spec_result.n_accepted; i++) {
            results.push_back({
                .token_id = spec_result.accepted_tokens[i],
                .logprob = spec_result.accepted_logprobs[i],
                .from_draft = true,
            });
        }

        results.push_back({
            .token_id = spec_result.bonus_token,
            .logprob = spec_result.bonus_logprob,
            .from_draft = false,
        });

        return results;
    }

    // Fallback: greedy prefix acceptance (stub path or no Sampler)
    uint32_t n = std::min(
        target.n_accepted,
        static_cast<uint32_t>(draft.candidate_tokens.size()));

    std::vector<TokenResult> results;
    results.reserve(n + 1);

    for (uint32_t i = 0; i < n; ++i) {
        results.push_back({
            .token_id = draft.candidate_tokens[i],
            .logprob  = (i < target.logprobs.size()) ? target.logprobs[i] : 0.0f,
            .from_draft = true,
        });
    }

    if (n < target.verified_tokens.size()) {
        results.push_back({
            .token_id = target.verified_tokens[n],
            .logprob  = (n < target.logprobs.size()) ? target.logprobs[n] : 0.0f,
            .from_draft = false,
        });
    }

    return results;
}

// ===========================================================================
// Adaptive control
// ===========================================================================

void USPPScheduler::adapt_k(float accept_rate) {
    accept_rate_ema_ = kEmaAlpha * accept_rate
                     + (1.0f - kEmaAlpha) * accept_rate_ema_;

    if (mode_ == USPPMetrics::Mode::Emergency) {
        current_k_ = config_.min_draft_k;
        return;
    }

    if (total_tokens_generated_ < config_.warmup_tokens)
        return;

    if (accept_rate_ema_ > config_.accept_threshold) {
        current_k_ = std::min(current_k_ + 1, config_.max_draft_k);
    } else if (accept_rate_ema_ < config_.accept_threshold * 0.5f) {
        current_k_ = (current_k_ > 2)
            ? std::max(current_k_ - 2, config_.min_draft_k)
            : config_.min_draft_k;
    } else if (accept_rate_ema_ < config_.accept_threshold) {
        current_k_ = (current_k_ > 1)
            ? std::max(current_k_ - 1, config_.min_draft_k)
            : config_.min_draft_k;
    }
}

void USPPScheduler::check_and_switch_mode() {
    float pressure = components_.buffers->memory_pressure();

    if (pressure > 0.9f) {
        mode_ = USPPMetrics::Mode::Emergency;
        current_k_ = config_.min_draft_k;
    } else if (cache_hit_rate_ema_ < config_.conservative_hit_threshold) {
        mode_ = USPPMetrics::Mode::Conservative;
    } else {
        mode_ = USPPMetrics::Mode::Normal;
    }
}

// ===========================================================================
// I/O thread
// ===========================================================================

void USPPScheduler::io_worker() {
    while (true) {
        std::function<void()> task;
        uint64_t gen = 0;
        {
            std::unique_lock lk(io_mutex_);
            io_cv_.wait(lk, [this] {
                return !io_queue_.empty() ||
                       stop_requested_.load(std::memory_order_acquire);
            });

            if (io_queue_.empty()) break;

            task = std::move(io_queue_.front().fn);
            gen = io_queue_.front().generation;
            io_queue_.pop_front();
        }

        task();

        {
            std::lock_guard lk(io_mutex_);
            io_completed_gen_ = gen;
        }
        io_done_cv_.notify_all();
    }

    // Mark all generations done so waiters unblock on shutdown.
    {
        std::lock_guard lk(io_mutex_);
        io_completed_gen_ = io_next_gen_;
    }
    io_done_cv_.notify_all();
}

uint64_t USPPScheduler::submit_io(std::function<void()> task) {
    uint64_t gen;
    {
        std::lock_guard lk(io_mutex_);
        gen = io_next_gen_++;
        io_queue_.push_back({std::move(task), gen});
    }
    io_cv_.notify_one();
    return gen;
}

void USPPScheduler::wait_io_complete() {
    std::unique_lock lk(io_mutex_);
    io_done_cv_.wait(lk, [this] {
        return io_completed_gen_ >= io_next_gen_ - 1 ||
               stop_requested_.load(std::memory_order_relaxed);
    });
}

}  // namespace mugen

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "mugen/mugen.h"
#include "cli/doctor.h"
#include "core/compute/metal_compute.h"
#include "core/memory/kv_cache.h"
#include "core/memory/mmap_loader.h"
#include "core/model/transformer.h"
#include "model/gguf_parser.h"
#include "model/ggml_types.h"
#include "core/scheduler/sampling.h"
#include "model/tokenizer.h"
#include "server/http_server.h"
#include "server/api_routes.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto models_dir() -> fs::path {
    const char* env = std::getenv("MUGEN_MODEL_DIR");
    if (env && env[0] != '\0') return fs::path(env);
    const char* home = std::getenv("HOME");
    if (!home) home = "/tmp";
    return fs::path(home) / ".mugen" / "models";
}

static auto format_bytes(uint64_t bytes) -> std::string {
    char buf[64];
    if (bytes >= (1ULL << 30))
        std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / (1ULL << 30));
    else if (bytes >= (1ULL << 20))
        std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1ULL << 20));
    else
        std::snprintf(buf, sizeof(buf), "%llu bytes", static_cast<unsigned long long>(bytes));
    return buf;
}

// ---------------------------------------------------------------------------
// Help / Version
// ---------------------------------------------------------------------------

static void print_version() {
    std::printf("mugen %s\n", mugen::kVersion);
}

static void print_help() {
    std::printf(
        "mugen %s -- MoE inference engine for Apple Silicon\n"
        "\n"
        "Usage:\n"
        "  mugen <command> [arguments] [options]\n"
        "\n"
        "Commands:\n"
        "  doctor                     Environment diagnostics\n"
        "  list                       List downloaded models\n"
        "  info  <model>              Show model metadata\n"
        "  chat  <model> [options]    Interactive chat session\n"
        "  bench <model> [options]    Run inference benchmark\n"
        "  serve <model> [options]    Start API server\n"
        "  pull  <url>                Download a model\n"
        "  rm    <model>              Remove a model\n"
        "\n"
        "Options:\n"
        "  --version                  Print version and exit\n"
        "  --help                     Print this help message\n"
        "\n"
        "Chat options:\n"
        "  --verbose                  Enable verbose output\n"
        "  --max-tokens N             Maximum tokens to generate (default: 512)\n"
        "  --temperature T            Sampling temperature (default: 0.7, 0=greedy)\n"
        "  --top-p P                  Nucleus sampling threshold (default: 0.9)\n"
        "  --draft-model <path>       Enable speculative decoding with draft model\n"
        "  --draft-k N                Draft tokens per step (default: 6)\n"
        "\n"
        "Bench options:\n"
        "  --compare <engine>         Compare against another engine (e.g. llama.cpp)\n"
        "\n"
        "Serve options:\n"
        "  --port N                   Port number (default: 8080)\n"
        "\n"
        "Environment:\n"
        "  MUGEN_MODEL_DIR            Override model directory (default: ~/.mugen/models/)\n"
        "\n"
        "Examples:\n"
        "  mugen doctor\n"
        "  mugen list\n"
        "  mugen chat deepseek-v3-q4k --max-tokens 1024\n"
        "  mugen bench deepseek-v3-q4k --compare llama.cpp\n"
        "  mugen serve deepseek-v3-q4k --port 9090\n",
        mugen::kVersion
    );
}

// ---------------------------------------------------------------------------
// Command: doctor
// ---------------------------------------------------------------------------

static int cmd_doctor() {
    auto info = mugen::gather_system_info();
    mugen::print_doctor_report(info);
    return 0;
}

// ---------------------------------------------------------------------------
// Command: list
// ---------------------------------------------------------------------------

static int cmd_list() {
    const auto dir = models_dir();

    if (!fs::exists(dir)) {
        std::printf("Model directory does not exist: %s\n", dir.c_str());
        std::printf("Run 'mugen pull <url>' to download a model.\n");
        return 0;
    }

    struct ModelEntry {
        std::string name;
        uint64_t    size;
    };

    std::vector<ModelEntry> entries;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".gguf") continue;
        entries.push_back({entry.path().stem().string(),
                           static_cast<uint64_t>(entry.file_size())});
    }

    if (entries.empty()) {
        std::printf("No models found in %s\n", dir.c_str());
        std::printf("Run 'mugen pull <url>' to download a model.\n");
        return 0;
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.name < b.name; });

    std::printf("%-40s  %s\n", "MODEL", "SIZE");
    std::printf("%-40s  %s\n",
                "----------------------------------------",
                "----------");

    for (const auto& e : entries) {
        std::printf("%-40s  %s\n", e.name.c_str(), format_bytes(e.size).c_str());
    }

    std::printf("\n%zu model(s) in %s\n",
                entries.size(), dir.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Command: info <model>
// ---------------------------------------------------------------------------

static auto resolve_model_path(const std::string& name) -> fs::path {
    // If the argument is an absolute or relative path that exists, use it directly
    fs::path direct(name);
    if (fs::exists(direct)) return direct;

    // Otherwise, look in the models directory
    auto dir = models_dir();
    fs::path candidate = dir / name;
    if (fs::exists(candidate)) return candidate;

    // Try appending .gguf
    candidate = dir / (name + ".gguf");
    if (fs::exists(candidate)) return candidate;

    return {};
}

static int cmd_info(const std::string& model) {
    auto path = resolve_model_path(model);
    if (path.empty()) {
        std::fprintf(stderr, "Error: model '%s' not found.\n", model.c_str());
        std::fprintf(stderr, "  Searched: %s\n", models_dir().c_str());
        std::fprintf(stderr, "  Tip: run 'mugen list' to see available models.\n");
        return 1;
    }

    auto result = mugen::GGUFParser::parse(path);
    if (!result) {
        std::fprintf(stderr, "Error: failed to parse '%s': %s\n",
                     path.c_str(), result.error().c_str());
        return 1;
    }

    const auto& meta = result->metadata();
    const auto& tensors = result->tensors();

    constexpr int kW = 22;

    std::printf("\n");
    std::printf("  %-*s  %s\n", kW, "File", path.c_str());
    std::printf("  %-*s  %s\n", kW, "File size", format_bytes(result->file_size()).c_str());
    std::printf("  %-*s  %u\n", kW, "GGUF version", meta.version);

    if (!meta.name.empty())
        std::printf("  %-*s  %s\n", kW, "Model name", meta.name.c_str());
    if (!meta.arch.empty())
        std::printf("  %-*s  %s\n", kW, "Architecture", meta.arch.c_str());

    if (meta.vocab_size > 0)
        std::printf("  %-*s  %u\n", kW, "Vocab size", meta.vocab_size);
    if (meta.context_length > 0)
        std::printf("  %-*s  %u\n", kW, "Context length", meta.context_length);
    if (meta.embedding_dim > 0)
        std::printf("  %-*s  %u\n", kW, "Embedding dim", meta.embedding_dim);
    if (meta.n_layers > 0)
        std::printf("  %-*s  %u\n", kW, "Layers", meta.n_layers);
    if (meta.n_heads > 0)
        std::printf("  %-*s  %u\n", kW, "Attention heads", meta.n_heads);
    if (meta.n_kv_heads > 0)
        std::printf("  %-*s  %u\n", kW, "KV heads", meta.n_kv_heads);

    if (result->is_moe()) {
        std::printf("  %-*s  %u (active: %u)\n", kW, "Experts",
                    meta.n_experts, meta.n_experts_used);
    }

    auto get_int = [&](const std::string& key) -> int64_t {
        auto it = meta.raw_int_kv.find(meta.arch + "." + key);
        return it != meta.raw_int_kv.end() ? it->second : 0;
    };
    auto get_float = [&](const std::string& key) -> double {
        auto it = meta.raw_float_kv.find(meta.arch + "." + key);
        return it != meta.raw_float_kv.end() ? it->second : 0.0;
    };

    int64_t kv_lora_rank = get_int("attention.kv_lora_rank");
    if (kv_lora_rank > 0) {
        std::printf("\n  --- MLA ---\n");
        std::printf("  %-*s  %lld\n", kW, "KV LoRA rank", kv_lora_rank);
        if (auto v = get_int("attention.q_lora_rank"); v > 0)
            std::printf("  %-*s  %lld\n", kW, "Q LoRA rank", v);
        if (auto v = get_int("rope.dimension_count"); v > 0)
            std::printf("  %-*s  %lld\n", kW, "RoPE dims", v);
        if (auto v = get_int("attention.key_length"); v > 0)
            std::printf("  %-*s  %lld\n", kW, "Key length", v);
        if (auto v = get_int("attention.value_length"); v > 0)
            std::printf("  %-*s  %lld\n", kW, "Value length", v);
        if (auto v = get_int("expert_shared_count"); v > 0)
            std::printf("  %-*s  %lld\n", kW, "Shared experts", v);
        if (auto v = get_float("rope.scaling.factor"); v > 0)
            std::printf("  %-*s  %.1f\n", kW, "YaRN factor", v);
    }

    std::printf("  %-*s  %zu\n", kW, "Tensors", tensors.size());
    std::printf("  %-*s  0x%llx\n", kW, "Data offset",
                static_cast<unsigned long long>(result->data_offset()));

    std::printf("\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Helpers: load model + tokenizer into a reusable bundle
// ---------------------------------------------------------------------------

struct ModelBundle {
    std::unique_ptr<mugen::MetalCompute> gpu;
    std::vector<mugen::MmapRegion> mmaps;
    std::unique_ptr<mugen::TransformerModel> model;
    std::optional<mugen::Tokenizer> tokenizer;
    std::unique_ptr<mugen::GGUFParser> parser;
};

static auto load_model_bundle(const fs::path& path) -> std::expected<ModelBundle, std::string> {
    ModelBundle b;

    auto shard_paths = mugen::GGUFParser::detect_shards(path);
    if (shard_paths.size() > 1)
        std::printf("  Split GGUF: %zu shards detected\n", shard_paths.size());

    auto parser_result = (shard_paths.size() > 1)
        ? mugen::GGUFParser::parse_sharded(shard_paths)
        : mugen::GGUFParser::parse(path);
    if (!parser_result)
        return std::unexpected("GGUF parse error: " + parser_result.error());
    b.parser = std::make_unique<mugen::GGUFParser>(std::move(*parser_result));

    auto gpu_result = mugen::MetalCompute::create();
    if (!gpu_result)
        return std::unexpected("Metal device error: " + gpu_result.error());
    b.gpu = std::move(*gpu_result);

    for (auto& sp : shard_paths) {
        auto mr = mugen::MmapLoader::map_file(sp);
        if (!mr) return std::unexpected("mmap error: " + mr.error());
        b.mmaps.push_back(std::move(*mr));
    }

    auto model_result = (b.mmaps.size() == 1)
        ? mugen::TransformerModel::from_gguf(b.gpu.get(), *b.parser, b.mmaps[0])
        : mugen::TransformerModel::from_gguf(b.gpu.get(), *b.parser, b.mmaps);
    if (!model_result)
        return std::unexpected("Model build error: " + model_result.error());
    b.model = std::move(*model_result);

    auto tok_result = mugen::Tokenizer::from_gguf(b.parser->metadata());
    if (!tok_result)
        return std::unexpected("Tokenizer error: " + tok_result.error());
    b.tokenizer.emplace(std::move(*tok_result));

    return std::move(b);
}

// ---------------------------------------------------------------------------
// Helpers: chat template + prompt building + stop token (shared by all paths)
// ---------------------------------------------------------------------------

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatTemplate {
    uint32_t im_start_id = UINT32_MAX;
    uint32_t im_end_id   = UINT32_MAX;
    uint32_t header_start_id = UINT32_MAX;
    uint32_t header_end_id   = UINT32_MAX;
    uint32_t eot_id          = UINT32_MAX;
    bool has_chatml = false;
    bool has_llama3 = false;
};

static auto detect_chat_template(const mugen::Tokenizer& tokenizer) -> ChatTemplate {
    ChatTemplate ct;
    auto find_token = [&](const std::string& text) -> uint32_t {
        for (uint32_t i = 0; i < tokenizer.vocab_size(); i++) {
            if (tokenizer.token_to_text(i) == text) return i;
        }
        return UINT32_MAX;
    };

    ct.im_start_id = find_token("<|im_start|>");
    ct.im_end_id   = find_token("<|im_end|>");
    ct.has_chatml   = (ct.im_start_id != UINT32_MAX && ct.im_end_id != UINT32_MAX);

    ct.header_start_id = find_token("<|start_header_id|>");
    ct.header_end_id   = find_token("<|end_header_id|>");
    ct.eot_id          = find_token("<|eot_id|>");
    ct.has_llama3 = (ct.header_start_id != UINT32_MAX &&
                     ct.header_end_id != UINT32_MAX &&
                     ct.eot_id != UINT32_MAX);
    return ct;
}

/// Build the full token sequence for a multi-turn conversation.
/// Includes all history messages + assistant turn prefix.
static auto build_chat_tokens(const std::vector<ChatMessage>& messages,
                              const mugen::Tokenizer& tokenizer,
                              const ChatTemplate& ct) -> std::vector<uint32_t>
{
    std::vector<uint32_t> toks;
    if (ct.has_chatml) {
        auto nl_tokens = tokenizer.encode("\n");
        for (auto& m : messages) {
            toks.push_back(ct.im_start_id);
            auto content = tokenizer.encode(m.role + "\n" + m.content);
            toks.insert(toks.end(), content.begin(), content.end());
            toks.push_back(ct.im_end_id);
            toks.insert(toks.end(), nl_tokens.begin(), nl_tokens.end());
        }
        toks.push_back(ct.im_start_id);
        auto asst = tokenizer.encode("assistant\n");
        toks.insert(toks.end(), asst.begin(), asst.end());
    } else if (ct.has_llama3) {
        auto nl2 = tokenizer.encode("\n\n");
        auto asst_label = tokenizer.encode("assistant");
        toks.push_back(tokenizer.bos_token());
        for (auto& m : messages) {
            toks.push_back(ct.header_start_id);
            auto rl = tokenizer.encode(m.role);
            toks.insert(toks.end(), rl.begin(), rl.end());
            toks.push_back(ct.header_end_id);
            toks.insert(toks.end(), nl2.begin(), nl2.end());
            auto content = tokenizer.encode(m.content);
            toks.insert(toks.end(), content.begin(), content.end());
            toks.push_back(ct.eot_id);
        }
        toks.push_back(ct.header_start_id);
        toks.insert(toks.end(), asst_label.begin(), asst_label.end());
        toks.push_back(ct.header_end_id);
        toks.insert(toks.end(), nl2.begin(), nl2.end());
    } else {
        // DeepSeek V2 style: "User: {msg}\n\nAssistant: {msg}<eos>"
        toks.push_back(tokenizer.bos_token());
        for (auto& m : messages) {
            std::string prefix = (m.role == "user") ? "User: " : "Assistant: ";
            auto content = tokenizer.encode(prefix + m.content);
            toks.insert(toks.end(), content.begin(), content.end());
            if (m.role == "assistant") {
                toks.push_back(tokenizer.eos_token());
            }
            auto nl2 = tokenizer.encode("\n\n");
            toks.insert(toks.end(), nl2.begin(), nl2.end());
        }
        auto asst = tokenizer.encode("Assistant:");
        toks.insert(toks.end(), asst.begin(), asst.end());
    }
    return toks;
}

static bool is_stop_token(uint32_t token, const mugen::Tokenizer& tokenizer,
                          const ChatTemplate& ct)
{
    if (token == tokenizer.eos_token()) return true;
    if (ct.has_chatml && token == ct.im_end_id) return true;
    if (ct.has_llama3 && token == ct.eot_id) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Speculative decoding chat (--draft-model)
// ---------------------------------------------------------------------------

static int cmd_speculative_chat(const std::string& target_path_str,
                                const std::string& draft_path_str,
                                bool verbose, int max_tokens,
                                double temperature, double top_p,
                                int draft_k)
{
    auto target_path = resolve_model_path(target_path_str);
    auto draft_path  = resolve_model_path(draft_path_str);
    if (target_path.empty()) {
        std::fprintf(stderr, "Error: target model '%s' not found.\n", target_path_str.c_str());
        return 1;
    }
    if (draft_path.empty()) {
        std::fprintf(stderr, "Error: draft model '%s' not found.\n", draft_path_str.c_str());
        return 1;
    }

    std::printf("Loading target model: %s\n", target_path.c_str());
    auto target_bundle = load_model_bundle(target_path);
    if (!target_bundle) {
        std::fprintf(stderr, "Error loading target: %s\n", target_bundle.error().c_str());
        return 1;
    }

    std::printf("Loading draft model: %s\n", draft_path.c_str());
    auto draft_bundle = load_model_bundle(draft_path);
    if (!draft_bundle) {
        std::fprintf(stderr, "Error loading draft: %s\n", draft_bundle.error().c_str());
        return 1;
    }

    auto& target = target_bundle->model;
    auto& draft  = draft_bundle->model;
    auto& tokenizer = *target_bundle->tokenizer;

    const auto& tcfg = target->config();
    const auto& dcfg = draft->config();

    std::printf("\nTarget: %s — %u layers, embed %u, vocab %u\n",
                target_bundle->parser->metadata().arch.c_str(),
                tcfg.n_layers, tcfg.embed_dim, tcfg.vocab_size);
    std::printf("Draft:  %s — %u layers, embed %u, vocab %u\n",
                draft_bundle->parser->metadata().arch.c_str(),
                dcfg.n_layers, dcfg.embed_dim, dcfg.vocab_size);
    std::printf("Draft K: %d, Temperature: %.2f, Top-p: %.2f\n",
                draft_k, temperature, top_p);

    uint32_t valid_vocab = std::min(tcfg.vocab_size, dcfg.vocab_size);
    uint32_t common_vocab = std::max(tcfg.vocab_size, dcfg.vocab_size);
    if (tcfg.vocab_size != dcfg.vocab_size) {
        std::printf("  Note: vocab size mismatch (target=%u, draft=%u). "
                    "Padding distributions to max.\n", tcfg.vocab_size, dcfg.vocab_size);
    }

    auto ct = detect_chat_template(tokenizer);
    mugen::Sampler::SamplingParams sp;
    sp.temperature = static_cast<float>(temperature);
    sp.top_p       = static_cast<float>(top_p);
    mugen::Sampler sampler(sp);

    std::printf("\nReady for speculative decoding. Type your message (Ctrl-D to exit).\n\n");

    std::vector<ChatMessage> history;
    std::vector<uint32_t> prev_prompt_tokens;
    uint32_t prev_target_kv = 0;
    uint32_t prev_draft_kv = 0;

    auto pad_probs = [common_vocab](std::vector<float>& probs) {
        if (probs.size() < common_vocab) probs.resize(common_vocab, 0.0f);
    };
    auto apply_sampling = [&sp](std::vector<float>& logits) {
        mugen::Sampler::temperature_scale(logits, sp.temperature);
        mugen::Sampler::top_p_filter(logits, sp.top_p);
        mugen::Sampler::softmax(logits);
    };

    std::string line;
    while (true) {
        std::printf(">>> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        history.push_back({"user", line});
        auto token_ids = build_chat_tokens(history, tokenizer, ct);

        // KV cache reuse for both models
        uint32_t common_len = 0;
        {
            uint32_t max_common = std::min(
                static_cast<uint32_t>(prev_prompt_tokens.size()),
                static_cast<uint32_t>(token_ids.size()));
            while (common_len < max_common &&
                   prev_prompt_tokens[common_len] == token_ids[common_len])
                ++common_len;

            bool reuse_target = (common_len > 0 && common_len <= prev_target_kv);
            bool reuse_draft  = (common_len > 0 && common_len <= prev_draft_kv);

            if (reuse_target) target->kv_cache()->truncate(common_len);
            else { target->kv_cache()->clear(); common_len = 0; }
            if (reuse_draft) draft->kv_cache()->truncate(common_len);
            else draft->kv_cache()->clear();
        }

        std::vector<uint32_t> new_tokens(
            token_ids.begin() + common_len, token_ids.end());

        if (verbose) {
            std::printf("[prompt: %zu, reused: %u, new: %zu]\n",
                        token_ids.size(), common_len, new_tokens.size());
        }

        if (new_tokens.empty()) {
            history.pop_back();
            continue;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        auto target_logits = target->forward(new_tokens, common_len);
        if (!target_logits) {
            std::fprintf(stderr, "\nTarget prefill error: %s\n", target_logits.error().c_str());
            history.pop_back();
            continue;
        }
        auto draft_logits = draft->forward(new_tokens, common_len);
        if (!draft_logits) {
            std::fprintf(stderr, "\nDraft prefill error: %s\n", draft_logits.error().c_str());
            history.pop_back();
            continue;
        }

        double prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();

        std::vector<float> saved_target_logits = std::move(*target_logits);
        prev_prompt_tokens = token_ids;
        uint32_t pos = static_cast<uint32_t>(token_ids.size());
        uint32_t total_generated = 0, total_draft_tokens = 0;
        uint32_t total_accepted = 0, total_steps = 0;
        bool first_token = true;
        double ttft_ms = 0.0;
        bool stop = false;
        std::string response_text;

        auto t_decode_start = std::chrono::high_resolution_clock::now();

        while (total_generated < static_cast<uint32_t>(max_tokens) && !stop) {
            uint32_t K = static_cast<uint32_t>(draft_k);

            // Step 1: Draft K tokens
            std::vector<uint32_t> draft_tokens;
            std::vector<std::vector<float>> draft_probs;
            draft_tokens.reserve(K);
            draft_probs.reserve(K);

            {
                auto& d_logits = *draft_logits;
                for (uint32_t i = 0; i < K; i++) {
                    std::vector<float> lc;
                    if (i == 0) {
                        lc = d_logits;
                    } else {
                        auto r = draft->forward({draft_tokens.back()}, pos + i - 1);
                        if (!r) break;
                        lc = std::move(*r);
                    }
                    apply_sampling(lc);
                    uint32_t token = sampler.sample_token(lc);
                    draft_tokens.push_back(token);
                    pad_probs(lc);
                    draft_probs.push_back(std::move(lc));
                }
            }

            if (draft_tokens.empty()) break;
            K = static_cast<uint32_t>(draft_tokens.size());
            total_draft_tokens += K;

            // Step 2: Target verify
            auto verify_logits = target->forward_prefill_all_logits(draft_tokens, pos);
            if (!verify_logits) {
                std::fprintf(stderr, "\nTarget verify error: %s\n", verify_logits.error().c_str());
                break;
            }

            std::vector<std::vector<float>> target_probs;
            target_probs.reserve(K + 1);

            { auto fp = saved_target_logits; apply_sampling(fp); pad_probs(fp); target_probs.push_back(std::move(fp)); }
            for (auto& vl : *verify_logits) { apply_sampling(vl); pad_probs(vl); target_probs.push_back(std::move(vl)); }

            // Step 3: Speculative sample
            auto spec = sampler.speculative_sample(draft_tokens, draft_probs, target_probs);
            total_accepted += spec.n_accepted;
            ++total_steps;

            // Step 4: Emit tokens, record timing
            auto emit = [&](uint32_t tok) -> bool {
                if (is_stop_token(tok, tokenizer, ct)) { stop = true; return false; }
                if (tok >= valid_vocab) { stop = true; return false; }
                auto text = tokenizer.decode({tok});
                std::printf("%s", text.c_str());
                std::fflush(stdout);
                response_text += text;
                ++total_generated;
                if (first_token) {
                    ttft_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_start).count();
                    first_token = false;
                }
                return total_generated < static_cast<uint32_t>(max_tokens);
            };

            for (uint32_t i = 0; i < spec.n_accepted && !stop; i++)
                if (!emit(spec.accepted_tokens[i])) break;
            if (!stop) emit(spec.bonus_token);
            if (stop) break;

            // Step 5: KV rollback
            {
                uint32_t n_acc = spec.n_accepted;
                uint32_t keep = pos + n_acc;
                if (target->kv_cache()->seq_len() > keep) target->kv_cache()->truncate(keep);
                if (draft->kv_cache()->seq_len() > keep)  draft->kv_cache()->truncate(keep);
            }

            // Step 6: Forward bonus through both models
            {
                uint32_t n_acc = spec.n_accepted;
                uint32_t sync_pos = pos + n_acc;

                if (n_acc == K && draft->kv_cache()->seq_len() < sync_pos) {
                    auto gap = draft->forward({draft_tokens[K - 1]}, draft->kv_cache()->seq_len());
                    if (!gap) { std::fprintf(stderr, "\nDraft gap fill error\n"); break; }
                }

                auto ts = target->forward({spec.bonus_token}, sync_pos);
                if (!ts) { std::fprintf(stderr, "\nTarget sync error\n"); break; }
                saved_target_logits = std::move(*ts);

                draft_logits = draft->forward({spec.bonus_token}, sync_pos);
                if (!draft_logits) { std::fprintf(stderr, "\nDraft sync error\n"); break; }

                pos = sync_pos + 1;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_decode_start).count();
        double tps = (total_generated > 0 && decode_ms > 0)
            ? total_generated / (decode_ms / 1000.0) : 0.0;
        float accept_rate = (total_draft_tokens > 0)
            ? static_cast<float>(total_accepted) / static_cast<float>(total_draft_tokens)
            : 0.0f;
        float avg_accepted = (total_steps > 0)
            ? static_cast<float>(total_accepted + total_steps) / static_cast<float>(total_steps)
            : 0.0f;

        std::printf("\n\n--- Speculative Decoding Stats ---\n");
        std::printf("  Generated:       %u tokens\n", total_generated);
        std::printf("  Draft steps:     %u (K=%d)\n", total_steps, draft_k);
        std::printf("  Accepted:        %u / %u (%.1f%%)\n", total_accepted, total_draft_tokens, accept_rate * 100.0f);
        std::printf("  Avg tok/step:    %.1f\n", avg_accepted);
        std::printf("  Prefill:         %.1f ms\n", prefill_ms);
        std::printf("  TTFT:            %.1f ms\n", ttft_ms);
        std::printf("  Decode:          %.1f tok/s\n", tps);
        std::printf("  Total:           %.1f ms\n", total_ms);
        std::printf("----------------------------------\n\n");

        prev_target_kv = pos;
        prev_draft_kv = pos;
        history.push_back({"assistant", response_text});
    }

    std::printf("\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Command: chat <model>
// ---------------------------------------------------------------------------

static int cmd_chat(const std::string& model_name, bool verbose,
                    int max_tokens, double temperature, double top_p)
{
    auto path = resolve_model_path(model_name);
    if (path.empty()) {
        std::fprintf(stderr, "Error: model '%s' not found.\n", model_name.c_str());
        std::fprintf(stderr, "  Tip: run 'mugen list' to see available models.\n");
        return 1;
    }

    std::printf("Loading model: %s\n", path.c_str());
    auto bundle = load_model_bundle(path);
    if (!bundle) {
        std::fprintf(stderr, "Error: %s\n", bundle.error().c_str());
        return 1;
    }

    auto& transformer = bundle->model;
    auto& tokenizer = *bundle->tokenizer;
    const auto& cfg = transformer->config();
    const auto& meta = bundle->parser->metadata();

    std::printf("Model: %s (%s)\n",
                meta.name.empty() ? path.stem().c_str() : meta.name.c_str(),
                meta.arch.c_str());
    std::printf("  Layers: %u, Heads: %u, Embed: %u, Vocab: %u\n",
                cfg.n_layers, cfg.n_heads, cfg.embed_dim, cfg.vocab_size);
    if (cfg.is_moe)
        std::printf("  MoE: %u experts (%u active)\n", cfg.n_experts, cfg.n_experts_used);
    std::printf("  GPU: %s\n", bundle->gpu->device_name().c_str());

    if (verbose) {
        std::printf("  FFN dim: %u, Head dim: %u, KV heads: %u\n",
                    cfg.ffn_dim, cfg.head_dim, cfg.n_kv_heads);
        std::printf("  RoPE theta: %.1f, RMS eps: %e\n", cfg.rope_theta, cfg.rms_norm_eps);
        std::printf("  Max tokens: %d, Temperature: %.2f, Top-p: %.2f\n",
                    max_tokens, temperature, top_p);
    }

    mugen::Sampler::SamplingParams sp;
    sp.temperature = static_cast<float>(temperature);
    sp.top_p       = static_cast<float>(top_p);
    mugen::Sampler sampler(sp);

    auto ct = detect_chat_template(tokenizer);
    if (verbose) {
        if (ct.has_chatml) std::printf("  ChatML detected\n");
        if (ct.has_llama3) std::printf("  Llama 3 template detected\n");
    }

    std::printf("\nReady. Type your message (Ctrl-D to exit).\n\n");

    std::vector<ChatMessage> history;
    std::vector<uint32_t> prev_prompt_tokens;
    uint32_t prev_kv_len = 0;

    std::string line;
    while (true) {
        std::printf(">>> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        history.push_back({"user", line});
        auto token_ids = build_chat_tokens(history, tokenizer, ct);

        uint32_t common_len = 0;
        uint32_t max_common = std::min(
            static_cast<uint32_t>(prev_prompt_tokens.size()),
            static_cast<uint32_t>(token_ids.size()));
        while (common_len < max_common &&
               prev_prompt_tokens[common_len] == token_ids[common_len])
            ++common_len;

        if (common_len > 0 && common_len <= prev_kv_len)
            transformer->kv_cache()->truncate(common_len);
        else { transformer->kv_cache()->clear(); common_len = 0; }

        std::vector<uint32_t> new_tokens(
            token_ids.begin() + common_len, token_ids.end());

        if (verbose) {
            std::printf("[prompt: %zu, reused: %u, new: %zu]\n",
                        token_ids.size(), common_len, new_tokens.size());
        }

        if (new_tokens.empty()) {
            std::fprintf(stderr, "Error: no new tokens to process.\n");
            history.pop_back();
            continue;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        auto logits = transformer->forward(new_tokens, common_len);
        if (!logits) {
            std::fprintf(stderr, "\nError during prefill: %s\n", logits.error().c_str());
            history.pop_back();
            continue;
        }

        prev_prompt_tokens = token_ids;
        uint32_t pos = static_cast<uint32_t>(token_ids.size());
        uint32_t gen_count = 0;
        std::string response_text;
        bool ttft_recorded = false;
        double ttft_ms = 0.0;

        for (int t = 0; t < max_tokens; t++) {
            auto& l = *logits;
            mugen::Sampler::temperature_scale(l, sp.temperature);
            mugen::Sampler::top_p_filter(l, sp.top_p);
            mugen::Sampler::softmax(l);
            uint32_t next_token = sampler.sample_token(l);

            if (is_stop_token(next_token, tokenizer, ct)) break;

            if (!ttft_recorded) {
                ttft_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_start).count();
                ttft_recorded = true;
            }

            auto text = tokenizer.decode({next_token});
            std::printf("%s", text.c_str());
            std::fflush(stdout);
            response_text += text;
            ++gen_count;

            logits = transformer->forward({next_token}, pos++);
            if (!logits) {
                std::fprintf(stderr, "\nError during generation: %s\n", logits.error().c_str());
                break;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double decode_tps = (gen_count > 0 && total_ms > ttft_ms)
            ? gen_count / ((total_ms - ttft_ms) / 1000.0) : 0.0;

        if (verbose && gen_count > 0) {
            std::printf("\n  [TTFT: %.0f ms | Decode: %.1f tok/s | %u tokens in %.0f ms]",
                        ttft_ms, decode_tps, gen_count, total_ms);
        }
        std::printf("\n\n");

        prev_kv_len = pos;
        history.push_back({"assistant", response_text});
    }

    std::printf("\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Command: bench <model>
// ---------------------------------------------------------------------------

static int cmd_bench(const std::string& model_name, const std::string& compare_engine) {
    auto path = resolve_model_path(model_name);
    if (path.empty()) {
        std::fprintf(stderr, "Error: model '%s' not found.\n", model_name.c_str());
        std::fprintf(stderr, "  Tip: run 'mugen list' to see available models.\n");
        return 1;
    }

    std::printf("Loading model for benchmark...\n");
    auto bundle = load_model_bundle(path);
    if (!bundle) {
        std::fprintf(stderr, "Error: %s\n", bundle.error().c_str());
        return 1;
    }

    auto& transformer = bundle->model;
    auto& tokenizer = *bundle->tokenizer;
    const auto& meta = bundle->parser->metadata();

    std::printf("Model: %s (%s), GPU: %s\n",
                meta.name.empty() ? path.stem().c_str() : meta.name.c_str(),
                meta.arch.c_str(), bundle->gpu->device_name().c_str());
    constexpr const char* kBenchPrompt =
        "The transformer architecture, introduced in the paper Attention Is All You Need, "
        "revolutionized natural language processing by replacing recurrence with self-attention "
        "mechanisms. This allowed for significantly more parallelizable training and better "
        "modeling of long-range dependencies in text sequences.";

    auto prompt_tokens = tokenizer.encode(kBenchPrompt);
    if (prompt_tokens.empty() || prompt_tokens.front() != tokenizer.bos_token())
        prompt_tokens.insert(prompt_tokens.begin(), tokenizer.bos_token());

    uint32_t n_prompt = static_cast<uint32_t>(prompt_tokens.size());
    constexpr uint32_t n_gen = 128;

    std::printf("Prompt tokens: %u, Generation tokens: %u\n\n", n_prompt, n_gen);

    // Warmup run
    {
        transformer->kv_cache()->clear();
        auto r = transformer->forward({prompt_tokens[0]}, 0);
        if (!r) {
            std::fprintf(stderr, "Warmup failed: %s\n", r.error().c_str());
            return 1;
        }
        transformer->kv_cache()->clear();
    }

    // Prefill benchmark
    transformer->reset_prefill_profile();
    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = transformer->forward(prompt_tokens, 0);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!logits) {
        std::fprintf(stderr, "Prefill failed: %s\n", logits.error().c_str());
        return 1;
    }

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ttft_ms = prefill_ms;

    uint32_t pos = n_prompt;
    uint32_t gen_count = 0;
    transformer->reset_decode_profile();
    auto t2 = std::chrono::high_resolution_clock::now();

    for (uint32_t t = 0; t < n_gen; t++) {
        auto& l = *logits;
        uint32_t next_token = 0;
        float max_val = l[0];
        for (uint32_t i = 1; i < static_cast<uint32_t>(l.size()); i++) {
            if (l[i] > max_val) { max_val = l[i]; next_token = i; }
        }
        if (next_token == tokenizer.eos_token()) break;

        logits = transformer->forward({next_token}, pos++);
        if (!logits) {
            std::fprintf(stderr, "Decode error at token %u: %s\n", t, logits.error().c_str());
            break;
        }
        gen_count++;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double prefill_tps = (n_prompt > 0 && prefill_ms > 0)
                         ? n_prompt / (prefill_ms / 1000.0) : 0.0;
    double decode_tps = (gen_count > 0 && decode_ms > 0)
                        ? gen_count / (decode_ms / 1000.0) : 0.0;

    std::printf("{\n");
    std::printf("  \"engine\": \"mugen\",\n");
    std::printf("  \"version\": \"%s\",\n", mugen::kVersion);
    std::printf("  \"model\": \"%s\",\n", path.stem().c_str());
    std::printf("  \"architecture\": \"%s\",\n", meta.arch.c_str());
    std::printf("  \"device\": \"%s\",\n", bundle->gpu->device_name().c_str());
    std::printf("  \"results\": {\n");
    std::printf("    \"prompt_eval\": {\n");
    std::printf("      \"tokens\": %u,\n", n_prompt);
    std::printf("      \"time_ms\": %.1f,\n", prefill_ms);
    std::printf("      \"tokens_per_sec\": %.1f\n", prefill_tps);
    std::printf("    },\n");
    std::printf("    \"generation\": {\n");
    std::printf("      \"tokens\": %u,\n", gen_count);
    std::printf("      \"time_ms\": %.1f,\n", decode_ms);
    std::printf("      \"tokens_per_sec\": %.1f\n", decode_tps);
    std::printf("    },\n");
    std::printf("    \"ttft_ms\": %.1f\n", ttft_ms);
    std::printf("  }");

    if (!compare_engine.empty()) {
        std::printf(",\n");
        std::printf("  \"comparison\": {\n");
        std::printf("    \"engine\": \"%s\",\n", compare_engine.c_str());
        std::printf("    \"note\": \"run the same model with %s to compare\"\n", compare_engine.c_str());
        std::printf("  }");
    }

    std::printf("\n}\n\n");

    std::printf("Summary:\n");
    std::printf("  TTFT:       %.1f ms\n", ttft_ms);
    std::printf("  Prefill:    %.1f tok/s (%u tokens)\n", prefill_tps, n_prompt);
    std::printf("  Decode:     %.1f tok/s (%u tokens)\n", decode_tps, gen_count);
    std::printf("  Total:      %.1f ms\n", prefill_ms + decode_ms);

    // Profiled prefill pass: re-run with per-kernel GPU timestamps
    {
        transformer->kv_cache()->clear();
        transformer->set_prefill_profile_enabled(true);
        transformer->reset_prefill_profile();

        auto prof_logits = transformer->forward(prompt_tokens, 0);

        transformer->set_prefill_profile_enabled(false);

        if (prof_logits) {
            const auto& pprof = transformer->prefill_profile();
            if (pprof.n_tokens > 0 && pprof.total_gpu_ms > 0) {
                double total = pprof.total_gpu_ms;
                std::printf("\n=== Prefill Profile (%u tokens, gpu timestamps) ===\n", pprof.n_tokens);
                if (pprof.matmul_ms > 0)
                    std::printf("  matmul (q4_0+f16):  %7.2f ms  (%4.1f%%)\n",
                                pprof.matmul_ms, pprof.matmul_ms / total * 100);
                if (pprof.rms_norm_ms > 0)
                    std::printf("  rms_norm:           %7.2f ms  (%4.1f%%)\n",
                                pprof.rms_norm_ms, pprof.rms_norm_ms / total * 100);
                if (pprof.attention_ms > 0)
                    std::printf("  attention:          %7.2f ms  (%4.1f%%)\n",
                                pprof.attention_ms, pprof.attention_ms / total * 100);
                if (pprof.rope_ms > 0)
                    std::printf("  rope:               %7.2f ms  (%4.1f%%)\n",
                                pprof.rope_ms, pprof.rope_ms / total * 100);
                if (pprof.elementwise_ms > 0)
                    std::printf("  elementwise:        %7.2f ms  (%4.1f%%)\n",
                                pprof.elementwise_ms, pprof.elementwise_ms / total * 100);
                if (pprof.silu_mul_ms > 0)
                    std::printf("  silu_mul:           %7.2f ms  (%4.1f%%)\n",
                                pprof.silu_mul_ms, pprof.silu_mul_ms / total * 100);
                if (pprof.other_ms > 0)
                    std::printf("  other:              %7.2f ms  (%4.1f%%)\n",
                                pprof.other_ms, pprof.other_ms / total * 100);
                std::printf("  total GPU:          %7.2f ms\n", pprof.total_gpu_ms);
                std::printf("  total wall:         %7.2f ms\n", pprof.total_wall_ms);
            }
        }
    }

    const auto& prof = transformer->decode_profile();
    if (prof.n_tokens > 0) {
        double n = prof.n_tokens;
        std::printf("\nDecode Profile (%u tokens, per-token averages):\n", prof.n_tokens);
        std::printf("  GPU time:           %6.2f ms  (Metal HW timestamps)\n", prof.gpu_ms / n);
        std::printf("  dispatch wall:      %6.2f ms  (CPU wall around dispatch_chain_sync)\n", prof.dispatch_wall_ms / n);
        std::printf("  encode overhead:    %6.2f ms  (dispatch_wall - GPU = CPU encoding + CB lifecycle)\n",
                    (prof.dispatch_wall_ms - prof.gpu_ms) / n);
        std::printf("  logits readback:    %6.2f ms  (read_buffer + f16→f32)\n", prof.logits_ms / n);
        std::printf("  embedding:          %6.2f ms\n", prof.embed_ms / n);
        std::printf("  DP cache patch:     %6.2f ms\n", prof.patch_ms / n);
        std::printf("  total forward():    %6.2f ms\n", prof.total_wall_ms / n);
        double accounted = prof.dispatch_wall_ms + prof.logits_ms + prof.embed_ms + prof.patch_ms;
        std::printf("  unaccounted:        %6.2f ms  (CPU argmax in bench loop + other)\n",
                    (prof.total_wall_ms - accounted) / n);
    }

    auto kernel_prof = transformer->profile_kernel_dispatch();
    if (kernel_prof) {
        std::printf("\nPer-Kernel GPU Profile (decode, single-group dispatch):\n");
        double total = 0;
        for (auto& [name, ms] : *kernel_prof) total += ms;
        for (auto& [name, ms] : *kernel_prof) {
            std::printf("  %-25s %7.2f ms  (%4.1f%%)\n",
                        name.c_str(), ms, ms / total * 100);
        }
        std::printf("  %-25s %7.2f ms\n", "TOTAL", total);
        std::printf("  (Note: per-group dispatch has ~30us/group CB overhead;"
                    " relative proportions are meaningful)\n");
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Command: serve <model>
// ---------------------------------------------------------------------------

static mugen::HttpServer* g_cli_server = nullptr;

static void cli_signal_handler(int /*sig*/) {
    if (g_cli_server) {
        std::fprintf(stderr, "\nShutting down...\n");
        g_cli_server->stop();
    }
}

static int cmd_serve(const std::string& model, uint16_t port) {
    auto path = resolve_model_path(model);
    if (path.empty()) {
        std::fprintf(stderr, "Error: model '%s' not found.\n", model.c_str());
        std::fprintf(stderr, "  Tip: run 'mugen list' to see available models.\n");
        return 1;
    }

    std::fprintf(stderr, "Mugen %s — OpenAI-compatible API server\n", mugen::kVersion);
    std::fprintf(stderr, "Loading model: %s\n", path.c_str());

    auto bundle = load_model_bundle(path);
    if (!bundle) {
        std::fprintf(stderr, "Error: %s\n", bundle.error().c_str());
        return 1;
    }

    auto& transformer = bundle->model;
    auto& tokenizer = *bundle->tokenizer;
    const auto& cfg = transformer->config();
    const auto& meta = bundle->parser->metadata();

    std::string model_name = meta.name.empty()
        ? path.stem().string() : meta.name;

    std::fprintf(stderr, "Model: %s (%s)\n", model_name.c_str(), meta.arch.c_str());
    std::fprintf(stderr, "  Layers: %u, Heads: %u, Embed: %u, Vocab: %u\n",
                 cfg.n_layers, cfg.n_heads, cfg.embed_dim, cfg.vocab_size);
    std::fprintf(stderr, "  GPU: %s\n", bundle->gpu->device_name().c_str());

    auto find_token = [&](const std::string& text) -> uint32_t {
        for (uint32_t i = 0; i < tokenizer.vocab_size(); i++) {
            if (tokenizer.token_to_text(i) == text) return i;
        }
        return UINT32_MAX;
    };

    mugen::InferenceContext ctx;
    ctx.model = transformer.get();
    ctx.tokenizer = &tokenizer;
    ctx.model_name = model_name;
    ctx.created_at = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    ctx.im_start_id = find_token("<|im_start|>");
    ctx.im_end_id = find_token("<|im_end|>");
    ctx.has_chatml = (ctx.im_start_id != UINT32_MAX && ctx.im_end_id != UINT32_MAX);

    if (ctx.has_chatml)
        std::fprintf(stderr, "  ChatML: im_start=%u, im_end=%u\n",
                     ctx.im_start_id, ctx.im_end_id);

    mugen::HttpServer::Config srv_cfg;
    srv_cfg.host = "127.0.0.1";
    srv_cfg.port = port;

    std::fprintf(stderr, "\nBinding to %s:%u\n\n", srv_cfg.host.c_str(), srv_cfg.port);
    std::fprintf(stderr, "Routes:\n");

    mugen::HttpServer server(srv_cfg);
    g_cli_server = &server;

    mugen::register_api_routes(server, &ctx);

    std::signal(SIGINT, cli_signal_handler);
    std::signal(SIGTERM, cli_signal_handler);

    auto result = server.start();
    if (!result) {
        std::fprintf(stderr, "\nFailed to start server: %s\n", result.error().c_str());
        return 1;
    }

    std::fprintf(stderr, "\nServer listening on http://%s:%u\n",
                 srv_cfg.host.c_str(), srv_cfg.port);
    std::fprintf(stderr, "Press Ctrl+C to stop.\n");

    while (server.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::fprintf(stderr, "Server stopped.\n");
    g_cli_server = nullptr;
    return 0;
}

// ---------------------------------------------------------------------------
// Command: pull <url>
// ---------------------------------------------------------------------------

static auto extract_filename_from_url(const std::string& url) -> std::string {
    auto pos = url.rfind('/');
    if (pos == std::string::npos || pos + 1 >= url.size())
        return "model.gguf";
    auto name = url.substr(pos + 1);
    auto query = name.find('?');
    if (query != std::string::npos)
        name = name.substr(0, query);
    if (name.empty()) name = "model.gguf";
    return name;
}

static int cmd_pull(const std::string& url) {
    auto dir = models_dir();

    if (!fs::exists(dir)) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            std::fprintf(stderr, "Error: cannot create model directory '%s': %s\n",
                         dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    auto filename = extract_filename_from_url(url);
    auto dest = dir / filename;

    if (fs::exists(dest)) {
        std::fprintf(stderr, "File already exists: %s\n", dest.c_str());
        std::fprintf(stderr, "Remove it first with: mugen rm %s\n", dest.stem().c_str());
        return 1;
    }

    std::printf("Downloading: %s\n", url.c_str());
    std::printf("Destination: %s\n", dest.c_str());
    std::printf("\n");

    // Use curl subprocess (zero C++ dependencies, available on all macOS)
    std::string cmd = "curl -fSL --progress-bar -o \"" +
                      dest.string() + "\" \"" + url + "\"";

    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "\nError: download failed (curl exit code %d).\n", rc);
        std::error_code ec;
        fs::remove(dest, ec);
        return 1;
    }

    if (!fs::exists(dest) || fs::file_size(dest) == 0) {
        std::fprintf(stderr, "\nError: downloaded file is empty or missing.\n");
        std::error_code ec;
        fs::remove(dest, ec);
        return 1;
    }

    std::printf("\nDone: %s (%s)\n",
                dest.c_str(), format_bytes(fs::file_size(dest)).c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Command: rm <model>
// ---------------------------------------------------------------------------

static int cmd_rm(const std::string& model) {
    auto path = resolve_model_path(model);
    if (path.empty()) {
        std::fprintf(stderr, "Error: model '%s' not found.\n", model.c_str());
        std::fprintf(stderr, "  Tip: run 'mugen list' to see available models.\n");
        return 1;
    }

    std::error_code ec;
    auto size = fs::file_size(path, ec);

    if (!fs::remove(path, ec)) {
        std::fprintf(stderr, "Error: failed to remove '%s': %s\n",
                     path.c_str(), ec.message().c_str());
        return 1;
    }

    std::printf("Removed: %s (%s)\n", path.c_str(), format_bytes(size).c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    std::string command;
    std::string positional;  // model name, url, etc.

    bool        verbose     = false;
    int         max_tokens  = 512;
    double      temperature = 0.7;
    double      top_p       = 0.9;
    uint16_t    port        = 8080;
    std::string compare;
    std::string draft_model;
    int         draft_k     = 6;
};

static auto find_opt(int argc, char* argv[], int start,
                     const char* name) -> int {
    for (int i = start; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return i;
    }
    return -1;
}

static auto parse_args(int argc, char* argv[]) -> Args {
    Args args;

    if (argc < 2) return args;

    std::string_view first(argv[1]);

    if (first == "--version" || first == "-v") {
        args.command = "version";
        return args;
    }
    if (first == "--help" || first == "-h") {
        args.command = "help";
        return args;
    }

    args.command = std::string(first);

    // Commands that take a positional argument (model or url)
    if (argc >= 3 && argv[2][0] != '-') {
        args.positional = argv[2];
    }

    int opt_start = args.positional.empty() ? 2 : 3;

    // --verbose
    if (find_opt(argc, argv, opt_start, "--verbose") >= 0)
        args.verbose = true;

    // --max-tokens N
    int idx = find_opt(argc, argv, opt_start, "--max-tokens");
    if (idx >= 0 && idx + 1 < argc)
        args.max_tokens = std::atoi(argv[idx + 1]);

    // --temperature T
    idx = find_opt(argc, argv, opt_start, "--temperature");
    if (idx >= 0 && idx + 1 < argc)
        args.temperature = std::atof(argv[idx + 1]);

    // --top-p P
    idx = find_opt(argc, argv, opt_start, "--top-p");
    if (idx >= 0 && idx + 1 < argc)
        args.top_p = std::atof(argv[idx + 1]);

    // --port N
    idx = find_opt(argc, argv, opt_start, "--port");
    if (idx >= 0 && idx + 1 < argc)
        args.port = static_cast<uint16_t>(std::atoi(argv[idx + 1]));

    // --compare <engine>
    idx = find_opt(argc, argv, opt_start, "--compare");
    if (idx >= 0 && idx + 1 < argc)
        args.compare = argv[idx + 1];

    // --draft-model <path>
    idx = find_opt(argc, argv, opt_start, "--draft-model");
    if (idx >= 0 && idx + 1 < argc)
        args.draft_model = argv[idx + 1];

    // --draft-k N
    idx = find_opt(argc, argv, opt_start, "--draft-k");
    if (idx >= 0 && idx + 1 < argc)
        args.draft_k = std::atoi(argv[idx + 1]);

    return args;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    if (args.command.empty() || args.command == "help") {
        print_help();
        return 0;
    }

    if (args.command == "version") {
        print_version();
        return 0;
    }

    if (args.command == "doctor") {
        return cmd_doctor();
    }

    if (args.command == "list" || args.command == "ls") {
        return cmd_list();
    }

    if (args.command == "info") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'info' requires a model name.\n");
            std::fprintf(stderr, "Usage: mugen info <model>\n");
            return 1;
        }
        return cmd_info(args.positional);
    }

    if (args.command == "chat") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'chat' requires a model name.\n");
            std::fprintf(stderr, "Usage: mugen chat <model> [--verbose] [--max-tokens N] [--temperature T]\n");
            return 1;
        }
        if (!args.draft_model.empty()) {
            return cmd_speculative_chat(args.positional, args.draft_model,
                                        args.verbose, args.max_tokens,
                                        args.temperature, args.top_p,
                                        args.draft_k);
        }
        return cmd_chat(args.positional, args.verbose,
                        args.max_tokens, args.temperature, args.top_p);
    }

    if (args.command == "bench") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'bench' requires a model name.\n");
            std::fprintf(stderr, "Usage: mugen bench <model> [--compare <engine>]\n");
            return 1;
        }
        return cmd_bench(args.positional, args.compare);
    }

    if (args.command == "serve") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'serve' requires a model name.\n");
            std::fprintf(stderr, "Usage: mugen serve <model> --port <port>\n");
            return 1;
        }
        return cmd_serve(args.positional, args.port);
    }

    if (args.command == "pull") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'pull' requires a URL.\n");
            std::fprintf(stderr, "Usage: mugen pull <url>\n");
            return 1;
        }
        return cmd_pull(args.positional);
    }

    if (args.command == "rm" || args.command == "remove") {
        if (args.positional.empty()) {
            std::fprintf(stderr, "Error: 'rm' requires a model name.\n");
            std::fprintf(stderr, "Usage: mugen rm <model>\n");
            return 1;
        }
        return cmd_rm(args.positional);
    }

    std::fprintf(stderr, "Error: unknown command '%s'.\n\n", args.command.c_str());
    print_help();
    return 1;
}

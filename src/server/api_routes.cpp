#include "server/api_routes.h"
#include "server/json_utils.h"
#include "core/model/transformer.h"
#include "core/memory/kv_cache.h"
#include "model/tokenizer.h"
#include "core/scheduler/sampling.h"

#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace mugen {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto current_unix_time() -> int64_t {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static auto generate_id() -> std::string {
    static thread_local std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    static constexpr char hex[] = "0123456789abcdef";
    std::string id = "chatcmpl-";
    for (int i = 0; i < 24; ++i) {
        id += hex[rng() % 16];
    }
    return id;
}

// ---------------------------------------------------------------------------
// ChatML token sequence builder
// ---------------------------------------------------------------------------

struct ChatMessage {
    std::string role;
    std::string content;
};

static auto parse_messages(const json::Parser::JsonValue& root)
    -> std::vector<ChatMessage>
{
    std::vector<ChatMessage> messages;
    auto* msgs = root.get("messages");
    if (!msgs || msgs->type != json::Parser::JsonValue::Array) return messages;

    for (const auto& msg : msgs->arr_val) {
        if (msg.type != json::Parser::JsonValue::Object) continue;
        auto* role = msg.get("role");
        auto* content = msg.get("content");
        if (!role || !content) continue;
        messages.push_back({
            std::string(role->as_string("user")),
            std::string(content->as_string(""))});
    }
    return messages;
}

static auto build_chatml_tokens(
    const std::vector<ChatMessage>& messages,
    const Tokenizer& tokenizer,
    uint32_t im_start_id,
    uint32_t im_end_id,
    bool has_chatml) -> std::vector<uint32_t>
{
    std::vector<uint32_t> tokens;

    if (!has_chatml) {
        std::string combined;
        for (const auto& msg : messages) {
            combined += msg.content + "\n";
        }
        tokens = tokenizer.encode(combined);
        if (tokens.empty() || tokens.front() != tokenizer.bos_token()) {
            tokens.insert(tokens.begin(), tokenizer.bos_token());
        }
        return tokens;
    }

    auto nl_tokens = tokenizer.encode("\n");

    for (const auto& msg : messages) {
        tokens.push_back(im_start_id);
        auto content_tokens = tokenizer.encode(msg.role + "\n" + msg.content);
        tokens.insert(tokens.end(), content_tokens.begin(), content_tokens.end());
        tokens.push_back(im_end_id);
        tokens.insert(tokens.end(), nl_tokens.begin(), nl_tokens.end());
    }

    // Assistant turn prefix
    tokens.push_back(im_start_id);
    auto asst_tokens = tokenizer.encode("assistant\n");
    tokens.insert(tokens.end(), asst_tokens.begin(), asst_tokens.end());

    return tokens;
}

// ---------------------------------------------------------------------------
// HTTP error helper
// ---------------------------------------------------------------------------

static void send_json_error(
    const std::function<void(const std::string&)>& send,
    int status_code,
    const std::string& status_text,
    const std::string& message)
{
    auto err_body = json::object({
        {"error", json::object({
            {"message", message},
            {"type", "invalid_request_error"},
        })}
    }).dump();

    std::string http;
    http += "HTTP/1.1 " + std::to_string(status_code) + " " + status_text + "\r\n";
    http += "Content-Type: application/json\r\n";
    http += "Content-Length: " + std::to_string(err_body.size()) + "\r\n";
    http += "Connection: close\r\n";
    http += "Access-Control-Allow-Origin: *\r\n";
    http += "\r\n";
    http += err_body;
    send(http);
}

// ---------------------------------------------------------------------------
// Route registration
// ---------------------------------------------------------------------------

void register_api_routes(HttpServer& server, InferenceContext* ctx) {

    // GET /health
    server.route("GET", "/health",
        [](const HttpServer::Request&) -> HttpServer::Response {
            auto body = json::object({
                {"status", "ok"},
                {"version", MUGEN_VERSION},
            });
            HttpServer::Response resp;
            resp.body = body.dump();
            return resp;
        });

    // GET /v1/models
    server.route("GET", "/v1/models",
        [ctx](const HttpServer::Request&) -> HttpServer::Response {
            auto body = json::object({
                {"object", "list"},
                {"data", json::array({
                    json::object({
                        {"id", ctx->model_name},
                        {"object", "model"},
                        {"created", ctx->created_at},
                        {"owned_by", "mugen"},
                    })
                })},
            });
            HttpServer::Response resp;
            resp.body = body.dump();
            return resp;
        });

    // GET /v1/metrics
    server.route("GET", "/v1/metrics",
        [](const HttpServer::Request&) -> HttpServer::Response {
            auto body = json::object({
                {"tok_per_sec", 0.0},
                {"cache_hit_rate", 0.0},
                {"memory_pressure", 0.0},
                {"mode", "inference"},
            });
            HttpServer::Response resp;
            resp.body = body.dump();
            return resp;
        });

    // POST /v1/chat/completions — unified stream router
    server.route_stream("POST", "/v1/chat/completions",
        [ctx](const HttpServer::Request& req,
              std::function<void(const std::string&)> send) {

            json::Parser parser(req.body);
            auto parsed = parser.parse();

            if (!parsed) {
                send_json_error(send, 400, "Bad Request",
                                "Invalid JSON in request body");
                return;
            }

            // --- Parse request parameters ---
            std::string model_name = ctx->model_name;
            bool stream = false;
            int max_tokens = 512;
            float temperature = 0.7f;
            float top_p = 1.0f;

            if (auto* m = parsed->get("model"))
                model_name = std::string(m->as_string(ctx->model_name));
            if (auto* s = parsed->get("stream"))
                stream = s->as_bool(false);
            if (auto* mt = parsed->get("max_tokens"))
                max_tokens = mt->as_int(512);
            if (auto* t = parsed->get("temperature"))
                temperature = static_cast<float>(t->as_double(0.7));
            if (auto* p = parsed->get("top_p"))
                top_p = static_cast<float>(p->as_double(1.0));

            if (max_tokens <= 0) max_tokens = 1;

            auto messages = parse_messages(*parsed);
            if (messages.empty()) {
                send_json_error(send, 400, "Bad Request",
                                "messages array is required and must not be empty");
                return;
            }

            auto prompt_tokens = build_chatml_tokens(
                messages, *ctx->tokenizer,
                ctx->im_start_id, ctx->im_end_id, ctx->has_chatml);

            auto id = generate_id();
            auto now = current_unix_time();

            Sampler::SamplingParams sp;
            sp.temperature = temperature;
            sp.top_p = top_p;
            Sampler sampler(sp);

            // =============================================================
            // Non-streaming path
            // =============================================================
            if (!stream) {
                std::string generated_text;
                int completion_tokens = 0;
                std::string finish_reason = "stop";

                {
                    std::lock_guard<std::mutex> lock(ctx->mtx);

                    uint32_t common_len = 0;
                    auto& prev = ctx->prev_prompt_tokens;
                    uint32_t max_common = std::min(
                        static_cast<uint32_t>(prev.size()),
                        static_cast<uint32_t>(prompt_tokens.size()));
                    while (common_len < max_common &&
                           prev[common_len] == prompt_tokens[common_len])
                        ++common_len;

                    if (common_len > 0 && common_len <= ctx->prev_kv_len) {
                        ctx->model->kv_cache()->truncate(common_len);
                    } else {
                        ctx->model->kv_cache()->clear();
                        common_len = 0;
                    }

                    std::vector<uint32_t> new_tokens(
                        prompt_tokens.begin() + common_len,
                        prompt_tokens.end());

                    std::expected<std::vector<float>, std::string> logits;
                    if (new_tokens.empty()) {
                        logits = std::unexpected(std::string("No new tokens to process"));
                    } else {
                        logits = ctx->model->forward(new_tokens, common_len);
                    }

                    ctx->prev_prompt_tokens = prompt_tokens;

                    if (!logits) {
                        send_json_error(send, 500, "Internal Server Error",
                                        "Prefill failed: " + logits.error());
                        return;
                    }

                    uint32_t pos = static_cast<uint32_t>(prompt_tokens.size());

                    for (int t = 0; t < max_tokens; t++) {
                        auto& l = *logits;
                        Sampler::temperature_scale(l, sp.temperature);
                        Sampler::top_p_filter(l, sp.top_p);
                        Sampler::softmax(l);
                        uint32_t next_token = sampler.sample_token(l);

                        if (next_token == ctx->tokenizer->eos_token() ||
                            (ctx->has_chatml && next_token == ctx->im_end_id)) {
                            break;
                        }

                        generated_text += ctx->tokenizer->decode({next_token});
                        completion_tokens++;

                        logits = ctx->model->forward({next_token}, pos++);
                        if (!logits) break;
                    }

                    if (completion_tokens >= max_tokens)
                        finish_reason = "length";

                    ctx->prev_kv_len = pos;
                }

                auto response_json = json::object({
                    {"id", id},
                    {"object", "chat.completion"},
                    {"created", now},
                    {"model", model_name},
                    {"choices", json::array({
                        json::object({
                            {"index", 0},
                            {"message", json::object({
                                {"role", "assistant"},
                                {"content", generated_text},
                            })},
                            {"finish_reason", finish_reason},
                        })
                    })},
                    {"usage", json::object({
                        {"prompt_tokens", static_cast<int>(prompt_tokens.size())},
                        {"completion_tokens", completion_tokens},
                        {"total_tokens", static_cast<int>(prompt_tokens.size()) + completion_tokens},
                    })},
                });

                std::string body = response_json.dump();
                std::string http;
                http += "HTTP/1.1 200 OK\r\n";
                http += "Content-Type: application/json\r\n";
                http += "Content-Length: " + std::to_string(body.size()) + "\r\n";
                http += "Connection: close\r\n";
                http += "Access-Control-Allow-Origin: *\r\n";
                http += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
                http += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
                http += "\r\n";
                http += body;
                send(http);
                return;
            }

            // =============================================================
            // Streaming path (SSE)
            // =============================================================

            // SSE response headers
            std::string header;
            header += "HTTP/1.1 200 OK\r\n";
            header += "Content-Type: text/event-stream\r\n";
            header += "Cache-Control: no-cache\r\n";
            header += "Connection: keep-alive\r\n";
            header += "Access-Control-Allow-Origin: *\r\n";
            header += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
            header += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
            header += "\r\n";
            send(header);

            // Role delta
            auto role_chunk = json::object({
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", now},
                {"model", model_name},
                {"choices", json::array({
                    json::object({
                        {"index", 0},
                        {"delta", json::object({
                            {"role", "assistant"},
                        })},
                        {"finish_reason", nullptr},
                    })
                })},
            });
            send("data: " + role_chunk.dump() + "\n\n");

            std::string finish_reason = "stop";

            {
                std::lock_guard<std::mutex> lock(ctx->mtx);

                uint32_t common_len = 0;
                auto& prev = ctx->prev_prompt_tokens;
                uint32_t max_common = std::min(
                    static_cast<uint32_t>(prev.size()),
                    static_cast<uint32_t>(prompt_tokens.size()));
                while (common_len < max_common &&
                       prev[common_len] == prompt_tokens[common_len])
                    ++common_len;

                if (common_len > 0 && common_len <= ctx->prev_kv_len) {
                    ctx->model->kv_cache()->truncate(common_len);
                } else {
                    ctx->model->kv_cache()->clear();
                    common_len = 0;
                }

                std::vector<uint32_t> new_tokens(
                    prompt_tokens.begin() + common_len,
                    prompt_tokens.end());

                std::expected<std::vector<float>, std::string> logits;
                if (!new_tokens.empty()) {
                    logits = ctx->model->forward(new_tokens, common_len);
                }

                ctx->prev_prompt_tokens = prompt_tokens;

                if (!logits) {
                    auto err_chunk = json::object({
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", now},
                        {"model", model_name},
                        {"choices", json::array({
                            json::object({
                                {"index", 0},
                                {"delta", json::object({})},
                                {"finish_reason", "stop"},
                            })
                        })},
                    });
                    send("data: " + err_chunk.dump() + "\n\n");
                    send("data: [DONE]\n\n");
                    return;
                }

                uint32_t pos = static_cast<uint32_t>(prompt_tokens.size());
                int completion_tokens = 0;

                for (int t = 0; t < max_tokens; t++) {
                    auto& l = *logits;
                    Sampler::temperature_scale(l, sp.temperature);
                    Sampler::top_p_filter(l, sp.top_p);
                    Sampler::softmax(l);
                    uint32_t next_token = sampler.sample_token(l);

                    if (next_token == ctx->tokenizer->eos_token() ||
                        (ctx->has_chatml && next_token == ctx->im_end_id)) {
                        break;
                    }

                    auto token_text = ctx->tokenizer->decode({next_token});
                    completion_tokens++;

                    auto chunk = json::object({
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", now},
                        {"model", model_name},
                        {"choices", json::array({
                            json::object({
                                {"index", 0},
                                {"delta", json::object({
                                    {"content", token_text},
                                })},
                                {"finish_reason", nullptr},
                            })
                        })},
                    });
                    send("data: " + chunk.dump() + "\n\n");

                    logits = ctx->model->forward({next_token}, pos++);
                    if (!logits) break;
                }

                if (completion_tokens >= max_tokens)
                    finish_reason = "length";

                ctx->prev_kv_len = pos;
            }

            // Finish chunk
            auto finish_chunk = json::object({
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", now},
                {"model", model_name},
                {"choices", json::array({
                    json::object({
                        {"index", 0},
                        {"delta", json::object({})},
                        {"finish_reason", finish_reason},
                    })
                })},
            });
            send("data: " + finish_chunk.dump() + "\n\n");
            send("data: [DONE]\n\n");
        });

    std::fprintf(stderr, "  POST /v1/chat/completions  (stream + non-stream)\n");
    std::fprintf(stderr, "  GET  /v1/models\n");
    std::fprintf(stderr, "  GET  /v1/metrics\n");
    std::fprintf(stderr, "  GET  /health\n");
}

}  // namespace mugen

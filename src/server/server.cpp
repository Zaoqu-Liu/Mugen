#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "mugen/mugen.h"
#include "server/http_server.h"
#include "server/api_routes.h"
#include "core/compute/metal_compute.h"
#include "core/memory/mmap_loader.h"
#include "core/model/transformer.h"
#include "model/gguf_parser.h"
#include "model/tokenizer.h"

namespace fs = std::filesystem;

static mugen::HttpServer* g_server = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    if (g_server) {
        std::fprintf(stderr, "\nShutting down...\n");
        g_server->stop();
    }
}

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s <model-path> [--host HOST] [--port PORT]\n", prog);
    std::fprintf(stderr, "  <model-path>  Path to GGUF model file\n");
    std::fprintf(stderr, "  --host HOST   Bind address (default: 127.0.0.1)\n");
    std::fprintf(stderr, "  --port PORT   Bind port    (default: 8080)\n");
}

int main(int argc, char* argv[]) {
    mugen::HttpServer::Config config;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--host") && i + 1 < argc) {
            config.host = argv[++i];
        } else if ((arg == "--port") && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] != '-' && model_path.empty()) {
            model_path = arg;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "Error: model path is required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!fs::exists(model_path)) {
        std::fprintf(stderr, "Error: model file not found: %s\n", model_path.c_str());
        return 1;
    }

    std::fprintf(stderr, "Mugen %s — OpenAI-compatible API server\n", mugen::kVersion);
    std::fprintf(stderr, "Loading model: %s\n", model_path.c_str());

    auto shard_paths = mugen::GGUFParser::detect_shards(model_path);
    if (shard_paths.size() > 1)
        std::fprintf(stderr, "Split GGUF: %zu shards detected\n", shard_paths.size());

    auto parser_result = (shard_paths.size() > 1)
        ? mugen::GGUFParser::parse_sharded(shard_paths)
        : mugen::GGUFParser::parse(model_path);
    if (!parser_result) {
        std::fprintf(stderr, "Error parsing GGUF: %s\n", parser_result.error().c_str());
        return 1;
    }
    auto& parser = *parser_result;
    const auto& meta = parser.metadata();

    auto gpu_result = mugen::MetalCompute::create();
    if (!gpu_result) {
        std::fprintf(stderr, "Error creating Metal device: %s\n", gpu_result.error().c_str());
        return 1;
    }
    auto& gpu = *gpu_result;

    std::vector<mugen::MmapRegion> mmaps;
    for (auto& sp : shard_paths) {
        auto mr = mugen::MmapLoader::map_file(sp);
        if (!mr) {
            std::fprintf(stderr, "Error mmapping shard: %s\n", mr.error().c_str());
            return 1;
        }
        mmaps.push_back(std::move(*mr));
    }

    auto model_result = (mmaps.size() == 1)
        ? mugen::TransformerModel::from_gguf(gpu.get(), parser, mmaps[0])
        : mugen::TransformerModel::from_gguf(gpu.get(), parser, mmaps);
    if (!model_result) {
        std::fprintf(stderr, "Error building model: %s\n", model_result.error().c_str());
        return 1;
    }
    auto& transformer = *model_result;

    auto tok_result = mugen::Tokenizer::from_gguf(meta);
    if (!tok_result) {
        std::fprintf(stderr, "Error building tokenizer: %s\n", tok_result.error().c_str());
        return 1;
    }
    auto& tokenizer = *tok_result;

    const auto& cfg = transformer->config();
    std::string model_name = meta.name.empty()
        ? fs::path(model_path).stem().string()
        : meta.name;

    std::fprintf(stderr, "Model: %s (%s)\n", model_name.c_str(), meta.arch.c_str());
    std::fprintf(stderr, "  Layers: %u, Heads: %u, Embed: %u, Vocab: %u\n",
                 cfg.n_layers, cfg.n_heads, cfg.embed_dim, cfg.vocab_size);
    std::fprintf(stderr, "  GPU: %s\n", gpu->device_name().c_str());

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
        std::fprintf(stderr, "  ChatML: im_start=%u, im_end=%u\n", ctx.im_start_id, ctx.im_end_id);

    std::fprintf(stderr, "\nBinding to %s:%u\n\n", config.host.c_str(), config.port);
    std::fprintf(stderr, "Routes:\n");

    mugen::HttpServer server(config);
    g_server = &server;

    mugen::register_api_routes(server, &ctx);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto result = server.start();
    if (!result) {
        std::fprintf(stderr, "\nFailed to start server: %s\n", result.error().c_str());
        return 1;
    }

    std::fprintf(stderr, "\nServer listening on http://%s:%u\n", config.host.c_str(), config.port);
    std::fprintf(stderr, "Press Ctrl+C to stop.\n");

    while (server.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::fprintf(stderr, "Server stopped.\n");
    g_server = nullptr;
    return 0;
}

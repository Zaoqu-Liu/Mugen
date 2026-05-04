#pragma once

#include "server/http_server.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mugen {

class TransformerModel;
class Tokenizer;

struct InferenceContext {
    TransformerModel* model = nullptr;
    Tokenizer* tokenizer = nullptr;
    std::string model_name;
    int64_t created_at = 0;
    uint32_t im_start_id = UINT32_MAX;
    uint32_t im_end_id = UINT32_MAX;
    bool has_chatml = false;
    std::mutex mtx;

    std::vector<uint32_t> prev_prompt_tokens;
    uint32_t prev_kv_len = 0;
};

void register_api_routes(HttpServer& server, InferenceContext* ctx);

}  // namespace mugen

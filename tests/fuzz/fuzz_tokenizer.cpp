#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "model/gguf_parser.h"
#include "model/tokenizer.h"

// Build a minimal SPM tokenizer once at startup for fuzzing.
static mugen::Tokenizer* get_test_tokenizer() {
    static auto tok = [] () -> mugen::Tokenizer* {
        mugen::GGUFMetadata meta;
        meta.raw_string_kv["tokenizer.ggml.model"] = "llama";

        std::vector<std::string> tokens;
        tokens.reserve(259);
        tokens.emplace_back("<unk>");
        tokens.emplace_back("<s>");
        tokens.emplace_back("</s>");
        // Byte-fallback tokens <0x00>..<0xFF>
        for (int i = 0; i < 256; ++i) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            tokens.emplace_back(buf);
        }
        meta.raw_string_array_kv["tokenizer.ggml.tokens"] = tokens;

        std::vector<float> scores(tokens.size(), -1.0f);
        meta.raw_float_array_kv["tokenizer.ggml.scores"] = scores;

        meta.raw_int_kv["tokenizer.ggml.bos_token_id"] = 1;
        meta.raw_int_kv["tokenizer.ggml.eos_token_id"] = 2;

        auto result = mugen::Tokenizer::from_gguf(meta);
        if (!result.has_value()) return nullptr;
        static auto owned = std::move(*result);
        return &owned;
    }();
    return tok;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    auto* tok = get_test_tokenizer();
    if (!tok) return 0;

    std::string_view input(reinterpret_cast<const char*>(data), size);

    // Encode must not crash on any byte sequence.
    auto ids = tok->encode(input);

    // Decode must not crash on any token sequence.
    [[maybe_unused]] auto text = tok->decode(ids);

    // Also try decoding the raw bytes as token IDs.
    if (size >= sizeof(uint32_t)) {
        size_t n_tokens = size / sizeof(uint32_t);
        std::vector<uint32_t> raw_ids(n_tokens);
        std::memcpy(raw_ids.data(), data, n_tokens * sizeof(uint32_t));
        [[maybe_unused]] auto raw_text = tok->decode(raw_ids);
    }

    return 0;
}

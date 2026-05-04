#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "model/gguf_parser.h"

// Write fuzz input to a temp file and attempt to parse it as GGUF.
// The parser must never crash, hang, or leak regardless of input.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static int counter = 0;
    auto path = std::filesystem::temp_directory_path()
              / ("mugen_fuzz_gguf_" + std::to_string(counter++) + ".gguf");

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return 0;
    std::fwrite(data, 1, size, fp);
    std::fclose(fp);

    auto result = mugen::GGUFParser::parse(path);
    // We don't care whether parsing succeeds — only that it doesn't crash.
    if (result.has_value()) {
        // Exercise accessors to ensure they're safe on parsed data.
        [[maybe_unused]] auto& meta = result->metadata();
        [[maybe_unused]] auto& tensors = result->tensors();
        [[maybe_unused]] auto offset = result->data_offset();
        [[maybe_unused]] auto fsize = result->file_size();
        [[maybe_unused]] auto moe = result->is_moe();

        for (const auto& ti : tensors) {
            [[maybe_unused]] auto* found = result->tensor_by_name(ti.name);
        }

        if (moe) {
            for (uint32_t layer = 0; layer < 2; ++layer) {
                for (uint32_t expert = 0; expert < result->expert_count(); ++expert) {
                    [[maybe_unused]] auto names =
                        result->expert_tensor_names(layer, expert);
                }
            }
        }
    }

    std::filesystem::remove(path);
    return 0;
}

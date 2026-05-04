#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mugen {

struct GGUFMetadata;

class Tokenizer {
public:
    /// Build a tokenizer from parsed GGUF metadata.
    /// Supports SPM (tokenizer.ggml.model == "llama"/"gemma") and
    /// BPE (tokenizer.ggml.model == "gpt2"/"starcoder"/"refact").
    static auto from_gguf(const GGUFMetadata& metadata)
        -> std::expected<Tokenizer, std::string>;

    /// Encode UTF-8 text to token IDs.
    auto encode(std::string_view text) const -> std::vector<uint32_t>;

    /// Decode token IDs back to UTF-8 text.
    auto decode(const std::vector<uint32_t>& tokens) const -> std::string;

    auto bos_token()  const -> uint32_t { return bos_id_; }
    auto eos_token()  const -> uint32_t { return eos_id_; }
    auto vocab_size() const -> uint32_t { return static_cast<uint32_t>(vocab_.size()); }

    /// Look up the text piece for a token ID.  Returns empty view if invalid.
    auto token_to_text(uint32_t token_id) const -> std::string_view;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

private:
    Tokenizer() = default;

    enum class Mode { SPM, BPE };

    auto encode_spm(std::string_view text) const -> std::vector<uint32_t>;
    auto encode_bpe(std::string_view text) const -> std::vector<uint32_t>;
    void bpe_merge(std::vector<std::string>& symbols) const;
    auto symbols_to_ids(const std::vector<std::string>& symbols) const
        -> std::vector<uint32_t>;

    Mode mode_ = Mode::SPM;

    std::vector<std::string>                    vocab_;
    std::unordered_map<std::string, uint32_t>   token_to_id_;
    std::vector<float>                          scores_;
    std::unordered_map<std::string, int>        merge_ranks_;

    uint32_t bos_id_ = 1;
    uint32_t eos_id_ = 2;
    uint32_t pad_id_ = 0;
    bool     add_space_prefix_ = true;
};

}  // namespace mugen

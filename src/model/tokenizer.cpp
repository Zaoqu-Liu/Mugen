#include "model/tokenizer.h"
#include "model/gguf_parser.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace mugen {

namespace {

// ▁ (U+2581) — SentencePiece space marker, 3 bytes in UTF-8.
constexpr char kSPSpace[] = "\xE2\x96\x81";
constexpr size_t kSPSpaceLen = 3;

// ---------------------------------------------------------------------------
// GPT-2 byte-to-unicode mapping (for BPE tokenizers)
// ---------------------------------------------------------------------------

// GPT-2 byte ↔ unicode tables (shared between encode and decode)
struct ByteUnicodeTables {
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
    std::unordered_map<uint8_t, std::string> byte_to_unicode;

    ByteUnicodeTables() {
        std::vector<int> bs;
        bs.reserve(256);
        for (int i = 0x21; i <= 0x7E; i++) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

        std::vector<int> cs(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            auto cp = static_cast<uint32_t>(cs[i]);
            auto byte = static_cast<uint8_t>(bs[i]);
            unicode_to_byte[cp] = byte;

            std::string utf8;
            if (cp < 0x80) {
                utf8 += static_cast<char>(cp);
            } else if (cp < 0x800) {
                utf8 += static_cast<char>(0xC0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                utf8 += static_cast<char>(0xE0 | (cp >> 12));
                utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            }
            byte_to_unicode[byte] = utf8;
        }
    }
};

auto get_tables() -> const ByteUnicodeTables& {
    static const ByteUnicodeTables tables;
    return tables;
}

auto decode_bpe_piece(std::string_view piece) -> std::string {
    static const auto& u2b = get_tables().unicode_to_byte;
    std::string out;
    out.reserve(piece.size());
    size_t i = 0;
    while (i < piece.size()) {
        uint32_t cp = 0;
        auto lead = static_cast<unsigned char>(piece[i]);
        size_t len = 1;
        if ((lead & 0x80) == 0)        { cp = lead; }
        else if ((lead & 0xE0) == 0xC0) { cp = lead & 0x1F; len = 2; }
        else if ((lead & 0xF0) == 0xE0) { cp = lead & 0x0F; len = 3; }
        else if ((lead & 0xF8) == 0xF0) { cp = lead & 0x07; len = 4; }
        for (size_t j = 1; j < len && i + j < piece.size(); j++)
            cp = (cp << 6) | (static_cast<unsigned char>(piece[i + j]) & 0x3F);
        auto it = u2b.find(cp);
        if (it != u2b.end())
            out += static_cast<char>(it->second);
        else
            out.append(piece.data() + i, len);
        i += len;
    }
    return out;
}

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

auto utf8_codepoint_len(unsigned char lead) -> size_t {
    if ((lead & 0x80) == 0x00) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;  // invalid lead byte — treat as single byte
}

auto split_utf8(std::string_view text) -> std::vector<std::string> {
    std::vector<std::string> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_codepoint_len(static_cast<unsigned char>(text[i]));
        if (i + len > text.size()) len = 1;
        out.emplace_back(text.substr(i, len));
        i += len;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Byte-fallback token helpers  (<0xHH>)
// ---------------------------------------------------------------------------

auto hex_digit_value(char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/// Returns 0-255 if text is a byte-fallback token, else -1.
auto parse_byte_token(std::string_view t) -> int {
    if (t.size() != 6) return -1;
    if (t[0] != '<' || t[1] != '0' || t[2] != 'x' || t[5] != '>') return -1;
    int hi = hex_digit_value(t[3]);
    int lo = hex_digit_value(t[4]);
    if (hi < 0 || lo < 0) return -1;
    return (hi << 4) | lo;
}

auto byte_to_fallback_str(unsigned char b) -> std::string {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// SentencePiece normalization
// ---------------------------------------------------------------------------

/// Replace ASCII space with ▁ throughout the string.
auto replace_space_with_sp(const std::string& input) -> std::string {
    std::string out;
    out.reserve(input.size() * 2);
    for (char c : input) {
        if (c == ' ')
            out.append(kSPSpace, kSPSpaceLen);
        else
            out += c;
    }
    return out;
}

/// Replace ▁ with ASCII space.
auto replace_sp_with_space(const std::string& input) -> std::string {
    std::string out;
    out.reserve(input.size());
    size_t i = 0;
    while (i < input.size()) {
        if (i + kSPSpaceLen <= input.size() &&
            static_cast<unsigned char>(input[i])     == 0xE2 &&
            static_cast<unsigned char>(input[i + 1]) == 0x96 &&
            static_cast<unsigned char>(input[i + 2]) == 0x81) {
            out += ' ';
            i += kSPSpaceLen;
        } else {
            out += input[i];
            ++i;
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tokenizer::from_gguf
// ---------------------------------------------------------------------------

auto Tokenizer::from_gguf(const GGUFMetadata& metadata)
    -> std::expected<Tokenizer, std::string>
{
    Tokenizer tok;

    // --- mode ---------------------------------------------------------------
    auto model_it = metadata.raw_string_kv.find("tokenizer.ggml.model");
    if (model_it == metadata.raw_string_kv.end())
        return std::unexpected("missing tokenizer.ggml.model");

    const auto& model_name = model_it->second;
    if (model_name == "llama" || model_name == "gemma")
        tok.mode_ = Mode::SPM;
    else if (model_name == "gpt2" || model_name == "starcoder" || model_name == "refact")
        tok.mode_ = Mode::BPE;
    else
        return std::unexpected("unsupported tokenizer model: " + model_name);

    // --- vocabulary ---------------------------------------------------------
    auto tokens_it = metadata.raw_string_array_kv.find("tokenizer.ggml.tokens");
    if (tokens_it == metadata.raw_string_array_kv.end())
        return std::unexpected("missing tokenizer.ggml.tokens");

    tok.vocab_ = tokens_it->second;
    for (uint32_t i = 0; i < static_cast<uint32_t>(tok.vocab_.size()); ++i)
        tok.token_to_id_[tok.vocab_[i]] = i;

    // --- scores (SPM priority) ----------------------------------------------
    auto scores_it = metadata.raw_float_array_kv.find("tokenizer.ggml.scores");
    if (scores_it != metadata.raw_float_array_kv.end())
        tok.scores_ = scores_it->second;
    if (tok.scores_.size() < tok.vocab_.size())
        tok.scores_.resize(tok.vocab_.size(), -1e9f);

    // --- merge rules (BPE priority) -----------------------------------------
    auto merges_it = metadata.raw_string_array_kv.find("tokenizer.ggml.merges");
    if (merges_it != metadata.raw_string_array_kv.end()) {
        const auto& merges = merges_it->second;
        for (size_t i = 0; i < merges.size(); ++i)
            tok.merge_ranks_[merges[i]] = static_cast<int>(i);
    }

    // --- special token IDs --------------------------------------------------
    auto get_u32 = [&](const std::string& key, uint32_t def) -> uint32_t {
        auto it = metadata.raw_int_kv.find(key);
        return it != metadata.raw_int_kv.end()
                   ? static_cast<uint32_t>(it->second)
                   : def;
    };
    tok.bos_id_ = get_u32("tokenizer.ggml.bos_token_id", 1);
    tok.eos_id_ = get_u32("tokenizer.ggml.eos_token_id", 2);
    tok.pad_id_ = get_u32("tokenizer.ggml.padding_token_id", 0);

    // --- add_space_prefix ---------------------------------------------------
    auto sp_it = metadata.raw_int_kv.find("tokenizer.ggml.add_space_prefix");
    tok.add_space_prefix_ =
        (sp_it != metadata.raw_int_kv.end()) ? (sp_it->second != 0)
                                              : (tok.mode_ == Mode::SPM);

    return tok;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

auto Tokenizer::encode(std::string_view text) const -> std::vector<uint32_t> {
    if (text.empty()) return {};
    return (mode_ == Mode::SPM) ? encode_spm(text) : encode_bpe(text);
}

auto Tokenizer::encode_spm(std::string_view text) const -> std::vector<uint32_t> {
    // 1. Normalize: optional space prefix, then ' ' → ▁.
    std::string raw;
    raw.reserve(text.size() + 4);
    if (add_space_prefix_) raw += ' ';
    raw.append(text);

    std::string normalized = replace_space_with_sp(raw);

    // 2. Split into UTF-8 codepoints, byte-fallback for unknown chars.
    auto codepoints = split_utf8(normalized);

    std::vector<std::string> symbols;
    symbols.reserve(codepoints.size());
    for (const auto& cp : codepoints) {
        if (token_to_id_.count(cp)) {
            symbols.push_back(cp);
        } else {
            for (unsigned char byte : cp)
                symbols.push_back(byte_to_fallback_str(byte));
        }
    }

    // 3. BPE merge (score-based priority).
    bpe_merge(symbols);

    // 4. Map to token IDs.
    return symbols_to_ids(symbols);
}

auto Tokenizer::encode_bpe(std::string_view text) const -> std::vector<uint32_t> {
    const auto& b2u = get_tables().byte_to_unicode;

    // 1. Map each byte to its GPT-2 unicode symbol.
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (auto byte : text) {
        auto it = b2u.find(static_cast<uint8_t>(byte));
        if (it != b2u.end()) {
            symbols.push_back(it->second);
        } else {
            symbols.emplace_back(1, byte);
        }
    }

    // 2. BPE merge (rank-based priority).
    bpe_merge(symbols);

    // 3. Map to token IDs.
    return symbols_to_ids(symbols);
}

// ---------------------------------------------------------------------------
// bpe_merge — unified merge loop for both SPM and BPE
// ---------------------------------------------------------------------------

void Tokenizer::bpe_merge(std::vector<std::string>& symbols) const {
    while (symbols.size() >= 2) {
        float  best_pri = -std::numeric_limits<float>::infinity();
        size_t best_pos = SIZE_MAX;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            float pri = -std::numeric_limits<float>::infinity();

            if (mode_ == Mode::SPM) {
                // Priority = score of the merged token (higher = merge first).
                std::string merged = symbols[i] + symbols[i + 1];
                auto it = token_to_id_.find(merged);
                if (it != token_to_id_.end())
                    pri = scores_[it->second];
            } else {
                // Priority = −rank in merge list (rank 0 → pri 0, rank 1 → pri −1).
                std::string key;
                key.reserve(symbols[i].size() + 1 + symbols[i + 1].size());
                key += symbols[i];
                key += ' ';
                key += symbols[i + 1];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end())
                    pri = -static_cast<float>(it->second);
            }

            if (pri > best_pri) {
                best_pri = pri;
                best_pos = i;
            }
        }

        if (best_pos == SIZE_MAX) break;

        symbols[best_pos] += symbols[best_pos + 1];
        symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_pos + 1));
    }
}

// ---------------------------------------------------------------------------
// symbols_to_ids — final symbol → token ID mapping with byte fallback
// ---------------------------------------------------------------------------

auto Tokenizer::symbols_to_ids(const std::vector<std::string>& symbols) const
    -> std::vector<uint32_t>
{
    std::vector<uint32_t> out;
    out.reserve(symbols.size());

    for (const auto& sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Already a byte-fallback token that isn't in vocab — skip.
        if (parse_byte_token(sym) >= 0)
            continue;
        // Last resort: decompose into bytes.
        for (unsigned char byte : sym) {
            auto fb = byte_to_fallback_str(byte);
            auto bit = token_to_id_.find(fb);
            if (bit != token_to_id_.end())
                out.push_back(bit->second);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

auto Tokenizer::decode(const std::vector<uint32_t>& tokens) const -> std::string {
    std::string raw;
    for (uint32_t id : tokens) {
        if (id == bos_id_ || id == eos_id_) continue;
        if (id >= static_cast<uint32_t>(vocab_.size())) continue;

        const std::string& piece = vocab_[id];
        int bv = parse_byte_token(piece);
        if (bv >= 0) {
            raw += static_cast<char>(static_cast<unsigned char>(bv));
        } else if (mode_ == Mode::BPE) {
            raw += decode_bpe_piece(piece);
        } else {
            raw += piece;
        }
    }

    if (mode_ == Mode::SPM) {
        auto result = replace_sp_with_space(raw);
        if (add_space_prefix_ && !result.empty() && result[0] == ' ')
            result.erase(result.begin());
        return result;
    }
    return raw;
}

// ---------------------------------------------------------------------------
// token_to_text
// ---------------------------------------------------------------------------

auto Tokenizer::token_to_text(uint32_t token_id) const -> std::string_view {
    if (token_id >= static_cast<uint32_t>(vocab_.size())) return {};
    return vocab_[token_id];
}

}  // namespace mugen

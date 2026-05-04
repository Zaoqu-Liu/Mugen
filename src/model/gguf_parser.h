#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "model/ggml_types.h"

namespace mugen {

/// Parsed metadata for a single tensor stored in a GGUF file.
struct TensorInfo {
    std::string          name;
    std::vector<int64_t> dimensions;
    GGMLType             type;
    uint64_t             offset;       // offset inside the data blob
    uint64_t             file_offset;  // absolute byte offset in the file
    size_t               byte_size;    // total data bytes for this tensor
    uint32_t             shard_index = 0;  // which shard file (0 for single-file)
};

/// Structured metadata extracted from GGUF key-value pairs.
struct GGUFMetadata {
    uint32_t    version        = 0;
    uint64_t    alignment      = 32;    // GGUF_DEFAULT_ALIGNMENT
    std::string arch;                   // general.architecture
    std::string name;                   // general.name
    uint32_t    n_layers       = 0;     // <arch>.block_count
    uint32_t    n_experts      = 0;     // <arch>.expert_count
    uint32_t    n_experts_used = 0;     // <arch>.expert_used_count
    uint32_t    n_heads        = 0;     // <arch>.attention.head_count
    uint32_t    n_kv_heads     = 0;     // <arch>.attention.head_count_kv
    uint32_t    vocab_size     = 0;     // <arch>.vocab_size
    uint32_t    context_length = 0;     // <arch>.context_length
    uint32_t    embedding_dim  = 0;     // <arch>.embedding_length

    std::unordered_map<std::string, std::string> raw_string_kv;
    std::unordered_map<std::string, int64_t>     raw_int_kv;
    std::unordered_map<std::string, double>      raw_float_kv;

    std::unordered_map<std::string, std::vector<std::string>> raw_string_array_kv;
    std::unordered_map<std::string, std::vector<float>>       raw_float_array_kv;
};

/// Zero-copy GGUF file parser.
///
/// Reads the GGUF header, metadata key-value pairs, and tensor descriptors
/// using sequential file I/O.  Tensor *data* is never read—callers are
/// expected to mmap the file and use the offsets recorded in TensorInfo.
class GGUFParser {
public:
    /// Parse the header, metadata, and tensor info from a GGUF file.
    /// The tensor data blob is not read.
    static auto parse(const std::filesystem::path& path)
        -> std::expected<GGUFParser, std::string>;

    /// Parse a sharded (split) GGUF model from multiple shard files.
    /// Metadata is taken from the first shard; tensors are merged from all.
    static auto parse_sharded(const std::vector<std::filesystem::path>& shard_paths)
        -> std::expected<GGUFParser, std::string>;

    /// Detect all shard paths if the given path is part of a split GGUF.
    /// Returns a single-element vector for non-split files.
    /// Pattern: <base>-NNNNN-of-MMMMM.gguf
    static auto detect_shards(const std::filesystem::path& path)
        -> std::vector<std::filesystem::path>;

    auto metadata()    const -> const GGUFMetadata&          { return metadata_; }
    auto tensors()     const -> const std::vector<TensorInfo>& { return tensors_; }
    auto data_offset() const -> uint64_t                     { return data_offset_; }
    auto file_size()   const -> uint64_t                     { return file_size_; }

    /// Look up a tensor by exact name.  Returns nullptr if not found.
    auto tensor_by_name(std::string_view name) const -> const TensorInfo*;

    /// Whether this model uses Mixture-of-Experts (expert_count > 1).
    auto is_moe() const -> bool { return metadata_.n_experts > 1; }

    /// Number of experts.  Returns 0 for dense models.
    auto expert_count() const -> uint32_t { return metadata_.n_experts; }

    /// Collect the tensor names that belong to a specific expert in a given
    /// layer.  Matches the llama.cpp naming convention:
    ///     blk.<layer>.ffn_*_exps.<expert_id>.weight
    auto expert_tensor_names(uint32_t layer, uint32_t expert_id) const
        -> std::vector<std::string>;

    ~GGUFParser() = default;
    GGUFParser(GGUFParser&&) noexcept = default;
    GGUFParser& operator=(GGUFParser&&) noexcept = default;
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

private:
    GGUFParser() = default;

    GGUFMetadata                           metadata_;
    std::vector<TensorInfo>                tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    uint64_t                               data_offset_ = 0;
    uint64_t                               file_size_   = 0;
};

}  // namespace mugen

#include "model/gguf_parser.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

namespace mugen {

// ---------------------------------------------------------------------------
// Internal helpers – BinaryReader and GGUF value type handling
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kGGUFMagic        = 0x46554747;  // "GGUF" as little-endian u32
constexpr uint64_t kMaxStringLen     = 1ULL << 20;   // 1 MiB per string
constexpr uint64_t kMaxKVCount       = 1'000'000;
constexpr uint64_t kMaxTensorCount   = 100'000'000;
constexpr uint64_t kMaxArrayCount    = 100'000'000;
constexpr uint32_t kMaxDims          = 8;

// GGUF metadata value types (distinct from GGMLType which describes tensor data).
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// RAII wrapper for FILE*.
struct FileHandle {
    FILE* fp = nullptr;

    FileHandle() = default;
    explicit FileHandle(FILE* f) : fp(f) {}
    ~FileHandle() { if (fp) std::fclose(fp); }

    FileHandle(FileHandle&& o) noexcept : fp(std::exchange(o.fp, nullptr)) {}
    FileHandle& operator=(FileHandle&& o) noexcept {
        if (this != &o) { if (fp) std::fclose(fp); fp = std::exchange(o.fp, nullptr); }
        return *this;
    }
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    explicit operator bool() const { return fp != nullptr; }
};

// Sequential binary reader with a sticky failure flag.  Once any read
// fails, all subsequent operations become no-ops returning zero/empty values.
struct BinaryReader {
    FILE* fp     = nullptr;
    bool  failed = false;

    void read_raw(void* dst, size_t n) {
        if (failed) return;
        if (std::fread(dst, 1, n, fp) != n) failed = true;
    }

    auto u8()  -> uint8_t  { uint8_t  v = 0; read_raw(&v, 1); return v; }
    auto i8()  -> int8_t   { int8_t   v = 0; read_raw(&v, 1); return v; }
    auto u16() -> uint16_t { uint16_t v = 0; read_raw(&v, 2); return v; }
    auto i16() -> int16_t  { int16_t  v = 0; read_raw(&v, 2); return v; }
    auto u32() -> uint32_t { uint32_t v = 0; read_raw(&v, 4); return v; }
    auto i32() -> int32_t  { int32_t  v = 0; read_raw(&v, 4); return v; }
    auto u64() -> uint64_t { uint64_t v = 0; read_raw(&v, 8); return v; }
    auto i64() -> int64_t  { int64_t  v = 0; read_raw(&v, 8); return v; }

    auto f32() -> float  { float  v = 0; read_raw(&v, 4); return v; }
    auto f64() -> double { double v = 0; read_raw(&v, 8); return v; }

    auto str(uint64_t max_len = kMaxStringLen) -> std::string {
        uint64_t len = u64();
        if (failed || len > max_len) { failed = true; return {}; }
        std::string s(static_cast<size_t>(len), '\0');
        if (len > 0) read_raw(s.data(), static_cast<size_t>(len));
        return s;
    }

    void skip(uint64_t n) {
        if (failed) return;
        if (fseeko(fp, static_cast<off_t>(n), SEEK_CUR) != 0) failed = true;
    }

    auto pos() const -> uint64_t {
        return static_cast<uint64_t>(ftello(fp));
    }
};

// Byte sizes of fixed-width GGUF value types.  Returns 0 for variable-width.
auto gguf_value_fixed_size(uint32_t raw) -> size_t {
    switch (raw) {
        case 0: case 1: case 7: return 1;  // UINT8, INT8, BOOL
        case 2: case 3:         return 2;  // UINT16, INT16
        case 4: case 5: case 6: return 4;  // UINT32, INT32, FLOAT32
        case 10: case 11: case 12: return 8;  // UINT64, INT64, FLOAT64
        default: return 0;
    }
}

// Skip over a GGUF array value (element_type + count + elements).
void skip_array(BinaryReader& r) {
    uint32_t elem_type = r.u32();
    uint64_t count     = r.u64();
    if (r.failed || count > kMaxArrayCount) { r.failed = true; return; }

    size_t fixed = gguf_value_fixed_size(elem_type);
    if (fixed > 0) {
        r.skip(count * fixed);
    } else if (elem_type == static_cast<uint32_t>(GGUFValueType::STRING)) {
        for (uint64_t i = 0; i < count && !r.failed; ++i) r.str();
    } else if (elem_type == static_cast<uint32_t>(GGUFValueType::ARRAY)) {
        for (uint64_t i = 0; i < count && !r.failed; ++i) skip_array(r);
    } else {
        r.failed = true;
    }
}

// Read a GGUF array value, storing STRING and FLOAT32 arrays into metadata.
// All other element types are skipped as before.
void read_or_skip_array(BinaryReader& r, GGUFMetadata& meta, const std::string& key) {
    uint32_t elem_type = r.u32();
    uint64_t count     = r.u64();
    if (r.failed || count > kMaxArrayCount) { r.failed = true; return; }

    if (elem_type == static_cast<uint32_t>(GGUFValueType::STRING)) {
        auto& vec = meta.raw_string_array_kv[key];
        vec.reserve(static_cast<size_t>(std::min(count, uint64_t{4'000'000})));
        for (uint64_t i = 0; i < count && !r.failed; ++i)
            vec.push_back(r.str());
    } else if (elem_type == static_cast<uint32_t>(GGUFValueType::FLOAT32)) {
        auto& vec = meta.raw_float_array_kv[key];
        vec.reserve(static_cast<size_t>(std::min(count, uint64_t{4'000'000})));
        for (uint64_t i = 0; i < count && !r.failed; ++i)
            vec.push_back(r.f32());
    } else {
        size_t fixed = gguf_value_fixed_size(elem_type);
        if (fixed > 0) {
            r.skip(count * fixed);
        } else if (elem_type == static_cast<uint32_t>(GGUFValueType::ARRAY)) {
            for (uint64_t i = 0; i < count && !r.failed; ++i) skip_array(r);
        } else {
            r.failed = true;
        }
    }
}

// Read a single KV value and store it into the appropriate raw map.
void read_kv_pair(BinaryReader& r, GGUFMetadata& meta) {
    std::string key = r.str();
    if (r.failed) return;

    auto vtype = static_cast<GGUFValueType>(r.u32());
    if (r.failed) return;

    switch (vtype) {
        case GGUFValueType::UINT8:   meta.raw_int_kv[key] = r.u8();  break;
        case GGUFValueType::INT8:    meta.raw_int_kv[key] = r.i8();  break;
        case GGUFValueType::UINT16:  meta.raw_int_kv[key] = r.u16(); break;
        case GGUFValueType::INT16:   meta.raw_int_kv[key] = r.i16(); break;
        case GGUFValueType::UINT32:  meta.raw_int_kv[key] = r.u32(); break;
        case GGUFValueType::INT32:   meta.raw_int_kv[key] = r.i32(); break;
        case GGUFValueType::UINT64:  meta.raw_int_kv[key] = static_cast<int64_t>(r.u64()); break;
        case GGUFValueType::INT64:   meta.raw_int_kv[key] = r.i64(); break;
        case GGUFValueType::FLOAT32: meta.raw_float_kv[key] = r.f32(); break;
        case GGUFValueType::FLOAT64: meta.raw_float_kv[key] = r.f64(); break;
        case GGUFValueType::BOOL:    meta.raw_int_kv[key] = r.u8() ? 1 : 0; break;
        case GGUFValueType::STRING:  meta.raw_string_kv[key] = r.str(); break;
        case GGUFValueType::ARRAY:   read_or_skip_array(r, meta, key); break;
        default:                     r.failed = true; break;
    }
}

// Extract well-known fields from the raw KV maps into the typed metadata.
void extract_structured_metadata(GGUFMetadata& m) {
    auto get_str = [&](const std::string& key) -> std::string {
        auto it = m.raw_string_kv.find(key);
        return it != m.raw_string_kv.end() ? it->second : std::string{};
    };
    auto get_u32 = [&](const std::string& key) -> uint32_t {
        auto it = m.raw_int_kv.find(key);
        return it != m.raw_int_kv.end() ? static_cast<uint32_t>(it->second) : 0u;
    };

    m.arch = get_str("general.architecture");
    m.name = get_str("general.name");

    auto align_it = m.raw_int_kv.find("general.alignment");
    if (align_it != m.raw_int_kv.end() && align_it->second > 0) {
        m.alignment = static_cast<uint64_t>(align_it->second);
    }

    const std::string& a = m.arch;
    if (a.empty()) return;

    m.n_layers       = get_u32(a + ".block_count");
    m.n_experts      = get_u32(a + ".expert_count");
    m.n_experts_used = get_u32(a + ".expert_used_count");
    m.n_heads        = get_u32(a + ".attention.head_count");
    m.n_kv_heads     = get_u32(a + ".attention.head_count_kv");
    m.vocab_size     = get_u32(a + ".vocab_size");
    m.context_length = get_u32(a + ".context_length");
    m.embedding_dim  = get_u32(a + ".embedding_length");

    if (m.vocab_size == 0) {
        auto it = m.raw_string_array_kv.find("tokenizer.ggml.tokens");
        if (it != m.raw_string_array_kv.end() && !it->second.empty())
            m.vocab_size = static_cast<uint32_t>(it->second.size());
    }
}

auto align_up(uint64_t offset, uint64_t alignment) -> uint64_t {
    uint64_t r = offset % alignment;
    return r == 0 ? offset : offset + alignment - r;
}

auto byteswap32(uint32_t v) -> uint32_t {
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8)
         | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF);
}

}  // namespace

// ---------------------------------------------------------------------------
// GGUFParser::parse
// ---------------------------------------------------------------------------

auto GGUFParser::parse(const std::filesystem::path& path)
    -> std::expected<GGUFParser, std::string>
{
    FileHandle fh{std::fopen(path.c_str(), "rb")};
    if (!fh) return std::unexpected("cannot open file: " + path.string());

    // File size via seek-to-end.
    if (fseeko(fh.fp, 0, SEEK_END) != 0)
        return std::unexpected("cannot seek to end of file");
    uint64_t file_size = static_cast<uint64_t>(ftello(fh.fp));
    if (fseeko(fh.fp, 0, SEEK_SET) != 0)
        return std::unexpected("cannot seek to start of file");

    BinaryReader r{fh.fp, false};

    // --- header -----------------------------------------------------------
    uint32_t magic = r.u32();
    if (r.failed)
        return std::unexpected("file too small to contain GGUF header");
    if (magic != kGGUFMagic)
        return std::unexpected("invalid GGUF magic number");

    uint32_t version = r.u32();
    if (r.failed)
        return std::unexpected("failed to read GGUF version");

    // Detect big-endian: a byte-swapped version of 2 or 3 would be huge.
    if (version > 1000) {
        uint32_t swapped = byteswap32(version);
        if (swapped >= 2 && swapped <= 3)
            return std::unexpected("big-endian GGUF files are not supported");
        return std::unexpected("unsupported GGUF version: " + std::to_string(version));
    }
    if (version < 2 || version > 3)
        return std::unexpected("unsupported GGUF version: " + std::to_string(version));

    uint64_t n_tensors = r.u64();
    uint64_t n_kv      = r.u64();
    if (r.failed)
        return std::unexpected("failed to read tensor/kv counts");
    if (n_tensors > kMaxTensorCount)
        return std::unexpected("tensor count exceeds limit: " + std::to_string(n_tensors));
    if (n_kv > kMaxKVCount)
        return std::unexpected("kv pair count exceeds limit: " + std::to_string(n_kv));

    // --- key-value pairs --------------------------------------------------
    GGUFParser parser;
    parser.metadata_.version = version;
    parser.file_size_ = file_size;

    for (uint64_t i = 0; i < n_kv; ++i) {
        read_kv_pair(r, parser.metadata_);
        if (r.failed)
            return std::unexpected("failed to read kv pair at index " + std::to_string(i));
    }

    extract_structured_metadata(parser.metadata_);

    // --- tensor descriptors -----------------------------------------------
    parser.tensors_.reserve(static_cast<size_t>(n_tensors));

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        ti.name = r.str();
        if (r.failed)
            return std::unexpected("failed to read tensor name at index " + std::to_string(i));

        uint32_t n_dims = r.u32();
        if (r.failed || n_dims > kMaxDims)
            return std::unexpected("invalid dimension count for tensor: " + ti.name);

        ti.dimensions.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d)
            ti.dimensions[d] = static_cast<int64_t>(r.u64());

        uint32_t raw_type = r.u32();
        if (!ggml_type_is_valid(raw_type))
            return std::unexpected("unknown ggml type " + std::to_string(raw_type)
                                  + " for tensor: " + ti.name);
        ti.type = static_cast<GGMLType>(raw_type);

        ti.offset = r.u64();
        if (r.failed)
            return std::unexpected("failed to read tensor info for: " + ti.name);

        ti.byte_size = ggml_tensor_byte_size(
            ti.type, std::span<const int64_t>(ti.dimensions));
        if (ti.byte_size == 0 && !ti.dimensions.empty())
            return std::unexpected("invalid dimensions for tensor: " + ti.name);

        parser.tensors_.push_back(std::move(ti));
    }

    // --- compute data offset (aligned) ------------------------------------
    uint64_t alignment = parser.metadata_.alignment;
    parser.data_offset_ = align_up(r.pos(), alignment);

    // Fill in absolute file offsets and build the name index.
    for (size_t i = 0; i < parser.tensors_.size(); ++i) {
        auto& ti = parser.tensors_[i];
        ti.file_offset = parser.data_offset_ + ti.offset;

        if (ti.file_offset + ti.byte_size > file_size)
            return std::unexpected("tensor data extends past end of file: " + ti.name);

        parser.tensor_index_[ti.name] = i;
    }

    return parser;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

auto GGUFParser::tensor_by_name(std::string_view name) const -> const TensorInfo* {
    auto it = tensor_index_.find(std::string(name));
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

auto GGUFParser::expert_tensor_names(uint32_t layer, uint32_t expert_id) const
    -> std::vector<std::string>
{
    // Matches the llama.cpp convention:
    //   blk.<layer>.ffn_<kind>_exps.<expert_id>.weight
    const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
    const std::string suffix = "." + std::to_string(expert_id) + ".weight";

    std::vector<std::string> result;
    for (const auto& ti : tensors_) {
        if (ti.name.size() > prefix.size() + suffix.size()
            && ti.name.compare(0, prefix.size(), prefix) == 0
            && ti.name.compare(ti.name.size() - suffix.size(),
                               suffix.size(), suffix) == 0)
        {
            result.push_back(ti.name);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// detect_shards — find all shard files for a split GGUF
// ---------------------------------------------------------------------------

auto GGUFParser::detect_shards(const std::filesystem::path& path)
    -> std::vector<std::filesystem::path>
{
    namespace fs = std::filesystem;
    std::string filename = path.filename().string();

    // Pattern: <base>-NNNNN-of-MMMMM.gguf
    // Find the "-NNNNN-of-MMMMM.gguf" suffix
    const std::string suffix = ".gguf";
    if (filename.size() < suffix.size() ||
        filename.substr(filename.size() - suffix.size()) != suffix) {
        return {path};
    }

    std::string stem = filename.substr(0, filename.size() - suffix.size());

    // Look for "-NNNNN-of-MMMMM" pattern at the end of stem
    auto of_pos = stem.rfind("-of-");
    if (of_pos == std::string::npos || of_pos < 6) return {path};

    std::string total_str = stem.substr(of_pos + 4);
    std::string before_of = stem.substr(0, of_pos);
    auto dash_pos = before_of.rfind('-');
    if (dash_pos == std::string::npos) return {path};

    std::string index_str = before_of.substr(dash_pos + 1);
    std::string base = before_of.substr(0, dash_pos);

    // Validate: index and total must be numeric
    for (char c : index_str) if (!std::isdigit(c)) return {path};
    for (char c : total_str) if (!std::isdigit(c)) return {path};
    if (index_str.empty() || total_str.empty()) return {path};

    int total = std::stoi(total_str);
    if (total <= 1) return {path};

    size_t pad_width = index_str.size();
    fs::path dir = path.parent_path();

    std::vector<fs::path> shards;
    shards.reserve(static_cast<size_t>(total));
    for (int i = 1; i <= total; i++) {
        std::string idx = std::to_string(i);
        while (idx.size() < pad_width) idx = "0" + idx;
        std::string shard_name = base + "-" + idx + "-of-" + total_str + suffix;
        fs::path shard_path = dir / shard_name;
        if (!fs::exists(shard_path)) return {path};
        shards.push_back(shard_path);
    }

    return shards;
}

// ---------------------------------------------------------------------------
// parse_sharded — parse metadata from first shard, tensors from all
// ---------------------------------------------------------------------------

auto GGUFParser::parse_sharded(const std::vector<std::filesystem::path>& shard_paths)
    -> std::expected<GGUFParser, std::string>
{
    if (shard_paths.empty())
        return std::unexpected(std::string("no shard paths provided"));

    if (shard_paths.size() == 1)
        return parse(shard_paths[0]);

    // Parse first shard for full metadata
    auto first = parse(shard_paths[0]);
    if (!first) return std::unexpected(first.error());

    GGUFParser merged = std::move(*first);

    // Tag first shard's tensors with shard_index = 0
    for (auto& ti : merged.tensors_)
        ti.shard_index = 0;

    // Parse remaining shards and merge tensors
    for (size_t s = 1; s < shard_paths.size(); s++) {
        auto shard = parse(shard_paths[s]);
        if (!shard) return std::unexpected(
            "failed to parse shard " + shard_paths[s].string() + ": " + shard.error());

        for (auto& ti : shard->tensors_) {
            ti.shard_index = static_cast<uint32_t>(s);
            merged.tensor_index_[ti.name] = merged.tensors_.size();
            merged.tensors_.push_back(std::move(ti));
        }
    }

    return merged;
}

}  // namespace mugen

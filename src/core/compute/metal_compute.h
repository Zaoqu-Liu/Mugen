#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mugen {

class MetalCompute {
public:
    static auto create() -> std::expected<std::unique_ptr<MetalCompute>, std::string>;

    ~MetalCompute();

    MetalCompute(const MetalCompute&) = delete;
    MetalCompute& operator=(const MetalCompute&) = delete;
    MetalCompute(MetalCompute&&) noexcept;
    MetalCompute& operator=(MetalCompute&&) noexcept;

    auto device_name() const -> std::string;
    auto max_buffer_length() const -> size_t;
    auto max_threadgroup_size() const -> size_t;
    auto recommended_working_set() const -> size_t;

    auto compile_library(const std::string& source,
                         const std::string& label = "") -> std::expected<void*, std::string>;

    auto get_function(void* library, const std::string& name) -> std::expected<void*, std::string>;

    auto create_pipeline(void* function) -> std::expected<void*, std::string>;

    auto create_buffer(size_t bytes, bool gpu_only = false) -> void*;
    auto create_buffer_from_data(const void* data, size_t bytes) -> void*;
    auto create_buffer_nocopy(void* data, size_t bytes) -> void*;

    struct DispatchParams {
        void* pipeline = nullptr;
        std::vector<void*> buffers;
        std::vector<std::pair<const void*, size_t>> constants;
        std::array<size_t, 3> grid_size  = {1, 1, 1};
        std::array<size_t, 3> group_size = {1, 1, 1};
        std::vector<uint8_t> const_data;  // owned storage for constants (safe after move)
        bool use_dispatch_threadgroups = false;

        DispatchParams() = default;
        DispatchParams(DispatchParams&&) = default;
        DispatchParams& operator=(DispatchParams&&) = default;

        DispatchParams(const DispatchParams& o)
            : pipeline(o.pipeline), buffers(o.buffers), constants(o.constants),
              grid_size(o.grid_size), group_size(o.group_size),
              const_data(o.const_data),
              use_dispatch_threadgroups(o.use_dispatch_threadgroups)
        {
            fixup_const_ptrs(o.const_data.data());
        }

        DispatchParams& operator=(const DispatchParams& o) {
            if (this != &o) {
                pipeline = o.pipeline; buffers = o.buffers;
                constants = o.constants; grid_size = o.grid_size;
                group_size = o.group_size; const_data = o.const_data;
                use_dispatch_threadgroups = o.use_dispatch_threadgroups;
                fixup_const_ptrs(o.const_data.data());
            }
            return *this;
        }

    private:
        void fixup_const_ptrs(const uint8_t* old_base) {
            if (const_data.empty() || !old_base) return;
            for (auto& [ptr, sz] : constants) {
                auto off = static_cast<const uint8_t*>(ptr) - old_base;
                if (off >= 0 && static_cast<size_t>(off) < const_data.size())
                    ptr = const_data.data() + off;
            }
        }
    };

    auto dispatch_sync(const DispatchParams& params) -> std::expected<double, std::string>;

    auto dispatch_batch_sync(const std::vector<DispatchParams>& batch) -> std::expected<double, std::string>;

    auto dispatch_chain_sync(const std::vector<std::vector<DispatchParams>>& groups)
        -> std::expected<double, std::string>;

    auto buffer_contents(void* buffer) const -> void*;
    auto read_buffer(void* buffer, void* dst, size_t bytes) const -> bool;

private:
    MetalCompute();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mugen

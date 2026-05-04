#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "core/compute/metal_compute.h"

#include <cassert>
#include <cstdio>

namespace mugen {

// ─────────────────────────────────────────────────────────────────────────────
// Impl: ObjC++ bridge holding all Metal objects
// ─────────────────────────────────────────────────────────────────────────────

struct MetalCompute::Impl {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;

    Impl() : device(nil), queue(nil) {}

    ~Impl() {
        @autoreleasepool {
            queue = nil;
            device = nil;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction / move
// ─────────────────────────────────────────────────────────────────────────────

MetalCompute::MetalCompute() : impl_(std::make_unique<Impl>()) {}

MetalCompute::~MetalCompute() = default;

MetalCompute::MetalCompute(MetalCompute&&) noexcept = default;
MetalCompute& MetalCompute::operator=(MetalCompute&&) noexcept = default;

auto MetalCompute::create() -> std::expected<std::unique_ptr<MetalCompute>, std::string> {
    @autoreleasepool {
        auto mc = std::unique_ptr<MetalCompute>(new MetalCompute());

        mc->impl_->device = MTLCreateSystemDefaultDevice();
        if (!mc->impl_->device) {
            return std::unexpected("No Metal-capable GPU found");
        }

        mc->impl_->queue = [mc->impl_->device newCommandQueue];
        if (!mc->impl_->queue) {
            return std::unexpected("Failed to create Metal command queue");
        }

        return mc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device info
// ─────────────────────────────────────────────────────────────────────────────

auto MetalCompute::device_name() const -> std::string {
    @autoreleasepool {
        return std::string([[impl_->device name] UTF8String]);
    }
}

auto MetalCompute::max_buffer_length() const -> size_t {
    return static_cast<size_t>([impl_->device maxBufferLength]);
}

auto MetalCompute::max_threadgroup_size() const -> size_t {
    // Conservative default; actual limit depends on pipeline state
    return 1024;
}

auto MetalCompute::recommended_working_set() const -> size_t {
    return static_cast<size_t>([impl_->device recommendedMaxWorkingSetSize]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader compilation (runtime)
// ─────────────────────────────────────────────────────────────────────────────

auto MetalCompute::compile_library(const std::string& source,
                                   const std::string& label)
    -> std::expected<void*, std::string>
{
    @autoreleasepool {
        NSError* error = nil;

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_0;

        NSString* src = [NSString stringWithUTF8String:source.c_str()];

        id<MTLLibrary> lib = [impl_->device newLibraryWithSource:src
                                                         options:opts
                                                           error:&error];
        if (!lib) {
            std::string msg = "Metal shader compilation failed";
            if (!label.empty()) {
                msg += " [" + label + "]";
            }
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            return std::unexpected(msg);
        }

        if (!label.empty()) {
            lib.label = [NSString stringWithUTF8String:label.c_str()];
        }

        // Transfer ownership: caller receives a retained pointer
        return (__bridge_retained void*)lib;
    }
}

auto MetalCompute::get_function(void* library, const std::string& name)
    -> std::expected<void*, std::string>
{
    @autoreleasepool {
        if (!library) {
            return std::unexpected("Null library pointer");
        }

        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
        NSString* funcName = [NSString stringWithUTF8String:name.c_str()];

        id<MTLFunction> func = [lib newFunctionWithName:funcName];
        if (!func) {
            return std::unexpected("Function '" + name + "' not found in library");
        }

        return (__bridge_retained void*)func;
    }
}

auto MetalCompute::create_pipeline(void* function)
    -> std::expected<void*, std::string>
{
    @autoreleasepool {
        if (!function) {
            return std::unexpected("Null function pointer");
        }

        NSError* error = nil;
        id<MTLFunction> func = (__bridge id<MTLFunction>)function;

        id<MTLComputePipelineState> pso =
            [impl_->device newComputePipelineStateWithFunction:func error:&error];

        if (!pso) {
            std::string msg = "Pipeline creation failed";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            return std::unexpected(msg);
        }

        return (__bridge_retained void*)pso;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer creation
// ─────────────────────────────────────────────────────────────────────────────

auto MetalCompute::create_buffer(size_t bytes, bool gpu_only) -> void* {
    @autoreleasepool {
        MTLResourceOptions opts = gpu_only
            ? MTLResourceStorageModePrivate
            : MTLResourceStorageModeShared;

        id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes options:opts];
        if (!buf) return nullptr;
        return (__bridge_retained void*)buf;
    }
}

auto MetalCompute::create_buffer_from_data(const void* data, size_t bytes) -> void* {
    @autoreleasepool {
        id<MTLBuffer> buf = [impl_->device newBufferWithBytes:data
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared];
        if (!buf) return nullptr;
        return (__bridge_retained void*)buf;
    }
}

auto MetalCompute::create_buffer_nocopy(void* data, size_t bytes) -> void* {
    @autoreleasepool {
        id<MTLBuffer> buf = [impl_->device newBufferWithBytesNoCopy:data
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
        if (!buf) return nullptr;
        return (__bridge_retained void*)buf;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

static void encode_dispatch(id<MTLComputeCommandEncoder> enc,
                            const MetalCompute::DispatchParams& params)
{
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)params.pipeline;
    [enc setComputePipelineState:pso];

    for (size_t i = 0; i < params.buffers.size(); ++i) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)params.buffers[i];
        [enc setBuffer:buf offset:0 atIndex:static_cast<NSUInteger>(i)];
    }

    NSUInteger buf_offset = static_cast<NSUInteger>(params.buffers.size());
    for (size_t i = 0; i < params.constants.size(); ++i) {
        [enc setBytes:params.constants[i].first
               length:params.constants[i].second
              atIndex:buf_offset + static_cast<NSUInteger>(i)];
    }

    MTLSize grid = MTLSizeMake(
        static_cast<NSUInteger>(params.grid_size[0]),
        static_cast<NSUInteger>(params.grid_size[1]),
        static_cast<NSUInteger>(params.grid_size[2])
    );
    MTLSize group = MTLSizeMake(
        static_cast<NSUInteger>(params.group_size[0]),
        static_cast<NSUInteger>(params.group_size[1]),
        static_cast<NSUInteger>(params.group_size[2])
    );

    if (params.use_dispatch_threadgroups) {
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    } else {
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
    }
}

auto MetalCompute::dispatch_sync(const DispatchParams& params)
    -> std::expected<double, std::string>
{
    @autoreleasepool {
        if (!params.pipeline) {
            return std::unexpected("Null pipeline in DispatchParams");
        }

        id<MTLCommandBuffer> cmdBuf = [impl_->queue commandBuffer];
        if (!cmdBuf) {
            return std::unexpected("Failed to create command buffer");
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (!enc) {
            return std::unexpected("Failed to create compute encoder");
        }

        encode_dispatch(enc, params);
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            std::string msg = "GPU execution error";
            if (cmdBuf.error) {
                msg += ": ";
                msg += [[cmdBuf.error localizedDescription] UTF8String];
            }
            return std::unexpected(msg);
        }

        double elapsed = [cmdBuf GPUEndTime] - [cmdBuf GPUStartTime];
        return elapsed;
    }
}

auto MetalCompute::dispatch_batch_sync(const std::vector<DispatchParams>& batch)
    -> std::expected<double, std::string>
{
    @autoreleasepool {
        if (batch.empty()) {
            return 0.0;
        }

        id<MTLCommandBuffer> cmdBuf = [impl_->queue commandBuffer];
        if (!cmdBuf) {
            return std::unexpected("Failed to create command buffer");
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (!enc) {
            return std::unexpected("Failed to create compute encoder");
        }

        for (const auto& params : batch) {
            if (!params.pipeline) {
                [enc endEncoding];
                return std::unexpected("Null pipeline in batch DispatchParams");
            }
            encode_dispatch(enc, params);
        }

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            std::string msg = "GPU batch execution error";
            if (cmdBuf.error) {
                msg += ": ";
                msg += [[cmdBuf.error localizedDescription] UTF8String];
            }
            return std::unexpected(msg);
        }

        double elapsed = [cmdBuf GPUEndTime] - [cmdBuf GPUStartTime];
        return elapsed;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dispatch_chain_sync — single CB + encoder; state tracking in encode_range
// ─────────────────────────────────────────────────────────────────────────────

auto MetalCompute::dispatch_chain_sync(const std::vector<std::vector<DispatchParams>>& groups)
    -> std::expected<double, std::string>
{
    @autoreleasepool {
        size_t n_nonempty = 0;
        for (const auto& g : groups) {
            if (!g.empty()) n_nonempty++;
        }
        if (n_nonempty == 0) return 0.0;

        // Encode groups[from, to) into enc with state tracking.
        // Metal encoder retains pipeline/buffer bindings across dispatches;
        // we skip redundant setComputePipelineState / setBuffer when unchanged.
        static constexpr size_t kMaxSlots = 31;
        auto encode_range = [&groups](id<MTLComputeCommandEncoder> enc,
                                      size_t from, size_t to) -> const char* {
            void* last_pso = nullptr;
            void* last_buf[kMaxSlots] = {};
            bool prev = false;

            for (size_t gi = from; gi < to; ++gi) {
                const auto& group = groups[gi];
                if (group.empty()) continue;
                if (prev) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                for (const auto& p : group) {
                    if (!p.pipeline) {
                        [enc endEncoding];
                        return "Null pipeline in chain DispatchParams";
                    }
                    if (p.pipeline != last_pso) {
                        [enc setComputePipelineState:
                            (__bridge id<MTLComputePipelineState>)p.pipeline];
                        last_pso = p.pipeline;
                    }
                    const size_t nb = p.buffers.size();
                    for (size_t i = 0; i < nb; ++i) {
                        if (p.buffers[i] != last_buf[i]) {
                            [enc setBuffer:(__bridge id<MTLBuffer>)p.buffers[i]
                                    offset:0
                                   atIndex:static_cast<NSUInteger>(i)];
                            last_buf[i] = p.buffers[i];
                        }
                    }
                    const NSUInteger boff = static_cast<NSUInteger>(nb);
                    for (size_t i = 0; i < p.constants.size(); ++i) {
                        const NSUInteger idx = boff + static_cast<NSUInteger>(i);
                        [enc setBytes:p.constants[i].first
                               length:p.constants[i].second
                              atIndex:idx];
                        if (idx < kMaxSlots) last_buf[idx] = nullptr;
                    }
                    MTLSize gs = MTLSizeMake(
                        static_cast<NSUInteger>(p.grid_size[0]),
                        static_cast<NSUInteger>(p.grid_size[1]),
                        static_cast<NSUInteger>(p.grid_size[2]));
                    MTLSize tg = MTLSizeMake(
                        static_cast<NSUInteger>(p.group_size[0]),
                        static_cast<NSUInteger>(p.group_size[1]),
                        static_cast<NSUInteger>(p.group_size[2]));
                    if (p.use_dispatch_threadgroups) {
                        [enc dispatchThreadgroups:gs threadsPerThreadgroup:tg];
                    } else {
                        [enc dispatchThreads:gs threadsPerThreadgroup:tg];
                    }
                }
                prev = true;
            }
            return nullptr;
        };

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        if (!cb) return std::unexpected("Failed to create command buffer");
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return std::unexpected("Failed to create compute encoder");
        if (auto err = encode_range(enc, 0, groups.size()))
            return std::unexpected(std::string(err));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string msg = "GPU chain execution error";
            if (cb.error) {
                msg += ": ";
                msg += [[cb.error localizedDescription] UTF8String];
            }
            return std::unexpected(msg);
        }

        return [cb GPUEndTime] - [cb GPUStartTime];
    }
}

auto MetalCompute::buffer_contents(void* buffer) const -> void* {
    auto* mtl_buf = (__bridge id<MTLBuffer>)buffer;
    return [mtl_buf contents];
}

auto MetalCompute::read_buffer(void* buffer, void* dst, size_t bytes) const -> bool {
    auto* mtl_buf = (__bridge id<MTLBuffer>)buffer;
    if (!mtl_buf) return false;
    std::memcpy(dst, [mtl_buf contents], bytes);
    return true;
}

}  // namespace mugen

# ---------------------------------------------------------------------------
# Original C++ benchmark (placeholder, depends on mugen-core)
# ---------------------------------------------------------------------------
add_executable(mugen-bench micro/bench_placeholder.cpp)

target_link_libraries(mugen-bench PRIVATE mugen-core)

target_include_directories(mugen-bench
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
)

# ---------------------------------------------------------------------------
# Standalone micro-benchmarks (no library dependencies)
# ---------------------------------------------------------------------------

add_executable(mugen-bench-ssd micro/bench_ssd_read.cpp)

add_executable(mugen-bench-madvise micro/bench_madvise.cpp)

add_executable(mugen-bench-expert-fault micro/bench_expert_fault.cpp)

# ---------------------------------------------------------------------------
# Metal micro-benchmarks (Objective-C++)
# ---------------------------------------------------------------------------

# Bench 1: MTLBuffer allocation / deallocation / heap / 2GB limit
add_executable(mugen-bench-metal-buffer
    "${CMAKE_CURRENT_SOURCE_DIR}/micro/bench_metal_buffer.mm"
)

target_compile_options(mugen-bench-metal-buffer PRIVATE -fobjc-arc)

target_link_libraries(mugen-bench-metal-buffer
    PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)

# Bench 2: UMA bandwidth (GPU read/write/copy + FP16 matmul)
add_executable(mugen-bench-uma-bandwidth
    "${CMAKE_CURRENT_SOURCE_DIR}/micro/bench_uma_bandwidth.mm"
)

target_compile_options(mugen-bench-uma-bandwidth PRIVATE -fobjc-arc)

target_link_libraries(mugen-bench-uma-bandwidth
    PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)

# Fuzz targets — requires Clang with -fsanitize=fuzzer support.
# Build with:  cmake -DMUGEN_BUILD_FUZZ=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..

set(FUZZ_COMPILE_FLAGS -fsanitize=fuzzer,address -fno-omit-frame-pointer)
set(FUZZ_LINK_FLAGS    -fsanitize=fuzzer,address)

# --- fuzz_gguf_parser ---
add_executable(fuzz_gguf_parser fuzz_gguf_parser.cpp)
target_compile_options(fuzz_gguf_parser PRIVATE ${FUZZ_COMPILE_FLAGS})
target_link_options(fuzz_gguf_parser    PRIVATE ${FUZZ_LINK_FLAGS})
target_link_libraries(fuzz_gguf_parser  PRIVATE mugen-core)
target_include_directories(fuzz_gguf_parser PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

# --- fuzz_tokenizer ---
add_executable(fuzz_tokenizer fuzz_tokenizer.cpp)
target_compile_options(fuzz_tokenizer PRIVATE ${FUZZ_COMPILE_FLAGS})
target_link_options(fuzz_tokenizer    PRIVATE ${FUZZ_LINK_FLAGS})
target_link_libraries(fuzz_tokenizer  PRIVATE mugen-core)
target_include_directories(fuzz_tokenizer PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

# --- fuzz_http_parser ---
add_executable(fuzz_http_parser fuzz_http_parser.cpp)
target_compile_options(fuzz_http_parser PRIVATE ${FUZZ_COMPILE_FLAGS})
target_link_options(fuzz_http_parser    PRIVATE ${FUZZ_LINK_FLAGS})
target_link_libraries(fuzz_http_parser  PRIVATE mugen-server-lib)
target_include_directories(fuzz_http_parser PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

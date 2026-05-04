cmake_minimum_required(VERSION 3.21)

project(mugen
    VERSION 0.1.0
    DESCRIPTION "MoE extreme inference engine for Apple Silicon"
    HOMEPAGE_URL "https://github.com/Zaoqu-Liu/Mugen"
    LANGUAGES CXX OBJCXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT APPLE)
    message(FATAL_ERROR "Mugen requires macOS with Apple Silicon")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(WARNING "Mugen is designed for Apple Silicon (arm64). Performance on x86_64 is not supported.")
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(MUGEN_ENABLE_ASAN  "Enable Address Sanitizer"  OFF)
option(MUGEN_ENABLE_TSAN  "Enable Thread Sanitizer"   OFF)
option(MUGEN_BUILD_TESTS  "Build unit tests"           ON)
option(MUGEN_BUILD_BENCH  "Build benchmarks"           ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-Werror)
endif()

if(MUGEN_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(MUGEN_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

if(MUGEN_ENABLE_ASAN AND MUGEN_ENABLE_TSAN)
    message(FATAL_ERROR "ASAN and TSAN cannot be enabled simultaneously")
endif()

# ---------------------------------------------------------------------------
# Metal shader compilation
# Requires Xcode.app (not just Command Line Tools) for the metal compiler.
# ---------------------------------------------------------------------------
set(MUGEN_HAS_METAL_COMPILER FALSE)

execute_process(
    COMMAND xcrun --find metal
    OUTPUT_VARIABLE METAL_COMPILER_PATH
    ERROR_QUIET
    RESULT_VARIABLE METAL_COMPILER_FOUND
)

if(METAL_COMPILER_FOUND EQUAL 0)
    set(MUGEN_HAS_METAL_COMPILER TRUE)
    string(STRIP "${METAL_COMPILER_PATH}" METAL_COMPILER_PATH)
    message(STATUS "Metal compiler found: ${METAL_COMPILER_PATH}")
else()
    message(WARNING
        "Metal compiler not found (requires Xcode.app, not just Command Line Tools). "
        "Shader compilation disabled -- .metallib will not be built."
    )
endif()

if(MUGEN_HAS_METAL_COMPILER)
    file(GLOB METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.metal")

    set(METAL_AIR_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal_air")
    file(MAKE_DIRECTORY "${METAL_AIR_DIR}")

    set(METAL_AIR_FILES "")

    foreach(METAL_SRC ${METAL_SOURCES})
        get_filename_component(METAL_NAME ${METAL_SRC} NAME_WE)
        set(AIR_FILE "${METAL_AIR_DIR}/${METAL_NAME}.air")

        add_custom_command(
            OUTPUT "${AIR_FILE}"
            COMMAND xcrun -sdk macosx metal
                -std=metal3.0
                -O2
                -o "${AIR_FILE}"
                -c "${METAL_SRC}"
            DEPENDS "${METAL_SRC}"
            COMMENT "Compiling Metal shader: ${METAL_NAME}.metal"
            VERBATIM
        )

        list(APPEND METAL_AIR_FILES "${AIR_FILE}")
    endforeach()

    set(METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mugen.metallib")

    if(METAL_AIR_FILES)
        add_custom_command(
            OUTPUT "${METALLIB_OUTPUT}"
            COMMAND xcrun -sdk macosx metallib
                -o "${METALLIB_OUTPUT}"
                ${METAL_AIR_FILES}
            DEPENDS ${METAL_AIR_FILES}
            COMMENT "Linking Metal library: mugen.metallib"
            VERBATIM
        )

        add_custom_target(mugen-metallib ALL DEPENDS "${METALLIB_OUTPUT}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(MUGEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MUGEN_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# mugen-core: static library containing the engine internals
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp"
)

add_library(mugen-core STATIC ${CORE_SOURCES})

# Enable ARC for Objective-C++ sources
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/core/compute/metal_compute.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/core/monitor/system_monitor.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(mugen-core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(mugen-core
    PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)

target_compile_definitions(mugen-core
    PUBLIC
        MUGEN_VERSION="${PROJECT_VERSION}"
        MUGEN_METALLIB_PATH="$<TARGET_FILE_DIR:mugen-cli>/mugen.metallib"
)

# ---------------------------------------------------------------------------
# mugen-cli: command-line inference interface
# ---------------------------------------------------------------------------
file(GLOB CLI_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.mm"
)

add_executable(mugen-cli ${CLI_SOURCES})

target_include_directories(mugen-cli PRIVATE "${PROJECT_SOURCE_DIR}/src")

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/cli/doctor.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

target_link_libraries(mugen-cli PRIVATE mugen-core mugen-server-lib)

if(TARGET mugen-metallib)
    add_dependencies(mugen-cli mugen-metallib)

    add_custom_command(TARGET mugen-cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/mugen.metallib"
            "$<TARGET_FILE_DIR:mugen-cli>/mugen.metallib"
        COMMENT "Copying mugen.metallib to output directory"
    )
endif()

# ---------------------------------------------------------------------------
# mugen-server-lib: static library for server components (shared by exe + tests)
# ---------------------------------------------------------------------------
file(GLOB SERVER_LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/server/http_server.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/server/api_routes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/server/json_utils.cpp"
)

add_library(mugen-server-lib STATIC ${SERVER_LIB_SOURCES})

target_include_directories(mugen-server-lib
    PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(mugen-server-lib PUBLIC mugen-core)

# ---------------------------------------------------------------------------
# mugen-server: HTTP API server executable
# ---------------------------------------------------------------------------
add_executable(mugen-server "${CMAKE_CURRENT_SOURCE_DIR}/server/server.cpp")

target_link_libraries(mugen-server PRIVATE mugen-server-lib)

if(TARGET mugen-metallib)
    add_dependencies(mugen-server mugen-metallib)
endif()

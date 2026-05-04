# Build each test source as its own executable and register it with CTest.

# C++ tests
file(GLOB_RECURSE TEST_SOURCES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES_CPP})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Server tests need mugen-server-lib; others just need mugen-core.
    if(TEST_NAME MATCHES "http_server")
        target_link_libraries(${TEST_NAME} PRIVATE mugen-server-lib)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE mugen-core)
    endif()

    target_include_directories(${TEST_NAME}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/src"
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Objective-C++ tests (.mm) — need ARC and Metal/Foundation frameworks
file(GLOB_RECURSE TEST_SOURCES_MM "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.mm")

foreach(TEST_SOURCE ${TEST_SOURCES_MM})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_compile_options(${TEST_NAME} PRIVATE -fobjc-arc)

    target_link_libraries(${TEST_NAME}
        PRIVATE
            mugen-core
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )

    target_include_directories(${TEST_NAME}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/src"
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

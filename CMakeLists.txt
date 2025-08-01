# 3.21 for IMPORTED_RUNTIME_ARTIFACTS
cmake_minimum_required(VERSION 3.21)

project(JaxAudioPlugins LANGUAGES C CXX)

include(FetchContent)
include(ExternalProject)
include(CTest)
include(TestData.cmake)

enable_testing()

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Enable generation of compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Consistently enable RTTI
set(WITH_RTTI "-frtti")
add_definitions(${WITH_RTTI})

# --- Enable CCACHE --- 
find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE)
    message(STATUS "Found ccache: ${CCACHE_EXECUTABLE}, enabling it for C and CXX.")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
else()
    message(WARNING "ccache executable not found. Building without ccache.")
endif()

# --- Find Packages ---

# Abseil for C++
FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20250127.1
)
FetchContent_MakeAvailable(abseil-cpp)

# Find or download Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.16.0
)
# Set variables to prevent GTest from installing and building tests globally
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # Needed on Windows, harmless elsewhere
FetchContent_MakeAvailable(googletest)

# Protocol buffers as a static library.
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.9
)
set(protobuf_INSTALL OFF)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_ALLOW_CCACHE ON)
set(protobuf_BUILD_EXAMPLES OFF)
set(protobuf_BUILD_SHARED_LIBS OFF)
set(protobuf_WITH_UTF8_VALIDATION OFF)
set(protobuf_DISABLE_RTTI OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)

include(StableHLO.cmake)
include(XLA.cmake)

# --- Fetch and build minizip-ng ---
FetchContent_Declare(
  minizip_ng
  GIT_REPOSITORY https://github.com/zlib-ng/minizip-ng.git
  GIT_TAG        4.0.10
)
set(MZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MZ_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(minizip_ng)

# --- Include directories ---

message(STATUS "Adding include dir: ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --- Targets ---

# Common libraries.
message(STATUS "Building JXAP common library")
add_library(jxap_common STATIC
    jxap/utils.h
    jxap/utils.cc
    jxap/packaged_plugin.h
    jxap/packaged_plugin.cc)
target_link_libraries(jxap_common
    PRIVATE
    absl::strings
    absl::log
    minizip

    PUBLIC
    absl::status
    absl::statusor
)
llvm_update_compile_flags(jxap_common)

# MLIR library.
message(STATUS "Building MLIR library")
add_subdirectory(jxap/mlir)

# PJRT plugin runner.
message(STATUS "Building PJRT plugin runner")
add_library(jxap_pjrt_plugin_runner STATIC
    jxap/pjrt_plugin_runner.h
    jxap/pjrt_plugin_runner.cc
)
target_link_libraries(jxap_pjrt_plugin_runner 
    PRIVATE
    absl::strings
    absl::log
    XLA::pjrt_c_api_cpu
    XLA_protos
    StablehloOps
    StablehloPasses
    jxap_mlir_pipeline

    PUBLIC
    XLA::pjrt_c_api
    jxap_common
    absl::status
    absl::statusor
)

# -- Pipewire ---
message(STATUS "Building pipewire module")
add_subdirectory(jxap/pipewire)

# --- Testing ---
message(STATUS "Building tests [working dir ${CMAKE_CURRENT_BINARY_DIR}]")

add_executable(jxap_tests
    jxap/packaged_plugin_test.cc
    jxap/mlir/pipeline_test.cc
    jxap/pjrt_plugin_runner_test.cc)
target_link_libraries(jxap_tests PRIVATE
    GTest::gtest_main
    jxap_common
    jxap_pjrt_plugin_runner
)
add_test_data_files(
    jxap_tests
    "jxap/testdata/test_plugin.jxap"
    "jxap/testdata/test_plugin.jxap-init"
    "jxap/testdata/test_plugin.jxap-update"
    "jxap/testdata/loop_plugin.jxap"
    "jxap/testdata/loop_plugin.jxap-update"
)
llvm_update_compile_flags(jxap_tests)
mlir_check_all_link_libraries(jxap_tests)

include(GoogleTest)
gtest_discover_tests(jxap_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")

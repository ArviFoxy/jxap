cmake_minimum_required(VERSION 3.15)

project(JaxAudioPlugins LANGUAGES C CXX)

include(FetchContent)
include(ExternalProject)
include(CTest)
include(TestData.cmake)

enable_testing()

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Enable generation of compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable RRTI to match LLVM
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

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

# Find PipeWire libraries using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3>=0.3.40)
pkg_check_modules(SPA REQUIRED IMPORTED_TARGET libspa-0.2)

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

# Protocol buffers
find_package(Protobuf REQUIRED)

include(StableHLO.cmake)
include(XLA.cmake)

# --- Directories ---

message(STATUS "Adding include dir: ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --- Targets ---

# PJRT plugin runner.
message(STATUS "Building MLIR passes")
add_llvm_library(jxap_mlir_passes STATIC
    jxap/mlir_passes.h
    jxap/mlir_passes.cc)
target_link_libraries(jxap_mlir_passes
    PUBLIC
    MLIRPass
)
llvm_update_compile_flags(jxap_mlir_passes)
mlir_check_all_link_libraries(jxap_mlir_passes)

message(STATUS "Building PJRT plugin runner")
add_library(jxap_pjrt_plugin_runner STATIC
    jxap/utils.h
    jxap/utils.cc
    jxap/mlir_pipeline.h
    jxap/mlir_pipeline.cc
    jxap/pjrt_plugin_runner.h
    jxap/pjrt_plugin_runner.cc
)
target_link_libraries(jxap_pjrt_plugin_runner 
    PRIVATE
    absl::strings
    absl::log

    XLA::pjrt_c_api
    XLA::pjrt_c_api_cpu
    XLA_protos

    StablehloOps
    StablehloPasses

    jxap_mlir_passes

    PUBLIC
    absl::status
    absl::statusor
)
llvm_update_compile_flags(jxap_pjrt_plugin_runner)
mlir_check_all_link_libraries(jxap_pjrt_plugin_runner)

# --- Testing ---
message(STATUS "Building tests [working dir ${CMAKE_CURRENT_BINARY_DIR}]")

add_executable(jxap_pjrt_plugin_runner_tests
    jxap/mlir_pipeline_test.cc
    jxap/pjrt_plugin_runner_test.cc)
target_link_libraries(jxap_pjrt_plugin_runner_tests PRIVATE
    GTest::gtest_main
    jxap_pjrt_plugin_runner
)
add_test_data_files(
    jxap_pjrt_plugin_runner_tests
    "jxap/testdata/test_plugin.jxap-init"
    "jxap/testdata/test_plugin.jxap-update"
)
llvm_update_compile_flags(jxap_pjrt_plugin_runner_tests)
mlir_check_all_link_libraries(jxap_pjrt_plugin_runner_tests)

include(GoogleTest)
gtest_discover_tests(jxap_pjrt_plugin_runner_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# --- Installation ---

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")

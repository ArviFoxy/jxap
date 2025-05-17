cmake_minimum_required(VERSION 3.15)

project(JaxAudioPlugins LANGUAGES C CXX)

include(FetchContent)
include(ExternalProject)
include(CTest)

enable_testing()

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Enable generation of compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# XLA - compile from source using Basel.
set(XLA_SOURCE_DIR ${CMAKE_BINARY_DIR}/xla_src)
set(XLA_INSTALL_DIR ${CMAKE_BINARY_DIR}/xla_install)
set(XLA_BAZEL_BIN ${XLA_SOURCE_DIR}/bazel-bin)
set(XLA_BAZEL_TARGETS
    "//xla/pjrt/c:pjrt_c_api.h"
    "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
    "//xla/pjrt/proto:compile_options_proto"
    "//xla/tools:hlo-opt"
)

set(XLA_PJRT_API_HDR_SRC "${XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_API_HDR_DST "${XLA_INSTALL_DIR}/include/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_CPU_HDR_SRC "${XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api_cpu.h")
set(XLA_PJRT_CPU_HDR_DST "${XLA_INSTALL_DIR}/include/xla/pjrt/c/pjrt_c_api_cpu.h")
set(XLA_PJRT_CPU_LIB_SRC "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so")
set(XLA_PJRT_CPU_LIB_DST "${XLA_INSTALL_DIR}/lib/pjrt_c_api_cpu_plugin.so")
set(XLA_HLO_OPT_SRC "${XLA_BAZEL_BIN}/xla/tools/hlo-opt")
set(XLA_HLO_OPT_DST "${XLA_INSTALL_DIR}/bin/hlo-opt")

ExternalProject_Add(xla_project
    GIT_REPOSITORY      https://github.com/openxla/xla.git
    GIT_TAG             fb601ce
    BUILD_IN_SOURCE     True
    SOURCE_DIR          ${XLA_SOURCE_DIR}
    INSTALL_DIR         ${XLA_INSTALL_DIR}

    GIT_PROGRESS        True  # Show git clone progress
    USES_TERMINAL_BUILD True  # Show build output

    CONFIGURE_COMMAND   ${XLA_SOURCE_DIR}/configure.py --backend=CPU
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}

    BUILD_COMMAND       ${CMAKE_COMMAND} -E rm -rf bazel-bin &&
                        bazel build --spawn_strategy=sandboxed ${XLA_BAZEL_TARGETS}
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}
    BUILD_BYPRODUCTS    "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
                        "${XLA_BAZEL_BIN}/xla/tools/xla-opt"

    # TODO: somehow INSTALL_BYPRODUCTS is not correctly picked up by ninja. As a workaround
    # we create the rules manually with add_custom_command.
    INSTALL_COMMAND     ""
)

add_custom_command(
    OUTPUT  "${XLA_PJRT_API_HDR_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/include/xla/pjrt/c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_PJRT_API_HDR_SRC}" "${XLA_PJRT_API_HDR_DST}"
    DEPENDS xla_project
    COMMENT "Copying XLA PJRT API header to ${XLA_PJRT_API_HDR_DST}"
    VERBATIM
)

add_custom_command(
    OUTPUT  "${XLA_PJRT_CPU_HDR_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/include/xla/pjrt/c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_PJRT_CPU_HDR_SRC}" "${XLA_PJRT_CPU_HDR_DST}"
    DEPENDS xla_project
    COMMENT "Copying XLA PJRT CPU header to ${XLA_PJRT_CPU_HDR_DST}"
    VERBATIM
)

add_custom_command(
    OUTPUT  "${XLA_PJRT_CPU_LIB_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_PJRT_CPU_LIB_SRC}" "${XLA_PJRT_CPU_LIB_DST}"
    DEPENDS "${XLA_PJRT_CPU_LIB_SRC}"
    COMMENT "Copying XLA PJRT CPU library to ${XLA_PJRT_CPU_LIB_DST}"
    VERBATIM
)

add_custom_command(
    OUTPUT  "${XLA_HLA_OPT_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_HLO_OPT_SRC}" "${XLA_HLA_OPT_DST}"
    DEPENDS "${XLA_HLO_OPT_SRC}"
    COMMENT "Copying XLA hlo-opt to ${XLA_HLA_OPT_DST}"
    VERBATIM
)

add_custom_target(xla_custom_installed_artifacts
    DEPENDS "${XLA_PJRT_API_HDR_DST}"
            "${XLA_PJRT_CPU_HDR_DST}"
            "${XLA_PJRT_CPU_LIB_DST}"
            "${XLA_HLA_OPT_DST}"
)

add_library(XLA::pjrt_c_api_cpu SHARED IMPORTED)
set_target_properties(XLA::pjrt_c_api_cpu PROPERTIES
    IMPORTED_LOCATION             "${XLA_PJRT_CPU_LIB_DST}"
    INTERFACE_INCLUDE_DIRECTORIES "${XLA_INSTALL_DIR}/include"
)
add_dependencies(XLA::pjrt_c_api_cpu xla_custom_installed_artifacts)

# --- Targets ---

# PJRT plugin runner.
message(STATUS "Building PJRT plugin runner")
add_library(pjrt_plugin_runner STATIC
    jxap/pjrt_plugin_runner.h
    jxap/pjrt_plugin_runner.cc
)
target_link_libraries(pjrt_plugin_runner 
    PRIVATE
    #PkgConfig::PIPEWIRE
    #PkgConfig::SPA
    absl::strings
    absl::log
    XLA::pjrt_c_api_cpu
    PUBLIC
    absl::status
    absl::statusor
)
target_include_directories(pjrt_plugin_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Testing ---
message(STATUS "Building Tests [working dir ${CMAKE_CURRENT_BINARY_DIR}]")

function(add_test_data_files target_name)
  # ARGN will contain all arguments passed after 'target_name'
  # These are expected to be the relative paths of the data files.
  set(copied_data_files "") # List to store all destination paths (outputs)
  foreach(relative_path ${ARGN})
      set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}")
      set(dest_path "${CMAKE_CURRENT_BINARY_DIR}/${relative_path}")
      # Ensure the source file actually exists
      if(NOT EXISTS "${source_path}")
          message(FATAL_ERROR "Test data source file not found: ${source_path} for target ${target_name}")
      endif()
      add_custom_command(
          TARGET "${target_name}"
          BYPRODUCTS "${dest_path}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  "${source_path}"
                  "${dest_path}"
          DEPENDS "${source_path}" # Depends on the original source file
          COMMENT "Copying test data ${relative_path} for target ${target_name}"
      )
  endforeach()
endfunction()

add_executable(pjrt_plugin_runner_test 
    jxap/pjrt_plugin_runner_test.cc)
target_link_libraries(pjrt_plugin_runner_test PRIVATE
    GTest::gtest_main
    pjrt_plugin_runner
)
target_include_directories(pjrt_plugin_runner_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test_data_files(
    pjrt_plugin_runner_test
    "jxap/testdata/test_plugin.jxap-init"
    "jxap/testdata/test_plugin.jxap-update"
)

include(GoogleTest)
gtest_discover_tests(pjrt_plugin_runner_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# --- Installation ---



message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")


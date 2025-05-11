cmake_minimum_required(VERSION 3.15)

project(JaxAudioPlugins LANGUAGES C CXX)

include(FetchContent)
include(ExternalProject)

# --- Options ---
option(BUILD_TESTING "Build the tests for the project" OFF)

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
)

set(XLA_PJRT_HEADER_FILE_SRC "${XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_HEADER_FILE_DST "${XLA_INSTALL_DIR}/include/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_LIB_FILE_SRC    "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so")
set(XLA_PJRT_LIB_FILE_DST    "${XLA_INSTALL_DIR}/lib/pjrt_c_api_cpu_plugin.so")

ExternalProject_Add(xla_project
    GIT_REPOSITORY      https://github.com/openxla/xla.git
    GIT_TAG             633c9ab
    SOURCE_DIR          ${XLA_SOURCE_DIR}
    BUILD_IN_SOURCE     True
    INSTALL_DIR         ${XLA_INSTALL_DIR}

    GIT_PROGRESS        True  # Show git clone progress
    USES_TERMINAL_BUILD True  # Show build output

    CONFIGURE_COMMAND   ${XLA_SOURCE_DIR}/configure.py --backend=CPU
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}

    BUILD_COMMAND       bazel build --spawn_strategy=sandboxed ${XLA_BAZEL_TARGETS}
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}
    BUILD_BYPRODUCTS    "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"

    # TODO: somehow INSTALL_BYPRODUCTS is not correctly picked up by ninja. As a workaround
    # we create the rules manually with add_custom_command.
    INSTALL_COMMAND     ""
)

add_custom_command(
    OUTPUT  "${XLA_PJRT_HEADER_FILE_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/include/xla/pjrt/c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_PJRT_HEADER_FILE_SRC}" "${XLA_PJRT_HEADER_FILE_DST}"
    DEPENDS xla_project
    COMMENT "Copying XLA PJRT header to ${XLA_PJRT_HEADER_FILE_DST}"
    VERBATIM
)

add_custom_command(
    OUTPUT  "${XLA_PJRT_LIB_FILE_DST}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${XLA_INSTALL_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XLA_PJRT_LIB_FILE_SRC}" "${XLA_PJRT_LIB_FILE_DST}"
    DEPENDS "${XLA_PJRT_LIB_FILE_SRC}"
    COMMENT "Copying XLA PJRT library to ${XLA_PJRT_LIB_FILE_DST}"
    VERBATIM
)

add_custom_target(xla_custom_installed_artifacts
    DEPENDS "${XLA_PJRT_HEADER_FILE_DST}"
            "${XLA_PJRT_LIB_FILE_DST}"
)

add_library(XLA::pjrt_c_api SHARED IMPORTED)
set_target_properties(XLA::pjrt_c_api PROPERTIES
    IMPORTED_LOCATION             "${XLA_PJRT_LIB_FILE_DST}"
    INTERFACE_INCLUDE_DIRECTORIES "${XLA_INSTALL_DIR}/include"
)
add_dependencies(XLA::pjrt_c_api xla_custom_installed_artifacts)

# --- Include Subdirectories ---

add_subdirectory(cpp/jxap)

# --- Installation ---

install(TARGETS jxap_spa_plugin LIBRARY DESTINATION lib/pipewire-0.3/spa-plugins)

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")


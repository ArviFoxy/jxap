message(STATUS "Building XLA")
set(XLA_SOURCE_DIR ${CMAKE_BINARY_DIR}/xla_src)
set(XLA_INSTALL_DIR ${CMAKE_BINARY_DIR}/xla_install)
set(XLA_BAZEL_BIN ${XLA_SOURCE_DIR}/bazel-bin)
set(XLA_BAZEL_TARGETS
    "//xla/pjrt/c:pjrt_c_api.h"
    "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
    "//xla/pjrt/proto:compile_options_proto"
)

set(XLA_PJRT_API_HDR_SRC "${XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_API_HDR_DST "${XLA_INSTALL_DIR}/include/xla/pjrt/c/pjrt_c_api.h")
set(XLA_PJRT_CPU_HDR_SRC "${XLA_SOURCE_DIR}/xla/pjrt/c/pjrt_c_api_cpu.h")
set(XLA_PJRT_CPU_HDR_DST "${XLA_INSTALL_DIR}/include/xla/pjrt/c/pjrt_c_api_cpu.h")
set(XLA_PJRT_CPU_LIB_SRC "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so")
set(XLA_PJRT_CPU_LIB_DST "${XLA_INSTALL_DIR}/lib/pjrt_c_api_cpu_plugin.so")

# Make sure the include directory exists so that cmake doesn't fail.
make_directory("${XLA_INSTALL_DIR}/include")

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

add_custom_target(xla_custom_installed_artifacts
    DEPENDS "${XLA_PJRT_API_HDR_DST}"
            "${XLA_PJRT_CPU_HDR_DST}"
            "${XLA_PJRT_CPU_LIB_DST}"
)

# PJRT C API
add_library(XLA::pjrt_c_api INTERFACE IMPORTED)
set_target_properties(XLA::pjrt_c_api PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${XLA_INSTALL_DIR}/include"
)
add_dependencies(XLA::pjrt_c_api xla_custom_installed_artifacts)

# PJRT CPU plugin
add_library(XLA::pjrt_c_api_cpu SHARED IMPORTED)
set_target_properties(XLA::pjrt_c_api_cpu PROPERTIES
    IMPORTED_LOCATION             "${XLA_PJRT_CPU_LIB_DST}"
    INTERFACE_INCLUDE_DIRECTORIES "${XLA_INSTALL_DIR}/include"
)
add_dependencies(XLA::pjrt_c_api_cpu xla_custom_installed_artifacts)

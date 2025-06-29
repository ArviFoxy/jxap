message(STATUS "Building XLA")
set(XLA_SOURCE_DIR ${CMAKE_BINARY_DIR}/xla_src)
set(XLA_BAZEL_BIN ${XLA_SOURCE_DIR}/bazel-bin)
set(XLA_BAZEL_TARGETS
    "//xla/pjrt/c:pjrt_c_api.h"
    "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
    "//xla:autotuning_proto_cc_impl"
    "//xla:autotune_results_proto_cc_impl"
    "//xla:xla_proto_cc_impl"
    "//xla:xla_data_proto_cc_impl"
    "//xla/service:hlo_proto_cc_impl"
    "//xla/service:metrics_proto_cc_impl"
    "//xla/tsl/protobuf:dnn_proto_cc_impl"
    "//xla/stream_executor:device_description_proto_cc_impl"
    "//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl"
    "//xla/pjrt/proto:compile_options_proto_cc_impl"
)

set(XLA_PJRT_CPU_LIB "${XLA_BAZEL_BIN}/xla/pjrt/c/pjrt_c_api_cpu_plugin.so")
set(XLA_AUTOTUNING_PROTO_LIB "${XLA_BAZEL_BIN}/xla/libautotuning_proto_cc_impl.so")
set(XLA_AUTOTUNE_RESULTS_PROTO_LIB "${XLA_BAZEL_BIN}/xla/libautotune_results_proto_cc_impl.so")
set(XLA_XLA_PROTO_LIB "${XLA_BAZEL_BIN}/xla/libxla_proto_cc_impl.so")
set(XLA_XLA_DATA_PROTO_LIB "${XLA_BAZEL_BIN}/xla/libxla_data_proto_cc_impl.so")
set(XLA_SERVICE_HLO_PROTO_LIB "${XLA_BAZEL_BIN}/xla/service/libhlo_proto_cc_impl.so")
set(XLA_SERVICE_METRICS_PROTO_LIB "${XLA_BAZEL_BIN}/xla/service/libmetrics_proto_cc_impl.so")
set(XLA_DNN_PROTO_LIB "${XLA_BAZEL_BIN}/xla/tsl/protobuf/libdnn_proto_cc_impl.so")
set(XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_PROTO_LIB "${XLA_BAZEL_BIN}/xla/stream_executor/libdevice_description_proto_cc_impl.so")
set(XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_CAPABILITY_PROTO_LIB "${XLA_BAZEL_BIN}/xla/stream_executor/cuda/libcuda_compute_capability_proto_cc_impl.so")
set(XLA_COMPILE_OPTIONS_PROTO_LIB "${XLA_BAZEL_BIN}/xla/pjrt/proto/libcompile_options_proto_cc_impl.so")

ExternalProject_Add(xla_project
    GIT_REPOSITORY      https://github.com/openxla/xla.git
    GIT_TAG             fb601ce
    BUILD_IN_SOURCE     True
    SOURCE_DIR          ${XLA_SOURCE_DIR}

    GIT_PROGRESS        True  # Show git clone progress
    USES_TERMINAL_BUILD True  # Show build output

    CONFIGURE_COMMAND   ${XLA_SOURCE_DIR}/configure.py --backend=CPU
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}

    BUILD_COMMAND       ${CMAKE_COMMAND} -E rm -rf bazel-bin &&
                        bazel build --spawn_strategy=sandboxed ${XLA_BAZEL_TARGETS}
                        WORKING_DIRECTORY ${XLA_SOURCE_DIR}
    BUILD_BYPRODUCTS    "${XLA_PJRT_CPU_LIB}"
                        "${XLA_AUTOTUNING_PROTO_LIB}"
                        "${XLA_AUTOTUNE_RESULTS_PROTO_LIB}"
                        "${XLA_XLA_PROTO_LIB}"
                        "${XLA_XLA_DATA_PROTO_LIB}"
                        "${XLA_SERVICE_HLO_PROTO_LIB}"
                        "${XLA_SERVICE_METRICS_PROTO_LIB}"
                        "${XLA_DNN_PROTO_LIB}"
                        "${XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_PROTO_LIB}"
                        "${XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_CAPABILITY_PROTO_LIB}"
                        "${XLA_COMPILE_OPTIONS_PROTO_LIB}"

    INSTALL_COMMAND     ""
)

# PJRT C API
add_library(XLA::pjrt_c_api INTERFACE IMPORTED)
set_target_properties(XLA::pjrt_c_api PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${XLA_SOURCE_DIR}"
)
add_dependencies(XLA::pjrt_c_api xla_project)

# Imported shared libraries
function(add_xla_shared_library target_name so_file)
    find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
    get_filename_component(so_filename "${so_file}" NAME)

    set(so_patched_file "${CMAKE_CURRENT_BINARY_DIR}/lib/${so_filename}")
    set(so_patched_target "${so_filename}_patched")
    message(STATUS "Patching SONAME for ${so_file} - target ${so_patched_target}")
  
    add_custom_command(
        OUTPUT ${so_patched_file}
        DEPENDS ${so_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/lib"
        COMMAND ${PATCHELF_EXECUTABLE} --set-soname "${so_filename}" "${so_file}" --output "${so_patched_file}"
        COMMENT "Patching SONAME for ${so_file}"
        VERBATIM
    )
    add_custom_target(
        ${so_patched_target}
        DEPENDS ${so_patched_file}
    )

    add_library(${target_name} SHARED IMPORTED)
    set_target_properties(${target_name} PROPERTIES
        IMPORTED_LOCATION             "${so_patched_file}"
        IMPORTED_SONAME               "${so_filename}"
        INTERFACE_INCLUDE_DIRECTORIES "${XLA_BAZEL_BIN}"
    )
    add_dependencies(${target_name} ${so_patched_target})
endfunction()

add_xla_shared_library(XLA::pjrt_c_api_cpu "${XLA_PJRT_CPU_LIB}")
add_xla_shared_library(XLA::autotuning_proto_cc "${XLA_AUTOTUNING_PROTO_LIB}")
add_xla_shared_library(XLA::autotune_results_proto_cc "${XLA_AUTOTUNE_RESULTS_PROTO_LIB}")
add_xla_shared_library(XLA::xla_proto_cc "${XLA_XLA_PROTO_LIB}")
add_xla_shared_library(XLA::xla_data_proto_cc "${XLA_XLA_DATA_PROTO_LIB}")
add_xla_shared_library(XLA::xla_service_hlo_proto_cc "${XLA_SERVICE_HLO_PROTO_LIB}")
add_xla_shared_library(XLA::xla_service_metrics_proto_cc "${XLA_SERVICE_METRICS_PROTO_LIB}")
add_xla_shared_library(XLA::xla_dnn_proto_cc "${XLA_DNN_PROTO_LIB}")
add_xla_shared_library(XLA::xla_streaming_executor_device_description_proto_cc "${XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_PROTO_LIB}")
add_xla_shared_library(XLA::xla_streaming_executor_cuda_cuda_device_capability_proto_cc "${XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_CAPABILITY_PROTO_LIB}")
add_xla_shared_library(XLA::compile_options_proto_cc "${XLA_COMPILE_OPTIONS_PROTO_LIB}")

add_library(XLA_protos INTERFACE)
target_link_libraries(XLA_protos
    INTERFACE
    "-Wl,--start-group"
    XLA::autotuning_proto_cc
    XLA::autotune_results_proto_cc
    XLA::xla_proto_cc
    XLA::xla_data_proto_cc
    XLA::xla_service_hlo_proto_cc
    XLA::xla_service_metrics_proto_cc
    XLA::xla_dnn_proto_cc
    XLA::xla_streaming_executor_device_description_proto_cc
    XLA::xla_streaming_executor_cuda_cuda_device_capability_proto_cc
    XLA::compile_options_proto_cc
    "-Wl,--end-group"
    protobuf::libprotobuf
)

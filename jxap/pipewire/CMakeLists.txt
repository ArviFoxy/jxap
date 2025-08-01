# --- Find public depdencenies ---
cmake_minimum_required(VERSION 3.16)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3>=0.3.40)
pkg_check_modules(SPA REQUIRED IMPORTED_TARGET libspa-0.2)

# --- Pipewire runner ---

message(STATUS "Building pipewire runner")
add_executable(jxap_pipewire_run
    run.cc)
target_link_libraries(jxap_pipewire_run
    PRIVATE
    absl::strings
    absl::log
    absl::log_initialize
    absl::flags
    absl::flags_parse
    jxap_pjrt_plugin_runner
    PkgConfig::PIPEWIRE
    PkgConfig::SPA
)
target_include_directories(jxap_pipewire_run PRIVATE
    ${PIPEWIRE_INCLUDE_DIRS}
    ${SPA_INCLUDE_DIRS}
)

#  --- Installation ---

install(TARGETS jxap_pipewire_run
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
set_target_properties(jxap_pipewire_run PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
)

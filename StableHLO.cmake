# --- Fetch and Build StableHLO ---
message(STATUS "Configuring StableHLO...")

# --- Fetch the stablehlo sources ---
FetchContent_Declare(
  stablehlo
  GIT_REPOSITORY https://github.com/openxla/stablehlo.git
  GIT_TAG a85bcc1
  GIT_PROGRESS True
)

FetchContent_GetProperties(stablehlo)
if(NOT stablehlo_POPULATED)
    message(STATUS "Fetching StableHLO sources...")
    FetchContent_Populate(stablehlo)
    message(STATUS "Fetching StableHLO sources... Done. Path: ${stablehlo_SOURCE_DIR}")
else()
    message(STATUS "StableHLO sources already available at ${stablehlo_SOURCE_DIR}")
endif()
set(STABLEHLO_SRC_DIR ${stablehlo_SOURCE_DIR}) # Convenience variable

# --- Get the required LLVM commit SHA ---
set(LLVM_VERSION_FILE "${STABLEHLO_SRC_DIR}/build_tools/llvm_version.txt")
file(READ ${LLVM_VERSION_FILE} LLVM_COMMIT_SHA)
string(STRIP "${LLVM_COMMIT_SHA}" LLVM_COMMIT_SHA)
message(STATUS "Required LLVM commit SHA for StableHLO: ${LLVM_COMMIT_SHA}")

# --- Configure LLVM (llvm-project) sources ---
message(STATUS "Configuring LLVM/MLIR...")
FetchContent_Declare(
    llvm_project
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG "${LLVM_COMMIT_SHA}"
    GIT_PROGRESS True
    SOURCE_SUBDIR llvm # LLVM's main CMakeLists.txt is in the 'llvm' subdirectory
)

# LLVM/MLIR specific options (taken from build_mlir.sh)
set(LLVM_INSTALL_UTILS ON CACHE BOOL "Install LLVM utilities")
set(LLVM_ENABLE_LLD ${JXAP_ENABLE_LLD} CACHE BOOL "Enable LLD linker")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "Build MLIR")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "Disable terminfo")
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "Include LLVM tools")
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable MLIR Python bindings")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "Disable LLVM C bindings")
set(LLVM_VERSION_SUFFIX "" CACHE STRING "LLVM version suffix")
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL "Disable versioned sonames")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "Build LLVM tools")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Exclude LLVM/MLIR tests")
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "Exclude MLIR tests")
set(LLVM_USE_SPLIT_DWARF ON CACHE BOOL "Enable split DWARF for debug info")
set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "Enable LLVM assertions")

# By default, FetchContent_MakeAvailable builds in CMAKE_BINARY_DIR/_deps/llvm_project-build
# And the LLVM/MLIR CMake system will make its targets and CMake functions available to our project.
FetchContent_MakeAvailable(llvm_project)
message(STATUS "Configured LLVM/MLIR.")

# --- Make StableHLO available, it will find the MLIR we just built ---
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO embedded in another project" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable Python bindings for StableHLO" FORCE)
set(STABLEHLO_ENABLE_PYTHON_TF_TESTS OFF CACHE BOOL "Disable Python TF tests for StableHLO" FORCE)
set(STABLEHLO_ENABLE_LLD ON CACHE BOOL "Enable/disable LLD for StableHLO" FORCE)

message(STATUS "Configuring StableHLO...")
FetchContent_MakeAvailable(stablehlo) 
message(STATUS "Configured StableHLO.")

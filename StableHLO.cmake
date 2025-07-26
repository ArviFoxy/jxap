# --- Configure LLVM (llvm-project) sources ---
message(STATUS "Configuring LLVM/MLIR...")
FetchContent_Declare(
    llvm_project
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG "741fef3a445339523500f614e0f752b9a74517a6"
    GIT_PROGRESS True
    SOURCE_SUBDIR llvm # LLVM's main CMakeLists.txt is in the 'llvm' subdirectory
)

# LLVM/MLIR specific options (taken from build_mlir.sh)
set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities" FORCE)
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "Install only toolchain files" FORCE)
set(LLVM_ENABLE_LLD OFF CACHE BOOL "Enable LLD linker" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "Build MLIR")
set(LLVM_TARGETS_TO_BUILD "host X86" CACHE STRING "LLVM targets to build")
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
# Make sure RTTI is consistently on.
set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI" FORCE)

# By default, FetchContent_MakeAvailable builds in CMAKE_BINARY_DIR/_deps/llvm_project-build
# And the LLVM/MLIR CMake system will make its targets and CMake functions available to our project.
FetchContent_MakeAvailable(llvm_project)
message(STATUS "Configured LLVM/MLIR.")

# --- Fetch and Build StableHLO ---
message(STATUS "Configuring StableHLO...")
FetchContent_Declare(
  stablehlo
  GIT_REPOSITORY https://github.com/openxla/stablehlo.git
  GIT_TAG a85bcc1
  GIT_PROGRESS True
)

# --- Set up options ---
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO embedded in another project")
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Disable Python bindings for StableHLO")
set(STABLEHLO_ENABLE_PYTHON_TF_TESTS OFF CACHE BOOL "Disable Python TF tests for StableHLO")
set(STABLEHLO_ENABLE_LLD OFF CACHE BOOL "Enable/disable LLD for StableHLO" FORCE)

# --- Set up the environment for the embedded build ---
set(LLVM_MAIN_SRC_DIR ${llvm_project_SOURCE_DIR}/llvm)
set(LLVM_MAIN_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
set(LLVM_GENERATED_INCLUDE_DIR ${llvm_project_BINARY_DIR}/include)
set(LLVM_BINARY_DIR ${llvm_project_BINARY_DIR})

set(MLIR_MAIN_SRC_DIR ${llvm_project_SOURCE_DIR}/mlir)
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_GENERATED_INCLUDE_DIR ${llvm_project_BINARY_DIR}/tools/mlir/include)
set(MLIR_BINARY_DIR ${llvm_project_BINARY_DIR}/tools/mlir)

list(APPEND CMAKE_MODULE_PATH ${llvm_project_SOURCE_DIR}/llvm/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${llvm_project_SOURCE_DIR}/mlir/cmake/modules)

message(STATUS "Adding include dir: ${LLVM_MAIN_INCLUDE_DIR}")
include_directories(${LLVM_MAIN_INCLUDE_DIR})
message(STATUS "Adding include dir: ${LLVM_GENERATED_INCLUDE_DIR}")
include_directories(${LLVM_GENERATED_INCLUDE_DIR})
message(STATUS "Adding include dir: ${MLIR_MAIN_INCLUDE_DIR}")
include_directories(${MLIR_MAIN_INCLUDE_DIR})
message(STATUS "Adding include dir: ${MLIR_GENERATED_INCLUDE_DIR}")
include_directories(${MLIR_GENERATED_INCLUDE_DIR})

FetchContent_MakeAvailable(stablehlo) 
message(STATUS "Configured StableHLO.")

# Add resulting include directories.
set(STABLEHLO_MAIN_SRC_DIR ${stablehlo_SOURCE_DIR})
set(STABLEHLO_MAIN_INCLUDE_DIR ${STABLEHLO_MAIN_SRC_DIR})
set(STABLEHLO_GENERATED_INCLUDE_DIR ${stablehlo_BINARY_DIR})

message(STATUS "Adding include dir: ${STABLEHLO_MAIN_INCLUDE_DIR}")
include_directories(${STABLEHLO_MAIN_INCLUDE_DIR})
message(STATUS "Adding include dir: ${STABLEHLO_GENERATED_INCLUDE_DIR}")
include_directories(${STABLEHLO_GENERATED_INCLUDE_DIR})

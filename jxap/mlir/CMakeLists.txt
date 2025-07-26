add_library(jxap_mlir_utils STATIC
    utils.h
    utils.cc)
target_link_libraries(jxap_mlir_utils
    PUBLIC
    absl::status
    absl::statusor

    PRIVATE
    LLVMCore
    LLVMSupport
    LLVMTarget
    LLVMCodeGen
    LLVMAsmPrinter
    LLVMX86CodeGen
    MLIRTargetLLVMIRExport
    MLIRExecutionEngine)
      
add_library(jxap_mlir_passes STATIC
    passes.h
    passes.cc)
target_link_libraries(jxap_mlir_passes
    PUBLIC
    MLIRPass

    PRIVATE
    absl::log
    absl::strings
    LLVMCore
    LLVMSupport
    MLIRIR
)

add_library(jxap_mlir_pipeline STATIC
    pipeline.h
    pipeline.cc
)
target_link_libraries(jxap_mlir_pipeline 
    PUBLIC
    absl::status
    absl::statusor

    PRIVATE
    jxap_common
    absl::strings
    absl::log
    LLVMCore
    LLVMSupport
    MLIRIR
    StablehloOps
    StablehloPasses
    jxap_mlir_utils
    jxap_mlir_passes
)
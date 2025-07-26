#include "jxap/utils.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "jxap/mlir/passes.h"
#include "jxap/mlir/pipeline.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jxap {

std::string MlirTensorType(const std::vector<int64_t>& shape, absl::string_view dtype) {
  std::vector<std::string> parts;
  for (int64_t dim : shape) {
    parts.push_back(std::to_string(dim));
  }
  parts.push_back(std::string(dtype));
  return absl::StrCat("tensor<", absl::StrJoin(parts, "x"), ">");
}

mlir::MLIRContext* GetMlirContext() {
  static mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();
  static mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  return &context;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModule(absl::string_view mlir_code,
                                                                  mlir::MLIRContext* context) {
  // Parse the MLIR.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(mlir_code, /*BufferName=*/"input.mlir",
                                         /*RequiresNullTerminator=*/false),
        llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, context);
  }

  if (!module) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse MLIR input."));
  }
  return module;
}

absl::Status AddDiagnostics(absl::Status status, const std::string& diagnostic_output) {
  if (!status.ok()) {
    return absl::Status(status.code(),
                        absl::StrCat(status.message(), "\nDiagnostics:\n", diagnostic_output));
  }
  return status;
}

absl::StatusOr<std::string> LowerToOptimizedLLVMIR(mlir::ModuleOp module) {
  // Translate the module, that contains the LLVM dialect, to LLVM IR. Use a
  // fresh LLVM IR context. (Note that LLVM is not thread-safe and any
  // concurrent use of a context requires external locking.)
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return absl::InternalError("Failed to emit LLVM IR.");
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        /*targetMachine=*/nullptr);

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (llvm::Error err = optPipeline(llvmModule.get())) {
    std::string msg_str;
    llvm::raw_string_ostream msg(msg_str);
    msg << "Failed to optimize LLVM IR: " << err;
    return absl::InvalidArgumentError(msg.str());
  }

  std::string moduleStr;
  llvm::raw_string_ostream os(moduleStr);
  llvmModule->print(os, nullptr);
  os.flush();
  return os.str();
}

}  // namespace jxap

#include "jxap/mlir_pipeline.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "jxap/mlir_passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace jxap {

constexpr char kMainFnName[] = "main";

std::string MlirTensorType(const std::vector<int64_t>& shape, absl::string_view dtype) {
  std::vector<std::string> parts;
  for (int64_t dim : shape) {
    parts.push_back(std::to_string(dim));
  }
  parts.push_back(std::string(dtype));
  return absl::StrCat("tensor<", absl::StrJoin(parts, "x"), ">");
}

absl::StatusOr<std::string> MlirPipeline(absl::string_view mlir_code,
                                         const std::vector<ArgumentTransform>& transforms) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string diagnostic_output;
  llvm::raw_string_ostream diagnostic_os(diagnostic_output);
  mlir::ScopedDiagnosticHandler diag_handler(&context, [&](mlir::Diagnostic& diag) {
    diag.print(diagnostic_os);
    diagnostic_os << "\n";
    return mlir::failure(diag.getSeverity() == mlir::DiagnosticSeverity::Error);
  });

  // Parse the MLIR.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(mlir_code, /*BufferName=*/"input.mlir",
                                         /*RequiresNullTerminator=*/false),
        llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context);
  }

  if (!module) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse MLIR input. Diagnostics:\n", diagnostic_output));
  }

  // Parse the refined types.
  llvm::SmallVector<mlir::Type> parsed_types;
  for (const auto& transform : transforms) {
    if (std::holds_alternative<RefineType>(transform)) {
      const auto& refine_type = std::get<RefineType>(transform);
      mlir::Type parsed_type = mlir::parseType(refine_type.type, &context);
      if (!parsed_type) {
        return absl::InvalidArgumentError(absl::StrCat("Failed to parse type ", refine_type.type,
                                                       ". Diagnostics:\n", diagnostic_output));
      }
      parsed_types.push_back(parsed_type);
    }
  }

  mlir::PassManager pm(module.get()->getName(), mlir::PassManager::Nesting::Implicit);
  // Disable printing ops on diagnostics as we capture them manually
  pm.getContext()->printOpOnDiagnostic(false);

  // Pass 1: replace constants.
  std::map<unsigned int, float> arg_to_value;
  for (unsigned int i = 0; i < transforms.size(); i++) {
    if (std::holds_alternative<ReplaceWithConstant>(transforms[i])) {
      const auto& replace_with_constant = std::get<ReplaceWithConstant>(transforms[i]);
      arg_to_value[i] = replace_with_constant.value;
    }
  }
  if (!arg_to_value.empty()) {
    pm.addPass(createReplaceFuncArgWithConstantPass(arg_to_value, kMainFnName));
  }
  // Pass 2: refine program input shapes to be static.
  pm.addPass(mlir::stablehlo::createStablehloRefineArgumentsPass(parsed_types));
  // Pass 3: propagates static shapes across the program.
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  // Pass 4: replaces dynamic shape ops with static shape ops if possible.
  pm.addPass(mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  // Pass 5: remove shape assertions.
  pm.addPass(createRemoveShapeAssertionsPass());
  // Pass 6: simplify and propagate constants.
  pm.addPass(mlir::stablehlo::createStablehloAggressiveSimplificationPass());

  if (mlir::failed(pm.run(*module))) {
    return absl::InternalError(
        absl::StrCat("Failed to refine input types. Diagnostics:\n", diagnostic_output));
  }

  // Print the transformed module back to a string
  std::string output_mlir;
  llvm::raw_string_ostream os(output_mlir);
  module->print(os);
  os.flush();
  return output_mlir;
}

}  // namespace jxap

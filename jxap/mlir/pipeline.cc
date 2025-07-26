#include "jxap/mlir/pipeline.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "jxap/mlir/passes.h"
#include "jxap/mlir/utils.h"
#include "jxap/utils.h"
#include "llvm/IR/LLVMContext.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace jxap {

constexpr char kMainFnName[] = "main";

absl::StatusOr<std::string> MlirPipeline(
    absl::string_view mlir_code, const std::vector<ArgumentTransform>& transforms,
    const std::map<std::string, ScalarValue>& global_constants) {
  mlir::MLIRContext* context = GetMlirContext();
  DiagnosticsHandler diag_handler(context);

  auto status_or_module = ParseMlirModule(mlir_code, context);
  RETURN_IF_ERROR(diag_handler.Annotate(status_or_module.status()));
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(status_or_module.value());

  // Parse the refined types.
  llvm::SmallVector<mlir::Type> parsed_types;
  for (const auto& transform : transforms) {
    if (std::holds_alternative<RefineType>(transform)) {
      const auto& refine_type = std::get<RefineType>(transform);
      mlir::Type parsed_type = mlir::parseType(refine_type.type, context);
      if (!parsed_type) {
        return diag_handler.Annotate(
            absl::InvalidArgumentError(absl::StrCat("Failed to parse type ", refine_type.type)));
      }
      parsed_types.push_back(parsed_type);
    }
  }

  mlir::PassManager pm(module.get()->getName(), mlir::PassManager::Nesting::Implicit);
  // Disable printing ops on diagnostics as we capture them manually
  pm.getContext()->printOpOnDiagnostic(false);

  // Pass 1: replace constant arguments.
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
  // Pass 3: replace global jax constants.
  pm.addPass(createReplaceGlobalConstantsPass(global_constants));
  // Pass 4: propagates static shapes across the program.
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  // Pass 5: remove shape assertions.
  pm.addPass(createRemoveShapeAssertionsPass());
  // Pass 6: replaces dynamic shape ops with static shape ops if possible.
  pm.addPass(mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  // Pass 7: simplify and propagate constants.
  pm.addPass(mlir::stablehlo::createStablehloAggressiveSimplificationPass());

  if (mlir::failed(pm.run(*module))) {
    return diag_handler.Annotate(absl::InternalError("Failed to refine input types"));
  }

  // Print the transformed module back to a string
  std::string output_mlir;
  llvm::raw_string_ostream os(output_mlir);
  module->print(os);
  os.flush();
  return output_mlir;
}

}  // namespace jxap

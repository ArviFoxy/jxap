#include "jxap/mlir_passes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jxap {
namespace {

struct ReplaceFuncArgWithConstantPass
    : public mlir::PassWrapper<ReplaceFuncArgWithConstantPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 protected:
  std::map<unsigned int, float> arg_to_value_;
  std::string target_function_name_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceFuncArgWithConstantPass)

  ReplaceFuncArgWithConstantPass(std::map<unsigned int, float> arg_to_value,
                                 std::string target_function_name)
      : arg_to_value_(std::move(arg_to_value)),
        target_function_name_(std::move(target_function_name)) {}

  ReplaceFuncArgWithConstantPass(const ReplaceFuncArgWithConstantPass &other) : PassWrapper(other) {
    arg_to_value_ = other.arg_to_value_;
    target_function_name_ = other.target_function_name_;
  }

  llvm::StringRef getArgument() const final { return "replace-func-arg-with-constant"; }
  llvm::StringRef getDescription() const final {
    return "Replaces a specified tensor<f32> function argument with a stablehlo.constant.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = funcOp.getContext();

    if (!target_function_name_.empty() && funcOp.getSymName() != target_function_name_) {
      return;
    }

    if (funcOp.isExternal() | funcOp.empty()) {
      return;
    }

    mlir::Block &entryBlock = funcOp.front();

    // Replace all occurences of the arguments with a constant.
    for (const auto &arg : arg_to_value_) {
      unsigned int index = arg.first;
      if (index >= entryBlock.getNumArguments()) {
        funcOp.emitError("Argument index ") << index << " is out of bounds.";
        return signalPassFailure();
      }

      mlir::BlockArgument blockArg = entryBlock.getArgument(index);

      auto tensorTy = llvm::dyn_cast<mlir::TensorType>(blockArg.getType());
      if (!tensorTy || !tensorTy.getElementType().isF32()) {
        funcOp.emitError("argument is not of type tensor<f32>.");
        return signalPassFailure();
      }

      // Create constant
      mlir::OpBuilder builder(context);
      builder.setInsertionPointToStart(&entryBlock);
      mlir::Type elementType = builder.getF32Type();
      mlir::ShapedType constantType = mlir::RankedTensorType::get({}, elementType);  // Scalar
      mlir::Attribute valueAttr =
          mlir::DenseElementsAttr::get(constantType, llvm::APFloat(arg.second));
      if (!valueAttr) {
        funcOp.emitError("Failed to create DenseElementsAttr for the constant value.");
        return signalPassFailure();
      }
      mlir::Location loc = funcOp.getLoc();
      auto constantOp = builder.create<mlir::stablehlo::ConstantOp>(loc, valueAttr);

      // Replace uses and erase
      blockArg.replaceAllUsesWith(constantOp.getResult());
    }

    llvm::BitVector should_erase(entryBlock.getNumArguments(), false);
    for (const auto &arg : arg_to_value_) {
      should_erase[arg.first] = true;
    }
    entryBlock.eraseArguments(should_erase);

    // Update function type
    mlir::FunctionType oldFuncType = funcOp.getFunctionType();
    llvm::SmallVector<mlir::Type, 4> newInputTypes;
    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      if (!should_erase[i]) {
        newInputTypes.push_back(oldFuncType.getInput(i));
      }
    }

    llvm::SmallVector<mlir::DictionaryAttr, 4> newArgAttrs;
    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      if (!should_erase[i]) {
        newArgAttrs.push_back(funcOp.getArgAttrDict(i));
      }
    }

    mlir::FunctionType newFuncType =
        mlir::FunctionType::get(context, newInputTypes, oldFuncType.getResults());
    funcOp.setType(newFuncType);

    if (!newArgAttrs.empty() | newInputTypes.empty()) {
      funcOp.setAllArgAttrs(newArgAttrs);
    }
  }
};

struct RemoveShapeAssertionsPass
    : public mlir::PassWrapper<RemoveShapeAssertionsPass, mlir::OperationPass<mlir::func::FuncOp>> {
 protected:
  std::map<unsigned int, float> arg_to_value_;
  std::string target_function_name_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveShapeAssertionsPass)

  RemoveShapeAssertionsPass() = default;
  RemoveShapeAssertionsPass(const RemoveShapeAssertionsPass &other) = default;

  llvm::StringRef getArgument() const final { return "remove-shape-assertions-pass"; }
  llvm::StringRef getDescription() const final { return "Removes shape_assertion custom calls."; }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    llvm::SmallVector<mlir::stablehlo::CustomCallOp, 4> customCallsToRemove;
    funcOp.walk([&](mlir::stablehlo::CustomCallOp customCallOp) {
      if (customCallOp.getCallTargetName() == "shape_assertion") {
        customCallsToRemove.push_back(customCallOp);
      }
    });
    for (mlir::stablehlo::CustomCallOp op : customCallsToRemove) {
      if (op->getNumResults() > 0) {
        funcOp.emitError("Expected stablehlo.custom_call @shape_assertion to have no results.");
        return signalPassFailure();
      }
      op.erase();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
    std::map<unsigned int, float> arg_to_value, std::string target_function_name) {
  return std::make_unique<ReplaceFuncArgWithConstantPass>(std::move(arg_to_value),
                                                          std::move(target_function_name));
}

std::unique_ptr<::mlir::Pass> createRemoveShapeAssertionsPass() {
  return std::make_unique<RemoveShapeAssertionsPass>();
}

}  // namespace jxap

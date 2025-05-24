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
  unsigned int arg_index_;
  float constant_float_value_;
  std::string target_function_name_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceFuncArgWithConstantPass)

  ReplaceFuncArgWithConstantPass(unsigned int arg_index, float constant_float_value,
                                 std::string target_function_name)
      : arg_index_(arg_index),
        constant_float_value_(constant_float_value),
        target_function_name_(target_function_name) {}

  ReplaceFuncArgWithConstantPass(const ReplaceFuncArgWithConstantPass &other) : PassWrapper(other) {
    arg_index_ = other.arg_index_;
    constant_float_value_ = other.constant_float_value_;
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

    if (arg_index_ >= entryBlock.getNumArguments()) {
      funcOp.emitError("Argument index ") << arg_index_ << " is out of bounds.";
      return signalPassFailure();
    }
    mlir::BlockArgument blockArg = entryBlock.getArgument(arg_index_);

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
        mlir::DenseElementsAttr::get(constantType, llvm::APFloat(constant_float_value_));
    if (!valueAttr) {
      funcOp.emitError("Failed to create DenseElementsAttr for the constant value.");
      return signalPassFailure();
    }
    mlir::Location loc = funcOp.getLoc();
    auto constantOp = builder.create<mlir::stablehlo::ConstantOp>(loc, valueAttr);

    // Replace uses and erase
    blockArg.replaceAllUsesWith(constantOp.getResult());
    entryBlock.eraseArgument(arg_index_);

    // Update function type
    mlir::FunctionType oldFuncType = funcOp.getFunctionType();
    llvm::SmallVector<mlir::Type, 4> newInputTypes;
    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      if (i != arg_index_) {
        newInputTypes.push_back(oldFuncType.getInput(i));
      }
    }

    llvm::SmallVector<mlir::DictionaryAttr, 4> newArgAttrs;
    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      if (i != arg_index_) {
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

}  // namespace

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
    unsigned int arg_index, float constant_float_value, std::string target_function_name) {
  return std::make_unique<ReplaceFuncArgWithConstantPass>(arg_index, constant_float_value,
                                                          target_function_name);
}

}  // namespace jxap

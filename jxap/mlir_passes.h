#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace jxap {

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
    std::map<unsigned int, float> arg_to_value, std::string target_function_name);

std::unique_ptr<::mlir::Pass> createRemoveShapeAssertionsPass();

}  // namespace jxap

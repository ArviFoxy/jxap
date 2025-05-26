#include <map>
#include <memory>
#include <string>
#include <variant>

#include "mlir/Pass/Pass.h"

namespace jxap {

using ScalarValue = std::variant<int, float>;

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
    std::map<unsigned int, float> arg_to_value, std::string target_function_name);

std::unique_ptr<::mlir::Pass> createReplaceGlobalConstantsPass(
    std::map<std::string, ScalarValue> global_to_value);

std::unique_ptr<::mlir::Pass> createRemoveShapeAssertionsPass();

}  // namespace jxap

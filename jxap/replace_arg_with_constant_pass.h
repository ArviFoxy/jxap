#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace jxap {

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
    unsigned int arg_index, float constant_float_value, std::string target_function_name);

}  // namespace jxap

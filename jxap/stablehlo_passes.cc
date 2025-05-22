#include "jxap/stablehlo_passes.h"

#include <filesystem>
#include <fstream>

#include "absl/log/log.h"
#include "stablehlo/transforms/Passes.h"

namespace jxap {

std::string RefineInputTypes(absl::string_view mlir, const std::vector<Shape>& input_shapes,
                             const std::vector<std::string>& input_dtypes) {
  mlir::stablehlo::
}

}  // namespace jxap

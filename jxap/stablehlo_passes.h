#ifndef JXAP_STABLEHLO_PASSES
#define JXAP_STABLEHLO_PASSES

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace jxap {

using Shape = std::vector<int64_t>;

/**
 * Refines input types to static shapes and removes dynamicism from the MLIR.
 */
std::string RefineInputTypes(absl::string_view mlir, const std::vector<Shape>& input_shapes,
                             const std::vector<std::string>& input_dtypes);

}  // namespace jxap

#endif

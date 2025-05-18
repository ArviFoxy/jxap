#ifndef JXAP_HLO_OPT
#define JXAP_HLO_OPT

#include <vector>
#include <string>

#include "absl/strings/string_view.h"

namespace jxap {

using Shape = std::vector<int64_t>;

/**
 * Looks for the jxap-hlo-opt binary in $PATH.
 */
std::string FindHloOptBinary();

/**
 * Refines input types to static shapes and removes dynamicism from the MLIR.
 */
std::string RefineInputTypes(
  const std::string& hlo_opt_path, 
  absl::string_view mlir, 
  const std::vector<Shape>& input_shapes,
  const std::vector<std::string>& input_dtypes
);

} // namespace jxap

#endif

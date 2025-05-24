#ifndef JXAP_STABLEHLO_PASSES
#define JXAP_STABLEHLO_PASSES

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
namespace jxap {

/*
 * Formats a string encoding an MLIR tensor type.
 *
 * @param shape Dimensions of the tensor.
 * @param dtype Data type, such as "f32" or "i32".
 */
std::string MlirTensorType(const std::vector<int64_t>& shape, absl::string_view dtype);

/**
 * Refines input types to static shapes and removes dynamicism from the MLIR.
 */
absl::StatusOr<std::string> RefineInputTypes(absl::string_view mlir_code,
                                             const std::vector<std::string>& mlir_types);

}  // namespace jxap

#endif

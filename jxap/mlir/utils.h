#ifndef JXAP_MLIR_MLIR_UTILS_H
#define JXAP_MLIR_MLIR_UTILS_H

#include <map>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "jxap/mlir/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace jxap {

/**
 * Adds diagnostics to an existing status.
 */
absl::Status AddDiagnostics(absl::Status status, const std::string& diagnostic_output);

/**
 * Global MLIR context.
 */
mlir::MLIRContext* GetMlirContext();

/*
 * Formats a string encoding an MLIR tensor type.
 *
 * @param shape Dimensions of the tensor.
 * @param dtype Data type, such as "f32" or "i32".
 */
std::string MlirTensorType(const std::vector<int64_t>& shape, absl::string_view dtype);

/**
 * Parses an MLIR module from a string.
 */
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModule(absl::string_view mlir_code,
                                                                  mlir::MLIRContext* context);

/**
 * Lowers and optimizes the MLIR module to LLVM IR. Returns a string containing the LLVM IR.
 * This is for debugging purposes.
 */
absl::StatusOr<std::string> LowerToOptimizedLLVMIR(mlir::ModuleOp module);

}  // namespace jxap

#endif  // JXAP_MLIR_MLIR_UTILS_H

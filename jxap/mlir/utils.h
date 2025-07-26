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

class DiagnosticsHandler {
 public:
  DiagnosticsHandler(mlir::MLIRContext* context);
  absl::Status Annotate(absl::Status status);

 private:
  std::string diagnostic_output_;
  llvm::raw_string_ostream diagnostic_os_;
  mlir::ScopedDiagnosticHandler diag_handler_;
};

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

}  // namespace jxap

#endif  // JXAP_MLIR_MLIR_UTILS_H

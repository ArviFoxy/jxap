#ifndef JXAP_UTILS
#define JXAP_UTILS

#include <filesystem>
#include <source_location>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace jxap {

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path);
absl::Status WriteFile(const std::filesystem::path& path, const std::string& contents);

// Appends source code information to a status.
absl::Status AppendLocToStatus(absl::Status status, std::source_location loc);

#define RETURN_IF_ERROR(status_expr)                        \
  {                                                         \
    absl::Status status_ = status_expr;                     \
    if (!status_.ok()) {                                    \
      constexpr auto loc = std::source_location::current(); \
      return ::jxap::AppendLocToStatus(status_, loc);       \
    }                                                       \
  }

}  // namespace jxap

#endif

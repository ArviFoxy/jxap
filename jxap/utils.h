#ifndef JXAP_UTILS
#define JXAP_UTILS

#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace jxap {

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path);
absl::Status WriteFile(const std::filesystem::path& path, const std::string& contents);

}  // namespace jxap

#endif

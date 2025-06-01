#include "jxap/utils.h"

#include <fstream>

#include "absl/strings/str_cat.h"

namespace jxap {

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to open file: ", path.string()));
  }
  std::stringstream string_stream;
  string_stream << file.rdbuf();
  if (string_stream.bad()) {
    return absl::InternalError(absl::StrCat("Error while reading file: ", path.string()));
  }
  return string_stream.str();
}

absl::Status WriteFile(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream file(path.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    return absl::InternalError(absl::StrCat("Failed to open file: ", path.string()));
  }
  file << contents;
  if (file.bad()) {
    return absl::InternalError(absl::StrCat("Error while writing file: ", path.string()));
  }
  return absl::OkStatus();
}

absl::Status AppendLocToStatus(absl::Status status, std::source_location loc) {
  return absl::Status(status.code(),
                      absl::StrCat(status.message(), "\nTraceback: ", loc.file_name(), ":",
                                   loc.line(), " in function ", loc.function_name(), "."));
}

}  // namespace jxap

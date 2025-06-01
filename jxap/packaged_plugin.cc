#include "jxap/packaged_plugin.h"  // Defines PackagedPlugin and LoadPluginFile declaration

#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "jxap/utils.h"
#include "mz.h"
#include "mz_strm_os.h"
#include "mz_zip.h"

namespace jxap {

namespace {

// Filenames to be read from the ZIP archive
const char* kNameFilename = "name.txt";
const char* kInitMlirFilename = "init.mlir";
const char* kUpdateMlirFilename = "update.mlir";
const char* kInputBufferNamesFilename = "input_buffer_names.txt";
const char* kOutputBufferNamesFilename = "output_buffer_names.txt";
// Define a reasonable maximum size for a single file to prevent excessive memory allocation.
constexpr int64_t kMaxIndividualFileSize = 256 * 1024 * 1024;  // 256 MB

class MzZip {
 public:
  void* handle = nullptr;
  bool is_open = false;

  MzZip() { handle = mz_zip_create(); }
  ~MzZip() {
    if (handle) {
      if (is_open) {
        mz_zip_close(handle);
      }
      mz_zip_delete(&handle);
    }
  }
  // Disable copy and assignment
  MzZip(const MzZip&) = delete;
  MzZip& operator=(const MzZip&) = delete;
};

class MzOsStream {
 public:
  void* handle = nullptr;
  bool is_open = false;

  MzOsStream() { handle = mz_stream_os_create(); }
  ~MzOsStream() {
    if (handle) {
      if (is_open) {
        mz_stream_os_close(handle);
      }
      mz_stream_os_delete(&handle);
    }
  }
  // Disable copy and assignment
  MzOsStream(const MzOsStream&) = delete;
  MzOsStream& operator=(const MzOsStream&) = delete;
};

// Helper function to read a specific entry from the zip file into a string.
absl::StatusOr<std::string> ReadEntryToString(void* zip_archive_handle, const char* entry_name) {
  int32_t err = mz_zip_locate_entry(zip_archive_handle, entry_name, 0);
  if (err == MZ_END_OF_LIST) {
    return absl::NotFoundError(
        absl::StrCat("Required file not found in ZIP archive: ", entry_name));
  }
  if (err != MZ_OK) {
    return absl::InternalError(
        absl::StrCat("Failed to locate file '", entry_name, "' in ZIP archive. Error code: ", err));
  }

  mz_zip_file* file_info = nullptr;
  err = mz_zip_entry_get_info(zip_archive_handle, &file_info);
  if (err != MZ_OK || file_info == nullptr) {
    return absl::InternalError(absl::StrCat("Failed to get information for file '", entry_name,
                                            "' in ZIP archive. Error code: ", err));
  }
  if (mz_zip_entry_is_dir(zip_archive_handle) == MZ_OK) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a file but found a directory: ", entry_name));
  }

  err = mz_zip_entry_read_open(zip_archive_handle, /*raw=*/0,
                               /*password=*/nullptr /* no password */);
  if (err != MZ_OK) {
    return absl::InternalError(absl::StrCat("Failed to open file '", entry_name,
                                            "' for reading from ZIP archive. Error code: ", err));
  }

  std::string content;
  if (file_info->uncompressed_size > 0) {
    if (file_info->uncompressed_size > kMaxIndividualFileSize) {
      mz_zip_entry_read_close(zip_archive_handle, nullptr, nullptr, nullptr);
      return absl::ResourceExhaustedError(absl::StrCat(
          "File '", entry_name, "' in ZIP archive is too large: ", file_info->uncompressed_size,
          " bytes. Maximum allowed size is ", kMaxIndividualFileSize, " bytes."));
    }
    content.resize(static_cast<size_t>(file_info->uncompressed_size));
    int32_t bytes_read = mz_zip_entry_read(zip_archive_handle, &content[0],
                                           static_cast<int32_t>(file_info->uncompressed_size));

    if (bytes_read < 0 || static_cast<int64_t>(bytes_read) != file_info->uncompressed_size) {
      mz_zip_entry_read_close(zip_archive_handle, nullptr, nullptr,
                              nullptr);  // Ensure entry is closed on error
      return absl::InternalError(
          absl::StrCat("Failed to read content of file '", entry_name, "' from ZIP archive. Read ",
                       bytes_read, " bytes, expected ", file_info->uncompressed_size, "."));
    }
  }
  // If file_info->uncompressed_size is 0, content will be an empty string.

  err = mz_zip_entry_read_close(zip_archive_handle, nullptr, nullptr, nullptr);
  if (err != MZ_OK) {
    return absl::InternalError(absl::StrCat("Failed to close file '", entry_name,
                                            "' in ZIP archive after reading. Error code: ", err));
  }

  return content;
}

// Helper function to parse a string containing multiple lines into a vector of non-empty strings.
std::vector<std::string> ParseLines(const std::string& multiline_string) {
  std::vector<std::string> lines;
  if (multiline_string.empty()) {
    return lines;
  }
  std::istringstream iss(multiline_string);
  std::string line;
  while (std::getline(iss, line)) {
    // Remove carriage return if present (for Windows-style \r\n line endings)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    // As per "No empty names are allowed."
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

}  // namespace

absl::StatusOr<PackagedPlugin> LoadPackagedPlugin(const std::string& path) {
  MzOsStream os_file_stream;
  if (os_file_stream.handle == nullptr) {
    return absl::InternalError("Failed to create OS file stream object.");
  }

  int32_t err = mz_stream_os_open(os_file_stream.handle, path.c_str(), MZ_OPEN_MODE_READ);
  if (err != MZ_OK) {
    return absl::NotFoundError(
        absl::StrCat("Cannot open file path '", path, "'. OS error code: ", err));
  }
  os_file_stream.is_open = true;

  MzZip zip_reader;  // Uses the MzZip class from the user's initial code
  if (zip_reader.handle == nullptr) {
    return absl::InternalError("Failed to create ZIP reader object.");
  }

  err = mz_zip_open(zip_reader.handle, os_file_stream.handle, MZ_OPEN_MODE_READ);
  if (err != MZ_OK) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to open '", path,
        "' as a ZIP archive. It might be corrupted or not a valid ZIP file. Error code: ", err));
  }
  zip_reader.is_open = true;

  PackagedPlugin plugin_data;
  absl::Status status;

  // Lambda to simplify reading a file entry and assigning its content.
  auto read_and_assign_entry = [&](const char* filename,
                                   std::string* target_field) -> absl::Status {
    absl::StatusOr<std::string> content = ReadEntryToString(zip_reader.handle, filename);
    if (!content.ok()) {
      return content.status();
    }
    *target_field = std::move(content.value());
    return absl::OkStatus();
  };

  // Lambda to simplify reading a file entry, parsing its lines, and assigning.
  auto read_parse_and_assign_entry = [&](const char* filename,
                                         std::vector<std::string>* target_vector) -> absl::Status {
    absl::StatusOr<std::string> content = ReadEntryToString(zip_reader.handle, filename);
    if (!content.ok()) {
      return content.status();
    }
    *target_vector = ParseLines(*content);
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(read_and_assign_entry(kNameFilename, &plugin_data.name));
  RETURN_IF_ERROR(read_and_assign_entry(kInitMlirFilename, &plugin_data.init_mlir));
  RETURN_IF_ERROR(read_and_assign_entry(kUpdateMlirFilename, &plugin_data.update_mlir));
  RETURN_IF_ERROR(
      read_parse_and_assign_entry(kInputBufferNamesFilename, &plugin_data.input_buffer_names));
  RETURN_IF_ERROR(
      read_parse_and_assign_entry(kOutputBufferNamesFilename, &plugin_data.output_buffer_names));

  err = mz_zip_close(zip_reader.handle);
  if (err != MZ_OK) {
    return absl::InternalError(
        absl::StrCat("Failed to cleanly close ZIP archive '", path, "'. Error code: ", err));
  }
  zip_reader.is_open = false;
  return plugin_data;
}

}  // namespace jxap
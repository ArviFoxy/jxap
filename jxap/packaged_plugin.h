// Loading plugins from a .jxap ZIP file.

#ifndef JXAP_PACKAGED_PLUGIN
#define JXAP_PACKAGED_PLUGIN

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace jxap {

class PackagedPlugin {
 public:
  std::string name;
  std::string init_mlir;
  std::string update_mlir;
  std::vector<std::string> input_buffer_names;
  std::vector<std::string> output_buffer_names;
};

// Loads the packaged plugin from a .jxap zip file.
absl::StatusOr<PackagedPlugin> LoadPackagedPlugin(const std::string& path);

}  // namespace jxap

#endif  // JXAP_PACKAGED_PLUGIN

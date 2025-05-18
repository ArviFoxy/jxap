#include <fstream>
#include <filesystem>

#include "absl/log/log.h"
#include "jxap/hlo_opt.h"

namespace jxap {
  
std::string FindHloOptBinary() {
  // TODO: Check if it's in PATH and print a nice error message.
  return "jxap-hlo-opt";
}

} // namespace jxap

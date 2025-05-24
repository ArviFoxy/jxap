#ifndef JXAP_MLIR_PLUGIN
#define JXAP_MLIR_PLUGIN

#include <cstdint>
#include <memory>
#include <set>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace jxap {

class PJRTContext;
class PJRTExecutable;
class PJRTPluginRunner;

class PJRTCompiledPlugin {
 protected:
  PJRTCompiledPlugin() = default;
  friend PJRTPluginRunner;

 public:
  ~PJRTCompiledPlugin();

  int buffer_size() const { return buffer_size_; }

  float sample_rate() const { return sample_rate_; }

  const std::set<std::string>& input_buffers() const { return input_buffers_; }

  const std::set<std::string>& output_buffers() const { return output_buffers_; }

 private:
  int buffer_size_;
  float sample_rate_;
  std::set<std::string> input_buffers_;
  std::set<std::string> output_buffers_;
  std::unique_ptr<PJRTExecutable> init_fn_;
  std::unique_ptr<PJRTExecutable> update_fn_;
};

/**
 * Audio plugin runner usig XLA's PJRT.
 *
 * Loads StableHLO from `jax.export` and handles shape refining and JIT compilation.
 */
class PJRTPluginRunner {
 protected:
  PJRTPluginRunner() = default;

 public:
  ~PJRTPluginRunner();

  /**
   * Loads the plugin from a jxap file.
   */
  static absl::StatusOr<std::unique_ptr<PJRTPluginRunner>> LoadPlugin(absl::string_view path);

  /**
   * Compiles the plugin, filling in static buffer shapes.
   */
  absl::StatusOr<std::unique_ptr<PJRTCompiledPlugin>> Compile(
      const std::set<std::string>& input_buffers, const std::set<std::string>& output_buffers,
      int buffer_size, float sample_rate);

 private:
  std::unique_ptr<PJRTContext> ctx_;
  std::string init_fn_mlir_;
  std::string update_fn_mlir_;
};

}  // namespace jxap

#endif  // JXAP_MLIR_PLUGIN

#ifndef JXAP_MLIR_PLUGIN
#define JXAP_MLIR_PLUGIN

#include <cstdint>
#include <memory>
#include <map>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace jxap {

class PJRTContext;
class PJRTExecutable;
class PJRTPluginRunner;

/**
 * Specification of an audio buffer size.
 */
struct AudioBufferSpec {
  int32_t num_channels;
  int32_t buffer_size;
};

class PJRTCompiledPlugin {
 protected:
  PJRTCompiledPlugin() = default;
  friend PJRTPluginRunner;

 public:
  ~PJRTCompiledPlugin();

  const std::map<std::string, AudioBufferSpec>& input_buffer_specs() const {
    return input_buffer_specs_;
  }

  const std::map<std::string, AudioBufferSpec>& output_buffer_specs() const {
    return output_buffer_specs_;
  }

 private:
  std::map<std::string, AudioBufferSpec> input_buffer_specs_;
  std::map<std::string, AudioBufferSpec> output_buffer_specs_;
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
   absl::StatusOr<PJRTCompiledPlugin> Compile(
    const std::map<std::string, AudioBufferSpec>& input_buffer_specs,
    const std::map<std::string, AudioBufferSpec>& output_buffer_specs,
    float sample_rate);

  private:
   std::unique_ptr<PJRTContext> ctx_;
   std::string init_fn_mlir_;
   std::string update_fn_mlir_;
};

}  // namespace jxap

#endif  // JXAP_MLIR_PLUGIN

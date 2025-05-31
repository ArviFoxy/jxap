#ifndef JXAP_MLIR_PLUGIN
#define JXAP_MLIR_PLUGIN

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <span>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace jxap {

class PJRTContext;
class PJRTExecutable;
class PJRTPluginRunner;

// Memory buffer.
using Buffer = std::vector<std::byte>;
// State of a plugin.
using PluginState = std::vector<Buffer>;

class PJRTCompiledPlugin {
 protected:
  PJRTCompiledPlugin() = default;
  friend PJRTPluginRunner;

 public:
  ~PJRTCompiledPlugin();

  int audio_buffer_size() const { return audio_buffer_size_; }

  float sample_rate() const { return sample_rate_; }

  const std::set<std::string>& input_buffer_names() const { return input_buffer_names_; }

  const std::set<std::string>& output_buffer_names() const { return output_buffer_names_; }

  // Initializes an instance of the plugin.
  absl::StatusOr<PluginState> Init(const std::vector<Buffer>& inputs) const;

  // Updates the state of the plugin.
  // Takes ownership of the input buffers to avoid copying.
  // Writes the outputs to the output buffers (resizing them if needed).
  absl::Status Update(const std::vector<Buffer>& inputs, PluginState* state,
                      std::vector<Buffer>* outputs) const;

 private:
  PJRTContext* ctx_;

  int64_t audio_buffer_size_;
  PJRT_Buffer_Type audio_buffer_type_ = PJRT_Buffer_Type_F32;
  float sample_rate_;

  size_t state_size_;
  std::vector<PJRT_Buffer_Type> state_types_;
  std::vector<std::vector<int64_t>> state_dimensions_;

  std::set<std::string> input_buffer_names_;
  std::set<std::string> output_buffer_names_;

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
  std::string path_;
  std::string init_fn_mlir_;
  std::string update_fn_mlir_;
};

}  // namespace jxap

#endif  // JXAP_MLIR_PLUGIN

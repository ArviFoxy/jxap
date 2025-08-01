// Pipewire client that runs a plugin using the JXAP PjRt runtime.
// This version uses the dual pw_stream architecture for robust session manager
// integration.

#include <absl/base/log_severity.h>
#include <absl/container/flat_hash_map.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_cat.h>
#include <errno.h>
#include <math.h>
#include <pipewire/pipewire.h>
#include <pipewire/stream.h>
#include <signal.h>
#include <spa/buffer/buffer.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/latency-utils.h>
#include <spa/param/props.h>
#include <spa/pod/builder.h>
#include <spa/pod/iter.h>
#include <stdio.h>

#include "jxap/packaged_plugin.h"
#include "jxap/pjrt_plugin_runner.h"
#include "jxap/utils.h"

ABSL_FLAG(std::string, plugin_path, "", "JXAP plugin path.");
ABSL_FLAG(std::string, node_name, "jxap-filter", "Pipewire node name.");

struct Data {
  Data() { spa_zero(audio_info); }

  std::mutex mutex;
  pw_main_loop *loop;
  pw_context *context;

  // Two streams for the dual-stream pattern
  pw_stream *capture_stream;
  pw_stream *playback_stream;

  // JXAP plugin state
  std::unique_ptr<jxap::PJRTPluginRunner> runner;
  std::unique_ptr<jxap::PJRTCompiledPlugin> compiled_plugin;
  std::optional<jxap::PluginState> plugin_state;

  // Position info, updated by the io_changed callback
  spa_io_position *position = nullptr;
  spa_audio_info_raw audio_info;
};

float FracToFloat(const spa_fraction &frac) {
  if (frac.denom == 0) return 0.0f;
  return static_cast<float>(frac.num) / static_cast<float>(frac.denom);
}

// RAII helper that returns buffers to PipeWire after processing.
class DequeuedBuffer {
 public:
  DequeuedBuffer(pw_buffer *buffer, pw_stream *stream) : buffer_(buffer), stream_(stream) {}

  ~DequeuedBuffer() {
    if (buffer_) {
      pw_stream_queue_buffer(stream_, buffer_);
    }
  }

  spa_buffer *get() { return buffer_->buffer; }

 private:
  pw_buffer *buffer_;
  pw_stream *stream_;
};

absl::Status RecompileIfNeeded(Data *data) {
  if (!data->position) {
    LOG(WARNING) << "No position info available, cannot determine buffer size.";
    return absl::InternalError("No position info available");
  }
  if (data->audio_info.rate == 0) {
    LOG(WARNING) << "No audio info available, cannot determine sample rate.";
    return absl::InternalError("No audio info available");
  }
  const uint32_t n_samples = data->position->clock.duration;
  const float sampling_rate = data->audio_info.rate;

  if (data->compiled_plugin) {
    if (data->compiled_plugin->audio_buffer_size() == n_samples &&
        data->compiled_plugin->sample_rate() == sampling_rate) {
      return absl::OkStatus();  // No need to recompile
    }
  }

  LOG(INFO) << "Compiling plugin for " << n_samples << " samples buffer size at " << sampling_rate
            << " Hz";
  auto plugin_or_status = data->runner->Compile(n_samples, sampling_rate);
  RETURN_IF_ERROR(plugin_or_status.status());
  data->compiled_plugin = std::move(plugin_or_status.value());
  data->plugin_state = std::nullopt;
  return absl::OkStatus();
}

extern "C" {
// This is the main processing callback, driven by the playback stream.
// It synchronizes buffers from both capture and playback streams.
void on_playback_process(void *user_data) {
  Data *data = static_cast<Data *>(user_data);
  std::lock_guard<std::mutex> lock(data->mutex);

  // Dequeue buffers from both streams.
  DequeuedBuffer capture_buffer(pw_stream_dequeue_buffer(data->capture_stream),
                                data->capture_stream);
  if (!capture_buffer.get()) {
    LOG(WARNING) << "Out of capture buffers.";
    return;
  }
  DequeuedBuffer playback_buffer(pw_stream_dequeue_buffer(data->playback_stream),
                                 data->playback_stream);
  if (!playback_buffer.get()) {
    LOG(WARNING) << "Out of playback buffers.";
    return;
  }

  if (!data->position) {
    LOG(WARNING) << "No position info available, cannot determine buffer size.";
    return;
  }
  const uint32_t n_samples = data->position->clock.duration;

  auto status = RecompileIfNeeded(data);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to recompile plugin: " << status;
    return;
  }

  // Prepare input buffers for the JXAP plugin.
  size_t buffer_size = n_samples * sizeof(float);
  std::vector<jxap::Buffer> input_buffers(1);  // Assuming mono input for now
  void *capture_data = capture_buffer.get()->datas[0].data;
  if (capture_data == nullptr) {
    LOG(WARNING) << "Capture buffer has no data.";
    return;
  }
  input_buffers[0].resize(buffer_size);
  std::memcpy(input_buffers[0].data(), capture_data, buffer_size);

  // Initialize the JXAP plugin if not already done.
  if (data->plugin_state == std::nullopt) {
    auto status_or_state = data->compiled_plugin->Init(input_buffers);
    if (!status_or_state.ok()) {
      LOG(ERROR) << "Failed to initialize plugin: " << status_or_state.status();
      return;
    }
    data->plugin_state = std::move(status_or_state.value());
  }

  // Run the JXAP plugin.
  std::vector<jxap::Buffer> output_buffers(1);
  status =
      data->compiled_plugin->Update(input_buffers, &data->plugin_state.value(), &output_buffers);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to update plugin: " << status;
    return;
  }

  // Copy plugin output to the playback buffer.
  if (!output_buffers.empty()) {
    void *playback_data = playback_buffer.get()->datas[0].data;
    if (playback_data) {
      size_t buffer_size = n_samples * sizeof(float);
      if (output_buffers[0].size() != buffer_size) {
        LOG(WARNING) << "Plugin output buffer size mismatch.";
        return;
      }
      std::memcpy(playback_data, output_buffers[0].data(), buffer_size);
      playback_buffer.get()->datas[0].chunk->offset = 0;
      playback_buffer.get()->datas[0].chunk->stride = sizeof(float);
      playback_buffer.get()->datas[0].chunk->size = buffer_size;
    }
  }
}

// The capture stream's process callback simply triggers the playback one.
// This makes the capture stream the "driver" of our processing chain.
void on_capture_process(void *user_data) {
  Data *data = static_cast<Data *>(user_data);
  if (pw_stream_trigger_process(data->playback_stream) < 0) {
    // Playback side is not ready, so we dequeue and immediately requeue
    // the capture buffer to avoid stalling the graph.
    pw_buffer *capture_buf;
    while ((capture_buf = pw_stream_dequeue_buffer(data->capture_stream)) != nullptr) {
      pw_stream_queue_buffer(data->capture_stream, capture_buf);
    }
  }
}

void on_capture_state_changed(void *user_data, enum pw_stream_state old, enum pw_stream_state state,
                              const char *error) {
  Data *data = static_cast<Data *>(user_data);

  LOG(INFO) << "Capture stream changed from " << pw_stream_state_as_string(old) << " to "
            << pw_stream_state_as_string(state);

  switch (state) {
    case PW_STREAM_STATE_STREAMING:
      break;

    case PW_STREAM_STATE_PAUSED:
      pw_stream_flush(data->playback_stream, false);
      break;

    case PW_STREAM_STATE_UNCONNECTED:
      break;

    case PW_STREAM_STATE_ERROR:
      LOG(ERROR) << "Capture stream entered error state: " << error;
      pw_main_loop_quit(data->loop);
      break;

    default:
      break;
  }
}

void on_playback_state_changed(void *user_data, enum pw_stream_state old,
                               enum pw_stream_state state, const char *error) {
  Data *data = static_cast<Data *>(user_data);
  std::lock_guard<std::mutex> lock(data->mutex);

  LOG(INFO) << "Playback stream changed from " << pw_stream_state_as_string(old) << " to "
            << pw_stream_state_as_string(state);

  switch (state) {
    case PW_STREAM_STATE_STREAMING: {
      auto status = RecompileIfNeeded(data);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to recompile plugin: " << status;
      }
      break;
    }

    case PW_STREAM_STATE_PAUSED: {
      auto status = RecompileIfNeeded(data);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to recompile plugin: " << status;
      }
      pw_stream_flush(data->playback_stream, false);
      break;
    }

    case PW_STREAM_STATE_UNCONNECTED: {
      data->compiled_plugin.reset();
      data->plugin_state = std::nullopt;
    } break;

    case PW_STREAM_STATE_ERROR:
      LOG(ERROR) << "Playback stream entered error state: " << error;
      pw_main_loop_quit(data->loop);
      break;

    default:
      break;
  }
}

void on_param_changed(void *user_data, uint32_t id, const struct spa_pod *param,
                      enum spa_direction direction) {
  if (param == nullptr) {
    return;
  }

  Data *data = static_cast<Data *>(user_data);
  pw_stream *stream =
      (direction == SPA_DIRECTION_INPUT) ? data->capture_stream : data->playback_stream;
  if (stream == nullptr) {
    LOG(ERROR) << "Stream is null, cannot handle param change.";
    return;
  }
  std::string stream_name = pw_stream_get_name(stream);

  switch (id) {
    case SPA_PARAM_Format: {
      std::lock_guard<std::mutex> lock(data->mutex);
      spa_zero(data->audio_info);
      if (spa_format_audio_raw_parse(param, &data->audio_info) >= 0) {
        LOG(INFO) << "Stream `" << stream_name << "` format set: " << data->audio_info.channels
                  << " channels at " << data->audio_info.rate << " Hz";
      } else {
        LOG(ERROR) << "Failed to parse playback stream format.";
        return;
      }
      break;
    }

    case SPA_PARAM_Buffers: {
      int num_buffers;
      int result = spa_pod_parse_object(param, SPA_TYPE_OBJECT_ParamBuffers, nullptr,
                                        SPA_PARAM_BUFFERS_buffers, SPA_POD_OPT_Int(&num_buffers));
      LOG(INFO) << "Stream `" << stream_name << "` number of buffers changed: " << num_buffers;
      break;
    }

    case SPA_PARAM_IO:
      // Handle IO area changes.
      break;

    default:
      LOG(INFO) << "Unhandled playback param change: " << id;
      return;
  }
  if (param == nullptr || id != SPA_PARAM_Format) {
    return;
  }
}

void on_capture_param_changed(void *user_data, uint32_t id, const struct spa_pod *param) {
  on_param_changed(user_data, id, param, SPA_DIRECTION_INPUT);
}

void on_playback_param_changed(void *user_data, uint32_t id, const struct spa_pod *param) {
  on_param_changed(user_data, id, param, SPA_DIRECTION_OUTPUT);
}

// This callback receives timing information from the graph driver.
void on_playback_io_changed(void *user_data, uint32_t id, void *area, uint32_t size) {
  Data *data = static_cast<Data *>(user_data);
  std::lock_guard<std::mutex> lock(data->mutex);
  if (id == SPA_IO_Position) {
    data->position = static_cast<spa_io_position *>(area);
    LOG(INFO) << "Playback stream position updated: "
              << "duration = " << data->position->clock.duration
              << ", rate = " << 1.0f / FracToFloat(data->position->clock.rate)
              << ", nsec = " << data->position->clock.nsec;
    auto status = RecompileIfNeeded(data);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to recompile plugin: " << status;
    }
  }
}

const struct pw_stream_events kCaptureStreamEvents = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_capture_state_changed,
    .param_changed = on_capture_param_changed,
    .process = on_capture_process,
};

const struct pw_stream_events kPlaybackStreamEvents = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_playback_state_changed,
    .io_changed = on_playback_io_changed,
    .param_changed = on_playback_param_changed,
    .process = on_playback_process,
};

void do_quit(void *user_data, int signal_number) {
  Data *data = static_cast<Data *>(user_data);
  pw_main_loop_quit(data->loop);
}

}  // extern "C"

int main(int argc, char **argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::SetProgramUsageMessage("Runs a JXAP plugin as a pipewire filter.");
  std::vector<char *> remaining_flags = absl::ParseCommandLine(argc, argv);
  int new_argc = remaining_flags.size();
  char **new_argv = remaining_flags.data();
  pw_init(&new_argc, &new_argv);

  // Load the plugin.
  auto plugin_or_status = jxap::LoadPackagedPlugin(absl::GetFlag(FLAGS_plugin_path));
  if (!plugin_or_status.ok()) {
    LOG(ERROR) << "Failed to load plugin: " << plugin_or_status.status();
    return -1;
  }
  auto plugin = std::move(plugin_or_status.value());

  // Create the plugin runner.
  auto runner_or_status = jxap::PJRTPluginRunner::LoadPlugin(plugin);
  if (!runner_or_status.ok()) {
    LOG(ERROR) << "Failed to create plugin runner: " << runner_or_status.status();
    return -1;
  }
  Data pw_data;
  pw_data.runner = std::move(runner_or_status.value());

  LOG(INFO) << "Creating Pipewire loop";
  pw_data.loop = pw_main_loop_new(nullptr);
  pw_data.context = pw_context_new(pw_main_loop_get_loop(pw_data.loop), nullptr, 0);
  pw_loop_add_signal(pw_main_loop_get_loop(pw_data.loop), SIGINT, do_quit, &pw_data);
  pw_loop_add_signal(pw_main_loop_get_loop(pw_data.loop), SIGTERM, do_quit, &pw_data);

  // Create Playback Stream (Source)
  std::string node_name = absl::GetFlag(FLAGS_node_name);
  std::string playback_name = absl::StrCat(node_name, "-playback");
  pw_data.playback_stream =
      pw_stream_new_simple(pw_main_loop_get_loop(pw_data.loop), playback_name.c_str(),
                           pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",                     //
                                             PW_KEY_MEDIA_CATEGORY, "Playback",              //
                                             PW_KEY_MEDIA_CLASS, "Audio/Source",             //
                                             PW_KEY_MEDIA_ROLE, "DSP",                       //
                                             PW_KEY_NODE_DESCRIPTION, "JXAP Plugin Output",  //
                                             PW_KEY_NODE_GROUP, node_name.c_str(),           //
                                             PW_KEY_NODE_LINK_GROUP, node_name.c_str(),      //
                                             PW_KEY_NODE_VIRTUAL, "true",                    //
                                             // "resample.prefill", "true",                     //
                                             nullptr),
                           &kPlaybackStreamEvents, &pw_data);

  // Create Capture Stream (Sink)
  std::string capture_name = absl::StrCat(absl::GetFlag(FLAGS_node_name), "-capture");
  pw_data.capture_stream =
      pw_stream_new_simple(pw_main_loop_get_loop(pw_data.loop), capture_name.c_str(),
                           pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",                    //
                                             PW_KEY_MEDIA_CATEGORY, "Capture",              //
                                             PW_KEY_MEDIA_CLASS, "Audio/Sink",              //
                                             PW_KEY_MEDIA_ROLE, "DSP",                      //
                                             PW_KEY_NODE_DESCRIPTION, "JXAP Plugin Input",  //
                                             PW_KEY_NODE_GROUP, node_name.c_str(),          //
                                             PW_KEY_NODE_LINK_GROUP, node_name.c_str(),     //
                                             PW_KEY_NODE_VIRTUAL, "true",                   //
                                             // "resample.prefill", "true",                    //
                                             nullptr),
                           &kCaptureStreamEvents, &pw_data);

  // Connect Streams
  uint8_t buffer[1024];
  spa_pod_builder pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
  const spa_pod *params[2];
  spa_audio_info_raw format_info =
      SPA_AUDIO_INFO_RAW_INIT(.format = SPA_AUDIO_FORMAT_F32P, .channels = 1,
                              .position = {SPA_AUDIO_CHANNEL_MONO});
  params[0] = spa_format_audio_raw_build(&pod_builder, SPA_PARAM_EnumFormat, &format_info);
  params[1] = reinterpret_cast<spa_pod *>(
      spa_pod_builder_add_object(&pod_builder, SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                                 SPA_PARAM_BUFFERS_stride, SPA_POD_Int(4),  //
                                 SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(8, 4, 32)));

  // Connect playback stream first, with the TRIGGER flag.
  if (pw_stream_connect(pw_data.playback_stream, PW_DIRECTION_OUTPUT, PW_ID_ANY,
                        (pw_stream_flags)(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS |
                                          PW_STREAM_FLAG_RT_PROCESS | PW_STREAM_FLAG_TRIGGER),
                        params, 2) < 0) {
    LOG(ERROR) << "Failed to connect playback stream.";
    return -1;
  }

  // Connect capture stream.
  if (pw_stream_connect(pw_data.capture_stream, PW_DIRECTION_INPUT, PW_ID_ANY,
                        (pw_stream_flags)(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS |
                                          PW_STREAM_FLAG_RT_PROCESS),
                        params, 2) < 0) {
    LOG(ERROR) << "Failed to connect capture stream.";
    return -1;
  }

  LOG(INFO) << "Streams connecting... Press Ctrl-C to exit.";
  pw_main_loop_run(pw_data.loop);

  LOG(INFO) << "Cleaning up...";
  pw_stream_destroy(pw_data.capture_stream);
  pw_stream_destroy(pw_data.playback_stream);
  pw_context_destroy(pw_data.context);
  pw_main_loop_destroy(pw_data.loop);
  pw_deinit();

  return EXIT_SUCCESS;
}

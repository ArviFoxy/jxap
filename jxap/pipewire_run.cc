// Pipewire client that runs a plugin using the JXAP PjRt runtime.

#include <absl/container/flat_hash_map.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <errno.h>
#include <math.h>
#include <pipewire/filter.h>
#include <pipewire/pipewire.h>
#include <signal.h>
#include <spa/param/latency-utils.h>
#include <spa/pod/builder.h>
#include <stdio.h>

#include "jxap/packaged_plugin.h"
#include "jxap/pjrt_plugin_runner.h"

ABSL_FLAG(std::string, plugin_path, "", "JXAP plugin path.");
ABSL_FLAG(std::string, node_name, "", "Pipewire node name.");

struct Port {};

struct Data {
  pw_main_loop *loop;
  pw_filter *filter;
  std::vector<Port *> in_ports;
  std::vector<Port *> out_ports;
  std::unique_ptr<jxap::PJRTPluginRunner> runner;
  std::unique_ptr<jxap::PJRTCompiledPlugin> compiled_plugin;
  jxap::PluginState plugin_state;
};

float FracToFloat(const spa_fraction &frac) {
  return static_cast<float>(frac.num) / static_cast<float>(frac.denom);
}

extern "C" {

void on_process(void *user_data, struct spa_io_position *position) {
  Data *data = reinterpret_cast<Data *>(user_data);
  const uint32_t n_samples = position->clock.duration;
  const float sampling_rate = FracToFloat(position->clock.rate);
  pw_log_trace("do process %d %d", n_samples, sampling_rate);

  std::vector<jxap::Buffer> input_buffers(data->in_ports.size());
  for (size_t i = 0; i < data->in_ports.size(); ++i) {
    // TODO: reduce copying
    void *dsp_buffer = pw_filter_get_dsp_buffer(data->in_ports[i], n_samples);
    if (dsp_buffer == nullptr) return;
    input_buffers[i].resize(n_samples);
    std::memcmp(input_buffers[i].data(), dsp_buffer, n_samples * sizeof(float));
  }

  if (!data->compiled_plugin) {
    auto plugin_or_status = data->runner->Compile(n_samples, sampling_rate);
    if (!plugin_or_status.ok()) {
      pw_log_error("Failed to compile plugin: %s", plugin_or_status.status().ToString().c_str());
      return;
    }
    data->compiled_plugin = std::move(plugin_or_status.value());
    auto status_or_state = data->compiled_plugin->Init(input_buffers);
    if (!status_or_state.ok()) {
      pw_log_error("Failed to initialize plugin: %s", status_or_state.status().ToString().c_str());
      return;
    }
    // TODO: reduce copying
    data->plugin_state = std::move(status_or_state.value());
  }

  std::vector<jxap::Buffer> output_buffers;
  auto status = data->compiled_plugin->Update(input_buffers, &data->plugin_state, &output_buffers);
  if (!status.ok()) {
    pw_log_error("Failed to update plugin: %s", status.ToString().c_str());
    return;
  }

  for (size_t i = 0; i < data->out_ports.size(); ++i) {
    // TODO: reduce copying
    void *dsp_buffer = pw_filter_get_dsp_buffer(data->out_ports[i], n_samples);
    if (dsp_buffer == nullptr) return;
    std::memcpy(dsp_buffer, output_buffers[i].data(), n_samples * sizeof(float));
  }
}

const struct pw_filter_events kFilterEvents = {
    .version = PW_VERSION_FILTER_EVENTS,
    .process = on_process,
};

void do_quit(void *user_data, int signal_number) {
  Data *data = reinterpret_cast<Data *>(user_data);
  pw_main_loop_quit(data->loop);
}

}  // namespace

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Runs a JXAP plugin as a pipewire filter.");
  std::vector<char *> remaining_flags = absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
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

  Data pw_data;
  pw_data.loop = pw_main_loop_new(nullptr);
  pw_loop_add_signal(pw_main_loop_get_loop(pw_data.loop), SIGINT, do_quit, &pw_data);
  pw_loop_add_signal(pw_main_loop_get_loop(pw_data.loop), SIGTERM, do_quit, &pw_data);

  std::vector<uint8_t> spa_pod_buffer(1024);
  struct spa_pod_builder b =
      SPA_POD_BUILDER_INIT(spa_pod_buffer.data(), static_cast<uint32_t>(spa_pod_buffer.size()));

  // Create the pipewire filter.
  std::string filter_name = absl::GetFlag(FLAGS_node_name);
  pw_data.filter = pw_filter_new_simple(
      //
      pw_main_loop_get_loop(pw_data.loop), filter_name.c_str(),
      pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",         //
                        PW_KEY_MEDIA_CATEGORY, "Playback",  //
                        PW_KEY_MEDIA_ROLE, "DSP",           //
                        nullptr),
      &kFilterEvents, &pw_data);

  // Create the input/output ports.
  for (const auto &input_buffer_name : plugin.input_buffer_names) {
    void *port_data = pw_filter_add_port(
        pw_data.filter, PW_DIRECTION_INPUT, PW_FILTER_PORT_FLAG_MAP_BUFFERS, sizeof(Port),
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",  //
                          PW_KEY_PORT_NAME, input_buffer_name.c_str(),   //
                          nullptr),
        /*params=*/nullptr, /*nparams=*/0);
    pw_data.in_ports.push_back(reinterpret_cast<Port *>(port_data));
  }
  for (const auto &output_buffer_name : plugin.output_buffer_names) {
    void *port_data = pw_filter_add_port(
        pw_data.filter, PW_DIRECTION_OUTPUT, PW_FILTER_PORT_FLAG_MAP_BUFFERS, sizeof(Port),
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",  //
                          PW_KEY_PORT_NAME, output_buffer_name.c_str(),  //
                          nullptr),
        /*params=*/nullptr, /*nparams=*/0);
    pw_data.in_ports.push_back(reinterpret_cast<Port *>(port_data));
  }

  // Connct the filter. We ask that our process function is called in a realtime thread.
  std::vector<const spa_pod *> params;
  auto process_latency = SPA_PROCESS_LATENCY_INFO_INIT(.ns = 10 * SPA_NSEC_PER_MSEC);
  params.push_back(spa_process_latency_build(&b, SPA_PARAM_ProcessLatency, &process_latency));
  if (pw_filter_connect(pw_data.filter, PW_FILTER_FLAG_RT_PROCESS, params.data(), params.size()) <
      0) {
    fprintf(stderr, "can't connect\n");
    return -1;
  }

  /* and wait while we let things run */
  pw_main_loop_run(pw_data.loop);

  pw_filter_destroy(pw_data.filter);
  pw_main_loop_destroy(pw_data.loop);
  pw_deinit();

  return EXIT_SUCCESS;
}

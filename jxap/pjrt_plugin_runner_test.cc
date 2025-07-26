#include "jxap/pjrt_plugin_runner.h"

#include <memory>
#include <span>

#include "absl/log/log.h"
#include "gtest/gtest.h"

namespace jxap {
namespace {

const std::string kTestPluginPath = "jxap/testdata/test_plugin.jxap";
const std::string kLoopPluginPath = "jxap/testdata/loop_plugin.jxap";

TEST(PJRTPluginRunnerTest, CompilesWithLoop) {
  auto packaged_plugin_or_status = LoadPackagedPlugin(kLoopPluginPath);
  ASSERT_TRUE(packaged_plugin_or_status.ok()) << packaged_plugin_or_status.status();
  auto plugin_or_status = PJRTPluginRunner::LoadPlugin(packaged_plugin_or_status.value());
  ASSERT_TRUE(plugin_or_status.ok()) << plugin_or_status.status();

  int64_t buffer_size = 128;
  auto compiled_plugin_or_status = plugin_or_status.value()->Compile(
      /*buffer_size=*/buffer_size,
      /*sample_rate=*/44100.f);
  ASSERT_TRUE(compiled_plugin_or_status.ok()) << compiled_plugin_or_status.status();

  auto compiled_plugin = std::move(compiled_plugin_or_status.value());
  ASSERT_EQ(compiled_plugin->input_buffer_names(), std::vector<std::string>{"input"});
  ASSERT_EQ(compiled_plugin->output_buffer_names(), std::vector<std::string>{"output"});
  ASSERT_EQ(compiled_plugin->audio_buffer_size(), buffer_size);
  ASSERT_EQ(compiled_plugin->sample_rate(), 44100.f);
}

TEST(PJRTPluginRunnerTest, CompilesAndRunsPlugin) {
  auto packaged_plugin_or_status = LoadPackagedPlugin(kTestPluginPath);
  ASSERT_TRUE(packaged_plugin_or_status.ok()) << packaged_plugin_or_status.status();
  auto plugin_or_status = PJRTPluginRunner::LoadPlugin(packaged_plugin_or_status.value());
  ASSERT_TRUE(plugin_or_status.ok()) << plugin_or_status.status();

  int64_t buffer_size = 4;
  auto compiled_plugin_or_status = plugin_or_status.value()->Compile(
      /*buffer_size=*/buffer_size,
      /*sample_rate=*/44100.f);
  ASSERT_TRUE(compiled_plugin_or_status.ok()) << compiled_plugin_or_status.status();

  auto compiled_plugin = std::move(compiled_plugin_or_status.value());
  ASSERT_EQ(compiled_plugin->input_buffer_names(), std::vector<std::string>{"input"});
  ASSERT_EQ(compiled_plugin->output_buffer_names(), std::vector<std::string>{"output"});
  ASSERT_EQ(compiled_plugin->audio_buffer_size(), buffer_size);
  ASSERT_EQ(compiled_plugin->sample_rate(), 44100.f);

  for (int try_num = 0; try_num < 100; ++try_num) {
    std::vector<Buffer> input_buffers = {Buffer(buffer_size * sizeof(float))};
    std::span<float> input_view(reinterpret_cast<float*>(input_buffers[0].data()), buffer_size);
    for (int i = 0; i < buffer_size; ++i) {
      input_view[i] = 1.0f;
    }
    LOG(INFO) << "Running init.";
    auto status_or_state = compiled_plugin->Init(input_buffers);
    ASSERT_TRUE(status_or_state.ok()) << status_or_state.status();
    PluginState state = std::move(status_or_state.value());
    LOG(INFO) << "Init done.";

    LOG(INFO) << "Running update.";
    std::vector<Buffer> output_buffers(1);
    auto status = compiled_plugin->Update(input_buffers, &state, &output_buffers);
    ASSERT_TRUE(status.ok()) << status;
    LOG(INFO) << "Update done.";

    {
      // check the output
      std::span<float> output_view(reinterpret_cast<float*>(output_buffers[0].data()), buffer_size);
      LOG(INFO) << "Outputs: " << output_view[0] << ", " << output_view[1] << ", " << output_view[2]
                << ", " << output_view[3];
      ASSERT_FLOAT_EQ(output_view[0], 1.0f);
      ASSERT_FLOAT_EQ(output_view[1], 2.0f);
      ASSERT_FLOAT_EQ(output_view[2], 2.0f);
      ASSERT_FLOAT_EQ(output_view[3], 2.0f);
    }

    input_view[0] = 0.0f;
    input_view[1] = 1.0f;
    input_view[2] = 2.0f;
    input_view[3] = 2.0f;

    LOG(INFO) << "Running update.";
    status = compiled_plugin->Update(input_buffers, &state, &output_buffers);
    ASSERT_TRUE(status.ok()) << status;
    LOG(INFO) << "Update done.";

    {
      // check the output
      std::span<float> output_view(reinterpret_cast<float*>(output_buffers[0].data()), buffer_size);
      LOG(INFO) << "Outputs: " << output_view[0] << ", " << output_view[1] << ", " << output_view[2]
                << ", " << output_view[3];
      ASSERT_FLOAT_EQ(output_view[0], 2.0f);
      ASSERT_FLOAT_EQ(output_view[1], 2.0f);
      ASSERT_FLOAT_EQ(output_view[2], 4.0f);
      ASSERT_FLOAT_EQ(output_view[3], 5.0f);
    }
  }
}

}  // namespace
}  // namespace jxap

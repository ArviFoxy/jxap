#include "jxap/pjrt_plugin_runner.h"

#include <memory>
#include <span>

#include "gtest/gtest.h"

namespace jxap {
namespace {

const std::string kTestPluginPath = "jxap/testdata/test_plugin.jxap";

TEST(PJRTPluginRunnerTest, CompilesAndRunsPlugin) {
  auto plugin_or_status = PJRTPluginRunner::LoadPlugin(kTestPluginPath);
  ASSERT_TRUE(plugin_or_status.ok()) << plugin_or_status.status();

  auto compiled_plugin_or_status = plugin_or_status.value()->Compile(
      /*input_buffers=*/{"input"},
      /*output_buffers=*/{"output"},
      /*buffer_size=*/128,
      /*sample_rate=*/44100.f);
  ASSERT_TRUE(compiled_plugin_or_status.ok()) << compiled_plugin_or_status.status();

  auto compiled_plugin = std::move(compiled_plugin_or_status.value());
  ASSERT_EQ(compiled_plugin->input_buffer_names(), std::set<std::string>{"input"});
  ASSERT_EQ(compiled_plugin->output_buffer_names(), std::set<std::string>{"output"});
  ASSERT_EQ(compiled_plugin->audio_buffer_size(), 128);
  ASSERT_EQ(compiled_plugin->sample_rate(), 44100.f);

  std::vector<Buffer> input_buffers = {Buffer(128 * sizeof(float))};
  std::span<float> input_view(reinterpret_cast<float*>(input_buffers[0].data()), 128);
  for (int i = 0; i < 128; ++i) {
    input_view[i] = 0.0f;
  }
  auto status = compiled_plugin->Init(std::move(input_buffers));
  ASSERT_TRUE(status.ok()) << status;
}

}  // namespace
}  // namespace jxap

#include "jxap/pjrt_plugin_runner.h"

#include <memory>

#include "gtest/gtest.h"

namespace jxap {
namespace {

const std::string kTestPluginPath = "jxap/testdata/test_plugin.jxap";

TEST(PJRTPluginRunnerTest, PluginLoading) {
  auto plugin_or_status = PJRTPluginRunner::LoadPlugin(kTestPluginPath);
  ASSERT_TRUE(plugin_or_status.ok()) << plugin_or_status.status();

  auto compiled_plugin_or_status = plugin_or_status.value()->Compile(
      /*input_buffers=*/{"input"},
      /*output_buffers=*/{"output"},
      /*buffer_size=*/128,
      /*sample_rate=*/44100.f);
  ASSERT_TRUE(compiled_plugin_or_status.ok()) << compiled_plugin_or_status.status();

  auto compiled_plugin = std::move(compiled_plugin_or_status.value());
  ASSERT_EQ(compiled_plugin->input_buffers(), std::set<std::string>{"input"});
  ASSERT_EQ(compiled_plugin->output_buffers(), std::set<std::string>{"output"});
  ASSERT_EQ(compiled_plugin->buffer_size(), 128);
  ASSERT_EQ(compiled_plugin->sample_rate(), 44100.f);
}

}  // namespace
}  // namespace jxap

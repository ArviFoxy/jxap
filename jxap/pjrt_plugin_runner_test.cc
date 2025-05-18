#include "gtest/gtest.h"
#include "jxap/pjrt_plugin_runner.h"

#include <memory>

namespace jxap {
namespace {

const std::string kTestPluginPath = "jxap/testdata/test_plugin.jxap";

TEST(PJRTPluginRunnerTest, PluginLoading) {
  auto plugin_or_status = PJRTPluginRunner::LoadPlugin(kTestPluginPath);
  ASSERT_TRUE(plugin_or_status.ok()) << plugin_or_status.status();
}

} // namespace
} // namespace jxap

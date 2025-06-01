#include "jxap/packaged_plugin.h"

#include <memory>
#include <span>

#include "gtest/gtest.h"

namespace jxap {
namespace {

const std::string kTestPluginPath = "jxap/testdata/test_plugin.jxap";

TEST(PackagedPlugin, LoadsPlugin) {
  auto status_or_plugin = LoadPackagedPlugin(kTestPluginPath);
  ASSERT_TRUE(status_or_plugin.ok()) << status_or_plugin.status();
  auto plugin = status_or_plugin.value();
  ASSERT_TRUE(plugin.init_mlir.find("func.func public @main") != std::string::npos);
  ASSERT_TRUE(plugin.update_mlir.find("func.func public @main") != std::string::npos);
  ASSERT_EQ(plugin.name, "TestPlugin()");
  ASSERT_EQ(plugin.input_buffer_names.size(), 1);
  ASSERT_EQ(plugin.input_buffer_names[0], "input");
  ASSERT_EQ(plugin.output_buffer_names.size(), 1);
  ASSERT_EQ(plugin.output_buffer_names[0], "output");
}

}  // namespace
}  // namespace jxap

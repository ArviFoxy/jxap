#include "jxap/mlir_pipeline.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"
#include "jxap/utils.h"

namespace jxap {
namespace {

const std::string kTestMlirPathInit = "jxap/testdata/test_plugin.jxap-init";
const std::string kTestMlirPathUpdate = "jxap/testdata/test_plugin.jxap-update";

TEST(MlirPipeline, PipelineTestPluginInit) {
  auto mlir = ReadFile(kTestMlirPathInit);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),    // platform index
      RefineType(MlirTensorType({32}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),             // sampling rate scalar
  };
  auto output = MlirPipeline(mlir.value(), transforms);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

TEST(MlirPipeline, PipelineTestPluginUpdate) {
  auto mlir = ReadFile(kTestMlirPathUpdate);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),    // platform index
      RefineType(MlirTensorType({}, "f32")),    // plugin state
      RefineType(MlirTensorType({32}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),             // sampling rate scalar
  };
  auto output = MlirPipeline(mlir.value(), transforms);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

}  // namespace
}  // namespace jxap

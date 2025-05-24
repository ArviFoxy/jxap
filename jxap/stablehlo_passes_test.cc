#include "jxap/stablehlo_passes.h"

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

const std::string kTestMlirPath = "jxap/testdata/test_plugin.jxap-init";

TEST(StablehloTest, TransformArguments) {
  auto mlir = ReadFile(kTestMlirPath);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),    // platform index
      RefineType(MlirTensorType({32}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),             // sampling rate scalar
  };
  auto output = MlirTransformArguments(mlir.value(), transforms);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

}  // namespace
}  // namespace jxap

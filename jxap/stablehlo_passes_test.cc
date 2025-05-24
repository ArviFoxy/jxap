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

TEST(StablehloTest, RefineTypes) {
  auto mlir = ReadFile(kTestMlirPath);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<std::string> input_types = {
      MlirTensorType({}, "i32"),       // platform index
      MlirTensorType({32, 2}, "f32"),  // input audio buffer
      MlirTensorType({}, "f32"),       // sampling rate scalar
  };
  auto output = RefineInputTypes(mlir.value(), input_types);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

}  // namespace
}  // namespace jxap

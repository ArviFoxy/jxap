#include "jxap/mlir/pipeline.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"
#include "jxap/mlir/utils.h"
#include "jxap/utils.h"

namespace jxap {
namespace {

const std::string kTestMlirPathInit = "jxap/testdata/test_plugin.jxap-init";
const std::string kTestMlirPathUpdate = "jxap/testdata/test_plugin.jxap-update";
const std::string kLoopMlirPath = "jxap/testdata/loop_plugin.jxap-update";

TEST(MlirPipeline, PipelineTestPluginInit) {
  auto mlir = ReadFile(kTestMlirPathInit);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),    // platform index
      RefineType(MlirTensorType({32}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),             // sampling rate scalar
  };
  std::map<std::string, ScalarValue> global_to_value;
  global_to_value["_platform_index"] = 0;
  global_to_value["BufferSize"] = 32;
  auto output = MlirPipeline(mlir.value(), transforms, global_to_value);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

TEST(MlirPipeline, PipelineTestPluginUpdate) {
  auto mlir = ReadFile(kTestMlirPathUpdate);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),    // platform index
      RefineType(MlirTensorType({}, "f32")),    // plugin state 1
      RefineType(MlirTensorType({32}, "f32")),  // plugin state 2
      RefineType(MlirTensorType({32}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),             // sampling rate scalar
  };
  std::map<std::string, ScalarValue> global_to_value;
  global_to_value["_platform_index"] = 0;
  global_to_value["BufferSize"] = 32;
  auto output = MlirPipeline(mlir.value(), transforms, global_to_value);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();
}

TEST(MlirPipeline, PipelineTestLoopOpt) {
  auto mlir = ReadFile(kLoopMlirPath);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  std::vector<ArgumentTransform> transforms = {
      RefineType(MlirTensorType({}, "i32")),     // platform index
      RefineType(MlirTensorType({}, "f32")),     // plugin state 1
      RefineType(MlirTensorType({}, "f32")),     // plugin state 2
      RefineType(MlirTensorType({128}, "f32")),  // input audio buffer
      ReplaceWithConstant(44100.f),              // sampling rate scalar
  };
  std::map<std::string, ScalarValue> global_to_value;
  global_to_value["_platform_index"] = 0;
  global_to_value["BufferSize"] = 128;
  auto output = MlirPipeline(mlir.value(), transforms, global_to_value);
  ASSERT_TRUE(output.ok()) << output.status();

  LOG(INFO) << "MLIR output:\n" << output.value();

  // Lower the MLIR to LLVM IR for debugging.
  // TODO: move to a tool.
  auto mlirModule = ParseMlirModule(output.value(), GetMlirContext());
  ASSERT_TRUE(mlirModule.ok()) << mlirModule.status();
  auto llvm_ir = LowerToOptimizedLLVMIR(mlirModule.value().get());
  ASSERT_TRUE(llvm_ir.ok()) << llvm_ir.status();
  LOG(INFO) << "LLVM IR output:\n" << llvm_ir.value();
}

}  // namespace
}  // namespace jxap

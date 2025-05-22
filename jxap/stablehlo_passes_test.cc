
#include <cstdlib>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "absl/status/statusor.h"
#include "absl/log/log.h"

#include "jxap/hlo_opt.h"
#include "jxap/utils.h"

namespace jxap {
namespace {

#ifndef JXAP_HLO_OPT_PATH
#error JXAP_HLO_OPT_PATH must be defined for tests.
#endif

#define _JXAP_QUOTE(x) #x

const std::string kTestMlirPath = "jxap/testdata/test_plugin.jxap-init";
const std::string kHloOptPath = _JXAP_QUOTE(#JXAP_HLO_OPT_PATH);

TEST(HloOptTest, HloOptExists) {
  std::vector<std::string> args = {
    kHloOptPath,
    "--version",
  };
  ASSERT_EQ(std::system(absl::StrJoin(args, " ").c_str()), EXIT_SUCCESS);
}

TEST(HloOptTest, RefineTypes) {
  auto mlir = ReadFile(kTestMlirPath);
  ASSERT_TRUE(mlir.ok()) << mlir.status();

  // Create temporary in/out files.
  auto tmp_dir = std::filesystem::temp_directory_path();
  auto mlir_in = tmp_dir / "mlir_in.mlir";
  auto mlir_out = tmp_dir / "mlir_out.mlir";

  // Write the file.
  auto write_status = WriteFile(mlir_in, mlir.value());
  ASSERT_TRUE(write_status.ok()) << write_status;

  std::vector<std::string> args = {
    kHloOptPath,
    mlir_in.string(),
    "--stablehlo-refine-shapes",
  };
  std::string cmd = absl::StrJoin(args, " ");
  LOG(INFO) << "Running command: " << cmd;
  ASSERT_EQ(std::system(cmd.c_str()), EXIT_SUCCESS);
}

} // namespace
} // namespace jxap

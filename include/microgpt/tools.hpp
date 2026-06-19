#pragma once

#include "microgpt/cli_arithmetic.hpp"
#include "microgpt/cli_artifacts.hpp"
#include "microgpt/cli_backend.hpp"
#include "microgpt/cli_bench.hpp"
#include "microgpt/cli_data.hpp"
#include "microgpt/cli_eval.hpp"
#include "microgpt/cli_generate.hpp"
#include "microgpt/cli_shared.hpp"
#include "microgpt/cli_train.hpp"
#include "microgpt/harness.hpp"

#include <string>
#include <vector>

namespace microgpt {

inline void register_microgpt_tools(HarnessRegistry& registry) {
  registry.register_tool("train", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_train_command(args, false, command_prefix("train", args), io.out, io.err);
  });
  registry.register_tool("resume", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_train_command(args, true, command_prefix("resume", args), io.out, io.err);
  });
  registry.register_tool("generate", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_generate_command(args, io.out, io.err);
  });
  registry.register_tool("eval", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_eval_command(args, io.out, io.err);
  });
  registry.register_tool("validate-data", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_validate_data_command(args, io.out);
  });
  registry.register_tool("split-data", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_split_data_command(args, io.out);
  });
  registry.register_tool("import-jsonl", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_import_jsonl_command(args, io.out);
  });
  registry.register_tool("make-arithmetic-data", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_make_arithmetic_data_command(args, io.out);
  });
  registry.register_tool("list-artifacts", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_list_artifacts_command(args, io.out);
  });
  registry.register_tool("clean-artifacts", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_clean_artifacts_command(args, io.out);
  });
  registry.register_tool("backends", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_backends_command(args, io.out);
  });
  registry.register_tool("bench", [](const std::vector<std::string>& args, HarnessIO& io) {
    return run_bench_command(args, io.out);
  });
  registry.register_tool("test", [](const std::vector<std::string>&, HarnessIO&) { return run_tests() ? 0 : 2; });
}

}  // namespace microgpt

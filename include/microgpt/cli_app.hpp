#pragma once

#include "microgpt/harness.hpp"
#include "microgpt/tools.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace microgpt {

inline constexpr int kExitUsage = 64;
inline constexpr int kExitFailure = 1;

inline std::vector<std::string> collect_cli_args(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return args;
}

inline void print_mgpt_usage(const HarnessRegistry& registry, std::ostream& err) {
  err << build_usage_text();
  err << "  available harness tools:";
  for (const std::string& name : registry.names()) {
    err << ' ' << name;
  }
  err << '\n';
}

inline int run_mgpt_cli(int argc, char** argv, std::ostream& out = std::cout, std::ostream& err = std::cerr) {
  try {
    HarnessRegistry registry;
    register_microgpt_tools(registry);

    if (argc < 2) {
      print_mgpt_usage(registry, err);
      return kExitUsage;
    }

    std::vector<std::string> args = collect_cli_args(argc, argv);
    std::string cmd = args.front();
    std::vector<std::string> rest(args.begin() + 1, args.end());
    HarnessIO io{out, err};
    return registry.run(cmd, rest, io);
  } catch (const std::exception& e) {
    err << "error";
    if (argc >= 2 && argv != nullptr && argv[1] != nullptr) {
      err << " in command '" << argv[1] << "'";
    }
    err << ": " << e.what() << '\n';
    return kExitFailure;
  }
}

}  // namespace microgpt

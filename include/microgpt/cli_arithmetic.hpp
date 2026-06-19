#pragma once

#include "microgpt/cli_shared.hpp"

namespace microgpt {

inline int run_make_arithmetic_data_command(const std::vector<std::string>& args, std::ostream& out) {
  std::string output = get_arg(args, "--output");
  if (output.empty()) {
    throw std::runtime_error("--output is required");
  }
  int max_a = get_arg_int(args, "--max-a", 19);
  int max_b = get_arg_int(args, "--max-b", 9);
  if (max_a < 0 || max_b < 0) {
    throw std::runtime_error("--max-a and --max-b must be non-negative");
  }
  std::ofstream out_file(output, std::ios::binary);
  if (!out_file) {
    throw std::runtime_error("failed to open output file: " + output);
  }
  int count = 0;
  for (int a = 0; a <= max_a; ++a) {
    for (int b = 0; b <= max_b; ++b) {
      out_file << "<BOS><USER>\n";
      out_file << "What is " << a << '+' << b << "?\n";
      out_file << "<ASSISTANT>\n";
      out_file << (a + b) << "\n";
      out_file << "<EOS>\n\n";
      ++count;
    }
  }
  out << "wrote " << count << " examples to " << output << '\n';
  return 0;
}

}  // namespace microgpt

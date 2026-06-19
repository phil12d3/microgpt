#pragma once

#include "microgpt/cli_shared.hpp"

#include <cstddef>
#include <iterator>

namespace microgpt {

inline bool is_session_format(const std::vector<std::string>& args) {
  return get_arg(args, "--format", "single") == "session";
}

inline int run_validate_data_command(const std::vector<std::string>& args, std::ostream& out) {
  std::string input = get_arg(args, "--input");
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  std::string text = read_file_text(input);
  bool session_format = is_session_format(args);
  out << "file " << input << '\n';
  if (session_format) {
    SessionValidation result = validate_session_text(text);
    size_t n = result.examples.size();
    float avg_turns = n > 0 ? static_cast<float>(result.total_turns) / static_cast<float>(n) : 0.0f;
    out << "bytes " << result.bytes << '\n';
    out << "sessions " << n << '\n';
    out << "markers BOS=" << result.bos_count << " USER=" << result.user_count << " ASSISTANT=" << result.assistant_count
        << " EOS=" << result.eos_count << '\n';
    out << std::fixed << std::setprecision(2);
    out << "turns min=" << result.min_turns << " avg=" << avg_turns << " max=" << result.max_turns << '\n';
    if (!result.errors.empty()) {
      out << "errors " << result.errors.size() << '\n';
      for (const std::string& error : result.errors) {
        out << "  " << error << '\n';
      }
      return 2;
    }
    out << "valid yes\n";
    return 0;
  }
  DatasetValidation result = validate_instruction_text(text);
  size_t n = result.examples.size();
  float avg_prompt = n > 0 ? static_cast<float>(result.prompt_chars) / static_cast<float>(n) : 0.0f;
  float avg_answer = n > 0 ? static_cast<float>(result.answer_chars) / static_cast<float>(n) : 0.0f;
  out << "bytes " << result.bytes << '\n';
  out << "examples " << n << '\n';
  out << "markers BOS=" << result.bos_count << " USER=" << result.user_count << " ASSISTANT=" << result.assistant_count
      << " EOS=" << result.eos_count << '\n';
  out << std::fixed << std::setprecision(2);
  out << "prompt_chars min=" << result.min_prompt_chars << " avg=" << avg_prompt << " max=" << result.max_prompt_chars
      << '\n';
  out << "answer_chars min=" << result.min_answer_chars << " avg=" << avg_answer << " max=" << result.max_answer_chars
      << '\n';
  if (!result.errors.empty()) {
    out << "errors " << result.errors.size() << '\n';
    for (const std::string& error : result.errors) {
      out << "  " << error << '\n';
    }
    return 2;
  }
  out << "valid yes\n";
  return 0;
}

inline int run_split_data_command(const std::vector<std::string>& args, std::ostream& out) {
  std::string input = get_arg(args, "--input");
  std::string train_path = get_arg(args, "--train");
  std::string val_path = get_arg(args, "--val");
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  if (train_path.empty()) {
    throw std::runtime_error("--train is required");
  }
  if (val_path.empty()) {
    throw std::runtime_error("--val is required");
  }
  float ratio = get_arg_float(args, "--ratio", 0.9f);
  if (ratio <= 0.0f || ratio >= 1.0f) {
    throw std::runtime_error("--ratio must be greater than 0 and less than 1");
  }
  uint32_t seed = static_cast<uint32_t>(get_arg_int(args, "--seed", 42));
  std::string text = read_file_text(input);
  if (is_session_format(args)) {
    SessionValidation validation = validate_session_text(text);
    if (!validation.errors.empty()) {
      throw std::runtime_error("input dataset is invalid; run validate-data for details");
    }
    std::vector<SessionExample> examples = validation.examples;
    auto split = split_session_examples(examples, ratio, seed);
    const std::vector<SessionExample>& train_examples = split.first;
    const std::vector<SessionExample>& val_examples = split.second;
    write_session_examples(train_path, train_examples);
    write_session_examples(val_path, val_examples);
    out << "input_sessions " << examples.size() << '\n';
    out << "train_sessions " << train_examples.size() << " wrote " << train_path << '\n';
    out << "val_sessions " << val_examples.size() << " wrote " << val_path << '\n';
    return 0;
  }
  DatasetValidation validation = validate_instruction_text(text);
  if (!validation.errors.empty()) {
    throw std::runtime_error("input dataset is invalid; run validate-data for details");
  }
  std::vector<EvalExample> examples = validation.examples;
  auto split = split_instruction_examples(examples, ratio, seed);
  const std::vector<EvalExample>& train_examples = split.first;
  const std::vector<EvalExample>& val_examples = split.second;
  write_instruction_examples(train_path, train_examples);
  write_instruction_examples(val_path, val_examples);
  out << "input_examples " << examples.size() << '\n';
  out << "train_examples " << train_examples.size() << " wrote " << train_path << '\n';
  out << "val_examples " << val_examples.size() << " wrote " << val_path << '\n';
  return 0;
}

inline int run_import_jsonl_command(const std::vector<std::string>& args, std::ostream& out) {
  std::string input = get_arg(args, "--input");
  std::string output = get_arg(args, "--output");
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  if (output.empty()) {
    throw std::runtime_error("--output is required");
  }
  std::ifstream in(input, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + input);
  }
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (is_session_format(args)) {
    std::vector<SessionExample> examples = parse_session_jsonl_examples(text);
    if (examples.empty()) {
      throw std::runtime_error("no JSONL examples found");
    }
    write_session_examples(output, examples);
    out << "imported " << examples.size() << " sessions to " << output << '\n';
    return 0;
  }
  std::vector<EvalExample> examples;
  std::istringstream lines(text);
  std::string line;
  int line_number = 0;
  while (std::getline(lines, line)) {
    ++line_number;
    size_t pos = 0;
    skip_json_ws(line, pos);
    if (pos == line.size()) {
      continue;
    }
    examples.push_back(parse_jsonl_example_line(line, line_number));
  }
  if (examples.empty()) {
    throw std::runtime_error("no JSONL examples found");
  }
  write_instruction_examples(output, examples);
  out << "imported " << examples.size() << " examples to " << output << '\n';
  return 0;
}

}  // namespace microgpt

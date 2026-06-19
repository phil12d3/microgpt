#pragma once

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace microgpt {

struct EvalExample {
  std::string prompt;
  std::string answer;
};

struct DatasetValidation {
  std::vector<EvalExample> examples;
  std::vector<std::string> errors;
  size_t bytes = 0;
  size_t prompt_chars = 0;
  size_t answer_chars = 0;
  size_t min_prompt_chars = std::numeric_limits<size_t>::max();
  size_t max_prompt_chars = 0;
  size_t min_answer_chars = std::numeric_limits<size_t>::max();
  size_t max_answer_chars = 0;
  int bos_count = 0;
  int user_count = 0;
  int assistant_count = 0;
  int eos_count = 0;
};

inline void skip_json_ws(const std::string& line, size_t& pos) {
  while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t' || line[pos] == '\r' || line[pos] == '\n')) {
    ++pos;
  }
}

inline std::string parse_json_string(const std::string& line, size_t& pos) {
  skip_json_ws(line, pos);
  if (pos >= line.size() || line[pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  ++pos;
  std::string out;
  while (pos < line.size()) {
    char c = line[pos++];
    if (c == '"') {
      return out;
    }
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (pos >= line.size()) {
      throw std::runtime_error("unterminated JSON escape");
    }
    char esc = line[pos++];
    switch (esc) {
      case '"':
      case '\\':
      case '/':
        out.push_back(esc);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      default:
        throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

inline int count_substr(const std::string& text, const std::string& needle) {
  int count = 0;
  size_t pos = 0;
  while (true) {
    pos = text.find(needle, pos);
    if (pos == std::string::npos) {
      break;
    }
    ++count;
    pos += needle.size();
  }
  return count;
}

inline std::string strip_trailing_space(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
  return s;
}

inline DatasetValidation validate_instruction_text(const std::string& text) {
  DatasetValidation result;
  result.bytes = text.size();
  result.bos_count = count_substr(text, "<BOS>");
  result.user_count = count_substr(text, "<USER>");
  result.assistant_count = count_substr(text, "<ASSISTANT>");
  result.eos_count = count_substr(text, "<EOS>");

  const std::string bos_user = "<BOS><USER>\n";
  const std::string assistant = "\n<ASSISTANT>\n";
  const std::string eos = "\n<EOS>";
  size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && (text[pos] == '\n' || text[pos] == '\r' || text[pos] == ' ' || text[pos] == '\t')) {
      ++pos;
    }
    if (pos >= text.size()) {
      break;
    }
    if (text.compare(pos, bos_user.size(), bos_user) != 0) {
      result.errors.push_back("expected <BOS><USER> at byte " + std::to_string(pos));
      break;
    }
    size_t prompt_start = pos + bos_user.size();
    size_t assistant_pos = text.find(assistant, prompt_start);
    if (assistant_pos == std::string::npos) {
      result.errors.push_back("missing <ASSISTANT> after byte " + std::to_string(prompt_start));
      break;
    }
    size_t answer_start = assistant_pos + assistant.size();
    size_t eos_pos = text.find(eos, answer_start);
    if (eos_pos == std::string::npos) {
      result.errors.push_back("missing <EOS> after byte " + std::to_string(answer_start));
      break;
    }

    EvalExample ex;
    ex.prompt = text.substr(prompt_start, assistant_pos - prompt_start);
    ex.answer = text.substr(answer_start, eos_pos - answer_start);
    if (strip_trailing_space(ex.prompt).empty()) {
      result.errors.push_back("empty prompt in example " + std::to_string(result.examples.size() + 1));
    }
    if (strip_trailing_space(ex.answer).empty()) {
      result.errors.push_back("empty answer in example " + std::to_string(result.examples.size() + 1));
    }

    result.prompt_chars += ex.prompt.size();
    result.answer_chars += ex.answer.size();
    result.min_prompt_chars = std::min(result.min_prompt_chars, ex.prompt.size());
    result.max_prompt_chars = std::max(result.max_prompt_chars, ex.prompt.size());
    result.min_answer_chars = std::min(result.min_answer_chars, ex.answer.size());
    result.max_answer_chars = std::max(result.max_answer_chars, ex.answer.size());
    result.examples.push_back(std::move(ex));
    pos = eos_pos + eos.size();
  }

  if (result.examples.empty() && result.errors.empty()) {
    result.errors.push_back("no instruction examples found");
  }
  if (result.bos_count != result.user_count || result.bos_count != result.assistant_count ||
      result.bos_count != result.eos_count) {
    result.errors.push_back("marker counts differ: BOS=" + std::to_string(result.bos_count) +
                            " USER=" + std::to_string(result.user_count) +
                            " ASSISTANT=" + std::to_string(result.assistant_count) +
                            " EOS=" + std::to_string(result.eos_count));
  }
  if (result.examples.empty()) {
    result.min_prompt_chars = 0;
    result.min_answer_chars = 0;
  }
  return result;
}

inline std::vector<EvalExample> parse_instruction_examples(const std::string& text) {
  return validate_instruction_text(text).examples;
}

inline void write_instruction_examples(const std::string& path, const std::vector<EvalExample>& examples) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  for (const EvalExample& ex : examples) {
    out << "<BOS><USER>\n";
    out << ex.prompt << '\n';
    out << "<ASSISTANT>\n";
    out << ex.answer << '\n';
    out << "<EOS>\n\n";
  }
}

inline EvalExample parse_jsonl_example_line(const std::string& line, int line_number) {
  size_t pos = 0;
  skip_json_ws(line, pos);
  if (pos >= line.size() || line[pos] != '{') {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected JSON object");
  }
  ++pos;
  EvalExample ex;
  bool has_user = false;
  bool has_assistant = false;
  while (true) {
    skip_json_ws(line, pos);
    if (pos < line.size() && line[pos] == '}') {
      ++pos;
      break;
    }
    std::string key = parse_json_string(line, pos);
    skip_json_ws(line, pos);
    if (pos >= line.size() || line[pos] != ':') {
      throw std::runtime_error("line " + std::to_string(line_number) + ": expected ':'");
    }
    ++pos;
    std::string value = parse_json_string(line, pos);
    if (key == "user") {
      ex.prompt = value;
      has_user = true;
    } else if (key == "assistant") {
      ex.answer = value;
      has_assistant = true;
    }
    skip_json_ws(line, pos);
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < line.size() && line[pos] == '}') {
      ++pos;
      break;
    }
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected ',' or '}'");
  }
  skip_json_ws(line, pos);
  if (pos != line.size()) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": trailing content after JSON object");
  }
  if (!has_user || !has_assistant) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": missing user or assistant field");
  }
  if (strip_trailing_space(ex.prompt).empty() || strip_trailing_space(ex.answer).empty()) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": user and assistant fields must be non-empty");
  }
  return ex;
}

inline std::pair<std::vector<EvalExample>, std::vector<EvalExample>> split_instruction_examples(const std::vector<EvalExample>& examples,
                                                                                                float ratio, uint32_t seed) {
  if (ratio <= 0.0f || ratio >= 1.0f) {
    throw std::runtime_error("--ratio must be greater than 0 and less than 1");
  }
  if (examples.size() < 2) {
    throw std::runtime_error("need at least two examples to split data");
  }
  std::vector<EvalExample> shuffled = examples;
  std::mt19937 rng(seed);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  size_t train_count = static_cast<size_t>(std::round(static_cast<float>(shuffled.size()) * ratio));
  train_count = std::max<size_t>(1, std::min(train_count, shuffled.size() - 1));
  std::vector<EvalExample> train_examples(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(train_count));
  std::vector<EvalExample> val_examples(shuffled.begin() + static_cast<std::ptrdiff_t>(train_count), shuffled.end());
  return {std::move(train_examples), std::move(val_examples)};
}

}  // namespace microgpt

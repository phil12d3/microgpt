#pragma once

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace microgpt {

struct EvalExample {
  std::string prompt;
  std::string answer;
};

struct SessionTurn {
  std::string role;
  std::string content;
};

struct SessionExample {
  std::vector<SessionTurn> turns;
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

struct SessionValidation {
  std::vector<SessionExample> examples;
  std::vector<std::string> errors;
  size_t bytes = 0;
  size_t total_turns = 0;
  size_t min_turns = std::numeric_limits<size_t>::max();
  size_t max_turns = 0;
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

inline std::string strip_cr(std::string s) {
  if (!s.empty() && s.back() == '\r') {
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
  if (result.bos_count != result.eos_count || result.user_count != result.assistant_count ||
      result.user_count < result.bos_count) {
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

inline std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(strip_cr(line));
  }
  return lines;
}

inline SessionExample parse_session_block(const std::vector<std::string>& lines, size_t& pos, int session_index,
                                          std::string& error) {
  SessionExample example;
  auto require_line = [&](const std::string& what) -> std::string {
    if (pos >= lines.size()) {
      error = what;
      return {};
    }
    return lines[pos++];
  };

  if (require_line("missing <BOS><USER> line") != "<BOS><USER>") {
    if (error.empty()) {
      error = "expected <BOS><USER> at session " + std::to_string(session_index);
    }
    return example;
  }

  std::string user = require_line("missing user content after <BOS><USER>");
  if (!error.empty()) {
    return example;
  }
  if (strip_trailing_space(user).empty()) {
    error = "empty user content in session " + std::to_string(session_index);
    return example;
  }
  if (require_line("missing <ASSISTANT> marker after user content") != "<ASSISTANT>") {
    if (error.empty()) {
      error = "expected <ASSISTANT> after user content in session " + std::to_string(session_index);
    }
    return example;
  }
  std::string assistant = require_line("missing assistant content after <ASSISTANT>");
  if (!error.empty()) {
    return example;
  }
  if (strip_trailing_space(assistant).empty()) {
    error = "empty assistant content in session " + std::to_string(session_index);
    return example;
  }
  example.turns.push_back({"user", std::move(user)});
  example.turns.push_back({"assistant", std::move(assistant)});

  while (true) {
    if (pos >= lines.size()) {
      error = "missing <EOS> after assistant content in session " + std::to_string(session_index);
      return example;
    }
    std::string marker = lines[pos++];
    if (marker == "<EOS>") {
      return example;
    }
    if (marker != "<USER>") {
      error = "expected <USER> or <EOS> in session " + std::to_string(session_index);
      return example;
    }
    user = require_line("missing user content after <USER>");
    if (!error.empty()) {
      return example;
    }
    if (strip_trailing_space(user).empty()) {
      error = "empty user content in session " + std::to_string(session_index);
      return example;
    }
    if (require_line("missing <ASSISTANT> marker after user content") != "<ASSISTANT>") {
      if (error.empty()) {
        error = "expected <ASSISTANT> after user content in session " + std::to_string(session_index);
      }
      return example;
    }
    assistant = require_line("missing assistant content after <ASSISTANT>");
    if (!error.empty()) {
      return example;
    }
    if (strip_trailing_space(assistant).empty()) {
      error = "empty assistant content in session " + std::to_string(session_index);
      return example;
    }
    example.turns.push_back({"user", std::move(user)});
    example.turns.push_back({"assistant", std::move(assistant)});
  }
}

inline SessionValidation validate_session_text(const std::string& text) {
  SessionValidation result;
  result.bytes = text.size();
  result.bos_count = count_substr(text, "<BOS>");
  result.user_count = count_substr(text, "<USER>");
  result.assistant_count = count_substr(text, "<ASSISTANT>");
  result.eos_count = count_substr(text, "<EOS>");
  std::vector<std::string> lines = split_lines(text);
  size_t pos = 0;
  int session_index = 0;
  while (pos < lines.size()) {
    while (pos < lines.size() && strip_trailing_space(lines[pos]).empty()) {
      ++pos;
    }
    if (pos >= lines.size()) {
      break;
    }
    ++session_index;
    std::string error;
    SessionExample ex = parse_session_block(lines, pos, session_index, error);
    if (!error.empty()) {
      result.errors.push_back(error);
      break;
    }
    result.total_turns += ex.turns.size() / 2;
    result.min_turns = std::min(result.min_turns, ex.turns.size() / 2);
    result.max_turns = std::max(result.max_turns, ex.turns.size() / 2);
    result.examples.push_back(std::move(ex));
  }
  if (result.examples.empty() && result.errors.empty()) {
    result.errors.push_back("no session examples found");
  }
  if (result.bos_count != result.eos_count || result.user_count != result.assistant_count ||
      result.user_count < result.bos_count) {
    result.errors.push_back("marker counts differ: BOS=" + std::to_string(result.bos_count) +
                            " USER=" + std::to_string(result.user_count) +
                            " ASSISTANT=" + std::to_string(result.assistant_count) +
                            " EOS=" + std::to_string(result.eos_count));
  }
  if (result.examples.empty()) {
    result.min_turns = 0;
  }
  return result;
}

inline std::vector<SessionExample> parse_session_examples(const std::string& text) {
  return validate_session_text(text).examples;
}

inline void write_session_examples(const std::string& path, const std::vector<SessionExample>& examples) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  for (const SessionExample& ex : examples) {
    if (ex.turns.empty() || ex.turns.size() % 2 != 0) {
      throw std::runtime_error("session examples must contain alternating user and assistant turns");
    }
    out << "<BOS><USER>\n";
    for (size_t i = 0; i < ex.turns.size(); ++i) {
      const SessionTurn& turn = ex.turns[i];
      if (i % 2 == 0) {
        if (turn.role != "user") {
          throw std::runtime_error("session examples must start with a user turn and alternate");
        }
        out << turn.content << '\n';
        out << "<ASSISTANT>\n";
      } else {
        if (turn.role != "assistant") {
          throw std::runtime_error("session examples must start with a user turn and alternate");
        }
        out << turn.content << '\n';
        if (i + 1 < ex.turns.size()) {
          out << "<USER>\n";
        } else {
          out << "<EOS>\n\n";
        }
      }
    }
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

inline SessionTurn parse_jsonl_turn_object(const std::string& line, size_t& pos, int line_number) {
  if (pos >= line.size() || line[pos] != '{') {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected turn object");
  }
  ++pos;
  SessionTurn turn;
  bool has_role = false;
  bool has_content = false;
  while (true) {
    skip_json_ws(line, pos);
    if (pos < line.size() && line[pos] == '}') {
      ++pos;
      break;
    }
    std::string key = parse_json_string(line, pos);
    skip_json_ws(line, pos);
    if (pos >= line.size() || line[pos] != ':') {
      throw std::runtime_error("line " + std::to_string(line_number) + ": expected ':' in turn object");
    }
    ++pos;
    if (key == "role") {
      turn.role = parse_json_string(line, pos);
      has_role = true;
    } else if (key == "content" || key == "text") {
      turn.content = parse_json_string(line, pos);
      has_content = true;
    } else {
      throw std::runtime_error("line " + std::to_string(line_number) + ": unsupported turn key " + key);
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
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected ',' or '}' in turn object");
  }
  if (!has_role || !has_content) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": turn object missing role or content");
  }
  if (turn.role != "user" && turn.role != "assistant") {
    throw std::runtime_error("line " + std::to_string(line_number) + ": turn role must be user or assistant");
  }
  if (strip_trailing_space(turn.content).empty()) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": turn content must be non-empty");
  }
  return turn;
}

inline std::vector<SessionTurn> parse_jsonl_turn_array(const std::string& line, size_t& pos, int line_number) {
  if (pos >= line.size() || line[pos] != '[') {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected turn array");
  }
  ++pos;
  std::vector<SessionTurn> turns;
  while (true) {
    skip_json_ws(line, pos);
    if (pos < line.size() && line[pos] == ']') {
      ++pos;
      break;
    }
    turns.push_back(parse_jsonl_turn_object(line, pos, line_number));
    skip_json_ws(line, pos);
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < line.size() && line[pos] == ']') {
      ++pos;
      break;
    }
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected ',' or ']' in turns array");
  }
  return turns;
}

inline SessionExample parse_jsonl_session_line(const std::string& line, int line_number) {
  size_t pos = 0;
  skip_json_ws(line, pos);
  if (pos >= line.size() || line[pos] != '{') {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected JSON object");
  }
  ++pos;
  SessionExample ex;
  bool has_turns = false;
  bool has_user = false;
  bool has_assistant = false;
  std::string user;
  std::string assistant;
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
    if (key == "user") {
      user = parse_json_string(line, pos);
      has_user = true;
    } else if (key == "assistant") {
      assistant = parse_json_string(line, pos);
      has_assistant = true;
    } else if (key == "turns" || key == "messages") {
      ex.turns = parse_jsonl_turn_array(line, pos, line_number);
      has_turns = true;
    } else {
      throw std::runtime_error("line " + std::to_string(line_number) + ": unsupported key " + key);
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
  if (has_turns) {
    if (ex.turns.empty() || ex.turns.size() % 2 != 0) {
      throw std::runtime_error("line " + std::to_string(line_number) + ": turns array must contain alternating user and assistant turns");
    }
    for (size_t i = 0; i < ex.turns.size(); ++i) {
      const SessionTurn& turn = ex.turns[i];
      if (i % 2 == 0 && turn.role != "user") {
        throw std::runtime_error("line " + std::to_string(line_number) + ": turns must start with user");
      }
      if (i % 2 == 1 && turn.role != "assistant") {
        throw std::runtime_error("line " + std::to_string(line_number) + ": turns must alternate user and assistant");
      }
    }
    return ex;
  }
  if (!has_user || !has_assistant) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": missing user or assistant field");
  }
  if (strip_trailing_space(user).empty() || strip_trailing_space(assistant).empty()) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": user and assistant fields must be non-empty");
  }
  ex.turns.push_back({"user", user});
  ex.turns.push_back({"assistant", assistant});
  return ex;
}

inline std::vector<SessionExample> parse_session_jsonl_examples(const std::string& text) {
  std::vector<SessionExample> examples;
  std::istringstream in(text);
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    size_t pos = 0;
    skip_json_ws(line, pos);
    if (pos == line.size()) {
      continue;
    }
    examples.push_back(parse_jsonl_session_line(line, line_number));
  }
  return examples;
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

inline std::pair<std::vector<SessionExample>, std::vector<SessionExample>> split_session_examples(
    const std::vector<SessionExample>& examples, float ratio, uint32_t seed) {
  if (ratio <= 0.0f || ratio >= 1.0f) {
    throw std::runtime_error("--ratio must be greater than 0 and less than 1");
  }
  if (examples.size() < 2) {
    throw std::runtime_error("need at least two examples to split data");
  }
  std::vector<SessionExample> shuffled = examples;
  std::mt19937 rng(seed);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  size_t train_count = static_cast<size_t>(std::round(static_cast<float>(shuffled.size()) * ratio));
  train_count = std::max<size_t>(1, std::min(train_count, shuffled.size() - 1));
  std::vector<SessionExample> train_examples(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(train_count));
  std::vector<SessionExample> val_examples(shuffled.begin() + static_cast<std::ptrdiff_t>(train_count), shuffled.end());
  return {std::move(train_examples), std::move(val_examples)};
}

}  // namespace microgpt

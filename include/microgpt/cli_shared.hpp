#pragma once

#include "microgpt.hpp"
#include "microgpt/data_format.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

inline std::string get_arg(const std::vector<std::string>& args, const std::string& key, const std::string& def = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return def;
}

inline int get_arg_int(const std::vector<std::string>& args, const std::string& key, int def) {
  std::string v = get_arg(args, key);
  return v.empty() ? def : std::stoi(v);
}

inline float get_arg_float(const std::vector<std::string>& args, const std::string& key, float def) {
  std::string v = get_arg(args, key);
  return v.empty() ? def : std::stof(v);
}

inline bool has_arg(const std::vector<std::string>& args, const std::string& key) {
  for (const std::string& arg : args) {
    if (arg == key) {
      return true;
    }
  }
  return false;
}

inline std::string command_arg_quote(const std::string& arg) {
  if (arg.find_first_of(" \t\n\"'\\") == std::string::npos) {
    return arg;
  }
  std::string out = "'";
  for (char c : arg) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

inline std::string join_command(const std::vector<std::string>& args) {
  std::ostringstream out;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0) {
      out << ' ';
    }
    out << command_arg_quote(args[i]);
  }
  return out.str();
}

inline std::string format_seconds(double seconds) {
  if (seconds < 0.0) {
    seconds = 0.0;
  }
  int total = static_cast<int>(std::round(seconds));
  int h = total / 3600;
  int m = (total % 3600) / 60;
  int s = total % 60;
  std::ostringstream out;
  out << std::setfill('0');
  if (h > 0) {
    out << h << ':';
  }
  out << std::setw(2) << m << ':' << std::setw(2) << s;
  return out.str();
}

inline bool eval_match(const std::string& actual, const std::string& expected, const std::string& mode) {
  if (mode == "exact") {
    return actual == expected;
  }
  if (mode == "prefix") {
    return actual.size() >= expected.size() && actual.compare(0, expected.size(), expected) == 0;
  }
  if (mode == "contains") {
    return actual.find(expected) != std::string::npos;
  }
  throw std::runtime_error("--match must be exact, prefix, or contains");
}

inline void print_training_progress(const TrainingProgress& p, std::ostream& out) {
  std::ostringstream line;
  line << std::fixed << std::setprecision(4);
  line << "step " << p.step << '/' << p.total_steps;
  line << " loss " << p.loss;
  line << " avg_loss " << p.average_loss;
  if (p.has_validation) {
    line << " val_loss " << p.val_loss;
    line << " ppl " << p.perplexity;
  }
  line << std::setprecision(2);
  line << " " << p.steps_per_second << " it/s";
  line << " elapsed " << format_seconds(p.elapsed_seconds);
  line << " eta " << format_seconds(p.eta_seconds);
  out << line.str() << '\n';
}

inline void print_generation_progress(const GenerationProgress& p, std::ostream& err) {
  std::ostringstream line;
  line << "token " << p.token_index << '/' << p.total_tokens;
  line << " last_token " << p.last_token;
  line << " " << std::fixed << std::setprecision(2) << p.tokens_per_second << " tok/s";
  line << " elapsed " << format_seconds(p.elapsed_seconds);
  err << line.str() << '\n';
}

inline std::string build_usage_text() {
  std::ostringstream out;
  out << "Usage:\n"
      << "  microgpt train --input data.txt --checkpoint model.bin [options]\n"
      << "  microgpt resume --input data.txt --checkpoint model.bin [options]\n"
      << "  microgpt generate --checkpoint model.bin --prompt \"text\" [options]\n"
      << "  microgpt generate --checkpoint model.bin --prompt-token 104 [options]\n"
      << "  microgpt eval --checkpoint model.bin --input eval.txt [options]\n"
      << "  microgpt validate-data --input data.txt [options]\n"
      << "  microgpt split-data --input data.txt --train train.txt --val val.txt [options]\n"
      << "  microgpt import-jsonl --input pairs.jsonl --output data.txt\n"
      << "  microgpt make-arithmetic-data --output data.txt [options]\n"
      << "  microgpt list-artifacts [--root artifacts]\n"
      << "  microgpt clean-artifacts [--root artifacts] [--yes]\n"
      << "  microgpt backends\n"
      << "  microgpt parity [options]\n"
      << "  microgpt bench [options]\n"
      << "  microgpt test\n"
      << "  training options include --steps, --batch-size, --context, --d-model,\n"
      << "  --layers, --heads, --lr, --eval-interval, --save-interval, and\n"
      << "  --progress-interval. Use --val-input for an explicit validation file.\n"
      << "  generation options include --max-new-tokens, --temperature, --top-k,\n"
      << "  --mode raw|instruction, --prompt-token, --greedy, and --quiet.\n"
      << "  evaluation options include --match, --max-examples, --hide-failures,\n"
      << "  --greedy, and --output for a JSON report.\n"
      << "  backend options include --backend cpu|metal|cuda.\n"
      << "  parity options include --backend cpu|metal|cuda and --tolerance.\n"
      << "  arithmetic data options include --max-a and --max-b.\n";
  return out.str();
}

inline std::string command_prefix(const std::string& tool, const std::vector<std::string>& args) {
  std::vector<std::string> full;
  full.reserve(args.size() + 1);
  full.push_back(tool);
  full.insert(full.end(), args.begin(), args.end());
  return join_command(full);
}

}  // namespace microgpt

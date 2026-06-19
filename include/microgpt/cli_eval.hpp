#pragma once

#include "microgpt/cli_backend.hpp"
#include "microgpt/cli_shared.hpp"

#include <chrono>
#include <ctime>
#include <fstream>

namespace microgpt {

struct EvalCaseResult {
  int index = 0;
  std::string prompt;
  std::string expected;
  std::string actual;
  bool correct = false;
};

inline std::string eval_timestamp_utc_now() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

inline bool eval_is_session_format(const std::vector<std::string>& args) {
  return get_arg(args, "--format", "single") == "session";
}

inline std::string build_session_eval_prompt(const SessionExample& ex) {
  if (ex.turns.size() < 2 || ex.turns.size() % 2 != 0) {
    throw std::runtime_error("session examples must contain alternating user and assistant turns");
  }
  std::ostringstream out;
  out << "<BOS>";
  for (size_t i = 0; i + 2 < ex.turns.size(); ++i) {
    const SessionTurn& turn = ex.turns[i];
    if (i % 2 == 0) {
      if (turn.role != "user") {
        throw std::runtime_error("session examples must start with a user turn");
      }
      out << "<USER>\n" << turn.content << "\n";
    } else {
      if (turn.role != "assistant") {
        throw std::runtime_error("session examples must alternate user and assistant turns");
      }
      out << "<ASSISTANT>\n" << turn.content << "\n";
    }
  }
  const SessionTurn& final_user = ex.turns[ex.turns.size() - 2];
  const SessionTurn& final_assistant = ex.turns.back();
  if (final_user.role != "user" || final_assistant.role != "assistant") {
    throw std::runtime_error("session examples must end with a user turn followed by an assistant turn");
  }
  out << "<USER>\n" << final_user.content << "\n<ASSISTANT>\n";
  return out.str();
}

inline void write_eval_report_json(const std::string& path, const std::string& command, const std::string& checkpoint,
                                   const std::string& input, const std::string& match_mode, int max_new,
                                   float temperature, int top_k, BackendKind backend, const std::string& started_at,
                                   const std::string& ended_at, double duration_seconds,
                                   const std::vector<EvalCaseResult>& results, int correct) {
  std::ofstream json(path, std::ios::binary);
  if (!json) {
    throw std::runtime_error("failed to write eval report: " + path);
  }
  int total = static_cast<int>(results.size());
  float accuracy = total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
  json << "{\n";
  json << "  \"type\": \"microgpt_eval_report\",\n";
  json << "  \"started_at\": \"" << json_escape(started_at) << "\",\n";
  json << "  \"ended_at\": \"" << json_escape(ended_at) << "\",\n";
  json << "  \"duration_seconds\": " << duration_seconds << ",\n";
  json << "  \"command\": \"" << json_escape(command) << "\",\n";
  json << "  \"backend\": \"" << json_escape(backend_name(backend)) << "\",\n";
  json << "  \"checkpoint\": \"" << json_escape(checkpoint) << "\",\n";
  json << "  \"input\": \"" << json_escape(input) << "\",\n";
  json << "  \"match\": \"" << json_escape(match_mode) << "\",\n";
  json << "  \"max_new_tokens\": " << max_new << ",\n";
  json << "  \"temperature\": " << temperature << ",\n";
  json << "  \"top_k\": " << top_k << ",\n";
  json << "  \"examples\": " << total << ",\n";
  json << "  \"correct\": " << correct << ",\n";
  json << "  \"accuracy\": " << accuracy << ",\n";
  json << "  \"cases\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const EvalCaseResult& r = results[i];
    json << "    {\n";
    json << "      \"index\": " << r.index << ",\n";
    json << "      \"correct\": " << (r.correct ? "true" : "false") << ",\n";
    json << "      \"prompt\": \"" << json_escape(r.prompt) << "\",\n";
    json << "      \"expected\": \"" << json_escape(r.expected) << "\",\n";
    json << "      \"actual\": \"" << json_escape(r.actual) << "\"\n";
    json << "    }";
    if (i + 1 < results.size()) {
      json << ',';
    }
    json << '\n';
  }
  json << "  ]\n";
  json << "}\n";
}

inline int run_eval_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  (void)err;
  BackendKind backend = require_backend_arg(args);
  std::string checkpoint = get_arg(args, "--checkpoint");
  std::string input = get_arg(args, "--input");
  std::string output = get_arg(args, "--output");
  if (checkpoint.empty()) {
    throw std::runtime_error("--checkpoint is required");
  }
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  AdamW opt;
  int step = 0;
  Model model = load_checkpoint(checkpoint, opt, step);
  model.set_backend(backend);
  int max_new = get_arg_int(args, "--max-new-tokens", model.cfg.max_new_tokens);
  float temperature = get_arg_float(args, "--temperature", 0.2f);
  int top_k = has_arg(args, "--greedy") ? 1 : get_arg_int(args, "--top-k", 1);
  int max_examples = get_arg_int(args, "--max-examples", 0);
  std::string match_mode = get_arg(args, "--match", "exact");
  bool show_failures = !has_arg(args, "--hide-failures");
  std::string text = read_file_text(input);
  int correct = 0;
  std::vector<EvalCaseResult> results;
  std::string started_at = eval_timestamp_utc_now();
  auto started = std::chrono::steady_clock::now();
  if (eval_is_session_format(args)) {
    std::vector<SessionExample> examples = parse_session_examples(text);
    if (examples.empty()) {
      throw std::runtime_error("no session examples found in input");
    }
    int limit = max_examples > 0 ? std::min<int>(max_examples, static_cast<int>(examples.size()))
                                 : static_cast<int>(examples.size());
    results.reserve(static_cast<size_t>(limit));
    for (int i = 0; i < limit; ++i) {
      const SessionExample& ex = examples[static_cast<size_t>(i)];
      std::string model_prompt = build_session_eval_prompt(ex);
      std::string actual = strip_trailing_space(
          generate_text(model, model_prompt, max_new, temperature, top_k, Tokenizer::kEos));
      const SessionTurn& expected_turn = ex.turns.back();
      std::string expected = strip_trailing_space(expected_turn.content);
      std::string prompt_text = strip_trailing_space(ex.turns[ex.turns.size() - 2].content);
      bool ok = eval_match(actual, expected, match_mode);
      if (ok) {
        ++correct;
      } else if (show_failures) {
        out << "FAIL " << (i + 1) << ": " << prompt_text << '\n'
            << "  expected: " << expected << '\n'
            << "  actual:   " << actual << '\n';
      }
      EvalCaseResult result;
      result.index = i + 1;
      result.prompt = prompt_text;
      result.expected = expected;
      result.actual = actual;
      result.correct = ok;
      results.push_back(std::move(result));
    }
    auto ended = std::chrono::steady_clock::now();
    std::string ended_at = eval_timestamp_utc_now();
    double duration_seconds = std::chrono::duration<double>(ended - started).count();
    float accuracy = limit > 0 ? static_cast<float>(correct) / static_cast<float>(limit) : 0.0f;
    out << std::fixed << std::setprecision(2);
    out << "match " << match_mode << " sessions " << limit << " correct " << correct << " accuracy "
        << (accuracy * 100.0f) << "%\n";
    if (!output.empty()) {
      write_eval_report_json(output, command_prefix("eval", args), checkpoint, input, match_mode, max_new, temperature,
                             top_k, backend, started_at, ended_at, duration_seconds, results, correct);
      out << "report " << output << '\n';
    }
    if (backend != BackendKind::Cpu) {
      out << "backend_note " << backend_name(backend)
          << " selected but accelerated kernels are incomplete or unavailable; CPU fallback may have been used\n";
    }
    return correct == limit ? 0 : 2;
  }

  std::vector<EvalExample> examples = parse_instruction_examples(text);
  if (examples.empty()) {
    throw std::runtime_error("no instruction examples found in input");
  }
  int limit = max_examples > 0 ? std::min<int>(max_examples, static_cast<int>(examples.size()))
                               : static_cast<int>(examples.size());
  results.reserve(static_cast<size_t>(limit));
  for (int i = 0; i < limit; ++i) {
    const EvalExample& ex = examples[static_cast<size_t>(i)];
    std::string model_prompt = "<BOS><USER>\n" + ex.prompt + "\n<ASSISTANT>\n";
    std::string actual = strip_trailing_space(generate_text(model, model_prompt, max_new, temperature, top_k, Tokenizer::kEos));
    std::string expected = strip_trailing_space(ex.answer);
    bool ok = eval_match(actual, expected, match_mode);
    if (ok) {
      ++correct;
    } else if (show_failures) {
      out << "FAIL " << (i + 1) << ": " << ex.prompt << '\n'
          << "  expected: " << expected << '\n'
          << "  actual:   " << actual << '\n';
    }
    EvalCaseResult result;
    result.index = i + 1;
    result.prompt = ex.prompt;
    result.expected = expected;
    result.actual = actual;
    result.correct = ok;
    results.push_back(std::move(result));
  }
  auto ended = std::chrono::steady_clock::now();
  std::string ended_at = eval_timestamp_utc_now();
  double duration_seconds = std::chrono::duration<double>(ended - started).count();
  float accuracy = limit > 0 ? static_cast<float>(correct) / static_cast<float>(limit) : 0.0f;
  out << std::fixed << std::setprecision(2);
  out << "match " << match_mode << " examples " << limit << " correct " << correct << " accuracy "
      << (accuracy * 100.0f) << "%\n";
  if (!output.empty()) {
    write_eval_report_json(output, command_prefix("eval", args), checkpoint, input, match_mode, max_new, temperature,
                           top_k, backend, started_at, ended_at, duration_seconds, results, correct);
    out << "report " << output << '\n';
  }
  if (backend != BackendKind::Cpu) {
    out << "backend_note " << backend_name(backend)
        << " selected but accelerated kernels are incomplete or unavailable; CPU fallback may have been used\n";
  }
  return correct == limit ? 0 : 2;
}

}  // namespace microgpt

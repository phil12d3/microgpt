#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/chat_config.hpp"
#include "microgpt/chat_session.hpp"
#include "microgpt/checkpoint.hpp"
#include "microgpt/generation.hpp"
#include "microgpt/chat_tui.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace microgpt {

inline std::string chat_get_arg(const std::vector<std::string>& args, const std::string& key, const std::string& def = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return def;
}

inline int chat_get_arg_int(const std::vector<std::string>& args, const std::string& key, int def) {
  std::string v = chat_get_arg(args, key);
  return v.empty() ? def : std::stoi(v);
}

inline float chat_get_arg_float(const std::vector<std::string>& args, const std::string& key, float def) {
  std::string v = chat_get_arg(args, key);
  return v.empty() ? def : std::stof(v);
}

inline bool chat_has_arg(const std::vector<std::string>& args, const std::string& key) {
  for (const std::string& arg : args) {
    if (arg == key) {
      return true;
    }
  }
  return false;
}

inline ChatConfig parse_chat_config(const std::vector<std::string>& args) {
  ChatConfig config;
  config.checkpoint = chat_get_arg(args, "--checkpoint");
  if (config.checkpoint.empty()) {
    throw std::runtime_error("--checkpoint is required");
  }
  config.backend = parse_backend_kind(chat_get_arg(args, "--backend", "cpu"));
  require_backend_available(config.backend);
  return config;
}

inline int run_chat_line_mode(Model& model, ChatConfig config, std::istream& in, std::ostream& out, std::ostream& err) {
  std::string prompt;
  std::vector<ChatMessage> history;
  Tokenizer tok(tokenizer_kind_from_int(model.cfg.tokenizer_kind));
  out << "microgpt chat. type /exit to quit.\n";
  out << "type /reset to reset the session, /clear to clear the conversation context, or /compress to compact context.\n";
  while (true) {
    out << "> ";
    if (!std::getline(in, prompt)) {
      break;
    }
    if (prompt == "/exit" || prompt == "/quit") {
      break;
    }
    if (prompt == "/reset") {
      history.clear();
      out << "session reset\n";
      continue;
    }
    if (prompt == "/clear") {
      history.clear();
      out << "context cleared\n";
      continue;
    }
    if (prompt == "/compress") {
      std::vector<ChatMessage> compressed = compress_chat_history(history, tok, model.cfg.context_length);
      bool changed = compressed.size() != history.size();
      history = std::move(compressed);
      std::string compressed_prompt = format_multi_turn_chat_prompt(history, "");
      size_t compressed_tokens = tok.encode_text(compressed_prompt).size();
      size_t effective_tokens = model.cfg.context_length > 0
                                    ? std::min(compressed_tokens, static_cast<size_t>(model.cfg.context_length))
                                    : compressed_tokens;
      int context_percent = model.cfg.context_length > 0
                                ? static_cast<int>((100.0 * static_cast<double>(effective_tokens)) /
                                                   static_cast<double>(model.cfg.context_length))
                                : 0;
      out << (changed ? "context compressed\n" : "context already compact\n");
      out << "[ctx " << effective_tokens << '/' << model.cfg.context_length << ' ' << context_percent << "%]\n";
      continue;
    }
    if (prompt.empty()) {
      continue;
    }
    std::string raw_prompt = format_multi_turn_chat_prompt(history, prompt);
    size_t prompt_tokens = tok.encode_text(raw_prompt).size();
    std::vector<ChatMessage> trimmed_history = trim_chat_history_to_context(history, prompt, model.cfg.context_length, tok);
    std::string model_prompt = format_multi_turn_chat_prompt(trimmed_history, prompt);
    std::string reply = generate_text(model, model_prompt, config.max_new_tokens, config.temperature, config.top_k,
                                      Tokenizer::kEos);
    out << reply << "\n";
    size_t effective_prompt_tokens = model.cfg.context_length > 0
                                         ? std::min(prompt_tokens, static_cast<size_t>(model.cfg.context_length))
                                         : prompt_tokens;
    int context_percent = model.cfg.context_length > 0
                              ? static_cast<int>((100.0 * static_cast<double>(effective_prompt_tokens)) /
                                                 static_cast<double>(model.cfg.context_length))
                              : 0;
    out << "[ctx " << effective_prompt_tokens << '/' << model.cfg.context_length << ' ' << context_percent << "%]\n";
    history.push_back({ChatRole::User, prompt});
    history.push_back({ChatRole::Assistant, reply});
    if (config.backend != BackendKind::Cpu) {
      err << "backend_note " << backend_name(config.backend)
          << " selected but accelerated kernels are incomplete or unavailable; CPU fallback may have been used\n";
    }
  }
  return 0;
}

inline bool chat_should_use_tui() {
#if !defined(_WIN32)
  return ::isatty(STDIN_FILENO) != 0 && ::isatty(STDOUT_FILENO) != 0;
#else
  return false;
#endif
}

inline int run_chat_cli(int argc, char** argv, std::ostream& out = std::cout, std::ostream& err = std::cerr) {
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }
    ChatConfig config = parse_chat_config(args);

    AdamW opt;
    int step = 0;
    Model model = load_checkpoint(config.checkpoint, opt, step);
    model.set_backend(config.backend);
    config.max_new_tokens = chat_get_arg_int(args, "--max-new-tokens", model.cfg.max_new_tokens);
    config.temperature = chat_get_arg_float(args, "--temperature", model.cfg.temperature);
    config.top_k = chat_has_arg(args, "--greedy") ? 1 : chat_get_arg_int(args, "--top-k", model.cfg.top_k);

    if (chat_should_use_tui()) {
      run_chat_tui_loop(model, config, err);
      return 0;
    }
    return run_chat_line_mode(model, config, std::cin, out, err);
  } catch (const std::exception& e) {
    err << "error: " << e.what() << '\n';
        err << "usage: chat --checkpoint model.bin [--max-new-tokens N] [--temperature T] [--top-k K] [--greedy] [--backend cpu|metal|cuda]\n";
        return 1;
      }
}

}  // namespace microgpt

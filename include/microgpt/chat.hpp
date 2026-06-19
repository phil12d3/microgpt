#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/checkpoint.hpp"
#include "microgpt/cli_shared.hpp"
#include "microgpt/generation.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

struct ChatConfig {
  std::string checkpoint;
  int max_new_tokens = 0;
  float temperature = 0.0f;
  int top_k = 0;
  BackendKind backend = BackendKind::Cpu;
};

inline std::string format_single_turn_chat_prompt(const std::string& user_prompt) {
  return "<BOS><USER>\n" + user_prompt + "\n<ASSISTANT>\n";
}

inline ChatConfig parse_chat_config(const std::vector<std::string>& args) {
  ChatConfig config;
  config.checkpoint = get_arg(args, "--checkpoint");
  if (config.checkpoint.empty()) {
    throw std::runtime_error("--checkpoint is required");
  }
  config.backend = parse_backend_kind(get_arg(args, "--backend", "cpu"));
  require_backend_available(config.backend);
  return config;
}

inline int run_chat_cli(int argc, char** argv, std::istream& in = std::cin, std::ostream& out = std::cout,
                        std::ostream& err = std::cerr) {
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
    config.max_new_tokens = get_arg_int(args, "--max-new-tokens", model.cfg.max_new_tokens);
    config.temperature = get_arg_float(args, "--temperature", model.cfg.temperature);
    config.top_k = has_arg(args, "--greedy") ? 1 : get_arg_int(args, "--top-k", model.cfg.top_k);

    out << "microgpt chat. type /exit to quit.\n";
    std::string prompt;
    while (true) {
      out << "> ";
      if (!std::getline(in, prompt)) {
        break;
      }
      if (prompt == "/exit" || prompt == "/quit") {
        break;
      }
      if (prompt.empty()) {
        continue;
      }
      std::string reply = generate_text(model, format_single_turn_chat_prompt(prompt), config.max_new_tokens,
                                        config.temperature, config.top_k, Tokenizer::kEos);
      out << reply << "\n";
      if (config.backend != BackendKind::Cpu) {
        err << "backend_note " << backend_name(config.backend)
            << " selected but accelerated kernels are incomplete or unavailable; CPU fallback may have been used\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    err << "error: " << e.what() << '\n';
    err << "usage: chat --checkpoint model.bin [--max-new-tokens N] [--temperature T] [--top-k K] [--greedy] [--backend cpu|metal|cuda]\n";
    return 1;
  }
}

}  // namespace microgpt

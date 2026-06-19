#pragma once

#include "microgpt/cli_backend.hpp"
#include "microgpt/cli_shared.hpp"

#include <functional>

namespace microgpt {

inline int run_generate_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  BackendKind backend = require_backend_arg(args);
  std::string checkpoint = get_arg(args, "--checkpoint");
  std::string prompt = get_arg(args, "--prompt");
  std::string prompt_token = get_arg(args, "--prompt-token");
  std::string mode = get_arg(args, "--mode", "instruction");
  if (checkpoint.empty()) {
    throw std::runtime_error("--checkpoint is required");
  }
  if (prompt.empty() && prompt_token.empty()) {
    throw std::runtime_error("--prompt is required");
  }
  AdamW opt;
  int step = 0;
  Model model = load_checkpoint(checkpoint, opt, step);
  model.set_backend(backend);
  int max_new = get_arg_int(args, "--max-new-tokens", model.cfg.max_new_tokens);
  float temperature = get_arg_float(args, "--temperature", model.cfg.temperature);
  int top_k = has_arg(args, "--greedy") ? 1 : get_arg_int(args, "--top-k", model.cfg.top_k);
  bool quiet = has_arg(args, "--quiet");
  if (prompt.empty()) {
    int token = std::stoi(prompt_token);
    if (token < 0 || token > 255) {
      throw std::runtime_error("--prompt-token must be in range 0..255");
    }
    prompt.push_back(static_cast<char>(token));
  }
  std::string model_prompt = prompt;
  if (mode == "instruction") {
    model_prompt = "<BOS><USER>\n" + prompt + "\n<ASSISTANT>\n";
  } else if (mode != "raw") {
    throw std::runtime_error("--mode must be raw or instruction");
  }
  std::string output_text =
      generate_text(model, model_prompt, max_new, temperature, top_k, Tokenizer::kEos,
                    quiet ? std::function<void(const GenerationProgress&)>() : [&err](const GenerationProgress& p) {
                      print_generation_progress(p, err);
                    });
  out << output_text;
  if (backend != BackendKind::Cpu && !quiet) {
    err << "backend_note " << backend_name(backend)
        << " selected but accelerated kernels are incomplete or unavailable; CPU fallback may have been used\n";
  }
  return 0;
}

}  // namespace microgpt

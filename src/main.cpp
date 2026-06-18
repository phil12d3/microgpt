#include "microgpt.hpp"

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace microgpt;

static std::string get_arg(const std::vector<std::string>& args, const std::string& key, const std::string& def = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return def;
}

static int get_arg_int(const std::vector<std::string>& args, const std::string& key, int def) {
  std::string v = get_arg(args, key);
  return v.empty() ? def : std::stoi(v);
}

static float get_arg_float(const std::vector<std::string>& args, const std::string& key, float def) {
  std::string v = get_arg(args, key);
  return v.empty() ? def : std::stof(v);
}

static std::string format_seconds(double seconds) {
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

static void print_progress(const TrainingProgress& p) {
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
  std::cout << line.str() << '\n';
}

static void print_generation_progress(const GenerationProgress& p) {
  std::ostringstream line;
  line << "token " << p.token_index << '/' << p.total_tokens;
  line << " last_token " << p.last_token;
  line << " " << std::fixed << std::setprecision(2) << p.tokens_per_second << " tok/s";
  line << " elapsed " << format_seconds(p.elapsed_seconds);
  std::cerr << line.str() << '\n';
}

static void print_usage() {
  std::cerr << "Usage:\n"
            << "  microgpt train --input data.txt --checkpoint model.bin [options]\n"
            << "  microgpt resume --input data.txt --checkpoint model.bin [options]\n"
            << "  microgpt generate --checkpoint model.bin --prompt \"text\" [options]\n"
            << "  microgpt generate --checkpoint model.bin --prompt-token 104 [options]\n"
            << "  microgpt test\n";
  std::cerr << "  training options include --steps, --batch-size, --context, --d-model,\n"
            << "  --layers, --heads, --lr, --eval-interval, --save-interval, and\n"
            << "  --progress-interval.\n"
            << "  generation options include --max-new-tokens, --temperature, --top-k,\n"
            << "  --mode raw|instruction and --prompt-token.\n";
}

static Config make_config_from_args(const std::vector<std::string>& args) {
  Config cfg;
  cfg.context_length = get_arg_int(args, "--context", cfg.context_length);
  cfg.d_model = get_arg_int(args, "--d-model", cfg.d_model);
  cfg.num_layers = get_arg_int(args, "--layers", cfg.num_layers);
  cfg.num_heads = get_arg_int(args, "--heads", cfg.num_heads);
  cfg.d_ff = get_arg_int(args, "--ff", cfg.d_ff);
  cfg.batch_size = get_arg_int(args, "--batch-size", cfg.batch_size);
  cfg.learning_rate = get_arg_float(args, "--lr", cfg.learning_rate);
  cfg.eval_interval = get_arg_int(args, "--eval-interval", cfg.eval_interval);
  cfg.save_interval = get_arg_int(args, "--save-interval", cfg.save_interval);
  cfg.progress_interval = get_arg_int(args, "--progress-interval", cfg.progress_interval);
  cfg.max_new_tokens = get_arg_int(args, "--max-new-tokens", cfg.max_new_tokens);
  cfg.temperature = get_arg_float(args, "--temperature", cfg.temperature);
  cfg.top_k = get_arg_int(args, "--top-k", cfg.top_k);
  cfg.seed = static_cast<uint32_t>(get_arg_int(args, "--seed", static_cast<int>(cfg.seed)));
  return cfg;
}

static int run_train(const std::vector<std::string>& args, bool resume) {
  std::string input = get_arg(args, "--input");
  std::string checkpoint = get_arg(args, "--checkpoint", "checkpoint.bin");
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  Config cfg = make_config_from_args(args);
  int steps = get_arg_int(args, "--steps", 1000);
  Tokenizer tok;
  std::vector<int> tokens = tok.encode_text(read_file_text(input));
  if (tokens.size() < 2) {
    throw std::runtime_error("training input is too small");
  }
  std::vector<int> train_tokens = split_train_val(tokens, true);
  std::vector<int> val_tokens = split_train_val(tokens, false);
  AdamW opt;
  int resume_step = 0;
  Model model(cfg);
  if (resume) {
    model = load_checkpoint(checkpoint, opt, resume_step);
  } else {
    opt.lr = cfg.learning_rate;
    opt.beta1 = cfg.beta1;
    opt.beta2 = cfg.beta2;
    opt.eps = cfg.adam_eps;
    opt.weight_decay = cfg.weight_decay;
  }
  if (!resume) {
    train_model(model, train_tokens, val_tokens, opt, steps, checkpoint, 0, print_progress);
  } else {
    train_model(model, train_tokens, val_tokens, opt, steps, checkpoint, resume_step, print_progress);
  }
  return 0;
}

static int run_generate(const std::vector<std::string>& args) {
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
  int max_new = get_arg_int(args, "--max-new-tokens", model.cfg.max_new_tokens);
  float temperature = get_arg_float(args, "--temperature", model.cfg.temperature);
  int top_k = get_arg_int(args, "--top-k", model.cfg.top_k);
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
  std::string output = generate_text(model, model_prompt, max_new, temperature, top_k, Tokenizer::kEos,
                                     print_generation_progress);
  std::cout << output;
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      print_usage();
      return 1;
    }
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }
    const std::string cmd = args.front();
    std::vector<std::string> rest(args.begin() + 1, args.end());
    if (cmd == "train") {
      return run_train(rest, false);
    }
    if (cmd == "resume") {
      return run_train(rest, true);
    }
    if (cmd == "generate") {
      return run_generate(rest);
    }
    if (cmd == "test") {
      return run_tests() ? 0 : 2;
    }
    print_usage();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

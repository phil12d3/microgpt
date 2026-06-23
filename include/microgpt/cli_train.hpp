#pragma once

#include "microgpt/cli_backend.hpp"
#include "microgpt/cli_shared.hpp"

#include <chrono>
#include <ctime>

namespace microgpt {

struct TrainingRunSummary {
  std::string started_at;
  std::string ended_at;
  double duration_seconds = 0.0;
  float last_train_loss = 0.0f;
  float last_average_loss = 0.0f;
  float last_val_loss = 0.0f;
  bool has_validation = false;
};

inline std::string timestamp_utc_now() {
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

inline size_t model_parameter_count(Model& model) {
  size_t total = 0;
  for (Parameter* p : model.parameters()) {
    total += p->data.size();
  }
  return total;
}

inline std::string previous_training_history_entries(const std::string& metadata_path) {
  std::ifstream in(metadata_path, std::ios::binary);
  if (!in) {
    return "";
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string text = ss.str();
  std::string marker = "\"training_history\": [";
  size_t start = text.find(marker);
  if (start == std::string::npos) {
    return "";
  }
  start = text.find('[', start);
  if (start == std::string::npos) {
    return "";
  }
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = start; i < text.size(); ++i) {
    char c = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (c == '\\' && in_string) {
      escape = true;
      continue;
    }
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) {
      continue;
    }
    if (c == '[') {
      ++depth;
    } else if (c == ']') {
      --depth;
      if (depth == 0) {
        return text.substr(start + 1, i - start - 1);
      }
    }
  }
  return "";
}

inline void write_checkpoint_metadata_json(const std::string& checkpoint, const Model& model, const AdamW& opt, int final_step,
                                           const std::string& command, const std::string& input,
                                           const std::string& val_input, const std::string& validation_source,
                                           int requested_steps, int start_step, BackendKind backend,
                                           const TrainingRunSummary& run) {
  std::string metadata_path = checkpoint_json_path(checkpoint);
  std::string previous_history = previous_training_history_entries(metadata_path);
  std::ofstream json(metadata_path, std::ios::binary);
  if (!json) {
    throw std::runtime_error("failed to write checkpoint metadata json");
  }
  size_t params = model_parameter_count(const_cast<Model&>(model));
  json << "{\n";
  Tokenizer tok(tokenizer_kind_from_int(model.cfg.tokenizer_kind));
  json << "  \"magic\": \"MICROGPT2\",\n";
  json << "  \"checkpoint\": \"" << json_escape(checkpoint) << "\",\n";
  json << "  \"step\": " << final_step << ",\n";
  json << "  \"optimizer_step\": " << opt.step << ",\n";
  json << "  \"parameter_count\": " << params << ",\n";
  json << "  \"parameter_bytes_f32\": " << (params * sizeof(float)) << ",\n";
  json << "  \"start_step\": " << start_step << ",\n";
  json << "  \"requested_steps\": " << requested_steps << ",\n";
  json << "  \"command\": \"" << json_escape(command) << "\",\n";
  json << "  \"backend\": \"" << json_escape(backend_name(backend)) << "\",\n";
  json << "  \"train_input\": \"" << json_escape(input) << "\",\n";
  json << "  \"val_input\": \"" << json_escape(val_input) << "\",\n";
  json << "  \"validation_source\": \"" << json_escape(validation_source) << "\",\n";
  json << "  \"config\": {\n";
  json << "    \"vocab_size\": " << model.cfg.vocab_size << ",\n";
  json << "    \"tokenizer\": \"" << json_escape(tok.name()) << "\",\n";
  json << "    \"tokenizer_kind\": " << model.cfg.tokenizer_kind << ",\n";
  json << "    \"context_length\": " << model.cfg.context_length << ",\n";
  json << "    \"d_model\": " << model.cfg.d_model << ",\n";
  json << "    \"num_layers\": " << model.cfg.num_layers << ",\n";
  json << "    \"num_heads\": " << model.cfg.num_heads << ",\n";
  json << "    \"d_ff\": " << model.cfg.d_ff << ",\n";
  json << "    \"batch_size\": " << model.cfg.batch_size << ",\n";
  json << "    \"learning_rate\": " << model.cfg.learning_rate << ",\n";
  json << "    \"beta1\": " << model.cfg.beta1 << ",\n";
  json << "    \"beta2\": " << model.cfg.beta2 << ",\n";
  json << "    \"adam_eps\": " << model.cfg.adam_eps << ",\n";
  json << "    \"weight_decay\": " << model.cfg.weight_decay << ",\n";
  json << "    \"max_grad_norm\": " << model.cfg.max_grad_norm << "\n";
  json << "  },\n";
  json << "  \"training_history\": [\n";
  if (!previous_history.empty()) {
    json << previous_history;
    if (previous_history.find_first_not_of(" \t\r\n") != std::string::npos) {
      json << ",\n";
    }
  }
  json << "    {\n";
  json << "      \"started_at\": \"" << json_escape(run.started_at) << "\",\n";
  json << "      \"ended_at\": \"" << json_escape(run.ended_at) << "\",\n";
  json << "      \"duration_seconds\": " << run.duration_seconds << ",\n";
  json << "      \"start_step\": " << start_step << ",\n";
  json << "      \"end_step\": " << final_step << ",\n";
  json << "      \"requested_steps\": " << requested_steps << ",\n";
  json << "      \"command\": \"" << json_escape(command) << "\",\n";
  json << "      \"backend\": \"" << json_escape(backend_name(backend)) << "\",\n";
  json << "      \"tokenizer\": \"" << json_escape(tok.name()) << "\",\n";
  json << "      \"tokenizer_kind\": " << model.cfg.tokenizer_kind << ",\n";
  json << "      \"train_input\": \"" << json_escape(input) << "\",\n";
  json << "      \"val_input\": \"" << json_escape(val_input) << "\",\n";
  json << "      \"validation_source\": \"" << json_escape(validation_source) << "\",\n";
  json << "      \"parameter_count\": " << params << ",\n";
  json << "      \"parameter_bytes_f32\": " << (params * sizeof(float)) << ",\n";
  json << "      \"last_train_loss\": " << run.last_train_loss << ",\n";
  json << "      \"last_average_loss\": " << run.last_average_loss << ",\n";
  json << "      \"last_val_loss\": " << run.last_val_loss << ",\n";
  json << "      \"has_validation\": " << (run.has_validation ? "true" : "false") << "\n";
  json << "    }\n";
  json << "  ]\n";
  json << "}\n";
}

inline int run_train_command(const std::vector<std::string>& args, bool resume, const std::string& command, std::ostream& out,
                             std::ostream& err) {
  (void)err;
  BackendKind backend = require_backend_arg(args);
  std::string input = get_arg(args, "--input");
  std::string val_input = get_arg(args, "--val-input");
  std::string checkpoint = get_arg(args, "--checkpoint", "checkpoint.bin");
  if (input.empty()) {
    throw std::runtime_error("--input is required");
  }
  Config cfg = Config{};
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
  float split_ratio = get_arg_float(args, "--split-ratio", 0.9f);
  bool no_val_split = has_arg(args, "--no-val-split");
  if (split_ratio <= 0.0f || split_ratio >= 1.0f) {
    throw std::runtime_error("--split-ratio must be greater than 0 and less than 1");
  }
  if (no_val_split && !val_input.empty()) {
    throw std::runtime_error("--no-val-split cannot be used with --val-input");
  }

  std::string tokenizer_arg = get_arg(args, "--tokenizer", "byte");
  cfg.tokenizer_kind = static_cast<int>(parse_tokenizer_kind(tokenizer_arg));
  Tokenizer tok(tokenizer_kind_from_int(cfg.tokenizer_kind));
  cfg.vocab_size = tok.vocab_size();
  AdamW opt;
  int resume_step = 0;
  Model model(cfg);
  if (resume) {
    model = load_checkpoint(checkpoint, opt, resume_step);
    if (has_arg(args, "--tokenizer") && model.cfg.tokenizer_kind != cfg.tokenizer_kind) {
      throw std::runtime_error("--tokenizer cannot change when resuming a checkpoint");
    }
    tok = Tokenizer(tokenizer_kind_from_int(model.cfg.tokenizer_kind));
  } else {
    opt.lr = cfg.learning_rate;
    opt.beta1 = cfg.beta1;
    opt.beta2 = cfg.beta2;
    opt.eps = cfg.adam_eps;
    opt.weight_decay = cfg.weight_decay;
  }
  std::vector<int> tokens = tok.encode_text(read_file_text(input));
  if (tokens.size() < 2) {
    throw std::runtime_error("training input is too small");
  }
  std::vector<int> train_tokens;
  std::vector<int> val_tokens;
  if (val_input.empty()) {
    if (no_val_split) {
      train_tokens = tokens;
      val_tokens = tokens;
    } else {
      train_tokens = split_train_val(tokens, true, split_ratio);
      val_tokens = split_train_val(tokens, false, split_ratio);
    }
  } else {
    train_tokens = tokens;
    val_tokens = tok.encode_text(read_file_text(val_input));
    if (val_tokens.size() < 2) {
      throw std::runtime_error("validation input is too small");
    }
  }
  model.set_backend(backend);
  opt.set_backend(backend);
  reset_backend_dispatch_stats();
  std::string validation_source = val_input.empty() ? (no_val_split ? "full_data" : "auto_split") : "explicit";
  TrainingRunSummary run;
  run.started_at = timestamp_utc_now();
  auto started = std::chrono::steady_clock::now();
  auto cb = [&out, &run](const TrainingProgress& p) {
    print_training_progress(p, out);
    run.last_train_loss = p.loss;
    run.last_average_loss = p.average_loss;
    if (p.has_validation) {
      run.has_validation = true;
      run.last_val_loss = p.val_loss;
    }
  };
  int requested_steps = get_arg_int(args, "--steps", 1000);
  if (!resume) {
    train_model(model, train_tokens, val_tokens, opt, requested_steps, checkpoint, 0, cb);
  } else {
    train_model(model, train_tokens, val_tokens, opt, requested_steps, checkpoint, resume_step, cb);
  }
  auto ended = std::chrono::steady_clock::now();
  run.ended_at = timestamp_utc_now();
  run.duration_seconds = std::chrono::duration<double>(ended - started).count();
  write_checkpoint_metadata_json(checkpoint, model, opt, resume_step + requested_steps, command, input, val_input,
                                 validation_source, requested_steps, resume_step, backend, run);
  if (backend != BackendKind::Cpu) {
    const BackendDispatchStats& stats = backend_dispatch_stats();
    out << "backend_accelerated_ops " << stats.accelerated_ops << '\n';
    out << "backend_cpu_fallback_ops " << stats.cpu_fallback_ops << '\n';
    if (stats.accelerated_ops == 0) {
      out << "backend_note " << backend_name(backend) << " selected but no accelerated kernels ran\n";
    } else if (stats.cpu_fallback_ops > 0) {
      out << "backend_note " << backend_name(backend)
          << " accelerated kernels ran; unsupported operations used CPU fallback\n";
    } else {
      out << "backend_note " << backend_name(backend) << " accelerated kernels ran\n";
    }
  }
  return 0;
}

}  // namespace microgpt

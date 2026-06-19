#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/checkpoint.hpp"
#include "microgpt/io.hpp"
#include "microgpt/training.hpp"
#include "microgpt/generation.hpp"

#include <string>
#include <vector>

namespace microgpt::api {

struct GenerationOptions {
  int max_new_tokens = 200;
  float temperature = 0.8f;
  int top_k = 20;
  int stop_token_id = Tokenizer::kEos;
};

struct TrainingOptions {
  int steps = 1000;
  int context_length = 64;
  int d_model = 128;
  int num_layers = 2;
  int num_heads = 4;
  int d_ff = 512;
  int batch_size = 16;
  float learning_rate = 0.0003f;
  uint32_t seed = 42;
  BackendKind backend = BackendKind::Cpu;
};

class ModelHandle {
 public:
  ModelHandle(Model loaded_model, AdamW loaded_optimizer, int loaded_step)
      : model_(std::move(loaded_model)), optimizer_(std::move(loaded_optimizer)), step_(loaded_step) {}

  Model& model() { return model_; }
  const Model& model() const { return model_; }
  AdamW& optimizer() { return optimizer_; }
  const AdamW& optimizer() const { return optimizer_; }
  int step() const { return step_; }

 private:
  Model model_;
  AdamW optimizer_;
  int step_ = 0;
};

inline Config config_from_training_options(const TrainingOptions& options) {
  Config cfg;
  cfg.context_length = options.context_length;
  cfg.d_model = options.d_model;
  cfg.num_layers = options.num_layers;
  cfg.num_heads = options.num_heads;
  cfg.d_ff = options.d_ff;
  cfg.batch_size = options.batch_size;
  cfg.learning_rate = options.learning_rate;
  cfg.seed = options.seed;
  return cfg;
}

inline ModelHandle load_model(const std::string& checkpoint_path, BackendKind backend = BackendKind::Cpu) {
  AdamW opt;
  int step = 0;
  Model model = load_checkpoint(checkpoint_path, opt, step);
  model.set_backend(backend);
  return ModelHandle(std::move(model), std::move(opt), step);
}

inline std::string generate(ModelHandle& handle, const std::string& prompt, const GenerationOptions& options = {}) {
  return generate_text(handle.model(), prompt, options.max_new_tokens, options.temperature, options.top_k,
                       options.stop_token_id);
}

inline ModelHandle train_text_file(const std::string& input_path, const std::string& checkpoint_path,
                                   const TrainingOptions& options = {}) {
  Config cfg = config_from_training_options(options);
  Model model(cfg);
  model.set_backend(options.backend);
  AdamW opt;
  opt.lr = cfg.learning_rate;
  opt.beta1 = cfg.beta1;
  opt.beta2 = cfg.beta2;
  opt.eps = cfg.adam_eps;
  opt.weight_decay = cfg.weight_decay;
  std::vector<int> tokens = bytes_to_tokens(read_file_bytes(input_path), cfg.vocab_size);
  std::vector<int> train_tokens = split_train_val(tokens, true);
  std::vector<int> val_tokens = split_train_val(tokens, false);
  train_model(model, train_tokens, val_tokens, opt, options.steps, checkpoint_path);
  return load_model(checkpoint_path, options.backend);
}

}  // namespace microgpt::api

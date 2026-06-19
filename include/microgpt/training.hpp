#pragma once

#include "microgpt/checkpoint.hpp"
#include "microgpt/optim.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <vector>

namespace microgpt {
inline float evaluate_loss(Model& model, const std::vector<int>& tokens, int batches) {
  float total = 0.0f;
  for (int i = 0; i < batches; ++i) {
    Batch batch = sample_batch(tokens, model.cfg.batch_size, model.cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    total += cross_entropy_loss(logits, batch.y);
  }
  return total / static_cast<float>(batches);
}

struct TrainingProgress {
  int step = 0;
  int total_steps = 0;
  float loss = 0.0f;
  float average_loss = 0.0f;
  bool has_validation = false;
  float val_loss = 0.0f;
  float perplexity = 0.0f;
  double steps_per_second = 0.0;
  double elapsed_seconds = 0.0;
  double eta_seconds = 0.0;
};

struct GenerationProgress {
  int token_index = 0;
  int total_tokens = 0;
  int last_token = -1;
  double elapsed_seconds = 0.0;
  double tokens_per_second = 0.0;
};

inline void train_model(Model& model, const std::vector<int>& train_tokens, const std::vector<int>& val_tokens, AdamW& opt,
                        int steps, const std::string& checkpoint_path, int resume_step = 0,
                        const std::function<void(const TrainingProgress&)>& on_progress = {}) {
  int start_step = resume_step;
  using clock = std::chrono::steady_clock;
  auto train_start = clock::now();
  float rolling_loss = 0.0f;
  for (int step = 1; step <= steps; ++step) {
    model.zero_grad();
    Batch batch = sample_batch(train_tokens, model.cfg.batch_size, model.cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    float loss = cross_entropy_loss(logits, batch.y);
    model.backward(logits);
    auto params = model.parameters();
    clip_gradients(params, model.cfg.max_grad_norm);
    opt.update(params);
    rolling_loss += loss;
    int global_step = start_step + step;
    bool should_log = (step % model.cfg.progress_interval == 0) || (step == steps);
    if (global_step % model.cfg.eval_interval == 0) {
      float val_loss = evaluate_loss(model, val_tokens, model.cfg.eval_batches);
      should_log = true;
      auto now = clock::now();
      double elapsed = std::chrono::duration<double>(now - train_start).count();
      double steps_per_sec = global_step > 0 ? static_cast<double>(global_step) / elapsed : 0.0;
      double remaining = steps > step && steps_per_sec > 0.0 ? static_cast<double>(steps - step) / steps_per_sec : 0.0;
      if (on_progress) {
        TrainingProgress progress;
        progress.step = global_step;
        progress.total_steps = start_step + steps;
        progress.loss = loss;
        progress.average_loss = rolling_loss / static_cast<float>(std::max(1, model.cfg.progress_interval));
        progress.has_validation = true;
        progress.val_loss = val_loss;
        progress.perplexity = std::exp(val_loss);
        progress.steps_per_second = steps_per_sec;
        progress.elapsed_seconds = elapsed;
        progress.eta_seconds = remaining;
        on_progress(progress);
      }
      rolling_loss = 0.0f;
    } else if (should_log) {
      auto now = clock::now();
      double elapsed = std::chrono::duration<double>(now - train_start).count();
      double steps_per_sec = step > 0 ? static_cast<double>(step) / elapsed : 0.0;
      double remaining = steps > step && steps_per_sec > 0.0 ? static_cast<double>(steps - step) / steps_per_sec : 0.0;
      if (on_progress) {
        int window = std::min(step, model.cfg.progress_interval);
        TrainingProgress progress;
        progress.step = global_step;
        progress.total_steps = start_step + steps;
        progress.loss = loss;
        progress.average_loss = rolling_loss / static_cast<float>(window);
        progress.has_validation = false;
        progress.steps_per_second = steps_per_sec;
        progress.elapsed_seconds = elapsed;
        progress.eta_seconds = remaining;
        on_progress(progress);
      }
      rolling_loss = 0.0f;
    }
    if (!checkpoint_path.empty() && global_step % model.cfg.save_interval == 0) {
      save_checkpoint(checkpoint_path, model, opt, global_step);
    }
  }
  if (!checkpoint_path.empty()) {
    save_checkpoint(checkpoint_path, model, opt, start_step + steps);
  }
}

inline std::vector<int> split_train_val(const std::vector<int>& tokens, bool train) {
  size_t split = static_cast<size_t>(tokens.size() * 9 / 10);
  if (train) {
    return std::vector<int>(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(split));
  }
  return std::vector<int>(tokens.begin() + static_cast<std::ptrdiff_t>(split), tokens.end());
}

}  // namespace microgpt

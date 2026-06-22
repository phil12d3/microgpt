#pragma once

#include "microgpt/model.hpp"
#include "microgpt/sampling.hpp"
#include "microgpt/training.hpp"
#include "microgpt/tokenizer.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace microgpt {
inline std::string generate_text(Model& model, const std::string& prompt, int max_new_tokens, float temperature, int top_k,
                                 int stop_token_id = Tokenizer::kEos,
                                 const std::function<void(const GenerationProgress&)>& on_progress = {}) {
  Tokenizer tok(tokenizer_kind_from_int(model.cfg.tokenizer_kind));
  std::vector<int> ids = tok.encode_text(prompt);
  std::vector<int> generated_ids;
  using clock = std::chrono::steady_clock;
  auto gen_start = clock::now();
  for (int i = 0; i < max_new_tokens; ++i) {
    std::vector<int> window = ids;
    if (static_cast<int>(window.size()) > model.cfg.context_length) {
      window.erase(window.begin(), window.end() - model.cfg.context_length);
    }
    std::vector<int> flat(model.cfg.context_length, 0);
    for (size_t t = 0; t < window.size(); ++t) {
      flat[t] = window[t];
    }
    SeqTensor logits = model.forward(flat);
    int last_pos = static_cast<int>(window.size()) - 1;
    const float* row = &logits.data[idx3(0, last_pos, 0, logits.T, logits.D)];
    std::vector<float> last(row, row + logits.D);
    int next = sample_from_logits(last, temperature, top_k, model.rng);
    ids.push_back(next);
    if (next == stop_token_id) {
      break;
    }
    generated_ids.push_back(next);
    if (on_progress) {
      auto now = clock::now();
      double elapsed = std::chrono::duration<double>(now - gen_start).count();
      double tps = elapsed > 0.0 ? static_cast<double>(i + 1) / elapsed : 0.0;
      GenerationProgress progress;
      progress.token_index = i + 1;
      progress.total_tokens = max_new_tokens;
      progress.last_token = next;
      progress.elapsed_seconds = elapsed;
      progress.tokens_per_second = tps;
      on_progress(progress);
    }
  }
  return tok.decode_text(generated_ids);
}

}  // namespace microgpt

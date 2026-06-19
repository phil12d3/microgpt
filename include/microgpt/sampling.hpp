#pragma once

#include "microgpt/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace microgpt {
inline std::vector<int> top_k_filter(const std::vector<float>& logits, int top_k) {
  int n = static_cast<int>(logits.size());
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  if (top_k > 0 && top_k < n) {
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(top_k);
    return idx;
  }
  return idx;
}

inline int sample_from_logits(std::vector<float> logits, float temperature, int top_k, RNG& rng) {
  if (temperature <= 0.0f) {
    throw std::runtime_error("temperature must be positive");
  }
  for (float& v : logits) {
    v /= temperature;
  }
  std::vector<int> ids = top_k_filter(logits, top_k);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int id : ids) {
    max_logit = std::max(max_logit, logits[static_cast<size_t>(id)]);
  }
  std::vector<float> probs;
  probs.reserve(ids.size());
  float sum = 0.0f;
  for (int id : ids) {
    float p = std::exp(logits[static_cast<size_t>(id)] - max_logit);
    probs.push_back(p);
    sum += p;
  }
  float r = rng.uniform();
  float cdf = 0.0f;
  for (size_t i = 0; i < ids.size(); ++i) {
    cdf += probs[i] / sum;
    if (r <= cdf || i + 1 == ids.size()) {
      return ids[i];
    }
  }
  return ids.back();
}

}  // namespace microgpt

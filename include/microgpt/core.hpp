#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace microgpt {
inline size_t product(const std::vector<int>& shape) {
  size_t n = 1;
  for (int v : shape) {
    if (v <= 0) {
      throw std::runtime_error("invalid tensor shape");
    }
    n *= static_cast<size_t>(v);
  }
  return n;
}

inline size_t idx3(int b, int t, int d, int T, int D) {
  return (static_cast<size_t>(b) * static_cast<size_t>(T) + static_cast<size_t>(t)) *
             static_cast<size_t>(D) +
         static_cast<size_t>(d);
}

struct RNG {
  std::mt19937 gen;
  explicit RNG(uint32_t seed = 42) : gen(seed) {}

  float uniform(float a = 0.0f, float b = 1.0f) {
    std::uniform_real_distribution<float> dist(a, b);
    return dist(gen);
  }

  int randint(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(gen);
  }

  float normal(float mean = 0.0f, float stddev = 1.0f) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(gen);
  }
};

struct Config {
  int vocab_size = 260;
  int context_length = 64;
  int d_model = 128;
  int num_layers = 2;
  int num_heads = 4;
  int d_ff = 512;
  float dropout = 0.0f;
  int batch_size = 16;
  float learning_rate = 0.0003f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float adam_eps = 1e-8f;
  float weight_decay = 0.01f;
  float max_grad_norm = 1.0f;
  int eval_interval = 500;
  int eval_batches = 20;
  int save_interval = 1000;
  int progress_interval = 10;
  int max_new_tokens = 200;
  float temperature = 0.8f;
  int top_k = 20;
  uint32_t seed = 42;
};

struct Batch {
  int batch = 0;
  int time = 0;
  std::vector<int> x;
  std::vector<int> y;
};

struct SeqTensor {
  int B = 0;
  int T = 0;
  int D = 0;
  std::vector<float> data;
  std::vector<float> grad;

  SeqTensor() = default;
  SeqTensor(int b, int t, int d) { resize(b, t, d); }

  void resize(int b, int t, int d) {
    B = b;
    T = t;
    D = d;
    data.assign(static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(D), 0.0f);
    grad.assign(data.size(), 0.0f);
  }

  size_t size() const { return data.size(); }

  void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
};

struct Parameter {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<float> m;
  std::vector<float> v;
  bool decay = true;

  Parameter() = default;
  Parameter(std::string n, std::vector<int> s, bool use_decay = true)
      : name(std::move(n)), shape(std::move(s)), decay(use_decay) {
    reset(shape);
  }

  void reset(const std::vector<int>& s) {
    shape = s;
    data.assign(product(shape), 0.0f);
    grad.assign(data.size(), 0.0f);
    m.assign(data.size(), 0.0f);
    v.assign(data.size(), 0.0f);
  }

  size_t size() const { return data.size(); }

  void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
};

inline void fill_normal(std::vector<float>& v, RNG& rng, float mean, float stddev) {
  for (float& x : v) {
    x = rng.normal(mean, stddev);
  }
}

inline void fill_zeros(std::vector<float>& v) { std::fill(v.begin(), v.end(), 0.0f); }

inline void fill_ones(std::vector<float>& v) { std::fill(v.begin(), v.end(), 1.0f); }

inline void apply_add_bias(SeqTensor& y, const std::vector<float>& bias) {
  for (int b = 0; b < y.B; ++b) {
    for (int t = 0; t < y.T; ++t) {
      size_t base = idx3(b, t, 0, y.T, y.D);
      for (int o = 0; o < y.D; ++o) {
        y.data[base + static_cast<size_t>(o)] += bias[static_cast<size_t>(o)];
      }
    }
  }
}

}  // namespace microgpt

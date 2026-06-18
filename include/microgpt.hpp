#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
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

struct Linear {
  Parameter w;
  Parameter b;
  int in_features = 0;
  int out_features = 0;

  Linear() = default;
  Linear(const std::string& prefix, int in_f, int out_f, RNG& rng, bool bias = true)
      : w(prefix + ".w", {in_f, out_f}, true), b(prefix + ".b", {out_f}, false), in_features(in_f), out_features(out_f) {
    fill_normal(w.data, rng, 0.0f, 0.02f);
    if (bias) {
      fill_zeros(b.data);
    } else {
      b.data.clear();
      b.grad.clear();
      b.m.clear();
      b.v.clear();
      b.shape.clear();
    }
  }

  SeqTensor forward(const SeqTensor& x) const {
    if (x.D != in_features) {
      throw std::runtime_error("linear forward: input feature mismatch");
    }
    SeqTensor y(x.B, x.T, out_features);
    for (int b_ix = 0; b_ix < x.B; ++b_ix) {
      for (int t = 0; t < x.T; ++t) {
        const float* xin = &x.data[idx3(b_ix, t, 0, x.T, x.D)];
        float* yout = &y.data[idx3(b_ix, t, 0, y.T, y.D)];
        for (int o = 0; o < out_features; ++o) {
          float sum = b.data.empty() ? 0.0f : b.data[static_cast<size_t>(o)];
          for (int i = 0; i < in_features; ++i) {
            sum += xin[static_cast<size_t>(i)] * w.data[static_cast<size_t>(i) * static_cast<size_t>(out_features) +
                                                       static_cast<size_t>(o)];
          }
          yout[static_cast<size_t>(o)] = sum;
        }
      }
    }
    return y;
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    if (x.grad.size() != x.data.size()) {
      x.grad.assign(x.data.size(), 0.0f);
    }
    for (int b_ix = 0; b_ix < x.B; ++b_ix) {
      for (int t = 0; t < x.T; ++t) {
        const float* xin = &x.data[idx3(b_ix, t, 0, x.T, x.D)];
        float* xg = &x.grad[idx3(b_ix, t, 0, x.T, x.D)];
        const float* yg = &y.grad[idx3(b_ix, t, 0, y.T, y.D)];
        for (int o = 0; o < out_features; ++o) {
          float g = yg[static_cast<size_t>(o)];
          if (!b.data.empty()) {
            b.grad[static_cast<size_t>(o)] += g;
          }
          for (int i = 0; i < in_features; ++i) {
            size_t wi = static_cast<size_t>(i) * static_cast<size_t>(out_features) + static_cast<size_t>(o);
            w.grad[wi] += xin[static_cast<size_t>(i)] * g;
            xg[static_cast<size_t>(i)] += w.data[wi] * g;
          }
        }
      }
    }
  }
};

struct LayerNorm {
  Parameter gamma;
  Parameter beta;
  float eps = 1e-5f;
  std::vector<float> mean;
  std::vector<float> inv_std;
  std::vector<float> xhat;
  int dim = 0;

  LayerNorm() = default;
  LayerNorm(const std::string& prefix, int d) : gamma(prefix + ".gamma", {d}, false), beta(prefix + ".beta", {d}, false), dim(d) {
    fill_ones(gamma.data);
    fill_zeros(beta.data);
  }

  SeqTensor forward(const SeqTensor& x) {
    if (x.D != dim) {
      throw std::runtime_error("layernorm forward: feature mismatch");
    }
    SeqTensor y(x.B, x.T, x.D);
    mean.assign(static_cast<size_t>(x.B) * static_cast<size_t>(x.T), 0.0f);
    inv_std.assign(mean.size(), 0.0f);
    xhat.assign(y.data.size(), 0.0f);
    for (int b = 0; b < x.B; ++b) {
      for (int t = 0; t < x.T; ++t) {
        size_t row = static_cast<size_t>(b) * static_cast<size_t>(x.T) + static_cast<size_t>(t);
        const float* xin = &x.data[idx3(b, t, 0, x.T, x.D)];
        float mu = 0.0f;
        for (int i = 0; i < x.D; ++i) {
          mu += xin[static_cast<size_t>(i)];
        }
        mu /= static_cast<float>(x.D);
        float var = 0.0f;
        for (int i = 0; i < x.D; ++i) {
          float c = xin[static_cast<size_t>(i)] - mu;
          var += c * c;
        }
        var /= static_cast<float>(x.D);
        float inv = 1.0f / std::sqrt(var + eps);
        mean[row] = mu;
        inv_std[row] = inv;
        float* yout = &y.data[idx3(b, t, 0, y.T, y.D)];
        for (int i = 0; i < x.D; ++i) {
          float xn = (xin[static_cast<size_t>(i)] - mu) * inv;
          xhat[idx3(b, t, i, x.T, x.D)] = xn;
          yout[static_cast<size_t>(i)] = xn * gamma.data[static_cast<size_t>(i)] + beta.data[static_cast<size_t>(i)];
        }
      }
    }
    return y;
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    if (x.grad.size() != x.data.size()) {
      x.grad.assign(x.data.size(), 0.0f);
    }
    std::vector<float> dxhat(x.D, 0.0f);
    std::vector<float> xmu(x.D, 0.0f);
    for (int b = 0; b < x.B; ++b) {
      for (int t = 0; t < x.T; ++t) {
        size_t row = static_cast<size_t>(b) * static_cast<size_t>(x.T) + static_cast<size_t>(t);
        const float* xin = &x.data[idx3(b, t, 0, x.T, x.D)];
        const float* yg = &y.grad[idx3(b, t, 0, y.T, y.D)];
        float sum_dxhat = 0.0f;
        float sum_dxhat_xmu = 0.0f;
        for (int i = 0; i < x.D; ++i) {
          float xn = xhat[idx3(b, t, i, x.T, x.D)];
          gamma.grad[static_cast<size_t>(i)] += yg[static_cast<size_t>(i)] * xn;
          beta.grad[static_cast<size_t>(i)] += yg[static_cast<size_t>(i)];
          float dxh = yg[static_cast<size_t>(i)] * gamma.data[static_cast<size_t>(i)];
          dxhat[static_cast<size_t>(i)] = dxh;
          xmu[static_cast<size_t>(i)] = xin[static_cast<size_t>(i)] - mean[row];
          sum_dxhat += dxh;
          sum_dxhat_xmu += dxh * xmu[static_cast<size_t>(i)];
        }
        float inv = inv_std[row];
        float dvar = -0.5f * std::pow(inv, 3.0f) * sum_dxhat_xmu;
        float dmu = -inv * sum_dxhat;
        float mean_xmu = 0.0f;
        for (int i = 0; i < x.D; ++i) {
          mean_xmu += xmu[static_cast<size_t>(i)];
        }
        dmu += dvar * (-2.0f * mean_xmu / static_cast<float>(x.D));
        float* xg = &x.grad[idx3(b, t, 0, x.T, x.D)];
        for (int i = 0; i < x.D; ++i) {
          xg[static_cast<size_t>(i)] += dxhat[static_cast<size_t>(i)] * inv +
                                       dvar * 2.0f * xmu[static_cast<size_t>(i)] / static_cast<float>(x.D) +
                                       dmu / static_cast<float>(x.D);
        }
      }
    }
  }
};

inline float gelu(float x) {
  const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
  float x3 = x * x * x;
  float t = std::tanh(c * (x + 0.044715f * x3));
  return 0.5f * x * (1.0f + t);
}

inline float gelu_derivative(float x) {
  const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
  float x2 = x * x;
  float x3 = x2 * x;
  float u = c * (x + 0.044715f * x3);
  float t = std::tanh(u);
  float sech2 = 1.0f - t * t;
  float term = c * (1.0f + 3.0f * 0.044715f * x2);
  return 0.5f * (1.0f + t) + 0.5f * x * sech2 * term;
}

struct FeedForward {
  Linear fc1;
  Linear fc2;
  SeqTensor hidden;

  FeedForward() = default;
  FeedForward(const std::string& prefix, int d_model, int d_ff, RNG& rng)
      : fc1(prefix + ".fc1", d_model, d_ff, rng), fc2(prefix + ".fc2", d_ff, d_model, rng) {}

  SeqTensor forward(const SeqTensor& x) {
    hidden = fc1.forward(x);
    for (float& v : hidden.data) {
      v = gelu(v);
    }
    return fc2.forward(hidden);
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    fc2.backward(hidden, y);
    for (size_t i = 0; i < hidden.grad.size(); ++i) {
      hidden.grad[i] *= gelu_derivative(hidden.data[i]);
    }
    fc1.backward(x, hidden);
  }
};

struct Attention {
  int d_model = 0;
  int num_heads = 0;
  int head_dim = 0;
  Linear q_proj;
  Linear k_proj;
  Linear v_proj;
  Linear o_proj;
  SeqTensor q;
  SeqTensor k;
  SeqTensor v;
  SeqTensor context;
  std::vector<float> weights;

  Attention() = default;
  Attention(const std::string& prefix, int d, int h, RNG& rng)
      : d_model(d),
        num_heads(h),
        head_dim(d / h),
        q_proj(prefix + ".q", d, d, rng),
        k_proj(prefix + ".k", d, d, rng),
        v_proj(prefix + ".v", d, d, rng),
        o_proj(prefix + ".o", d, d, rng) {
    if (d_model % num_heads != 0) {
      throw std::runtime_error("d_model must be divisible by num_heads");
    }
  }

  SeqTensor forward(const SeqTensor& x) {
    q = q_proj.forward(x);
    k = k_proj.forward(x);
    v = v_proj.forward(x);
    context = SeqTensor(x.B, x.T, x.D);
    weights.assign(static_cast<size_t>(x.B) * static_cast<size_t>(num_heads) * static_cast<size_t>(x.T) *
                       static_cast<size_t>(x.T),
                   0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int b = 0; b < x.B; ++b) {
      for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        for (int t = 0; t < x.T; ++t) {
          float max_score = -std::numeric_limits<float>::infinity();
          for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              score += q.data[idx3(b, t, offset + d, x.T, x.D)] *
                       k.data[idx3(b, s, offset + d, x.T, x.D)];
            }
            score *= scale;
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            weights[wi] = score;
            if (score > max_score) {
              max_score = score;
            }
          }
          float sum = 0.0f;
          for (int s = 0; s <= t; ++s) {
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            float e = std::exp(weights[wi] - max_score);
            weights[wi] = e;
            sum += e;
          }
          for (int s = 0; s <= t; ++s) {
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            float w = weights[wi] / sum;
            weights[wi] = w;
            for (int d = 0; d < head_dim; ++d) {
              context.data[idx3(b, t, offset + d, x.T, x.D)] +=
                  w * v.data[idx3(b, s, offset + d, x.T, x.D)];
            }
          }
        }
      }
    }
    return o_proj.forward(context);
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    o_proj.backward(context, y);
    if (context.grad.size() != context.data.size()) {
      context.grad.assign(context.data.size(), 0.0f);
    }
    q.grad.assign(q.data.size(), 0.0f);
    k.grad.assign(k.data.size(), 0.0f);
    v.grad.assign(v.data.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> dweights(static_cast<size_t>(x.T), 0.0f);
    for (int b = 0; b < x.B; ++b) {
      for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        for (int t = 0; t < x.T; ++t) {
          std::fill(dweights.begin(), dweights.end(), 0.0f);
          const float* gctx = &context.grad[idx3(b, t, offset, x.T, x.D)];
          for (int s = 0; s <= t; ++s) {
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            float w = weights[wi];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              float vv = v.data[idx3(b, s, offset + d, x.T, x.D)];
              dot += gctx[static_cast<size_t>(d)] * vv;
              v.grad[idx3(b, s, offset + d, x.T, x.D)] += w * gctx[static_cast<size_t>(d)];
            }
            dweights[static_cast<size_t>(s)] = dot;
          }
          float sum = 0.0f;
          for (int s = 0; s <= t; ++s) {
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            sum += dweights[static_cast<size_t>(s)] * weights[wi];
          }
          for (int s = 0; s <= t; ++s) {
            size_t wi = (((static_cast<size_t>(b) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) *
                              static_cast<size_t>(x.T) +
                          static_cast<size_t>(t)) *
                             static_cast<size_t>(x.T)) +
                         static_cast<size_t>(s);
            float dscore = weights[wi] * (dweights[static_cast<size_t>(s)] - sum);
            for (int d = 0; d < head_dim; ++d) {
              q.grad[idx3(b, t, offset + d, x.T, x.D)] += dscore * k.data[idx3(b, s, offset + d, x.T, x.D)] * scale;
              k.grad[idx3(b, s, offset + d, x.T, x.D)] += dscore * q.data[idx3(b, t, offset + d, x.T, x.D)] * scale;
            }
          }
        }
      }
    }
    q_proj.backward(x, q);
    k_proj.backward(x, k);
    v_proj.backward(x, v);
  }
};

struct Block {
  LayerNorm ln1;
  Attention attn;
  LayerNorm ln2;
  FeedForward ff;
  SeqTensor norm1;
  SeqTensor attn_out;
  SeqTensor resid1;
  SeqTensor norm2;
  SeqTensor ff_out;

  Block() = default;
  Block(const std::string& prefix, int d_model, int num_heads, int d_ff, RNG& rng)
      : ln1(prefix + ".ln1", d_model),
        attn(prefix + ".attn", d_model, num_heads, rng),
        ln2(prefix + ".ln2", d_model),
        ff(prefix + ".ff", d_model, d_ff, rng) {}

  SeqTensor forward(const SeqTensor& x) {
    norm1 = ln1.forward(x);
    attn_out = attn.forward(norm1);
    resid1 = SeqTensor(x.B, x.T, x.D);
    for (size_t i = 0; i < resid1.data.size(); ++i) {
      resid1.data[i] = x.data[i] + attn_out.data[i];
    }
    norm2 = ln2.forward(resid1);
    ff_out = ff.forward(norm2);
    SeqTensor y(x.B, x.T, x.D);
    for (size_t i = 0; i < y.data.size(); ++i) {
      y.data[i] = resid1.data[i] + ff_out.data[i];
    }
    return y;
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    if (resid1.grad.size() != resid1.data.size()) {
      resid1.grad.assign(resid1.data.size(), 0.0f);
    }
    if (norm2.grad.size() != norm2.data.size()) {
      norm2.grad.assign(norm2.data.size(), 0.0f);
    }
    if (attn_out.grad.size() != attn_out.data.size()) {
      attn_out.grad.assign(attn_out.data.size(), 0.0f);
    }
    if (norm1.grad.size() != norm1.data.size()) {
      norm1.grad.assign(norm1.data.size(), 0.0f);
    }
    for (size_t i = 0; i < y.grad.size(); ++i) {
      resid1.grad[i] += y.grad[i];
      ff_out.grad[i] += y.grad[i];
    }
    ff.backward(norm2, ff_out);
    ln2.backward(resid1, norm2);
    for (size_t i = 0; i < resid1.grad.size(); ++i) {
      x.grad[i] += resid1.grad[i];
      attn_out.grad[i] += resid1.grad[i];
    }
    attn.backward(norm1, attn_out);
    ln1.backward(x, norm1);
  }
};

struct Tokenizer {
  static constexpr int kByteVocab = 256;
  static constexpr int kBos = 256;
  static constexpr int kEos = 257;
  static constexpr int kUser = 258;
  static constexpr int kAssistant = 259;

  int vocab_size() const { return 260; }

  std::vector<int> encode_text(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
      if (text.compare(i, 5, "<BOS>") == 0) {
        ids.push_back(kBos);
        i += 5;
      } else if (text.compare(i, 5, "<EOS>") == 0) {
        ids.push_back(kEos);
        i += 5;
      } else if (text.compare(i, 6, "<USER>") == 0) {
        ids.push_back(kUser);
        i += 6;
      } else if (text.compare(i, 11, "<ASSISTANT>") == 0) {
        ids.push_back(kAssistant);
        i += 11;
      } else {
        ids.push_back(static_cast<int>(static_cast<unsigned char>(text[i])));
        ++i;
      }
    }
    return ids;
  }

  std::string decode_text(const std::vector<int>& ids) const {
    std::string s;
    s.reserve(ids.size());
    for (int id : ids) {
      switch (id) {
        case kBos:
          s += "<BOS>";
          break;
        case kEos:
          s += "<EOS>";
          break;
        case kUser:
          s += "<USER>";
          break;
        case kAssistant:
          s += "<ASSISTANT>";
          break;
        default:
          if (id >= 0 && id < kByteVocab) {
            s.push_back(static_cast<char>(id));
          }
          break;
      }
    }
    return s;
  }

  std::string decode_bytes(const std::vector<int>& ids) const { return decode_text(ids); }
};

inline std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }
  in.seekg(0, std::ios::end);
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (size > 0 && !in.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("failed to read input file: " + path);
  }
  return data;
}

inline std::string read_file_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline void write_file_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to write file: " + path);
  }
  out << text;
}

inline std::vector<int> bytes_to_tokens(const std::vector<uint8_t>& bytes) {
  std::vector<int> tokens;
  tokens.reserve(bytes.size());
  for (uint8_t b : bytes) {
    tokens.push_back(static_cast<int>(b));
  }
  return tokens;
}

inline Batch sample_batch(const std::vector<int>& tokens, int batch_size, int context_length, RNG& rng) {
  if (tokens.size() < 2) {
    throw std::runtime_error("not enough tokens for batch sampling");
  }
  Batch batch;
  batch.batch = batch_size;
  batch.time = context_length;
  batch.x.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(context_length));
  batch.y.resize(batch.x.size());
  for (int b = 0; b < batch_size; ++b) {
    int start = rng.randint(0, static_cast<int>(tokens.size()) - 1);
    for (int t = 0; t < context_length; ++t) {
      size_t ix = (static_cast<size_t>(start) + static_cast<size_t>(t)) % tokens.size();
      size_t iy = (ix + 1) % tokens.size();
      batch.x[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[ix];
      batch.y[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[iy];
    }
  }
  return batch;
}

inline Batch make_fixed_batch(const std::vector<int>& tokens, int batch_size, int context_length) {
  Batch batch;
  batch.batch = batch_size;
  batch.time = context_length;
  batch.x.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(context_length));
  batch.y.resize(batch.x.size());
  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < context_length; ++t) {
      size_t src = static_cast<size_t>(t % std::max<int>(1, static_cast<int>(tokens.size()) - 1));
      batch.x[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[src];
      batch.y[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[src + 1];
    }
  }
  return batch;
}

inline SeqTensor batch_to_seqtensor(const Batch& b, int d) {
  SeqTensor t(b.batch, b.time, d);
  return t;
}

struct Model {
  Config cfg;
  RNG rng;
  Parameter token_embedding;
  Parameter position_embedding;
  std::vector<Block> blocks;
  LayerNorm final_norm;
  Linear lm_head;
  std::vector<int> last_tokens;
  int last_batch_size = 0;
  SeqTensor embed_out;
  std::vector<SeqTensor> block_inputs;
  SeqTensor pre_norm;
  SeqTensor norm_out;

  explicit Model(const Config& c)
      : cfg(c),
        rng(c.seed),
        token_embedding("tok_emb", {c.vocab_size, c.d_model}, true),
        position_embedding("pos_emb", {c.context_length, c.d_model}, true),
        final_norm("final_norm", c.d_model),
        lm_head("lm_head", c.d_model, c.vocab_size, rng) {
    fill_normal(token_embedding.data, rng, 0.0f, 0.02f);
    fill_normal(position_embedding.data, rng, 0.0f, 0.02f);
    blocks.reserve(static_cast<size_t>(cfg.num_layers));
    for (int i = 0; i < cfg.num_layers; ++i) {
      blocks.emplace_back("block" + std::to_string(i), cfg.d_model, cfg.num_heads, cfg.d_ff, rng);
    }
  }

  void zero_grad() {
    token_embedding.zero_grad();
    position_embedding.zero_grad();
    final_norm.gamma.zero_grad();
    final_norm.beta.zero_grad();
    lm_head.w.zero_grad();
    lm_head.b.zero_grad();
    for (auto& block : blocks) {
      block.ln1.gamma.zero_grad();
      block.ln1.beta.zero_grad();
      block.ln2.gamma.zero_grad();
      block.ln2.beta.zero_grad();
      block.attn.q_proj.w.zero_grad();
      block.attn.q_proj.b.zero_grad();
      block.attn.k_proj.w.zero_grad();
      block.attn.k_proj.b.zero_grad();
      block.attn.v_proj.w.zero_grad();
      block.attn.v_proj.b.zero_grad();
      block.attn.o_proj.w.zero_grad();
      block.attn.o_proj.b.zero_grad();
      block.ff.fc1.w.zero_grad();
      block.ff.fc1.b.zero_grad();
      block.ff.fc2.w.zero_grad();
      block.ff.fc2.b.zero_grad();
    }
  }

  std::vector<Parameter*> parameters() {
    std::vector<Parameter*> params;
    params.push_back(&token_embedding);
    params.push_back(&position_embedding);
    params.push_back(&final_norm.gamma);
    params.push_back(&final_norm.beta);
    params.push_back(&lm_head.w);
    params.push_back(&lm_head.b);
    for (auto& block : blocks) {
      params.push_back(&block.ln1.gamma);
      params.push_back(&block.ln1.beta);
      params.push_back(&block.attn.q_proj.w);
      params.push_back(&block.attn.q_proj.b);
      params.push_back(&block.attn.k_proj.w);
      params.push_back(&block.attn.k_proj.b);
      params.push_back(&block.attn.v_proj.w);
      params.push_back(&block.attn.v_proj.b);
      params.push_back(&block.attn.o_proj.w);
      params.push_back(&block.attn.o_proj.b);
      params.push_back(&block.ln2.gamma);
      params.push_back(&block.ln2.beta);
      params.push_back(&block.ff.fc1.w);
      params.push_back(&block.ff.fc1.b);
      params.push_back(&block.ff.fc2.w);
      params.push_back(&block.ff.fc2.b);
    }
    return params;
  }

  SeqTensor embed_tokens(const std::vector<int>& tokens) {
    if (tokens.size() % static_cast<size_t>(cfg.context_length) != 0) {
      throw std::runtime_error("embed_tokens expects a whole number of context windows");
    }
    int batch = static_cast<int>(tokens.size() / static_cast<size_t>(cfg.context_length));
    SeqTensor x(batch, cfg.context_length, cfg.d_model);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < cfg.context_length; ++t) {
        int tok = tokens[static_cast<size_t>(b) * static_cast<size_t>(cfg.context_length) + static_cast<size_t>(t)];
        if (tok < 0 || tok >= cfg.vocab_size) {
          throw std::runtime_error("token out of range");
        }
        for (int d = 0; d < cfg.d_model; ++d) {
          x.data[idx3(b, t, d, x.T, x.D)] =
              token_embedding.data[static_cast<size_t>(tok) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +
              position_embedding.data[static_cast<size_t>(t) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)];
        }
      }
    }
    return x;
  }

  SeqTensor forward(const std::vector<int>& tokens) {
    if (tokens.size() % static_cast<size_t>(cfg.context_length) != 0) {
      throw std::runtime_error("model forward received invalid token count");
    }
    last_tokens = tokens;
    last_batch_size = static_cast<int>(tokens.size() / static_cast<size_t>(cfg.context_length));
    embed_out = embed_tokens(tokens);
    block_inputs.clear();
    SeqTensor x = embed_out;
    for (auto& block : blocks) {
      block_inputs.push_back(x);
      x = block.forward(x);
    }
    pre_norm = x;
    norm_out = final_norm.forward(x);
    return lm_head.forward(norm_out);
  }

  void backward(SeqTensor& logits) {
    lm_head.backward(norm_out, logits);
    final_norm.backward(pre_norm, norm_out);
    SeqTensor* current = &pre_norm;
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
      blocks[static_cast<size_t>(i)].backward(block_inputs[static_cast<size_t>(i)], *current);
      current = &block_inputs[static_cast<size_t>(i)];
    }
    if (embed_out.grad.size() != embed_out.data.size()) {
      embed_out.grad.assign(embed_out.data.size(), 0.0f);
    }
    for (int b = 0; b < last_batch_size; ++b) {
      for (int t = 0; t < cfg.context_length; ++t) {
        int tok = last_tokens[static_cast<size_t>(b) * static_cast<size_t>(cfg.context_length) + static_cast<size_t>(t)];
        float* eg = &embed_out.grad[idx3(b, t, 0, embed_out.T, embed_out.D)];
        for (int d = 0; d < cfg.d_model; ++d) {
          token_embedding.grad[static_cast<size_t>(tok) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +=
              eg[static_cast<size_t>(d)];
          position_embedding.grad[static_cast<size_t>(t) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +=
              eg[static_cast<size_t>(d)];
        }
      }
    }
  }
};

inline float cross_entropy_loss(SeqTensor& logits, const std::vector<int>& targets) {
  if (static_cast<int>(targets.size()) != logits.B * logits.T) {
    throw std::runtime_error("cross_entropy_loss target count mismatch");
  }
  std::fill(logits.grad.begin(), logits.grad.end(), 0.0f);
  float loss = 0.0f;
  const float inv_n = 1.0f / static_cast<float>(logits.B * logits.T);
  for (int b = 0; b < logits.B; ++b) {
    for (int t = 0; t < logits.T; ++t) {
      const float* row = &logits.data[idx3(b, t, 0, logits.T, logits.D)];
      float* g = &logits.grad[idx3(b, t, 0, logits.T, logits.D)];
      int target = targets[static_cast<size_t>(b) * static_cast<size_t>(logits.T) + static_cast<size_t>(t)];
      if (target < 0 || target >= logits.D) {
        throw std::runtime_error("target token out of range");
      }
      float max_logit = row[0];
      for (int v = 1; v < logits.D; ++v) {
        max_logit = std::max(max_logit, row[static_cast<size_t>(v)]);
      }
      float sum_exp = 0.0f;
      for (int v = 0; v < logits.D; ++v) {
        sum_exp += std::exp(row[static_cast<size_t>(v)] - max_logit);
      }
      float log_prob = row[static_cast<size_t>(target)] - max_logit - std::log(sum_exp);
      loss += -log_prob;
      for (int v = 0; v < logits.D; ++v) {
        float p = std::exp(row[static_cast<size_t>(v)] - max_logit) / sum_exp;
        g[static_cast<size_t>(v)] = p * inv_n;
      }
      g[static_cast<size_t>(target)] -= inv_n;
    }
  }
  return loss * inv_n;
}

inline void clip_gradients(std::vector<Parameter*>& params, float max_norm) {
  double sumsq = 0.0;
  for (Parameter* p : params) {
    for (float g : p->grad) {
      sumsq += static_cast<double>(g) * static_cast<double>(g);
    }
  }
  double norm = std::sqrt(sumsq);
  if (norm > static_cast<double>(max_norm) && norm > 0.0) {
    float scale = static_cast<float>(max_norm / norm);
    for (Parameter* p : params) {
      for (float& g : p->grad) {
        g *= scale;
      }
    }
  }
}

struct AdamW {
  float lr = 0.0003f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  int step = 0;

  void update(std::vector<Parameter*>& params) {
    ++step;
    for (Parameter* p : params) {
      if (p->m.size() != p->data.size()) {
        p->m.assign(p->data.size(), 0.0f);
        p->v.assign(p->data.size(), 0.0f);
      }
      for (size_t i = 0; i < p->data.size(); ++i) {
        float g = p->grad[i];
        p->m[i] = beta1 * p->m[i] + (1.0f - beta1) * g;
        p->v[i] = beta2 * p->v[i] + (1.0f - beta2) * g * g;
        float mhat = p->m[i] / (1.0f - std::pow(beta1, static_cast<float>(step)));
        float vhat = p->v[i] / (1.0f - std::pow(beta2, static_cast<float>(step)));
        float update = mhat / (std::sqrt(vhat) + eps);
        if (p->decay) {
          update += weight_decay * p->data[i];
        }
        p->data[i] -= lr * update;
      }
    }
  }
};

inline std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

inline std::string checkpoint_json_path(const std::string& checkpoint) {
  size_t dot = checkpoint.find_last_of('.');
  if (dot == std::string::npos) {
    return checkpoint + ".json";
  }
  return checkpoint.substr(0, dot) + ".json";
}

inline void write_u64(std::ostream& out, uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline void write_i32(std::ostream& out, int32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline void write_f32(std::ostream& out, float v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline uint64_t read_u64(std::istream& in) {
  uint64_t v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline int32_t read_i32(std::istream& in) {
  int32_t v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline float read_f32(std::istream& in) {
  float v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline void write_string(std::ostream& out, const std::string& s) {
  write_u64(out, static_cast<uint64_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline std::string read_string(std::istream& in) {
  uint64_t n = read_u64(in);
  std::string s(static_cast<size_t>(n), '\0');
  in.read(s.data(), static_cast<std::streamsize>(n));
  return s;
}

inline void write_shape(std::ostream& out, const std::vector<int>& shape) {
  write_u64(out, static_cast<uint64_t>(shape.size()));
  for (int v : shape) {
    write_i32(out, static_cast<int32_t>(v));
  }
}

inline std::vector<int> read_shape(std::istream& in) {
  uint64_t n = read_u64(in);
  std::vector<int> shape(static_cast<size_t>(n));
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = static_cast<int>(read_i32(in));
  }
  return shape;
}

inline void write_vector_f32(std::ostream& out, const std::vector<float>& v) {
  write_u64(out, static_cast<uint64_t>(v.size()));
  if (!v.empty()) {
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  }
}

inline std::vector<float> read_vector_f32(std::istream& in) {
  uint64_t n = read_u64(in);
  std::vector<float> v(static_cast<size_t>(n));
  if (!v.empty()) {
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  }
  return v;
}

inline void save_checkpoint(const std::string& path, const Model& model, const AdamW& opt, int step) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open checkpoint for write: " + path);
  }
  out.write("MICROGPT1", 9);
  write_i32(out, step);
  write_i32(out, model.cfg.vocab_size);
  write_i32(out, model.cfg.context_length);
  write_i32(out, model.cfg.d_model);
  write_i32(out, model.cfg.num_layers);
  write_i32(out, model.cfg.num_heads);
  write_i32(out, model.cfg.d_ff);
  write_i32(out, model.cfg.batch_size);
  write_f32(out, model.cfg.learning_rate);
  write_f32(out, model.cfg.beta1);
  write_f32(out, model.cfg.beta2);
  write_f32(out, model.cfg.adam_eps);
  write_f32(out, model.cfg.weight_decay);
  write_f32(out, model.cfg.max_grad_norm);
  write_i32(out, opt.step);
  auto params = const_cast<Model&>(model).parameters();
  write_u64(out, static_cast<uint64_t>(params.size()));
  for (Parameter* p : params) {
    write_string(out, p->name);
    write_shape(out, p->shape);
    write_vector_f32(out, p->data);
    write_vector_f32(out, p->m);
    write_vector_f32(out, p->v);
    write_u64(out, p->decay ? 1 : 0);
  }
  out.close();

  std::ofstream json(checkpoint_json_path(path), std::ios::binary);
  if (!json) {
    throw std::runtime_error("failed to write config json");
  }
  json << "{\n";
  json << "  \"magic\": \"MICROGPT1\",\n";
  json << "  \"step\": " << step << ",\n";
  json << "  \"vocab_size\": " << model.cfg.vocab_size << ",\n";
  json << "  \"context_length\": " << model.cfg.context_length << ",\n";
  json << "  \"d_model\": " << model.cfg.d_model << ",\n";
  json << "  \"num_layers\": " << model.cfg.num_layers << ",\n";
  json << "  \"num_heads\": " << model.cfg.num_heads << ",\n";
  json << "  \"d_ff\": " << model.cfg.d_ff << ",\n";
  json << "  \"batch_size\": " << model.cfg.batch_size << ",\n";
  json << "  \"learning_rate\": " << model.cfg.learning_rate << ",\n";
  json << "  \"beta1\": " << model.cfg.beta1 << ",\n";
  json << "  \"beta2\": " << model.cfg.beta2 << ",\n";
  json << "  \"adam_eps\": " << model.cfg.adam_eps << ",\n";
  json << "  \"weight_decay\": " << model.cfg.weight_decay << ",\n";
  json << "  \"max_grad_norm\": " << model.cfg.max_grad_norm << "\n";
  json << "}\n";
}

inline Model load_checkpoint(const std::string& path, AdamW& opt, int& step) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open checkpoint for read: " + path);
  }
  char magic[10] = {};
  in.read(magic, 9);
  if (std::string(magic, 9) != "MICROGPT1") {
    throw std::runtime_error("invalid checkpoint magic");
  }
  step = read_i32(in);
  Config cfg;
  cfg.vocab_size = read_i32(in);
  cfg.context_length = read_i32(in);
  cfg.d_model = read_i32(in);
  cfg.num_layers = read_i32(in);
  cfg.num_heads = read_i32(in);
  cfg.d_ff = read_i32(in);
  cfg.batch_size = read_i32(in);
  cfg.learning_rate = read_f32(in);
  cfg.beta1 = read_f32(in);
  cfg.beta2 = read_f32(in);
  cfg.adam_eps = read_f32(in);
  cfg.weight_decay = read_f32(in);
  cfg.max_grad_norm = read_f32(in);
  opt.step = read_i32(in);
  Model model(cfg);
  uint64_t nparams = read_u64(in);
  auto params = model.parameters();
  if (params.size() != nparams) {
    throw std::runtime_error("checkpoint parameter count mismatch");
  }
  for (size_t i = 0; i < params.size(); ++i) {
    std::string name = read_string(in);
    std::vector<int> shape = read_shape(in);
    std::vector<float> data = read_vector_f32(in);
    std::vector<float> m = read_vector_f32(in);
    std::vector<float> v = read_vector_f32(in);
    uint64_t decay_flag = read_u64(in);
    Parameter* p = params[i];
    if (p->name != name) {
      throw std::runtime_error("checkpoint parameter name mismatch: " + p->name + " vs " + name);
    }
    if (p->shape != shape) {
      throw std::runtime_error("checkpoint parameter shape mismatch");
    }
    p->data = std::move(data);
    p->m = std::move(m);
    p->v = std::move(v);
    p->decay = decay_flag != 0;
    p->grad.assign(p->data.size(), 0.0f);
  }
  return model;
}

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

inline std::string generate_text(Model& model, const std::string& prompt, int max_new_tokens, float temperature, int top_k,
                                 int stop_token_id = Tokenizer::kEos,
                                 const std::function<void(const GenerationProgress&)>& on_progress = {}) {
  Tokenizer tok;
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
    int offset = model.cfg.context_length - static_cast<int>(window.size());
    for (size_t t = 0; t < window.size(); ++t) {
      flat[static_cast<size_t>(offset) + t] = window[t];
    }
    SeqTensor logits = model.forward(flat);
    const float* row = &logits.data[idx3(0, logits.T - 1, 0, logits.T, logits.D)];
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

inline bool gradient_check_linear() {
  RNG rng(123);
  Linear lin("gc", 3, 2, rng);
  SeqTensor x(1, 1, 3);
  x.data = {0.3f, -0.2f, 0.5f};
  SeqTensor y = lin.forward(x);
  y.grad.assign(y.data.size(), 0.0f);
  y.grad[0] = 1.0f;
  y.grad[1] = -0.5f;
  lin.backward(x, y);
  float eps = 1e-3f;
  size_t wi = 1;
  float orig = lin.w.data[wi];
  lin.w.data[wi] = orig + eps;
  SeqTensor yp = lin.forward(x);
  float lp = yp.data[0] * 1.0f + yp.data[1] * -0.5f;
  lin.w.data[wi] = orig - eps;
  SeqTensor ym = lin.forward(x);
  float lm = ym.data[0] * 1.0f + ym.data[1] * -0.5f;
  lin.w.data[wi] = orig;
  float numerical = (lp - lm) / (2.0f * eps);
  float analytical = lin.w.grad[wi];
  return std::fabs(numerical - analytical) < 1e-2f;
}

inline bool causal_mask_test() {
  RNG rng(321);
  Attention attn("mask", 8, 2, rng);
  SeqTensor x(1, 4, 8);
  for (size_t i = 0; i < x.data.size(); ++i) {
    x.data[i] = static_cast<float>(i + 1) * 0.01f;
  }
  SeqTensor y = attn.forward(x);
  (void)y;
  for (int h = 0; h < attn.num_heads; ++h) {
    for (int t = 0; t < x.T; ++t) {
      for (int s = t + 1; s < x.T; ++s) {
        size_t wi = (((static_cast<size_t>(0) * static_cast<size_t>(attn.num_heads) + static_cast<size_t>(h)) *
                          static_cast<size_t>(x.T) +
                      static_cast<size_t>(t)) *
                         static_cast<size_t>(x.T)) +
                     static_cast<size_t>(s);
        if (std::fabs(attn.weights[wi]) > 1e-6f) {
          return false;
        }
      }
    }
  }
  return true;
}

inline bool tiny_overfit_test() {
  Config cfg;
  cfg.batch_size = 4;
  cfg.context_length = 8;
  cfg.d_model = 32;
  cfg.num_layers = 1;
  cfg.num_heads = 4;
  cfg.d_ff = 64;
  cfg.learning_rate = 0.001f;
  cfg.eval_interval = 50;
  cfg.save_interval = 10000;
  cfg.seed = 7;
  Model model(cfg);
  AdamW opt;
  opt.lr = cfg.learning_rate;
  opt.beta1 = cfg.beta1;
  opt.beta2 = cfg.beta2;
  opt.eps = cfg.adam_eps;
  opt.weight_decay = cfg.weight_decay;
  std::vector<int> tokens;
  std::string text = "hello hello hello hello hello hello ";
  for (unsigned char c : text) {
    tokens.push_back(static_cast<int>(c));
  }
  std::vector<int> train = tokens;
  std::vector<int> val = tokens;
  float before = evaluate_loss(model, train, 2);
  for (int i = 0; i < 100; ++i) {
    model.zero_grad();
    Batch batch = sample_batch(train, cfg.batch_size, cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    cross_entropy_loss(logits, batch.y);
    model.backward(logits);
    auto params = model.parameters();
    clip_gradients(params, cfg.max_grad_norm);
    opt.update(params);
  }
  float after = evaluate_loss(model, val, 2);
  return after < before * 0.9f;
}

inline bool alternating_pattern_test() {
  Config cfg;
  cfg.batch_size = 4;
  cfg.context_length = 8;
  cfg.d_model = 32;
  cfg.num_layers = 1;
  cfg.num_heads = 4;
  cfg.d_ff = 64;
  cfg.learning_rate = 0.001f;
  cfg.seed = 11;
  Model model(cfg);
  AdamW opt;
  opt.lr = cfg.learning_rate;
  std::vector<int> tokens;
  std::string text = "abababababababababababab";
  for (unsigned char c : text) {
    tokens.push_back(static_cast<int>(c));
  }
  for (int i = 0; i < 80; ++i) {
    model.zero_grad();
    Batch batch = sample_batch(tokens, cfg.batch_size, cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    cross_entropy_loss(logits, batch.y);
    model.backward(logits);
    auto params = model.parameters();
    clip_gradients(params, cfg.max_grad_norm);
    opt.update(params);
  }
  std::vector<int> flat = {
      static_cast<int>('b'), static_cast<int>('a'), static_cast<int>('b'), static_cast<int>('a'),
      static_cast<int>('b'), static_cast<int>('a'), static_cast<int>('b'), static_cast<int>('a')};
  SeqTensor logits = model.forward(flat);
  const float* row = &logits.data[idx3(0, logits.T - 1, 0, logits.T, logits.D)];
  int next = sample_from_logits(std::vector<float>(row, row + logits.D), 0.8f, 1, model.rng);
  return next == static_cast<int>('b');
}

inline bool run_tests() {
  bool ok1 = gradient_check_linear();
  bool ok2 = causal_mask_test();
  bool ok3 = tiny_overfit_test();
  bool ok4 = alternating_pattern_test();
  std::cout << "gradient_check_linear: " << (ok1 ? "PASS" : "FAIL") << '\n';
  std::cout << "causal_mask_test: " << (ok2 ? "PASS" : "FAIL") << '\n';
  std::cout << "tiny_overfit_test: " << (ok3 ? "PASS" : "FAIL") << '\n';
  std::cout << "alternating_pattern_test: " << (ok4 ? "PASS" : "FAIL") << '\n';
  return ok1 && ok2 && ok3 && ok4;
}

}  // namespace microgpt

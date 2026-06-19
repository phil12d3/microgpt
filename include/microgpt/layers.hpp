#pragma once

#include "microgpt/backend_ops.hpp"
#include "microgpt/core.hpp"

#include <cmath>
#include <limits>
#include <memory>

namespace microgpt {
struct Linear {
  Parameter w;
  Parameter b;
  int in_features = 0;
  int out_features = 0;
  BackendKind backend = BackendKind::Cpu;
  mutable std::shared_ptr<LinearBackendCache> backend_cache;

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
    return linear_forward_op(backend, backend_cache.get(), x, w, b, in_features, out_features);
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    linear_backward_op(backend, backend_cache.get(), x, w, b, in_features, out_features, y);
  }

  void set_backend(BackendKind kind) {
    backend = kind;
    if (kind == BackendKind::Cpu) {
      backend_cache.reset();
    } else {
      if (!backend_cache) {
        backend_cache = std::make_shared<LinearBackendCache>(kind);
      } else {
        backend_cache->set_backend(kind);
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
  BackendKind backend = BackendKind::Cpu;

  LayerNorm() = default;
  LayerNorm(const std::string& prefix, int d) : gamma(prefix + ".gamma", {d}, false), beta(prefix + ".beta", {d}, false), dim(d) {
    fill_ones(gamma.data);
    fill_zeros(beta.data);
  }

  SeqTensor forward(const SeqTensor& x) {
    if (x.D != dim) {
      throw std::runtime_error("layernorm forward: feature mismatch");
    }
    LayerNormForwardResult result = layernorm_forward_op(backend, x, gamma, beta, eps);
    mean = std::move(result.mean);
    inv_std = std::move(result.inv_std);
    xhat = std::move(result.xhat);
    return result.y;
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    layernorm_backward_op(backend, x, gamma, beta, y, mean, inv_std, xhat);
  }

  void set_backend(BackendKind kind) { backend = kind; }
};

struct FeedForward {
  Linear fc1;
  Linear fc2;
  SeqTensor pre_activation;
  SeqTensor hidden;
  BackendKind backend = BackendKind::Cpu;
  struct BackendCache {
    BackendBuffer pre;
    BackendBuffer hidden;
    explicit BackendCache(BackendKind kind) : pre(kind), hidden(kind) {}
    void set_backend(BackendKind kind) {
      pre = BackendBuffer(kind);
      hidden = BackendBuffer(kind);
    }
  };
  mutable std::shared_ptr<BackendCache> backend_cache;

  FeedForward() = default;
  FeedForward(const std::string& prefix, int d_model, int d_ff, RNG& rng)
      : fc1(prefix + ".fc1", d_model, d_ff, rng), fc2(prefix + ".fc2", d_ff, d_model, rng) {}

  void set_backend(BackendKind kind) {
    backend = kind;
    fc1.set_backend(kind);
    fc2.set_backend(kind);
    if (kind == BackendKind::Cpu) {
      backend_cache.reset();
    } else {
      if (!backend_cache) {
        backend_cache = std::make_shared<BackendCache>(kind);
      } else {
        backend_cache->set_backend(kind);
      }
    }
  }

  SeqTensor forward(const SeqTensor& x) {
    if (backend == BackendKind::Metal && microgpt_metal_runtime_available() && backend_cache) {
      pre_activation = SeqTensor(x.B, x.T, fc1.out_features);
      hidden = SeqTensor(x.B, x.T, fc1.out_features);
      backend_cache->pre.ensure_size(pre_activation.data.size());
      backend_cache->hidden.ensure_size(hidden.data.size());
      SeqTensor y(x.B, x.T, fc2.out_features);
      bool ok = microgpt_metal_feedforward_forward(
          x.data.data(), fc1.w.data.data(), fc1.b.data.empty() ? nullptr : fc1.b.data.data(), fc2.w.data.data(),
          fc2.b.data.empty() ? nullptr : fc2.b.data.data(), backend_cache->pre.device_contents(),
          backend_cache->hidden.device_contents(),
          y.data.data(), x.B * x.T, fc1.in_features, fc1.out_features, !fc1.b.data.empty(), !fc2.b.data.empty());
      if (ok) {
        backend_cache->pre.device_dirty = true;
        backend_cache->hidden.device_dirty = true;
        backend_cache->pre.download();
        backend_cache->hidden.download();
        pre_activation.data = backend_cache->pre.host;
        hidden.data = backend_cache->hidden.host;
        return y;
      }
    }
    pre_activation = fc1.forward(x);
    hidden = pre_activation;
    hidden.data = gelu_forward_op(fc1.backend, pre_activation.data);
    return fc2.forward(hidden);
  }

  void backward(SeqTensor& x, const SeqTensor& y) {
    if (backend == BackendKind::Metal && microgpt_metal_runtime_available() && backend_cache) {
      if (x.grad.size() != x.data.size()) {
        x.grad.assign(x.data.size(), 0.0f);
      }
      std::vector<float> dx(x.data.size(), 0.0f);
      std::vector<float> dw1(fc1.w.data.size(), 0.0f);
      std::vector<float> db1(fc1.b.data.size(), 0.0f);
      std::vector<float> dw2(fc2.w.data.size(), 0.0f);
      std::vector<float> db2(fc2.b.data.size(), 0.0f);
      bool ok = microgpt_metal_feedforward_backward(
          x.data.data(), backend_cache->pre.device_contents(), backend_cache->hidden.device_contents(),
          fc1.w.data.data(), fc2.w.data.data(),
          y.grad.data(), dx.data(), dw1.data(), fc1.b.data.empty() ? nullptr : db1.data(), dw2.data(),
          fc2.b.data.empty() ? nullptr : db2.data(), x.B * x.T, fc1.in_features, fc1.out_features,
          !fc1.b.data.empty(), !fc2.b.data.empty());
      if (ok) {
        for (size_t i = 0; i < x.grad.size(); ++i) {
          x.grad[i] += dx[i];
        }
        for (size_t i = 0; i < fc1.w.grad.size(); ++i) {
          fc1.w.grad[i] += dw1[i];
        }
        for (size_t i = 0; i < fc1.b.grad.size(); ++i) {
          fc1.b.grad[i] += db1[i];
        }
        for (size_t i = 0; i < fc2.w.grad.size(); ++i) {
          fc2.w.grad[i] += dw2[i];
        }
        for (size_t i = 0; i < fc2.b.grad.size(); ++i) {
          fc2.b.grad[i] += db2[i];
        }
        return;
      }
    }
    fc2.backward(hidden, y);
    pre_activation.grad = gelu_backward_op(fc1.backend, pre_activation.data, hidden.grad);
    fc1.backward(x, pre_activation);
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

  void set_backend(BackendKind kind) {
    q_proj.set_backend(kind);
    k_proj.set_backend(kind);
    v_proj.set_backend(kind);
    o_proj.set_backend(kind);
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

  void set_backend(BackendKind kind) {
    ln1.set_backend(kind);
    attn.set_backend(kind);
    ln2.set_backend(kind);
    ff.set_backend(kind);
  }

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

}  // namespace microgpt

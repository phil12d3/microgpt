#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace microgpt {

struct BackendTransferStats {
  size_t uploads = 0;
  size_t upload_bytes = 0;
  size_t downloads = 0;
  size_t download_bytes = 0;
};

inline BackendTransferStats& backend_transfer_stats() {
  static BackendTransferStats stats;
  return stats;
}

inline void reset_backend_transfer_stats() {
  backend_transfer_stats() = BackendTransferStats{};
}

struct BackendBuffer {
  BackendKind backend = BackendKind::Cpu;
  std::vector<float> host;
  void* device = nullptr;
  bool host_dirty = false;
  bool device_dirty = false;

  BackendBuffer() = default;
  explicit BackendBuffer(BackendKind kind) : backend(kind) {}
  BackendBuffer(const BackendBuffer&) = delete;
  BackendBuffer& operator=(const BackendBuffer&) = delete;

  BackendBuffer(BackendBuffer&& other) noexcept
      : backend(other.backend),
        host(std::move(other.host)),
        device(other.device),
        host_dirty(other.host_dirty),
        device_dirty(other.device_dirty) {
    other.device = nullptr;
  }

  BackendBuffer& operator=(BackendBuffer&& other) noexcept {
    if (this != &other) {
      release_device();
      backend = other.backend;
      host = std::move(other.host);
      device = other.device;
      host_dirty = other.host_dirty;
      device_dirty = other.device_dirty;
      other.device = nullptr;
    }
    return *this;
  }

  ~BackendBuffer() { release_device(); }

  void release_device() {
    if (device != nullptr && backend == BackendKind::Metal) {
      microgpt_metal_buffer_destroy(device);
    }
    device = nullptr;
  }

  void resize(size_t n) {
    host.assign(n, 0.0f);
    release_device();
    if (backend == BackendKind::Metal && n > 0 && microgpt_metal_runtime_available()) {
      device = microgpt_metal_buffer_create(n * sizeof(float));
      if (device == nullptr) {
        throw std::runtime_error("failed to allocate Metal backend buffer");
      }
    }
    host_dirty = true;
    device_dirty = false;
  }

  size_t size() const { return host.size(); }
  void ensure_size(size_t n) {
    if (host.size() != n) {
      resize(n);
    }
  }
  float* data() { return host.data(); }
  const float* data() const { return host.data(); }

  void upload() {
    require_backend_available(backend);
    if (backend == BackendKind::Metal && device != nullptr && host_dirty) {
      if (!microgpt_metal_buffer_write(device, host.data(), host.size() * sizeof(float))) {
        throw std::runtime_error("failed to upload Metal backend buffer");
      }
      BackendTransferStats& stats = backend_transfer_stats();
      stats.uploads += 1;
      stats.upload_bytes += host.size() * sizeof(float);
    }
    host_dirty = false;
  }

  void download() {
    require_backend_available(backend);
    if (backend == BackendKind::Metal && device != nullptr && device_dirty) {
      if (!microgpt_metal_buffer_read(device, host.data(), host.size() * sizeof(float))) {
        throw std::runtime_error("failed to download Metal backend buffer");
      }
      BackendTransferStats& stats = backend_transfer_stats();
      stats.downloads += 1;
      stats.download_bytes += host.size() * sizeof(float);
    }
    device_dirty = false;
  }

  bool has_device() const { return device != nullptr; }

  float* device_contents() {
    if (backend == BackendKind::Metal && device != nullptr) {
      return static_cast<float*>(microgpt_metal_buffer_contents(device));
    }
    return host.data();
  }

  const float* device_contents() const {
    if (backend == BackendKind::Metal && device != nullptr) {
      return static_cast<const float*>(microgpt_metal_buffer_contents(device));
    }
    return host.data();
  }
};

struct LinearBackendCache {
  BackendKind backend = BackendKind::Cpu;
  BackendBuffer x;
  BackendBuffer w;
  BackendBuffer b;
  BackendBuffer y;
  BackendBuffer dy;
  BackendBuffer dx;
  BackendBuffer dw;
  BackendBuffer db;
  int rows = 0;
  int in_features = 0;
  int out_features = 0;
  bool has_bias = false;
  bool x_device_current = false;
  bool w_device_current = false;
  bool b_device_current = false;

  explicit LinearBackendCache(BackendKind kind = BackendKind::Cpu)
      : backend(kind), x(kind), w(kind), b(kind), y(kind), dy(kind), dx(kind), dw(kind), db(kind) {}

  void set_backend(BackendKind kind) {
    if (backend == kind) {
      return;
    }
    backend = kind;
    x = BackendBuffer(kind);
    w = BackendBuffer(kind);
    b = BackendBuffer(kind);
    y = BackendBuffer(kind);
    dy = BackendBuffer(kind);
    dx = BackendBuffer(kind);
    dw = BackendBuffer(kind);
    db = BackendBuffer(kind);
    rows = 0;
    in_features = 0;
    out_features = 0;
    has_bias = false;
    x_device_current = false;
    w_device_current = false;
    b_device_current = false;
  }

  bool usable_for_metal() const {
    return backend == BackendKind::Metal && x.has_device() && w.has_device() && y.has_device();
  }
};

inline void matmul_forward_cpu(const float* a, const float* b, float* c, int rows, int inner, int cols) {
  if (a == nullptr || b == nullptr || c == nullptr || rows <= 0 || inner <= 0 || cols <= 0) {
    throw std::runtime_error("matmul forward: invalid input");
  }
  for (int r = 0; r < rows; ++r) {
    for (int col = 0; col < cols; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < inner; ++k) {
        sum += a[static_cast<size_t>(r) * static_cast<size_t>(inner) + static_cast<size_t>(k)] *
               b[static_cast<size_t>(k) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
      }
      c[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(col)] = sum;
    }
  }
}

inline std::vector<float> matmul_forward_op(BackendKind backend, const float* a, const float* b, int rows, int inner,
                                            int cols) {
  require_backend_available(backend);
  if (a == nullptr || b == nullptr || rows <= 0 || inner <= 0 || cols <= 0) {
    throw std::runtime_error("matmul forward: invalid input");
  }
  std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  matmul_forward_cpu(a, b, out.data(), rows, inner, cols);
  return out;
}

inline float gelu_cpu(float x) {
  const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
  float x3 = x * x * x;
  float t = std::tanh(c * (x + 0.044715f * x3));
  return 0.5f * x * (1.0f + t);
}

inline float gelu_derivative_cpu(float x) {
  const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
  float x2 = x * x;
  float x3 = x2 * x;
  float u = c * (x + 0.044715f * x3);
  float t = std::tanh(u);
  float sech2 = 1.0f - t * t;
  float term = c * (1.0f + 3.0f * 0.044715f * x2);
  return 0.5f * (1.0f + t) + 0.5f * x * sech2 * term;
}

inline void gelu_forward_cpu(const std::vector<float>& x, std::vector<float>& y) {
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y[i] = gelu_cpu(x[i]);
  }
}

inline void gelu_backward_cpu(const std::vector<float>& x, const std::vector<float>& dy, std::vector<float>& dx) {
  if (x.size() != dy.size()) {
    throw std::runtime_error("gelu backward: size mismatch");
  }
  dx.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    dx[i] = dy[i] * gelu_derivative_cpu(x[i]);
  }
}

inline std::vector<float> gelu_forward_op(BackendKind backend, const std::vector<float>& x) {
  require_backend_available(backend);
  std::vector<float> y(x.size(), 0.0f);
  if (backend == BackendKind::Metal && !x.empty() &&
      microgpt_metal_gelu_forward(x.data(), y.data(), static_cast<int>(x.size()))) {
    return y;
  }
  gelu_forward_cpu(x, y);
  return y;
}

inline std::vector<float> gelu_backward_op(BackendKind backend, const std::vector<float>& x,
                                           const std::vector<float>& dy) {
  require_backend_available(backend);
  if (x.size() != dy.size()) {
    throw std::runtime_error("gelu backward: size mismatch");
  }
  std::vector<float> dx(x.size(), 0.0f);
  if (backend == BackendKind::Metal && !x.empty() &&
      microgpt_metal_gelu_backward(x.data(), dy.data(), dx.data(), static_cast<int>(x.size()))) {
    return dx;
  }
  gelu_backward_cpu(x, dy, dx);
  return dx;
}

struct LayerNormForwardResult {
  SeqTensor y;
  std::vector<float> mean;
  std::vector<float> inv_std;
  std::vector<float> xhat;
};

inline void layernorm_forward_cpu(const SeqTensor& x, const Parameter& gamma, const Parameter& beta, float eps,
                                  LayerNormForwardResult& out) {
  out.y = SeqTensor(x.B, x.T, x.D);
  out.mean.assign(static_cast<size_t>(x.B) * static_cast<size_t>(x.T), 0.0f);
  out.inv_std.assign(out.mean.size(), 0.0f);
  out.xhat.assign(x.data.size(), 0.0f);
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
      out.mean[row] = mu;
      out.inv_std[row] = inv;
      float* yout = &out.y.data[idx3(b, t, 0, out.y.T, out.y.D)];
      for (int i = 0; i < x.D; ++i) {
        float xn = (xin[static_cast<size_t>(i)] - mu) * inv;
        out.xhat[idx3(b, t, i, x.T, x.D)] = xn;
        yout[static_cast<size_t>(i)] = xn * gamma.data[static_cast<size_t>(i)] + beta.data[static_cast<size_t>(i)];
      }
    }
  }
}

inline LayerNormForwardResult layernorm_forward_op(BackendKind backend, const SeqTensor& x, const Parameter& gamma,
                                                   const Parameter& beta, float eps) {
  require_backend_available(backend);
  LayerNormForwardResult out;
  out.y = SeqTensor(x.B, x.T, x.D);
  out.mean.assign(static_cast<size_t>(x.B) * static_cast<size_t>(x.T), 0.0f);
  out.inv_std.assign(out.mean.size(), 0.0f);
  out.xhat.assign(x.data.size(), 0.0f);
  if (backend == BackendKind::Metal && !x.data.empty() &&
      microgpt_metal_layernorm_forward(x.data.data(), gamma.data.data(), beta.data.data(), out.y.data.data(),
                                       out.mean.data(), out.inv_std.data(), out.xhat.data(), x.B * x.T, x.D, eps)) {
    return out;
  }
  layernorm_forward_cpu(x, gamma, beta, eps, out);
  return out;
}

inline void layernorm_backward_cpu(SeqTensor& x, Parameter& gamma, Parameter& beta, const SeqTensor& y,
                                   const std::vector<float>& mean, const std::vector<float>& inv_std,
                                   const std::vector<float>& xhat) {
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

inline void layernorm_backward_op(BackendKind backend, SeqTensor& x, Parameter& gamma, Parameter& beta,
                                  const SeqTensor& y, const std::vector<float>& mean,
                                  const std::vector<float>& inv_std, const std::vector<float>& xhat) {
  require_backend_available(backend);
  if (x.grad.size() != x.data.size()) {
    x.grad.assign(x.data.size(), 0.0f);
  }
  if (backend == BackendKind::Metal && !x.data.empty()) {
    std::vector<float> dx(x.data.size(), 0.0f);
    std::vector<float> dgamma(gamma.data.size(), 0.0f);
    std::vector<float> dbeta(beta.data.size(), 0.0f);
    if (microgpt_metal_layernorm_backward(x.data.data(), gamma.data.data(), y.grad.data(), mean.data(), inv_std.data(),
                                          xhat.data(), dx.data(), dgamma.data(), dbeta.data(), x.B * x.T, x.D)) {
      for (size_t i = 0; i < x.grad.size(); ++i) {
        x.grad[i] += dx[i];
      }
      for (size_t i = 0; i < gamma.grad.size(); ++i) {
        gamma.grad[i] += dgamma[i];
      }
      for (size_t i = 0; i < beta.grad.size(); ++i) {
        beta.grad[i] += dbeta[i];
      }
      return;
    }
  }
  layernorm_backward_cpu(x, gamma, beta, y, mean, inv_std, xhat);
}

inline void linear_forward_cpu(const SeqTensor& x, const Parameter& w, const Parameter& b, int in_features,
                               int out_features, SeqTensor& y) {
  int rows = x.B * x.T;
  matmul_forward_cpu(x.data.data(), w.data.data(), y.data.data(), rows, in_features, out_features);
  if (!b.data.empty()) {
    for (int r = 0; r < rows; ++r) {
      for (int o = 0; o < out_features; ++o) {
        y.data[static_cast<size_t>(r) * static_cast<size_t>(out_features) + static_cast<size_t>(o)] +=
            b.data[static_cast<size_t>(o)];
      }
    }
  }
}

inline void linear_backward_cpu(SeqTensor& x, Parameter& w, Parameter& b, int in_features, int out_features,
                                const SeqTensor& y) {
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

inline SeqTensor linear_forward_op(BackendKind backend, const SeqTensor& x, const Parameter& w, const Parameter& b,
                                   int in_features, int out_features) {
  require_backend_available(backend);
  if (x.D != in_features) {
    throw std::runtime_error("linear forward: input feature mismatch");
  }
  SeqTensor y(x.B, x.T, out_features);
  if (backend == BackendKind::Metal &&
      microgpt_metal_linear_forward(x.data.data(), w.data.data(), b.data.empty() ? nullptr : b.data.data(),
                                    y.data.data(), x.B * x.T, in_features, out_features, !b.data.empty())) {
    return y;
  }
  linear_forward_cpu(x, w, b, in_features, out_features, y);
  return y;
}

inline SeqTensor linear_forward_op(BackendKind backend, LinearBackendCache* cache, const SeqTensor& x,
                                   const Parameter& w, const Parameter& b, int in_features, int out_features) {
  require_backend_available(backend);
  if (x.D != in_features) {
    throw std::runtime_error("linear forward: input feature mismatch");
  }
  SeqTensor y(x.B, x.T, out_features);
  int rows = x.B * x.T;
  if (backend == BackendKind::Metal && cache != nullptr && microgpt_metal_runtime_available()) {
    cache->set_backend(backend);
    cache->rows = rows;
    cache->in_features = in_features;
    cache->out_features = out_features;
    cache->has_bias = !b.data.empty();
    cache->x.ensure_size(x.data.size());
    cache->w.ensure_size(w.data.size());
    cache->y.ensure_size(y.data.size());
    if (!b.data.empty()) {
      cache->b.ensure_size(b.data.size());
    }
    cache->x.host = x.data;
    cache->w.host = w.data;
    if (!b.data.empty()) {
      cache->b.host = b.data;
    }
    cache->x.host_dirty = true;
    cache->w.host_dirty = true;
    cache->b.host_dirty = !b.data.empty();
    cache->x.upload();
    cache->w.upload();
    if (!b.data.empty()) {
      cache->b.upload();
    }
    cache->x_device_current = true;
    cache->w_device_current = true;
    cache->b_device_current = !b.data.empty();
    if (cache->usable_for_metal() &&
        microgpt_metal_linear_forward_buffers(cache->x.device, cache->w.device,
                                              b.data.empty() ? nullptr : cache->b.device, cache->y.device, rows,
                                              in_features, out_features, !b.data.empty())) {
      cache->y.device_dirty = true;
      cache->y.download();
      y.data = cache->y.host;
      return y;
    }
  }
  linear_forward_cpu(x, w, b, in_features, out_features, y);
  return y;
}

inline void linear_backward_op(BackendKind backend, SeqTensor& x, Parameter& w, Parameter& b, int in_features,
                               int out_features, const SeqTensor& y) {
  require_backend_available(backend);
  if (x.grad.size() != x.data.size()) {
    x.grad.assign(x.data.size(), 0.0f);
  }
  if (backend == BackendKind::Metal) {
    std::vector<float> dx(x.data.size(), 0.0f);
    std::vector<float> dw(w.data.size(), 0.0f);
    std::vector<float> db(b.data.size(), 0.0f);
    if (microgpt_metal_linear_backward(x.data.data(), w.data.data(), y.grad.data(), dx.data(), dw.data(),
                                       b.data.empty() ? nullptr : db.data(), x.B * x.T, in_features, out_features,
                                       !b.data.empty())) {
      for (size_t i = 0; i < x.grad.size(); ++i) {
        x.grad[i] += dx[i];
      }
      for (size_t i = 0; i < w.grad.size(); ++i) {
        w.grad[i] += dw[i];
      }
      for (size_t i = 0; i < b.grad.size(); ++i) {
        b.grad[i] += db[i];
      }
      return;
    }
  }
  linear_backward_cpu(x, w, b, in_features, out_features, y);
}

inline void linear_backward_op(BackendKind backend, LinearBackendCache* cache, SeqTensor& x, Parameter& w, Parameter& b,
                               int in_features, int out_features, const SeqTensor& y) {
  require_backend_available(backend);
  if (x.grad.size() != x.data.size()) {
    x.grad.assign(x.data.size(), 0.0f);
  }
  int rows = x.B * x.T;
  if (backend == BackendKind::Metal && cache != nullptr && microgpt_metal_runtime_available()) {
    cache->set_backend(backend);
    size_t old_x_size = cache->x.size();
    size_t old_w_size = cache->w.size();
    cache->x.ensure_size(x.data.size());
    cache->w.ensure_size(w.data.size());
    cache->dy.ensure_size(y.grad.size());
    cache->dx.ensure_size(x.data.size());
    cache->dw.ensure_size(w.data.size());
    if (cache->x.size() != old_x_size) {
      cache->x_device_current = false;
    }
    if (cache->w.size() != old_w_size) {
      cache->w_device_current = false;
    }
    if (!b.data.empty()) {
      cache->db.ensure_size(b.data.size());
    }
    cache->dy.host = y.grad;
    if (!cache->x_device_current) {
      cache->x.host = x.data;
      cache->x.host_dirty = true;
    }
    if (!cache->w_device_current) {
      cache->w.host = w.data;
      cache->w.host_dirty = true;
    }
    cache->dy.host_dirty = true;
    if (!cache->x_device_current) {
      cache->x.upload();
      cache->x_device_current = true;
    }
    if (!cache->w_device_current) {
      cache->w.upload();
      cache->w_device_current = true;
    }
    cache->dy.upload();
    bool ok = microgpt_metal_linear_backward_buffers(cache->x.device, cache->w.device, cache->dy.device,
                                                     cache->dx.device, cache->dw.device,
                                                     b.data.empty() ? nullptr : cache->db.device, rows, in_features,
                                                     out_features, !b.data.empty());
    if (ok) {
      cache->dx.device_dirty = true;
      cache->dw.device_dirty = true;
      cache->dx.download();
      cache->dw.download();
      for (size_t i = 0; i < x.grad.size(); ++i) {
        x.grad[i] += cache->dx.host[i];
      }
      for (size_t i = 0; i < w.grad.size(); ++i) {
        w.grad[i] += cache->dw.host[i];
      }
      if (!b.data.empty()) {
        cache->db.device_dirty = true;
        cache->db.download();
        for (size_t i = 0; i < b.grad.size(); ++i) {
          b.grad[i] += cache->db.host[i];
        }
      }
      return;
    }
  }
  linear_backward_op(backend, x, w, b, in_features, out_features, y);
}

}  // namespace microgpt

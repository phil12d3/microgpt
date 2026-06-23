#pragma once

#include "microgpt/data_format.hpp"
#include "microgpt/chat.hpp"
#include "microgpt/generation.hpp"
#include "microgpt/io.hpp"
#include "microgpt/optim.hpp"
#include "microgpt/training.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace microgpt {
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

inline bool backend_linear_op_test() {
  Parameter w("backend_linear.w", {3, 2}, true);
  Parameter b("backend_linear.b", {2}, false);
  w.data = {0.2f, -0.1f, 0.4f, 0.3f, -0.5f, 0.7f};
  b.data = {0.05f, -0.02f};
  w.grad.assign(w.data.size(), 0.0f);
  b.grad.assign(b.data.size(), 0.0f);
  SeqTensor x(2, 1, 3);
  x.data = {1.0f, 2.0f, -1.0f, -0.5f, 0.25f, 3.0f};
  SeqTensor y = linear_forward_op(BackendKind::Cpu, x, w, b, 3, 2);
  std::vector<float> expected = {
      1.0f * 0.2f + 2.0f * 0.4f + -1.0f * -0.5f + 0.05f,
      1.0f * -0.1f + 2.0f * 0.3f + -1.0f * 0.7f - 0.02f,
      -0.5f * 0.2f + 0.25f * 0.4f + 3.0f * -0.5f + 0.05f,
      -0.5f * -0.1f + 0.25f * 0.3f + 3.0f * 0.7f - 0.02f,
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(y.data[i] - expected[i]) > 1e-6f) {
      return false;
    }
  }
  y.grad = {0.1f, -0.2f, 0.3f, 0.4f};
  linear_backward_op(BackendKind::Cpu, x, w, b, 3, 2, y);
  std::vector<float> expected_bg = {0.4f, 0.2f};
  std::vector<float> expected_wg = {
      1.0f * 0.1f + -0.5f * 0.3f,
      1.0f * -0.2f + -0.5f * 0.4f,
      2.0f * 0.1f + 0.25f * 0.3f,
      2.0f * -0.2f + 0.25f * 0.4f,
      -1.0f * 0.1f + 3.0f * 0.3f,
      -1.0f * -0.2f + 3.0f * 0.4f,
  };
  std::vector<float> expected_xg = {
      0.1f * 0.2f + -0.2f * -0.1f,
      0.1f * 0.4f + -0.2f * 0.3f,
      0.1f * -0.5f + -0.2f * 0.7f,
      0.3f * 0.2f + 0.4f * -0.1f,
      0.3f * 0.4f + 0.4f * 0.3f,
      0.3f * -0.5f + 0.4f * 0.7f,
  };
  for (size_t i = 0; i < expected_bg.size(); ++i) {
    if (std::fabs(b.grad[i] - expected_bg[i]) > 1e-6f) {
      return false;
    }
  }
  for (size_t i = 0; i < expected_wg.size(); ++i) {
    if (std::fabs(w.grad[i] - expected_wg[i]) > 1e-6f) {
      return false;
    }
  }
  for (size_t i = 0; i < expected_xg.size(); ++i) {
    if (std::fabs(x.grad[i] - expected_xg[i]) > 1e-6f) {
      return false;
    }
  }
  return true;
}

inline bool backend_matmul_op_test() {
  std::vector<float> a = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.5f, 4.0f,
  };
  std::vector<float> b = {
      0.25f, -0.5f,
      1.0f, 2.0f,
      -1.5f, 0.75f,
  };
  std::vector<float> actual = matmul_forward_op(BackendKind::Cpu, a.data(), b.data(), 2, 3, 2);
  std::vector<float> expected = {
      1.0f * 0.25f + 2.0f * 1.0f + 3.0f * -1.5f,
      1.0f * -0.5f + 2.0f * 2.0f + 3.0f * 0.75f,
      -1.0f * 0.25f + 0.5f * 1.0f + 4.0f * -1.5f,
      -1.0f * -0.5f + 0.5f * 2.0f + 4.0f * 0.75f,
  };
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
      return false;
    }
  }
  return true;
}

inline bool backend_gelu_op_test() {
  std::vector<float> x = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f};
  std::vector<float> dy = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
  std::vector<float> y_cpu = gelu_forward_op(BackendKind::Cpu, x);
  std::vector<float> dx_cpu = gelu_backward_op(BackendKind::Cpu, x, dy);
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  std::vector<float> y_metal = gelu_forward_op(BackendKind::Metal, x);
  std::vector<float> dx_metal = gelu_backward_op(BackendKind::Metal, x, dy);
  for (size_t i = 0; i < x.size(); ++i) {
    if (std::fabs(y_cpu[i] - y_metal[i]) > 1e-5f) {
      return false;
    }
    if (std::fabs(dx_cpu[i] - dx_metal[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool backend_layernorm_op_test() {
  Parameter gamma_cpu("backend_layernorm.gamma", {4}, false);
  Parameter beta_cpu("backend_layernorm.beta", {4}, false);
  gamma_cpu.data = {1.0f, 0.5f, -1.2f, 2.0f};
  beta_cpu.data = {0.05f, -0.1f, 0.2f, 0.3f};
  gamma_cpu.grad.assign(gamma_cpu.data.size(), 0.0f);
  beta_cpu.grad.assign(beta_cpu.data.size(), 0.0f);
  Parameter gamma_metal = gamma_cpu;
  Parameter beta_metal = beta_cpu;
  SeqTensor x_cpu(2, 1, 4);
  x_cpu.data = {0.2f, -1.0f, 0.7f, 1.5f, -0.3f, 0.8f, 1.2f, -1.1f};
  SeqTensor x_metal = x_cpu;
  LayerNormForwardResult out_cpu = layernorm_forward_op(BackendKind::Cpu, x_cpu, gamma_cpu, beta_cpu, 1e-5f);
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  LayerNormForwardResult out_metal = layernorm_forward_op(BackendKind::Metal, x_metal, gamma_metal, beta_metal, 1e-5f);
  for (size_t i = 0; i < out_cpu.y.data.size(); ++i) {
    if (std::fabs(out_cpu.y.data[i] - out_metal.y.data[i]) > 1e-5f ||
        std::fabs(out_cpu.xhat[i] - out_metal.xhat[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < out_cpu.mean.size(); ++i) {
    if (std::fabs(out_cpu.mean[i] - out_metal.mean[i]) > 1e-5f ||
        std::fabs(out_cpu.inv_std[i] - out_metal.inv_std[i]) > 1e-5f) {
      return false;
    }
  }
  out_cpu.y.grad = {0.1f, -0.2f, 0.3f, -0.4f, 0.25f, 0.05f, -0.15f, 0.35f};
  out_metal.y.grad = out_cpu.y.grad;
  layernorm_backward_op(BackendKind::Cpu, x_cpu, gamma_cpu, beta_cpu, out_cpu.y, out_cpu.mean, out_cpu.inv_std,
                        out_cpu.xhat);
  layernorm_backward_op(BackendKind::Metal, x_metal, gamma_metal, beta_metal, out_metal.y, out_metal.mean,
                        out_metal.inv_std, out_metal.xhat);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < gamma_cpu.grad.size(); ++i) {
    if (std::fabs(gamma_cpu.grad[i] - gamma_metal.grad[i]) > 1e-5f ||
        std::fabs(beta_cpu.grad[i] - beta_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool backend_feedforward_forward_test() {
  RNG rng(222);
  FeedForward cpu("backend_ff", 3, 5, rng);
  FeedForward metal = cpu;
  metal.set_backend(BackendKind::Metal);
  SeqTensor x(2, 1, 3);
  x.data = {0.2f, -0.4f, 0.7f, -1.0f, 0.5f, 0.25f};
  SeqTensor y_cpu = cpu.forward(x);
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  SeqTensor y_metal = metal.forward(x);
  for (size_t i = 0; i < y_cpu.data.size(); ++i) {
    if (std::fabs(y_cpu.data[i] - y_metal.data[i]) > 1e-5f) {
      return false;
    }
  }
  if (!metal.backend_cache || !metal.backend_cache->pre.has_device() || !metal.backend_cache->hidden.has_device()) {
    return false;
  }
  for (size_t i = 0; i < cpu.pre_activation.data.size(); ++i) {
    if (std::fabs(cpu.pre_activation.data[i] - metal.pre_activation.data[i]) > 1e-5f ||
        std::fabs(cpu.hidden.data[i] - metal.hidden.data[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool backend_feedforward_backward_test() {
  RNG rng(333);
  FeedForward cpu("backend_ff_backward", 3, 5, rng);
  FeedForward metal = cpu;
  metal.set_backend(BackendKind::Metal);
  SeqTensor x_cpu(2, 1, 3);
  x_cpu.data = {0.2f, -0.4f, 0.7f, -1.0f, 0.5f, 0.25f};
  SeqTensor x_metal = x_cpu;
  SeqTensor y_cpu = cpu.forward(x_cpu);
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  SeqTensor y_metal = metal.forward(x_metal);
  y_cpu.grad = {0.1f, -0.2f, 0.3f, -0.15f, 0.05f, 0.25f};
  y_metal.grad = y_cpu.grad;
  cpu.backward(x_cpu, y_cpu);
  metal.backward(x_metal, y_metal);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc1.w.grad.size(); ++i) {
    if (std::fabs(cpu.fc1.w.grad[i] - metal.fc1.w.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc1.b.grad.size(); ++i) {
    if (std::fabs(cpu.fc1.b.grad[i] - metal.fc1.b.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc2.w.grad.size(); ++i) {
    if (std::fabs(cpu.fc2.w.grad[i] - metal.fc2.w.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc2.b.grad.size(); ++i) {
    if (std::fabs(cpu.fc2.b.grad[i] - metal.fc2.b.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool cuda_feedforward_forward_test() {
  if (!microgpt_cuda_runtime_available()) {
    return true;
  }
  RNG rng(224);
  FeedForward cpu("backend_cuda_ff", 3, 5, rng);
  FeedForward cuda = cpu;
  cuda.set_backend(BackendKind::Cuda);
  SeqTensor x(2, 1, 3);
  x.data = {0.2f, -0.4f, 0.7f, -1.0f, 0.5f, 0.25f};
  SeqTensor y_cpu = cpu.forward(x);
  SeqTensor y_cuda = cuda.forward(x);
  for (size_t i = 0; i < y_cpu.data.size(); ++i) {
    if (std::fabs(y_cpu.data[i] - y_cuda.data[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.pre_activation.data.size(); ++i) {
    if (std::fabs(cpu.pre_activation.data[i] - cuda.pre_activation.data[i]) > 1e-5f ||
        std::fabs(cpu.hidden.data[i] - cuda.hidden.data[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool cuda_feedforward_backward_test() {
  if (!microgpt_cuda_runtime_available()) {
    return true;
  }
  RNG rng(334);
  FeedForward cpu("backend_cuda_ff_backward", 3, 5, rng);
  FeedForward cuda = cpu;
  cuda.set_backend(BackendKind::Cuda);
  SeqTensor x_cpu(2, 1, 3);
  x_cpu.data = {0.2f, -0.4f, 0.7f, -1.0f, 0.5f, 0.25f};
  SeqTensor x_cuda = x_cpu;
  SeqTensor y_cpu = cpu.forward(x_cpu);
  SeqTensor y_cuda = cuda.forward(x_cuda);
  y_cpu.grad = {0.1f, -0.2f, 0.3f, -0.15f, 0.05f, 0.25f};
  y_cuda.grad = y_cpu.grad;
  cpu.backward(x_cpu, y_cpu);
  cuda.backward(x_cuda, y_cuda);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_cuda.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc1.w.grad.size(); ++i) {
    if (std::fabs(cpu.fc1.w.grad[i] - cuda.fc1.w.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc1.b.grad.size(); ++i) {
    if (std::fabs(cpu.fc1.b.grad[i] - cuda.fc1.b.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc2.w.grad.size(); ++i) {
    if (std::fabs(cpu.fc2.w.grad[i] - cuda.fc2.w.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < cpu.fc2.b.grad.size(); ++i) {
    if (std::fabs(cpu.fc2.b.grad[i] - cuda.fc2.b.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool backend_buffer_roundtrip_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  BackendBuffer buffer(BackendKind::Metal);
  buffer.resize(4);
  buffer.host = {1.25f, -2.0f, 3.5f, 0.75f};
  buffer.host_dirty = true;
  buffer.upload();
  buffer.host.assign(4, 0.0f);
  buffer.device_dirty = true;
  buffer.download();
  std::vector<float> expected = {1.25f, -2.0f, 3.5f, 0.75f};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(buffer.host[i] - expected[i]) > 1e-6f) {
      return false;
    }
  }
  return buffer.has_device();
}

inline bool gradient_check_feedforward() {
  RNG rng(456);
  FeedForward ff("gc_ff", 3, 5, rng);
  SeqTensor x(1, 1, 3);
  x.data = {0.15f, -0.35f, 0.45f};
  SeqTensor y = ff.forward(x);
  y.grad.assign(y.data.size(), 0.0f);
  y.grad = {0.2f, -0.1f, 0.05f};
  ff.backward(x, y);
  float eps = 1e-3f;
  size_t wi = 2;
  float orig = ff.fc1.w.data[wi];
  ff.fc1.w.data[wi] = orig + eps;
  SeqTensor yp = ff.forward(x);
  float lp = 0.0f;
  for (size_t i = 0; i < yp.data.size(); ++i) {
    lp += yp.data[i] * y.grad[i];
  }
  ff.fc1.w.data[wi] = orig - eps;
  SeqTensor ym = ff.forward(x);
  float lm = 0.0f;
  for (size_t i = 0; i < ym.data.size(); ++i) {
    lm += ym.data[i] * y.grad[i];
  }
  ff.fc1.w.data[wi] = orig;
  float numerical = (lp - lm) / (2.0f * eps);
  float analytical = ff.fc1.w.grad[wi];
  return std::fabs(numerical - analytical) < 1e-2f;
}

inline bool metal_linear_forward_kernel_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  std::vector<float> x = {
      1.0f, -2.0f, 0.5f,
      0.25f, 3.0f, -1.5f,
  };
  std::vector<float> w = {
      0.2f, -0.1f,
      0.4f, 0.3f,
      -0.5f, 0.7f,
  };
  std::vector<float> bias = {0.05f, -0.02f};
  std::vector<float> metal(4, 0.0f);
  bool ok = microgpt_metal_linear_forward(x.data(), w.data(), bias.data(), metal.data(), 2, 3, 2, true);
  if (!ok) {
    return false;
  }
  std::vector<float> cpu(4, 0.0f);
  matmul_forward_cpu(x.data(), w.data(), cpu.data(), 2, 3, 2);
  for (int r = 0; r < 2; ++r) {
    for (int o = 0; o < 2; ++o) {
      cpu[static_cast<size_t>(r) * 2 + static_cast<size_t>(o)] += bias[static_cast<size_t>(o)];
    }
  }
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (std::fabs(cpu[i] - metal[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool metal_linear_backward_kernel_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  Parameter w_cpu("metal_backward.w", {3, 2}, true);
  Parameter b_cpu("metal_backward.b", {2}, false);
  w_cpu.data = {0.2f, -0.1f, 0.4f, 0.3f, -0.5f, 0.7f};
  b_cpu.data = {0.05f, -0.02f};
  w_cpu.grad.assign(w_cpu.data.size(), 0.0f);
  b_cpu.grad.assign(b_cpu.data.size(), 0.0f);
  Parameter w_metal = w_cpu;
  Parameter b_metal = b_cpu;
  SeqTensor x_cpu(2, 1, 3);
  x_cpu.data = {1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f};
  SeqTensor x_metal = x_cpu;
  SeqTensor y(2, 1, 2);
  y.grad = {0.3f, -0.2f, 0.1f, 0.4f};
  linear_backward_op(BackendKind::Cpu, x_cpu, w_cpu, b_cpu, 3, 2, y);
  linear_backward_op(BackendKind::Metal, x_metal, w_metal, b_metal, 3, 2, y);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < w_cpu.grad.size(); ++i) {
    if (std::fabs(w_cpu.grad[i] - w_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < b_cpu.grad.size(); ++i) {
    if (std::fabs(b_cpu.grad[i] - b_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

inline bool cached_linear_backend_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  Parameter w_cpu("cached_linear.w", {3, 2}, true);
  Parameter b_cpu("cached_linear.b", {2}, false);
  w_cpu.data = {0.2f, -0.1f, 0.4f, 0.3f, -0.5f, 0.7f};
  b_cpu.data = {0.05f, -0.02f};
  w_cpu.grad.assign(w_cpu.data.size(), 0.0f);
  b_cpu.grad.assign(b_cpu.data.size(), 0.0f);
  Parameter w_metal = w_cpu;
  Parameter b_metal = b_cpu;
  SeqTensor x_cpu(2, 1, 3);
  x_cpu.data = {1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f};
  SeqTensor x_metal = x_cpu;
  LinearBackendCache cache(BackendKind::Metal);
  SeqTensor y_cpu = linear_forward_op(BackendKind::Cpu, x_cpu, w_cpu, b_cpu, 3, 2);
  SeqTensor y_metal = linear_forward_op(BackendKind::Metal, &cache, x_metal, w_metal, b_metal, 3, 2);
  for (size_t i = 0; i < y_cpu.data.size(); ++i) {
    if (std::fabs(y_cpu.data[i] - y_metal.data[i]) > 1e-5f) {
      return false;
    }
  }
  y_cpu.grad = {0.3f, -0.2f, 0.1f, 0.4f};
  y_metal.grad = y_cpu.grad;
  linear_backward_op(BackendKind::Cpu, x_cpu, w_cpu, b_cpu, 3, 2, y_cpu);
  linear_backward_op(BackendKind::Metal, &cache, x_metal, w_metal, b_metal, 3, 2, y_metal);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < w_cpu.grad.size(); ++i) {
    if (std::fabs(w_cpu.grad[i] - w_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < b_cpu.grad.size(); ++i) {
    if (std::fabs(b_cpu.grad[i] - b_metal.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return cache.x.has_device() && cache.w.has_device() && cache.y.has_device() && cache.dx.has_device() &&
         cache.dw.has_device();
}

inline bool cuda_cached_linear_backend_test() {
  if (!microgpt_cuda_runtime_available()) {
    return true;
  }
  Parameter w_cpu("cached_cuda_linear.w", {3, 2}, true);
  Parameter b_cpu("cached_cuda_linear.b", {2}, false);
  w_cpu.data = {0.2f, -0.1f, 0.4f, 0.3f, -0.5f, 0.7f};
  b_cpu.data = {0.05f, -0.02f};
  w_cpu.grad.assign(w_cpu.data.size(), 0.0f);
  b_cpu.grad.assign(b_cpu.data.size(), 0.0f);
  Parameter w_cuda = w_cpu;
  Parameter b_cuda = b_cpu;
  SeqTensor x_cpu(2, 1, 3);
  x_cpu.data = {1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f};
  SeqTensor x_cuda = x_cpu;
  LinearBackendCache cache(BackendKind::Cuda);
  SeqTensor y_cpu = linear_forward_op(BackendKind::Cpu, x_cpu, w_cpu, b_cpu, 3, 2);
  SeqTensor y_cuda = linear_forward_op(BackendKind::Cuda, &cache, x_cuda, w_cuda, b_cuda, 3, 2);
  for (size_t i = 0; i < y_cpu.data.size(); ++i) {
    if (std::fabs(y_cpu.data[i] - y_cuda.data[i]) > 1e-5f) {
      return false;
    }
  }
  y_cpu.grad = {0.3f, -0.2f, 0.1f, 0.4f};
  y_cuda.grad = y_cpu.grad;
  linear_backward_op(BackendKind::Cpu, x_cpu, w_cpu, b_cpu, 3, 2, y_cpu);
  linear_backward_op(BackendKind::Cuda, &cache, x_cuda, w_cuda, b_cuda, 3, 2, y_cuda);
  for (size_t i = 0; i < x_cpu.grad.size(); ++i) {
    if (std::fabs(x_cpu.grad[i] - x_cuda.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < w_cpu.grad.size(); ++i) {
    if (std::fabs(w_cpu.grad[i] - w_cuda.grad[i]) > 1e-5f) {
      return false;
    }
  }
  for (size_t i = 0; i < b_cpu.grad.size(); ++i) {
    if (std::fabs(b_cpu.grad[i] - b_cuda.grad[i]) > 1e-5f) {
      return false;
    }
  }
  return cache.x.has_device() && cache.w.has_device() && cache.y.has_device() && cache.dx.has_device() &&
         cache.dw.has_device();
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

inline bool generation_alignment_test() {
  Config cfg;
  cfg.batch_size = 1;
  cfg.context_length = 8;
  cfg.d_model = 2;
  cfg.num_layers = 0;
  cfg.num_heads = 1;
  cfg.d_ff = 4;
  cfg.seed = 13;
  Model model(cfg);
  fill_zeros(model.token_embedding.data);
  fill_zeros(model.position_embedding.data);
  fill_zeros(model.final_norm.beta.data);
  fill_ones(model.final_norm.gamma.data);
  fill_zeros(model.lm_head.w.data);
  fill_zeros(model.lm_head.b.data);

  auto set_position = [&](int pos, float a, float b) {
    model.position_embedding.data[static_cast<size_t>(pos) * static_cast<size_t>(cfg.d_model)] = a;
    model.position_embedding.data[static_cast<size_t>(pos) * static_cast<size_t>(cfg.d_model) + 1] = b;
  };
  set_position(1, 1.0f, -1.0f);
  set_position(7, -1.0f, 1.0f);

  int good = static_cast<int>('X');
  int bad = static_cast<int>('Y');
  model.lm_head.w.data[static_cast<size_t>(0) * static_cast<size_t>(cfg.vocab_size) + static_cast<size_t>(good)] = 5.0f;
  model.lm_head.w.data[static_cast<size_t>(1) * static_cast<size_t>(cfg.vocab_size) + static_cast<size_t>(bad)] = 5.0f;

  std::string generated = generate_text(model, "ab", 1, 0.2f, 1, Tokenizer::kEos);
  return generated == "X";
}

inline std::string test_temp_path(const std::string& stem, const std::string& suffix) {
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::ostringstream out;
  const char* tmpdir = std::getenv("TMPDIR");
  out << (tmpdir != nullptr && tmpdir[0] != '\0' ? tmpdir : "/tmp") << '/' << stem << '_' << now << suffix;
  return out.str();
}

inline bool instruction_roundtrip_test() {
  std::vector<EvalExample> examples = {{"What is 2+2?", "4"}, {"Say hi", "hello"}};
  std::string path = test_temp_path("microgpt-roundtrip", ".txt");
  write_instruction_examples(path, examples);
  DatasetValidation validation = validate_instruction_text(read_file_text(path));
  if (!validation.errors.empty() || validation.examples.size() != examples.size()) {
    return false;
  }
  for (size_t i = 0; i < examples.size(); ++i) {
    if (validation.examples[i].prompt != examples[i].prompt || validation.examples[i].answer != examples[i].answer) {
      return false;
    }
  }
  return true;
}

inline bool jsonl_import_roundtrip_test() {
  std::string path = test_temp_path("microgpt-jsonl", ".txt");
  std::vector<EvalExample> examples;
  examples.push_back(parse_jsonl_example_line(R"({"user":"What is 3+4?","assistant":"7"})", 1));
  examples.push_back(parse_jsonl_example_line(R"({"user":"Return JSON with key answer and value 42.","assistant":"{\"answer\":42}"})", 2));
  write_instruction_examples(path, examples);
  DatasetValidation validation = validate_instruction_text(read_file_text(path));
  if (!validation.errors.empty() || validation.examples.size() != examples.size()) {
    return false;
  }
  for (size_t i = 0; i < examples.size(); ++i) {
    if (validation.examples[i].prompt != examples[i].prompt || validation.examples[i].answer != examples[i].answer) {
      return false;
    }
  }
  return true;
}

inline bool split_dataset_contract_test() {
  std::vector<EvalExample> examples;
  for (int i = 0; i < 10; ++i) {
    examples.push_back({"Prompt " + std::to_string(i), "Answer " + std::to_string(i)});
  }
  auto split = split_instruction_examples(examples, 0.7f, 42);
  if (split.first.size() != 7 || split.second.size() != 3) {
    return false;
  }
  std::string train_path = test_temp_path("microgpt-train", ".txt");
  std::string val_path = test_temp_path("microgpt-val", ".txt");
  write_instruction_examples(train_path, split.first);
  write_instruction_examples(val_path, split.second);
  DatasetValidation train_validation = validate_instruction_text(read_file_text(train_path));
  DatasetValidation val_validation = validate_instruction_text(read_file_text(val_path));
  return train_validation.errors.empty() && val_validation.errors.empty() &&
         train_validation.examples.size() == split.first.size() &&
         val_validation.examples.size() == split.second.size();
}

inline bool session_roundtrip_test() {
  std::vector<SessionExample> examples;
  SessionExample first;
  first.turns.push_back({"user", "Hi"});
  first.turns.push_back({"assistant", "Hello"});
  first.turns.push_back({"user", "How are you?"});
  first.turns.push_back({"assistant", "Fine"});
  examples.push_back(first);
  SessionExample second;
  second.turns.push_back({"user", "Tell me a fact"});
  second.turns.push_back({"assistant", "Facts are useful."});
  examples.push_back(second);
  std::string path = test_temp_path("microgpt-session", ".txt");
  write_session_examples(path, examples);
  SessionValidation validation = validate_session_text(read_file_text(path));
  if (!validation.errors.empty() || validation.examples.size() != examples.size()) {
    return false;
  }
  for (size_t i = 0; i < examples.size(); ++i) {
    if (validation.examples[i].turns.size() != examples[i].turns.size()) {
      return false;
    }
    for (size_t j = 0; j < examples[i].turns.size(); ++j) {
      if (validation.examples[i].turns[j].role != examples[i].turns[j].role ||
          validation.examples[i].turns[j].content != examples[i].turns[j].content) {
        return false;
      }
    }
  }
  return true;
}

inline bool session_jsonl_import_roundtrip_test() {
  std::string path = test_temp_path("microgpt-session-jsonl", ".txt");
  std::vector<SessionExample> examples;
  SessionExample first;
  first.turns.push_back({"user", "Hi"});
  first.turns.push_back({"assistant", "Hello"});
  first.turns.push_back({"user", "What now?"});
  first.turns.push_back({"assistant", "Keep going."});
  examples.push_back(first);
  SessionExample second;
  second.turns.push_back({"user", "Give JSON"});
  second.turns.push_back({"assistant", "{\"ok\":true}"});
  examples.push_back(second);
  std::vector<SessionExample> parsed;
  parsed.push_back(parse_jsonl_session_line(
      R"({"turns":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"What now?"},{"role":"assistant","content":"Keep going."}]})",
      1));
  parsed.push_back(parse_jsonl_session_line(
      R"({"turns":[{"role":"user","content":"Give JSON"},{"role":"assistant","content":"{\"ok\":true}"}]})", 2));
  write_session_examples(path, parsed);
  SessionValidation validation = validate_session_text(read_file_text(path));
  if (!validation.errors.empty() || validation.examples.size() != parsed.size()) {
    return false;
  }
  for (size_t i = 0; i < parsed.size(); ++i) {
    if (validation.examples[i].turns.size() != parsed[i].turns.size()) {
      return false;
    }
  }
  return true;
}

inline bool session_split_contract_test() {
  std::vector<SessionExample> examples;
  for (int i = 0; i < 6; ++i) {
    SessionExample ex;
    ex.turns.push_back({"user", "Prompt " + std::to_string(i)});
    ex.turns.push_back({"assistant", "Answer " + std::to_string(i)});
    if (i % 2 == 0) {
      ex.turns.push_back({"user", "Follow up " + std::to_string(i)});
      ex.turns.push_back({"assistant", "Reply " + std::to_string(i)});
    }
    examples.push_back(ex);
  }
  auto split = split_session_examples(examples, 0.5f, 7);
  if (split.first.size() != 3 || split.second.size() != 3) {
    return false;
  }
  std::string train_path = test_temp_path("microgpt-session-train", ".txt");
  std::string val_path = test_temp_path("microgpt-session-val", ".txt");
  write_session_examples(train_path, split.first);
  write_session_examples(val_path, split.second);
  SessionValidation train_validation = validate_session_text(read_file_text(train_path));
  SessionValidation val_validation = validate_session_text(read_file_text(val_path));
  return train_validation.errors.empty() && val_validation.errors.empty() &&
         train_validation.examples.size() == split.first.size() &&
         val_validation.examples.size() == split.second.size();
}

inline bool chat_history_prompt_test() {
  std::vector<ChatMessage> history;
  history.push_back({ChatRole::User, "Hi"});
  history.push_back({ChatRole::Assistant, "Hello"});
  std::string prompt = format_multi_turn_chat_prompt(history, "What now?");
  std::string expected = "<BOS><USER>\nHi\n<ASSISTANT>\nHello\n<USER>\nWhat now?\n<ASSISTANT>\n";
  return prompt == expected;
}

inline bool chat_history_trimming_test() {
  std::vector<ChatMessage> history;
  history.push_back({ChatRole::User, "one"});
  history.push_back({ChatRole::Assistant, "two"});
  history.push_back({ChatRole::User, "three"});
  history.push_back({ChatRole::Assistant, "four"});
  std::vector<ChatMessage> trimmed = trim_chat_history_to_context(history, "five", 20);
  if (trimmed.size() >= history.size()) {
    return false;
  }
  std::string prompt = format_multi_turn_chat_prompt(trimmed, "five");
  if (prompt.find("five") == std::string::npos) {
    return false;
  }
  return prompt.find("one") == std::string::npos && prompt.find("two") == std::string::npos &&
         prompt.find("three") == std::string::npos && prompt.find("four") == std::string::npos;
}

inline bool chat_history_compression_test() {
  Tokenizer tok(TokenizerKind::Byte);
  std::vector<ChatMessage> history;
  for (int i = 0; i < 8; ++i) {
    std::string repeated = "This is a long message about compressing context. ";
    repeated += "It keeps enough signal for the summary to be shorter than the original. ";
    repeated += "Turn " + std::to_string(i) + ". ";
    repeated += repeated;
    history.push_back({ChatRole::User, repeated});
    history.push_back({ChatRole::Assistant, repeated});
  }

  std::string original_prompt = format_multi_turn_chat_prompt(history, "");
  size_t original_tokens = tok.encode_text(original_prompt).size();
  std::vector<ChatMessage> compressed = compress_chat_history(history, tok, static_cast<int>(original_tokens));
  if (compressed.size() >= history.size()) {
    return false;
  }
  std::string compressed_prompt = format_multi_turn_chat_prompt(compressed, "");
  size_t compressed_tokens = tok.encode_text(compressed_prompt).size();
  if (compressed_tokens >= original_tokens) {
    return false;
  }
  if (compressed.size() < 4) {
    return false;
  }
  const size_t tail_start = history.size() - 4;
  for (size_t i = 0; i < 4; ++i) {
    const ChatMessage& expected = history[tail_start + i];
    const ChatMessage& actual = compressed[compressed.size() - 4 + i];
    if (expected.role != actual.role || expected.content != actual.content) {
      return false;
    }
  }
  return true;
}

inline bool io_contract_test() {
  std::vector<uint8_t> bytes = {0, 127, 255};
  std::vector<int> tokens = bytes_to_tokens(bytes);
  if (tokens != std::vector<int>({0, 127, 255})) {
    return false;
  }
  bool rejected = false;
  try {
    (void)bytes_to_tokens(bytes, 128);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  return rejected;
}

inline bool tokenizer_byte_roundtrip_test() {
  Tokenizer tok(TokenizerKind::Byte);
  std::string text = "<BOS><USER>\nHello microgpt\n<ASSISTANT>\nOK<EOS>";
  std::vector<int> ids = tok.encode_text(text);
  return tok.vocab_size() == Tokenizer::kSpecialVocab && tok.decode_text(ids) == text;
}

inline bool tokenizer_bpe_roundtrip_test() {
  Tokenizer byte_tok(TokenizerKind::Byte);
  Tokenizer bpe_tok(TokenizerKind::Bpe);
  std::string text = "<BOS><USER>\nSummarize the training data and validation data for microgpt.\n<ASSISTANT>\nClean data helps.<EOS>";
  std::vector<int> byte_ids = byte_tok.encode_text(text);
  std::vector<int> bpe_ids = bpe_tok.encode_text(text);
  if (bpe_tok.vocab_size() <= byte_tok.vocab_size()) {
    return false;
  }
  if (bpe_ids.size() >= byte_ids.size()) {
    return false;
  }
  return bpe_tok.decode_text(bpe_ids) == text;
}

inline bool tokenizer_checkpoint_roundtrip_test() {
  Config cfg;
  cfg.tokenizer_kind = static_cast<int>(TokenizerKind::Bpe);
  cfg.vocab_size = Tokenizer(TokenizerKind::Bpe).vocab_size();
  cfg.context_length = 8;
  cfg.d_model = 8;
  cfg.num_layers = 1;
  cfg.num_heads = 2;
  cfg.d_ff = 16;
  cfg.batch_size = 2;
  Model model(cfg);
  AdamW opt;
  opt.step = 3;
  std::string checkpoint = test_temp_path("microgpt-tokenizer", ".bin");
  save_checkpoint(checkpoint, model, opt, 7);
  AdamW loaded_opt;
  int loaded_step = 0;
  Model loaded = load_checkpoint(checkpoint, loaded_opt, loaded_step);
  std::remove(checkpoint.c_str());
  std::remove(checkpoint_json_path(checkpoint).c_str());
  return loaded_step == 7 && loaded_opt.step == 3 && loaded.cfg.tokenizer_kind == cfg.tokenizer_kind &&
         loaded.cfg.vocab_size == cfg.vocab_size;
}

inline Config backend_parity_test_config() {
  Config cfg;
  cfg.batch_size = 2;
  cfg.context_length = 8;
  cfg.d_model = 8;
  cfg.num_layers = 1;
  cfg.num_heads = 2;
  cfg.d_ff = 16;
  cfg.learning_rate = 0.0005f;
  cfg.max_grad_norm = 1.0f;
  cfg.seed = 2024;
  return cfg;
}

inline std::vector<int> backend_parity_tokens() {
  std::vector<int> tokens;
  std::string text = "the quick brown fox answers tiny questions. ";
  while (tokens.size() < 256) {
    for (unsigned char c : text) {
      tokens.push_back(static_cast<int>(c));
    }
  }
  return tokens;
}

inline void backend_parity_train_steps(Model& model, AdamW& opt, const std::vector<int>& tokens, int steps) {
  opt.set_backend(model.backend);
  for (int i = 0; i < steps; ++i) {
    model.zero_grad();
    Batch batch = sample_batch(tokens, model.cfg.batch_size, model.cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    cross_entropy_loss(logits, batch.y);
    model.backward(logits);
    auto params = model.parameters();
    clip_gradients(params, model.cfg.max_grad_norm);
    opt.update(params);
  }
}

inline bool backend_adamw_update_test() {
  Parameter cpu_p("adamw.cpu", {4}, true);
  cpu_p.data = {0.2f, -0.4f, 0.7f, -1.1f};
  cpu_p.grad = {0.05f, -0.1f, 0.2f, -0.3f};
  cpu_p.m = {0.01f, -0.02f, 0.03f, -0.04f};
  cpu_p.v = {0.001f, 0.002f, 0.003f, 0.004f};
  Parameter metal_p = cpu_p;
  AdamW cpu_opt;
  cpu_opt.lr = 0.001f;
  cpu_opt.beta1 = 0.9f;
  cpu_opt.beta2 = 0.999f;
  cpu_opt.eps = 1e-8f;
  cpu_opt.weight_decay = 0.01f;
  AdamW metal_opt = cpu_opt;
  metal_opt.set_backend(BackendKind::Metal);
  std::vector<Parameter*> cpu_params = {&cpu_p};
  std::vector<Parameter*> metal_params = {&metal_p};
  cpu_opt.update(cpu_params);
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  metal_opt.update(metal_params);
  for (size_t i = 0; i < cpu_p.data.size(); ++i) {
    if (std::fabs(cpu_p.data[i] - metal_p.data[i]) > 1e-5f || std::fabs(cpu_p.m[i] - metal_p.m[i]) > 1e-5f ||
        std::fabs(cpu_p.v[i] - metal_p.v[i]) > 1e-5f) {
      return false;
    }
  }
  return cpu_p.version == 1 && metal_p.version == 1;
}

inline bool backend_adamw_multi_param_parity_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }

  std::vector<Parameter> cpu_storage;
  std::vector<Parameter> metal_storage;
  for (int i = 0; i < 3; ++i) {
    Parameter cpu_p("adamw.batch." + std::to_string(i), {4 + i}, true);
    for (size_t j = 0; j < cpu_p.data.size(); ++j) {
      float base = static_cast<float>(i + 1) * 0.1f;
      cpu_p.data[j] = base + static_cast<float>(j) * 0.05f;
      cpu_p.grad[j] = static_cast<float>(j + 1) * 0.02f - static_cast<float>(i) * 0.01f;
      cpu_p.m[j] = static_cast<float>(j) * 0.03f;
      cpu_p.v[j] = static_cast<float>(j + 1) * 0.004f;
    }
    if (i == 1) {
      cpu_p.decay = false;
    }
    cpu_storage.push_back(cpu_p);
    metal_storage.push_back(cpu_p);
  }

  AdamW cpu_opt;
  AdamW metal_opt;
  cpu_opt.lr = 0.001f;
  cpu_opt.beta1 = 0.9f;
  cpu_opt.beta2 = 0.999f;
  cpu_opt.eps = 1e-8f;
  cpu_opt.weight_decay = 0.01f;
  metal_opt = cpu_opt;
  metal_opt.set_backend(BackendKind::Metal);
  metal_opt.batch_updates = false;
  std::vector<Parameter*> cpu_params;
  std::vector<Parameter*> metal_params;
  for (size_t i = 0; i < cpu_storage.size(); ++i) {
    cpu_params.push_back(&cpu_storage[i]);
    metal_params.push_back(&metal_storage[i]);
  }
  cpu_opt.update(cpu_params);
  metal_opt.update(metal_params);
  for (size_t p = 0; p < cpu_storage.size(); ++p) {
    if (cpu_storage[p].version != 1 || metal_storage[p].version != 1) {
      return false;
    }
    for (size_t i = 0; i < cpu_storage[p].data.size(); ++i) {
      if (std::fabs(cpu_storage[p].data[i] - metal_storage[p].data[i]) > 1e-5f ||
          std::fabs(cpu_storage[p].m[i] - metal_storage[p].m[i]) > 1e-5f ||
          std::fabs(cpu_storage[p].v[i] - metal_storage[p].v[i]) > 1e-5f) {
        return false;
      }
    }
  }
  return true;
}

inline float max_parameter_abs_diff(Model& a, Model& b) {
  auto ap = a.parameters();
  auto bp = b.parameters();
  if (ap.size() != bp.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (size_t p = 0; p < ap.size(); ++p) {
    if (ap[p]->data.size() != bp[p]->data.size()) {
      return std::numeric_limits<float>::infinity();
    }
    for (size_t i = 0; i < ap[p]->data.size(); ++i) {
      max_diff = std::max(max_diff, std::fabs(ap[p]->data[i] - bp[p]->data[i]));
    }
  }
  return max_diff;
}

inline float max_tensor_abs_diff(const SeqTensor& a, const SeqTensor& b) {
  if (a.B != b.B || a.T != b.T || a.D != b.D || a.data.size() != b.data.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.data.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a.data[i] - b.data[i]));
  }
  return max_diff;
}

inline float max_vector_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

template <typename Getter>
inline float max_parameter_component_abs_diff(Model& a, Model& b, Getter getter) {
  auto ap = a.parameters();
  auto bp = b.parameters();
  if (ap.size() != bp.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (size_t p = 0; p < ap.size(); ++p) {
    const std::vector<float>& lhs = getter(*ap[p]);
    const std::vector<float>& rhs = getter(*bp[p]);
    max_diff = std::max(max_diff, max_vector_abs_diff(lhs, rhs));
  }
  return max_diff;
}

struct ParityTraceEntry {
  std::string stage;
  float max_abs_diff = 0.0f;
};

inline void add_parity_trace(std::vector<ParityTraceEntry>& trace, const std::string& stage, float diff) {
  trace.push_back({stage, diff});
}

inline bool parity_trace_ok(const std::vector<ParityTraceEntry>& trace, float tolerance) {
  for (const auto& entry : trace) {
    if (!std::isfinite(entry.max_abs_diff) || entry.max_abs_diff > tolerance) {
      return false;
    }
  }
  return true;
}

inline std::string first_parity_trace_failure(const std::vector<ParityTraceEntry>& trace, float tolerance) {
  for (const auto& entry : trace) {
    if (!std::isfinite(entry.max_abs_diff) || entry.max_abs_diff > tolerance) {
      return entry.stage;
    }
  }
  return "";
}

inline void print_parity_trace(const std::vector<ParityTraceEntry>& trace, std::ostream& out) {
  out << "parity_trace\n";
  for (const auto& entry : trace) {
    out << entry.stage << " max_abs_diff " << std::fixed << std::setprecision(6) << entry.max_abs_diff << '\n';
  }
}

inline void trace_block_state(std::vector<ParityTraceEntry>& trace, const Block& cpu, const Block& metal,
                              const std::string& prefix) {
  add_parity_trace(trace, prefix + ".ln1.mean", max_vector_abs_diff(cpu.ln1.mean, metal.ln1.mean));
  add_parity_trace(trace, prefix + ".ln1.inv_std", max_vector_abs_diff(cpu.ln1.inv_std, metal.ln1.inv_std));
  add_parity_trace(trace, prefix + ".ln1.xhat", max_vector_abs_diff(cpu.ln1.xhat, metal.ln1.xhat));
  add_parity_trace(trace, prefix + ".attn.q", max_tensor_abs_diff(cpu.attn.q, metal.attn.q));
  add_parity_trace(trace, prefix + ".attn.k", max_tensor_abs_diff(cpu.attn.k, metal.attn.k));
  add_parity_trace(trace, prefix + ".attn.v", max_tensor_abs_diff(cpu.attn.v, metal.attn.v));
  add_parity_trace(trace, prefix + ".attn.context", max_tensor_abs_diff(cpu.attn.context, metal.attn.context));
  add_parity_trace(trace, prefix + ".attn.weights", max_vector_abs_diff(cpu.attn.weights, metal.attn.weights));
  add_parity_trace(trace, prefix + ".resid1", max_tensor_abs_diff(cpu.resid1, metal.resid1));
  add_parity_trace(trace, prefix + ".ln2.mean", max_vector_abs_diff(cpu.ln2.mean, metal.ln2.mean));
  add_parity_trace(trace, prefix + ".ln2.inv_std", max_vector_abs_diff(cpu.ln2.inv_std, metal.ln2.inv_std));
  add_parity_trace(trace, prefix + ".ln2.xhat", max_vector_abs_diff(cpu.ln2.xhat, metal.ln2.xhat));
  add_parity_trace(trace, prefix + ".ff.pre_activation", max_tensor_abs_diff(cpu.ff.pre_activation, metal.ff.pre_activation));
  add_parity_trace(trace, prefix + ".ff.hidden", max_tensor_abs_diff(cpu.ff.hidden, metal.ff.hidden));
  add_parity_trace(trace, prefix + ".ff.fc1.w.grad", max_vector_abs_diff(cpu.ff.fc1.w.grad, metal.ff.fc1.w.grad));
  add_parity_trace(trace, prefix + ".ff.fc1.b.grad", max_vector_abs_diff(cpu.ff.fc1.b.grad, metal.ff.fc1.b.grad));
  add_parity_trace(trace, prefix + ".ff.fc2.w.grad", max_vector_abs_diff(cpu.ff.fc2.w.grad, metal.ff.fc2.w.grad));
  add_parity_trace(trace, prefix + ".ff.fc2.b.grad", max_vector_abs_diff(cpu.ff.fc2.b.grad, metal.ff.fc2.b.grad));
}

inline void trace_model_state(std::vector<ParityTraceEntry>& trace, const Model& cpu, const Model& metal,
                              const std::string& prefix) {
  add_parity_trace(trace, prefix + ".embed_out", max_tensor_abs_diff(cpu.embed_out, metal.embed_out));
  add_parity_trace(trace, prefix + ".pre_norm", max_tensor_abs_diff(cpu.pre_norm, metal.pre_norm));
  add_parity_trace(trace, prefix + ".norm_out", max_tensor_abs_diff(cpu.norm_out, metal.norm_out));
  add_parity_trace(trace, prefix + ".token_embedding.data", max_vector_abs_diff(cpu.token_embedding.data, metal.token_embedding.data));
  add_parity_trace(trace, prefix + ".position_embedding.data",
                   max_vector_abs_diff(cpu.position_embedding.data, metal.position_embedding.data));
  add_parity_trace(trace, prefix + ".token_embedding.grad", max_vector_abs_diff(cpu.token_embedding.grad, metal.token_embedding.grad));
  add_parity_trace(trace, prefix + ".position_embedding.grad",
                   max_vector_abs_diff(cpu.position_embedding.grad, metal.position_embedding.grad));
  add_parity_trace(trace, prefix + ".final_norm.gamma.grad",
                   max_vector_abs_diff(cpu.final_norm.gamma.grad, metal.final_norm.gamma.grad));
  add_parity_trace(trace, prefix + ".final_norm.beta.grad",
                   max_vector_abs_diff(cpu.final_norm.beta.grad, metal.final_norm.beta.grad));
  add_parity_trace(trace, prefix + ".lm_head.w.grad", max_vector_abs_diff(cpu.lm_head.w.grad, metal.lm_head.w.grad));
  add_parity_trace(trace, prefix + ".lm_head.b.grad", max_vector_abs_diff(cpu.lm_head.b.grad, metal.lm_head.b.grad));
  for (size_t i = 0; i < cpu.blocks.size(); ++i) {
    trace_block_state(trace, cpu.blocks[i], metal.blocks[i], prefix + ".block" + std::to_string(i));
  }
}

inline bool cpu_metal_staged_parity_trace_report(std::ostream& out) {
  if (!microgpt_metal_runtime_available()) {
    out << "parity_report unavailable\n";
    out << "reason no_runtime_metal_device\n";
    return true;
  }
  Config cfg = backend_parity_test_config();
  std::vector<int> tokens = backend_parity_tokens();
  RNG batch_rng(777);
  Batch batch = sample_batch(tokens, cfg.batch_size, cfg.context_length, batch_rng);

  Model cpu_model(cfg);
  Model metal_model(cfg);
  metal_model.set_backend(BackendKind::Metal);
  AdamW cpu_opt;
  AdamW metal_opt;
  cpu_opt.lr = cfg.learning_rate;
  metal_opt.lr = cfg.learning_rate;
  metal_opt.set_backend(BackendKind::Metal);
  std::vector<ParityTraceEntry> trace;

  SeqTensor cpu_logits = cpu_model.forward(batch.x);
  SeqTensor metal_logits = metal_model.forward(batch.x);
  add_parity_trace(trace, "forward.embed_out", max_tensor_abs_diff(cpu_model.embed_out, metal_model.embed_out));
  add_parity_trace(trace, "forward.pre_norm", max_tensor_abs_diff(cpu_model.pre_norm, metal_model.pre_norm));
  add_parity_trace(trace, "forward.norm_out", max_tensor_abs_diff(cpu_model.norm_out, metal_model.norm_out));
  for (size_t i = 0; i < cpu_model.blocks.size(); ++i) {
    trace_block_state(trace, cpu_model.blocks[i], metal_model.blocks[i], "forward.block" + std::to_string(i));
  }
  add_parity_trace(trace, "forward.logits", max_tensor_abs_diff(cpu_logits, metal_logits));
  out << "parity_report stage forward\n";
  print_parity_trace(trace, out);
  out << "first_divergent_stage " << first_parity_trace_failure(trace, 5e-3f) << '\n';
  bool forward_ok = parity_trace_ok(trace, 5e-3f);

  trace.clear();
  cross_entropy_loss(cpu_logits, batch.y);
  cross_entropy_loss(metal_logits, batch.y);
  add_parity_trace(trace, "backward.logits.grad", max_vector_abs_diff(cpu_logits.grad, metal_logits.grad));
  cpu_model.backward(cpu_logits);
  metal_model.backward(metal_logits);
  trace_model_state(trace, cpu_model, metal_model, "backward");
  add_parity_trace(trace, "backward.param_data", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.data;
                    }));
  add_parity_trace(trace, "backward.param_grad", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.grad;
                    }));
  out << "parity_report stage backward\n";
  print_parity_trace(trace, out);
  out << "first_divergent_stage " << first_parity_trace_failure(trace, 5e-3f) << '\n';
  bool backward_ok = parity_trace_ok(trace, 5e-3f);

  trace.clear();
  std::vector<Parameter*> cpu_params = cpu_model.parameters();
  std::vector<Parameter*> metal_params = metal_model.parameters();
  cpu_opt.update(cpu_params);
  metal_opt.update(metal_params);
  add_parity_trace(trace, "update.param_data", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.data;
                    }));
  add_parity_trace(trace, "update.param_grad", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.grad;
                    }));
  add_parity_trace(trace, "update.param_m", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.m;
                    }));
  add_parity_trace(trace, "update.param_v", max_parameter_component_abs_diff(cpu_model, metal_model, [](const Parameter& p) {
                      return p.v;
                    }));
  out << "parity_report stage update\n";
  print_parity_trace(trace, out);
  out << "first_divergent_stage " << first_parity_trace_failure(trace, 5e-3f) << '\n';
  bool update_ok = parity_trace_ok(trace, 5e-3f);
  out << "parity_report status " << ((forward_ok && backward_ok && update_ok) ? "pass" : "fail") << '\n';
  return forward_ok && backward_ok && update_ok;
}

inline bool cpu_metal_staged_parity_trace_test() {
  std::ostringstream sink;
  bool ok = cpu_metal_staged_parity_trace_report(sink);
  if (!ok) {
    std::cerr << sink.str();
  }
  return ok;
}

inline bool cpu_metal_training_parity_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  Config cfg = backend_parity_test_config();
  std::vector<int> tokens = backend_parity_tokens();
  Model cpu_model(cfg);
  Model metal_model(cfg);
  metal_model.set_backend(BackendKind::Metal);
  AdamW cpu_opt;
  AdamW metal_opt;
  cpu_opt.lr = cfg.learning_rate;
  metal_opt.lr = cfg.learning_rate;
  backend_parity_train_steps(cpu_model, cpu_opt, tokens, 2);
  backend_parity_train_steps(metal_model, metal_opt, tokens, 2);
  if (max_parameter_abs_diff(cpu_model, metal_model) > 5e-3f) {
    return false;
  }
  std::vector<int> prompt(cfg.context_length, static_cast<int>('a'));
  SeqTensor cpu_logits = cpu_model.forward(prompt);
  SeqTensor metal_logits = metal_model.forward(prompt);
  return max_tensor_abs_diff(cpu_logits, metal_logits) < 5e-3f;
}

inline bool metal_checkpoint_cpu_generation_interop_test() {
  if (!microgpt_metal_runtime_available()) {
    return true;
  }
  Config cfg = backend_parity_test_config();
  std::vector<int> tokens = backend_parity_tokens();
  Model metal_model(cfg);
  metal_model.set_backend(BackendKind::Metal);
  AdamW metal_opt;
  metal_opt.lr = cfg.learning_rate;
  backend_parity_train_steps(metal_model, metal_opt, tokens, 2);

  std::string checkpoint = test_temp_path("microgpt-metal-interop", ".bin");
  save_checkpoint(checkpoint, metal_model, metal_opt, 2);
  AdamW cpu_loaded_opt;
  AdamW metal_loaded_opt;
  int cpu_step = 0;
  int metal_step = 0;
  Model cpu_loaded = load_checkpoint(checkpoint, cpu_loaded_opt, cpu_step);
  Model metal_loaded = load_checkpoint(checkpoint, metal_loaded_opt, metal_step);
  metal_loaded.set_backend(BackendKind::Metal);
  std::remove(checkpoint.c_str());
  std::remove(checkpoint_json_path(checkpoint).c_str());
  if (cpu_step != 2 || metal_step != 2 || cpu_loaded_opt.step != metal_opt.step ||
      metal_loaded_opt.step != metal_opt.step) {
    return false;
  }
  std::vector<int> prompt(cfg.context_length, static_cast<int>('q'));
  SeqTensor cpu_logits = cpu_loaded.forward(prompt);
  SeqTensor metal_logits = metal_loaded.forward(prompt);
  if (max_tensor_abs_diff(cpu_logits, metal_logits) > 5e-3f) {
    return false;
  }
  std::string cpu_generated = generate_text(cpu_loaded, "the", 4, 0.2f, 1, Tokenizer::kEos);
  std::string metal_generated = generate_text(metal_loaded, "the", 4, 0.2f, 1, Tokenizer::kEos);
  return cpu_generated == metal_generated;
}

inline bool cuda_checkpoint_cpu_generation_interop_test() {
  if (!microgpt_cuda_runtime_compiled()) {
    return true;
  }
  Config cfg = backend_parity_test_config();
  std::vector<int> tokens = backend_parity_tokens();
  Model cuda_model(cfg);
  cuda_model.set_backend(BackendKind::Cuda);
  AdamW cuda_opt;
  cuda_opt.lr = cfg.learning_rate;
  cuda_opt.set_backend(BackendKind::Cuda);
  backend_parity_train_steps(cuda_model, cuda_opt, tokens, 2);

  std::string checkpoint = test_temp_path("microgpt-cuda-interop", ".bin");
  save_checkpoint(checkpoint, cuda_model, cuda_opt, 2);
  AdamW cpu_loaded_opt;
  AdamW cuda_loaded_opt;
  int cpu_step = 0;
  int cuda_step = 0;
  Model cpu_loaded = load_checkpoint(checkpoint, cpu_loaded_opt, cpu_step);
  Model cuda_loaded = load_checkpoint(checkpoint, cuda_loaded_opt, cuda_step);
  cuda_loaded.set_backend(BackendKind::Cuda);
  std::remove(checkpoint.c_str());
  std::remove(checkpoint_json_path(checkpoint).c_str());
  if (cpu_step != 2 || cuda_step != 2 || cpu_loaded_opt.step != cuda_opt.step ||
      cuda_loaded_opt.step != cuda_opt.step) {
    return false;
  }
  std::vector<int> prompt(cfg.context_length, static_cast<int>('q'));
  SeqTensor cpu_logits = cpu_loaded.forward(prompt);
  SeqTensor cuda_logits = cuda_loaded.forward(prompt);
  if (max_tensor_abs_diff(cpu_logits, cuda_logits) > 5e-3f) {
    return false;
  }
  std::string cpu_generated = generate_text(cpu_loaded, "the", 4, 0.2f, 1, Tokenizer::kEos);
  std::string cuda_generated = generate_text(cuda_loaded, "the", 4, 0.2f, 1, Tokenizer::kEos);
  return cpu_generated == cuda_generated;
}

inline bool run_tests() {
  bool ok1 = gradient_check_linear();
  bool ok2 = backend_linear_op_test();
  bool ok3 = backend_matmul_op_test();
  bool ok4 = backend_gelu_op_test();
  bool ok5 = backend_layernorm_op_test();
  bool ok6 = backend_feedforward_forward_test();
  bool ok7 = backend_feedforward_backward_test();
  bool ok8 = backend_buffer_roundtrip_test();
  bool ok9 = metal_linear_forward_kernel_test();
  bool ok10 = metal_linear_backward_kernel_test();
  bool ok11 = cached_linear_backend_test();
  bool ok12 = gradient_check_feedforward();
  bool ok13 = causal_mask_test();
  bool ok14 = tiny_overfit_test();
  bool ok15 = alternating_pattern_test();
  bool ok16 = generation_alignment_test();
  bool ok17 = instruction_roundtrip_test();
  bool ok18 = jsonl_import_roundtrip_test();
  bool ok19 = split_dataset_contract_test();
  bool ok20 = io_contract_test();
  bool ok21 = tokenizer_byte_roundtrip_test();
  bool ok22 = tokenizer_bpe_roundtrip_test();
  bool ok23 = tokenizer_checkpoint_roundtrip_test();
  bool ok24 = session_roundtrip_test();
  bool ok25 = session_jsonl_import_roundtrip_test();
  bool ok26 = session_split_contract_test();
  bool ok27 = chat_history_prompt_test();
  bool ok28 = chat_history_trimming_test();
  bool ok29 = chat_history_compression_test();
  bool ok30 = cpu_metal_training_parity_test();
  bool ok31 = backend_adamw_update_test();
  bool ok32 = backend_adamw_multi_param_parity_test();
  bool ok33 = cpu_metal_staged_parity_trace_test();
  bool ok34 = metal_checkpoint_cpu_generation_interop_test();
  bool ok35 = cuda_checkpoint_cpu_generation_interop_test();
  bool ok36 = cuda_feedforward_forward_test();
  bool ok37 = cuda_feedforward_backward_test();
  bool ok38 = cuda_cached_linear_backend_test();
  std::cout << "gradient_check_linear: " << (ok1 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_linear_op_test: " << (ok2 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_matmul_op_test: " << (ok3 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_gelu_op_test: " << (ok4 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_layernorm_op_test: " << (ok5 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_feedforward_forward_test: " << (ok6 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_feedforward_backward_test: " << (ok7 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_buffer_roundtrip_test: " << (ok8 ? "PASS" : "FAIL") << '\n';
  std::cout << "metal_linear_forward_kernel_test: " << (ok9 ? "PASS" : "FAIL") << '\n';
  std::cout << "metal_linear_backward_kernel_test: " << (ok10 ? "PASS" : "FAIL") << '\n';
  std::cout << "cached_linear_backend_test: " << (ok11 ? "PASS" : "FAIL") << '\n';
  std::cout << "gradient_check_feedforward: " << (ok12 ? "PASS" : "FAIL") << '\n';
  std::cout << "causal_mask_test: " << (ok13 ? "PASS" : "FAIL") << '\n';
  std::cout << "tiny_overfit_test: " << (ok14 ? "PASS" : "FAIL") << '\n';
  std::cout << "alternating_pattern_test: " << (ok15 ? "PASS" : "FAIL") << '\n';
  std::cout << "generation_alignment_test: " << (ok16 ? "PASS" : "FAIL") << '\n';
  std::cout << "instruction_roundtrip_test: " << (ok17 ? "PASS" : "FAIL") << '\n';
  std::cout << "jsonl_import_roundtrip_test: " << (ok18 ? "PASS" : "FAIL") << '\n';
  std::cout << "split_dataset_contract_test: " << (ok19 ? "PASS" : "FAIL") << '\n';
  std::cout << "io_contract_test: " << (ok20 ? "PASS" : "FAIL") << '\n';
  std::cout << "tokenizer_byte_roundtrip_test: " << (ok21 ? "PASS" : "FAIL") << '\n';
  std::cout << "tokenizer_bpe_roundtrip_test: " << (ok22 ? "PASS" : "FAIL") << '\n';
  std::cout << "tokenizer_checkpoint_roundtrip_test: " << (ok23 ? "PASS" : "FAIL") << '\n';
  std::cout << "session_roundtrip_test: " << (ok24 ? "PASS" : "FAIL") << '\n';
  std::cout << "session_jsonl_import_roundtrip_test: " << (ok25 ? "PASS" : "FAIL") << '\n';
  std::cout << "session_split_contract_test: " << (ok26 ? "PASS" : "FAIL") << '\n';
  std::cout << "chat_history_prompt_test: " << (ok27 ? "PASS" : "FAIL") << '\n';
  std::cout << "chat_history_trimming_test: " << (ok28 ? "PASS" : "FAIL") << '\n';
  std::cout << "chat_history_compression_test: " << (ok29 ? "PASS" : "FAIL") << '\n';
  std::cout << "cpu_metal_training_parity_test: " << (ok30 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_adamw_update_test: " << (ok31 ? "PASS" : "FAIL") << '\n';
  std::cout << "backend_adamw_multi_param_parity_test: " << (ok32 ? "PASS" : "FAIL") << '\n';
  std::cout << "cpu_metal_staged_parity_trace_test: " << (ok33 ? "PASS" : "FAIL") << '\n';
  std::cout << "metal_checkpoint_cpu_generation_interop_test: " << (ok34 ? "PASS" : "FAIL") << '\n';
  std::cout << "cuda_checkpoint_cpu_generation_interop_test: " << (ok35 ? "PASS" : "FAIL") << '\n';
  std::cout << "cuda_feedforward_forward_test: " << (ok36 ? "PASS" : "FAIL") << '\n';
  std::cout << "cuda_feedforward_backward_test: " << (ok37 ? "PASS" : "FAIL") << '\n';
  std::cout << "cuda_cached_linear_backend_test: " << (ok38 ? "PASS" : "FAIL") << '\n';
  return ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8 && ok9 && ok10 && ok11 && ok12 && ok13 && ok14 &&
         ok15 && ok16 && ok17 && ok18 && ok19 && ok20 && ok21 && ok22 && ok23 && ok24 && ok25 && ok26 && ok27 &&
         ok28 && ok29 && ok30 && ok31 && ok32 && ok33 && ok34 && ok35 && ok36 && ok37 && ok38;
}


}  // namespace microgpt

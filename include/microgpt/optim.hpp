#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace microgpt {
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
  BackendKind backend = BackendKind::Cpu;
  bool batch_updates = false;

  void set_backend(BackendKind kind) { backend = kind; }

  void update(std::vector<Parameter*>& params) {
    ++step;
    if (params.empty()) {
      return;
    }
    if (backend == BackendKind::Metal && microgpt_metal_runtime_available() && batch_updates) {
      struct MetalParamUpdate {
        Parameter* param = nullptr;
        BackendBuffer data_buf;
        BackendBuffer grad_buf;
        BackendBuffer m_buf;
        BackendBuffer v_buf;
      };

      struct CommandBatchGuard {
        bool active = false;
        CommandBatchGuard() {
          microgpt_metal_command_batch_begin();
          active = true;
        }
        void finish() {
          if (active) {
            if (!microgpt_metal_command_batch_end()) {
              throw std::runtime_error("Metal command batch failed");
            }
            active = false;
          }
        }
        ~CommandBatchGuard() {
          if (active) {
            (void)microgpt_metal_command_batch_end();
          }
        }
      };

      std::vector<MetalParamUpdate> updates;
      updates.reserve(params.size());
      CommandBatchGuard batch_guard;
      for (Parameter* p : params) {
        if (p->m.size() != p->data.size()) {
          p->m.assign(p->data.size(), 0.0f);
          p->v.assign(p->data.size(), 0.0f);
        }
        updates.push_back(MetalParamUpdate{});
        MetalParamUpdate& update = updates.back();
        update.param = p;
        update.data_buf = BackendBuffer(BackendKind::Metal);
        update.grad_buf = BackendBuffer(BackendKind::Metal);
        update.m_buf = BackendBuffer(BackendKind::Metal);
        update.v_buf = BackendBuffer(BackendKind::Metal);
        update.data_buf.resize(p->data.size());
        update.grad_buf.resize(p->grad.size());
        update.m_buf.resize(p->m.size());
        update.v_buf.resize(p->v.size());
        update.data_buf.host = p->data;
        update.grad_buf.host = p->grad;
        update.m_buf.host = p->m;
        update.v_buf.host = p->v;
        update.data_buf.host_dirty = true;
        update.grad_buf.host_dirty = true;
        update.m_buf.host_dirty = true;
        update.v_buf.host_dirty = true;
        update.data_buf.upload();
        update.grad_buf.upload();
        update.m_buf.upload();
        update.v_buf.upload();
        bool ok = microgpt_metal_adamw_update(update.data_buf.device_contents(), update.grad_buf.device_contents(),
                                              update.m_buf.device_contents(), update.v_buf.device_contents(),
                                              static_cast<int>(p->data.size()), lr, beta1, beta2, eps,
                                              weight_decay, step, p->decay);
        if (!ok) {
          throw std::runtime_error("Metal AdamW update failed");
        }
        update.data_buf.device_dirty = true;
        update.m_buf.device_dirty = true;
        update.v_buf.device_dirty = true;
      }
      batch_guard.finish();
      for (MetalParamUpdate& update : updates) {
        update.data_buf.download();
        update.m_buf.download();
        update.v_buf.download();
        update.param->data = std::move(update.data_buf.host);
        update.param->m = std::move(update.m_buf.host);
        update.param->v = std::move(update.v_buf.host);
        update.param->version += 1;
      }
      return;
    }
    if (backend == BackendKind::Cuda && microgpt_cuda_runtime_available()) {
      for (Parameter* p : params) {
        if (p->m.size() != p->data.size()) {
          p->m.assign(p->data.size(), 0.0f);
          p->v.assign(p->data.size(), 0.0f);
        }
        bool ok = microgpt_cuda_adamw_update(p->data.data(), p->grad.data(), p->m.data(), p->v.data(),
                                             static_cast<int>(p->data.size()), lr, beta1, beta2, eps, weight_decay,
                                             step, p->decay);
        if (!ok) {
          throw std::runtime_error("CUDA AdamW update failed");
        }
        record_backend_accelerated_op(backend);
        p->version += 1;
      }
      return;
    }
    record_backend_cpu_fallback_op(backend);
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
      p->version += 1;
    }
  }
};

}  // namespace microgpt

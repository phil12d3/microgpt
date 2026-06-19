#pragma once

#include <stdexcept>
#include <string>
#include <vector>

extern "C" void* microgpt_metal_buffer_create(size_t bytes);
extern "C" void microgpt_metal_buffer_destroy(void* buffer);
extern "C" bool microgpt_metal_buffer_write(void* buffer, const void* data, size_t bytes);
extern "C" bool microgpt_metal_buffer_read(void* buffer, void* data, size_t bytes);
extern "C" void* microgpt_metal_buffer_contents(void* buffer);
extern "C" bool microgpt_metal_runtime_available();
extern "C" bool microgpt_metal_runtime_compiled();
extern "C" const char* microgpt_metal_runtime_device_name();
extern "C" const char* microgpt_compiled_acceleration_backend();
extern "C" bool microgpt_metal_linear_forward(const float* x, const float* w, const float* bias, float* y, int rows,
                                               int in_features, int out_features, bool has_bias);
extern "C" bool microgpt_metal_linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw,
                                                float* db, int rows, int in_features, int out_features,
                                                bool has_bias);
extern "C" bool microgpt_metal_linear_forward_buffers(void* x, void* w, void* bias, void* y, int rows, int in_features,
                                                       int out_features, bool has_bias);
extern "C" bool microgpt_metal_linear_backward_buffers(void* x, void* w, void* dy, void* dx, void* dw, void* db,
                                                        int rows, int in_features, int out_features, bool has_bias);
extern "C" bool microgpt_metal_linear_forward_backward_buffers(void* x, void* w, void* bias, void* y, void* dy,
                                                                void* dx, void* dw, void* db, int rows,
                                                                int in_features, int out_features, bool has_bias);
extern "C" bool microgpt_metal_linear_forward_backward_repeat_buffers(void* x, void* w, void* bias, void* y, void* dy,
                                                                       void* dx, void* dw, void* db, int rows,
                                                                       int in_features, int out_features,
                                                                       bool has_bias, int iterations);
extern "C" bool microgpt_metal_gelu_forward(const float* x, float* y, int n);
extern "C" bool microgpt_metal_gelu_backward(const float* x, const float* dy, float* dx, int n);
extern "C" bool microgpt_metal_layernorm_forward(const float* x, const float* gamma, const float* beta, float* y,
                                                  float* mean, float* inv_std, float* xhat, int rows, int dim,
                                                  float eps);
extern "C" bool microgpt_metal_layernorm_backward(const float* x, const float* gamma, const float* dy,
                                                   const float* mean, const float* inv_std, const float* xhat,
                                                   float* dx, float* dgamma, float* dbeta, int rows, int dim);
extern "C" bool microgpt_metal_feedforward_forward(const float* x, const float* w1, const float* b1, const float* w2,
                                                    const float* b2, float* pre, float* hidden, float* y, int rows,
                                                    int d_model, int d_ff, bool has_b1, bool has_b2);
extern "C" bool microgpt_metal_feedforward_backward(const float* x, const float* pre, const float* hidden,
                                                     const float* w1, const float* w2, const float* dy, float* dx,
                                                     float* dw1, float* db1, float* dw2, float* db2, int rows,
                                                     int d_model, int d_ff, bool has_b1, bool has_b2);
extern "C" bool microgpt_metal_adamw_update(float* data, float* grad, float* m, float* v, int n, float lr, float beta1,
                                             float beta2, float eps, float weight_decay, int step, bool decay);
extern "C" void microgpt_metal_command_batch_begin();
extern "C" bool microgpt_metal_command_batch_end();
extern "C" size_t microgpt_metal_command_buffer_submissions();
extern "C" void microgpt_metal_reset_command_buffer_submissions();

namespace microgpt {

enum class BackendKind {
  Cpu,
  Metal,
  Cuda,
};

inline std::string backend_name(BackendKind backend) {
  switch (backend) {
    case BackendKind::Cpu:
      return "cpu";
    case BackendKind::Metal:
      return "metal";
    case BackendKind::Cuda:
      return "cuda";
  }
  return "unknown";
}

inline BackendKind parse_backend_kind(const std::string& name) {
  if (name.empty() || name == "cpu") {
    return BackendKind::Cpu;
  }
  if (name == "metal") {
    return BackendKind::Metal;
  }
  if (name == "cuda") {
    return BackendKind::Cuda;
  }
  throw std::runtime_error("--backend must be cpu, metal, or cuda");
}

inline bool backend_available(BackendKind backend) {
  switch (backend) {
    case BackendKind::Cpu:
      return true;
    case BackendKind::Metal:
      return microgpt_metal_runtime_compiled();
    case BackendKind::Cuda:
      return false;
  }
  return false;
}

inline std::string backend_detail(BackendKind backend) {
  if (backend == BackendKind::Metal) {
    if (microgpt_metal_runtime_available()) {
      const char* name = microgpt_metal_runtime_device_name();
      if (name && name[0] != '\0') {
        return name;
      }
    }
    if (microgpt_metal_runtime_compiled()) {
      return "compiled, no runtime device detected; CPU fallback only";
    }
    return "not compiled";
  }
  if (backend == BackendKind::Cpu) {
    return "portable scalar reference backend";
  }
  if (backend == BackendKind::Cuda) {
    return "not compiled or unavailable";
  }
  return "unavailable";
}

inline void require_backend_available(BackendKind backend) {
  if (backend_available(backend)) {
    return;
  }
  if (backend == BackendKind::Metal) {
    throw std::runtime_error("Metal backend requested but unavailable; run `make BACKEND=metal all` and `make deps`");
  }
  if (backend == BackendKind::Cuda) {
    throw std::runtime_error("CUDA backend requested but unavailable on this build");
  }
  throw std::runtime_error("backend unavailable: " + backend_name(backend));
}

inline std::vector<BackendKind> known_backends() {
  return {BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda};
}

}  // namespace microgpt

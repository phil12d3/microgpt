#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>

namespace {

bool cuda_ok(cudaError_t status) { return status == cudaSuccess; }

int cuda_block_count(int n, int block_size = 256) { return (n + block_size - 1) / block_size; }

__device__ float gelu_device(float x) {
  const float c = sqrtf(2.0f / 3.14159265358979323846f);
  float x3 = x * x * x;
  float t = tanhf(c * (x + 0.044715f * x3));
  return 0.5f * x * (1.0f + t);
}

__device__ float gelu_derivative_device(float x) {
  const float c = sqrtf(2.0f / 3.14159265358979323846f);
  float x2 = x * x;
  float x3 = x2 * x;
  float u = c * (x + 0.044715f * x3);
  float t = tanhf(u);
  float sech2 = 1.0f - t * t;
  float term = c * (1.0f + 3.0f * 0.044715f * x2);
  return 0.5f * (1.0f + t) + 0.5f * x * sech2 * term;
}

__global__ void linear_forward_kernel(const float* x, const float* w, const float* bias, float* y, int rows,
                                      int in_features, int out_features, bool has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * out_features;
  if (idx >= total) {
    return;
  }
  int r = idx / out_features;
  int o = idx % out_features;
  float sum = has_bias ? bias[o] : 0.0f;
  for (int i = 0; i < in_features; ++i) {
    sum += x[r * in_features + i] * w[i * out_features + o];
  }
  y[idx] = sum;
}

__global__ void linear_backward_dx_kernel(const float* dy, const float* w, float* dx, int rows, int in_features,
                                          int out_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * in_features;
  if (idx >= total) {
    return;
  }
  int r = idx / in_features;
  int i = idx % in_features;
  float sum = 0.0f;
  for (int o = 0; o < out_features; ++o) {
    sum += dy[r * out_features + o] * w[i * out_features + o];
  }
  dx[idx] = sum;
}

__global__ void linear_backward_dw_kernel(const float* x, const float* dy, float* dw, int rows, int in_features,
                                          int out_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = in_features * out_features;
  if (idx >= total) {
    return;
  }
  int i = idx / out_features;
  int o = idx % out_features;
  float sum = 0.0f;
  for (int r = 0; r < rows; ++r) {
    sum += x[r * in_features + i] * dy[r * out_features + o];
  }
  dw[idx] = sum;
}

__global__ void linear_backward_db_kernel(const float* dy, float* db, int rows, int out_features) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_features) {
    return;
  }
  float sum = 0.0f;
  for (int r = 0; r < rows; ++r) {
    sum += dy[r * out_features + o];
  }
  db[o] = sum;
}

__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = gelu_device(x[idx]);
  }
}

__global__ void gelu_backward_kernel(const float* x, const float* dy, float* dx, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dx[idx] = dy[idx] * gelu_derivative_device(x[idx]);
  }
}

__global__ void layernorm_forward_kernel(const float* x, const float* gamma, const float* beta, float* y, float* mean,
                                         float* inv_std, float* xhat, int rows, int dim, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xin = x + row * dim;
  float mu = 0.0f;
  for (int i = 0; i < dim; ++i) {
    mu += xin[i];
  }
  mu /= static_cast<float>(dim);
  float var = 0.0f;
  for (int i = 0; i < dim; ++i) {
    float c = xin[i] - mu;
    var += c * c;
  }
  var /= static_cast<float>(dim);
  float inv = rsqrtf(var + eps);
  mean[row] = mu;
  inv_std[row] = inv;
  for (int i = 0; i < dim; ++i) {
    float xn = (xin[i] - mu) * inv;
    xhat[row * dim + i] = xn;
    y[row * dim + i] = xn * gamma[i] + beta[i];
  }
}

__global__ void layernorm_backward_kernel(const float* x, const float* gamma, const float* dy, const float* mean,
                                          const float* inv_std, const float* xhat, float* dx, float* dgamma,
                                          float* dbeta, int rows, int dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xin = x + row * dim;
  const float* dyi = dy + row * dim;
  float sum_dxhat = 0.0f;
  float sum_dxhat_xmu = 0.0f;
  float mean_row = mean[row];
  float inv = inv_std[row];
  for (int i = 0; i < dim; ++i) {
    float xn = xhat[row * dim + i];
    atomicAdd(dgamma + i, dyi[i] * xn);
    atomicAdd(dbeta + i, dyi[i]);
    float dxh = dyi[i] * gamma[i];
    float xmu = xin[i] - mean_row;
    sum_dxhat += dxh;
    sum_dxhat_xmu += dxh * xmu;
  }
  float dvar = -0.5f * inv * inv * inv * sum_dxhat_xmu;
  float dmu = -inv * sum_dxhat;
  float mean_xmu = 0.0f;
  for (int i = 0; i < dim; ++i) {
    mean_xmu += xin[i] - mean_row;
  }
  dmu += dvar * (-2.0f * mean_xmu / static_cast<float>(dim));
  for (int i = 0; i < dim; ++i) {
    float xmu = xin[i] - mean_row;
    float dxh = dyi[i] * gamma[i];
    dx[row * dim + i] = dxh * inv + dvar * 2.0f * xmu / static_cast<float>(dim) + dmu / static_cast<float>(dim);
  }
}

__global__ void adamw_update_kernel(float* data, float* grad, float* m, float* v, int n, float lr, float beta1,
                                    float beta2, float eps, float weight_decay, int step, bool decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float g = grad[idx];
  m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
  v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
  float mhat = m[idx] / (1.0f - powf(beta1, static_cast<float>(step)));
  float vhat = v[idx] / (1.0f - powf(beta2, static_cast<float>(step)));
  float update = mhat / (sqrtf(vhat) + eps);
  if (decay) {
    update += weight_decay * data[idx];
  }
  data[idx] -= lr * update;
}

bool allocate_and_copy(const float* host, float** device, size_t count) {
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(device), count * sizeof(float)))) {
    return false;
  }
  if (!cuda_ok(cudaMemcpy(*device, host, count * sizeof(float), cudaMemcpyHostToDevice))) {
    cudaFree(*device);
    *device = nullptr;
    return false;
  }
  return true;
}

bool copy_and_free(float* host, float* device, size_t count) {
  bool ok = cuda_ok(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(device);
  return ok;
}

bool cuda_runtime_has_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

size_t cuda_submission_count = 0;

bool cuda_finish() {
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize());
  if (ok) {
    ++cuda_submission_count;
  }
  return ok;
}

}  // namespace

extern "C" void* microgpt_cuda_buffer_create(size_t bytes) {
  void* buffer = nullptr;
  if (!cuda_ok(cudaMalloc(&buffer, bytes))) {
    return nullptr;
  }
  return buffer;
}

extern "C" void microgpt_cuda_buffer_destroy(void* buffer) {
  if (buffer != nullptr) {
    cudaFree(buffer);
  }
}

extern "C" bool microgpt_cuda_buffer_write(void* buffer, const void* data, size_t bytes) {
  return buffer != nullptr && data != nullptr && cuda_ok(cudaMemcpy(buffer, data, bytes, cudaMemcpyHostToDevice));
}

extern "C" bool microgpt_cuda_buffer_read(void* buffer, void* data, size_t bytes) {
  return buffer != nullptr && data != nullptr && cuda_ok(cudaMemcpy(data, buffer, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void* microgpt_cuda_buffer_contents(void* buffer) { return buffer; }

extern "C" bool microgpt_cuda_runtime_available() { return cuda_runtime_has_device(); }

extern "C" bool microgpt_cuda_runtime_compiled() { return true; }

extern "C" const char* microgpt_cuda_runtime_device_name() {
  static std::string name;
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    name.clear();
  } else {
    name = prop.name;
  }
  return name.c_str();
}

extern "C" const char* microgpt_compiled_acceleration_backend() { return "cuda"; }

extern "C" void* microgpt_metal_buffer_create(size_t) { return nullptr; }
extern "C" void microgpt_metal_buffer_destroy(void*) {}
extern "C" bool microgpt_metal_buffer_write(void*, const void*, size_t) { return false; }
extern "C" bool microgpt_metal_buffer_read(void*, void*, size_t) { return false; }
extern "C" void* microgpt_metal_buffer_contents(void*) { return nullptr; }
extern "C" bool microgpt_metal_runtime_available() { return false; }
extern "C" bool microgpt_metal_runtime_compiled() { return false; }
extern "C" const char* microgpt_metal_runtime_device_name() { return ""; }
extern "C" bool microgpt_metal_linear_forward(const float*, const float*, const float*, float*, int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_metal_linear_backward(const float*, const float*, const float*, float*, float*, float*, int,
                                                int, int, bool) {
  return false;
}
extern "C" bool microgpt_metal_linear_forward_buffers(void*, void*, void*, void*, int, int, int, bool) { return false; }
extern "C" bool microgpt_metal_linear_backward_buffers(void*, void*, void*, void*, void*, void*, int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_metal_linear_forward_backward_buffers(void*, void*, void*, void*, void*, void*, void*, void*,
                                                                int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_metal_linear_forward_backward_repeat_buffers(void*, void*, void*, void*, void*, void*, void*,
                                                                       void*, int, int, int, bool, int) {
  return false;
}
extern "C" bool microgpt_metal_gelu_forward(const float*, float*, int) { return false; }
extern "C" bool microgpt_metal_gelu_backward(const float*, const float*, float*, int) { return false; }
extern "C" bool microgpt_metal_layernorm_forward(const float*, const float*, const float*, float*, float*, float*,
                                                  float*, int, int, float) {
  return false;
}
extern "C" bool microgpt_metal_layernorm_backward(const float*, const float*, const float*, const float*, const float*,
                                                   const float*, float*, float*, float*, int, int) {
  return false;
}
extern "C" bool microgpt_metal_feedforward_forward(const float*, const float*, const float*, const float*, const float*,
                                                    float*, float*, float*, int, int, int, bool, bool) {
  return false;
}
extern "C" bool microgpt_metal_feedforward_backward(const float*, const float*, const float*, const float*, const float*,
                                                     const float*, float*, float*, float*, float*, float*, int, int,
                                                     int, bool, bool) {
  return false;
}
extern "C" bool microgpt_metal_adamw_update(float*, float*, float*, float*, int, float, float, float, float, float, int,
                                             bool) {
  return false;
}
extern "C" void microgpt_metal_command_batch_begin() {}
extern "C" bool microgpt_metal_command_batch_end() { return false; }
extern "C" size_t microgpt_metal_command_buffer_submissions() { return 0; }
extern "C" void microgpt_metal_reset_command_buffer_submissions() {}

extern "C" bool microgpt_cuda_linear_forward_buffers(void* x, void* w, void* bias, void* y, int rows, int in_features,
                                                      int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || y == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (has_bias && bias == nullptr)) {
    return false;
  }
  int total = rows * out_features;
  linear_forward_kernel<<<cuda_block_count(total), 256>>>(static_cast<const float*>(x), static_cast<const float*>(w),
                                                          static_cast<const float*>(bias), static_cast<float*>(y), rows,
                                                          in_features, out_features, has_bias);
  return cuda_finish();
}

extern "C" bool microgpt_cuda_linear_backward_buffers(void* x, void* w, void* dy, void* dx, void* dw, void* db,
                                                       int rows, int in_features, int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || dy == nullptr || dx == nullptr || dw == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || (has_bias && db == nullptr)) {
    return false;
  }
  int dx_total = rows * in_features;
  int dw_total = in_features * out_features;
  linear_backward_dx_kernel<<<cuda_block_count(dx_total), 256>>>(static_cast<const float*>(dy),
                                                                 static_cast<const float*>(w), static_cast<float*>(dx),
                                                                 rows, in_features, out_features);
  linear_backward_dw_kernel<<<cuda_block_count(dw_total), 256>>>(static_cast<const float*>(x),
                                                                 static_cast<const float*>(dy), static_cast<float*>(dw),
                                                                 rows, in_features, out_features);
  if (has_bias) {
    linear_backward_db_kernel<<<cuda_block_count(out_features), 256>>>(static_cast<const float*>(dy),
                                                                       static_cast<float*>(db), rows, out_features);
  }
  return cuda_finish();
}

extern "C" bool microgpt_cuda_linear_forward_backward_buffers(void* x, void* w, void* bias, void* y, void* dy,
                                                              void* dx, void* dw, void* db, int rows, int in_features,
                                                              int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || y == nullptr || dy == nullptr || dx == nullptr || dw == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || (has_bias && (bias == nullptr || db == nullptr))) {
    return false;
  }
  int forward_total = rows * out_features;
  int dx_total = rows * in_features;
  int dw_total = in_features * out_features;
  linear_forward_kernel<<<cuda_block_count(forward_total), 256>>>(
      static_cast<const float*>(x), static_cast<const float*>(w), static_cast<const float*>(bias),
      static_cast<float*>(y), rows, in_features, out_features, has_bias);
  linear_backward_dx_kernel<<<cuda_block_count(dx_total), 256>>>(static_cast<const float*>(dy),
                                                                 static_cast<const float*>(w), static_cast<float*>(dx),
                                                                 rows, in_features, out_features);
  linear_backward_dw_kernel<<<cuda_block_count(dw_total), 256>>>(static_cast<const float*>(x),
                                                                 static_cast<const float*>(dy), static_cast<float*>(dw),
                                                                 rows, in_features, out_features);
  if (has_bias) {
    linear_backward_db_kernel<<<cuda_block_count(out_features), 256>>>(static_cast<const float*>(dy),
                                                                       static_cast<float*>(db), rows, out_features);
  }
  return cuda_finish();
}

extern "C" bool microgpt_cuda_linear_forward_backward_repeat_buffers(void* x, void* w, void* bias, void* y, void* dy,
                                                                     void* dx, void* dw, void* db, int rows,
                                                                     int in_features, int out_features, bool has_bias,
                                                                     int iterations) {
  if (x == nullptr || w == nullptr || y == nullptr || dy == nullptr || dx == nullptr || dw == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || iterations <= 0 || (has_bias && (bias == nullptr || db == nullptr))) {
    return false;
  }
  int forward_total = rows * out_features;
  int dx_total = rows * in_features;
  int dw_total = in_features * out_features;
  for (int iter = 0; iter < iterations; ++iter) {
    linear_forward_kernel<<<cuda_block_count(forward_total), 256>>>(
        static_cast<const float*>(x), static_cast<const float*>(w), static_cast<const float*>(bias),
        static_cast<float*>(y), rows, in_features, out_features, has_bias);
    linear_backward_dx_kernel<<<cuda_block_count(dx_total), 256>>>(static_cast<const float*>(dy),
                                                                   static_cast<const float*>(w), static_cast<float*>(dx),
                                                                   rows, in_features, out_features);
    linear_backward_dw_kernel<<<cuda_block_count(dw_total), 256>>>(
        static_cast<const float*>(x), static_cast<const float*>(dy), static_cast<float*>(dw), rows, in_features,
        out_features);
    if (has_bias) {
      linear_backward_db_kernel<<<cuda_block_count(out_features), 256>>>(static_cast<const float*>(dy),
                                                                         static_cast<float*>(db), rows, out_features);
    }
  }
  return cuda_finish();
}

extern "C" bool microgpt_cuda_linear_forward(const float* x, const float* w, const float* bias, float* y, int rows,
                                              int in_features, int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || y == nullptr || (has_bias && bias == nullptr)) {
    return false;
  }
  float* dx = nullptr;
  float* dw = nullptr;
  float* db = nullptr;
  float* dy = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(in_features);
  size_t w_count = static_cast<size_t>(in_features) * static_cast<size_t>(out_features);
  size_t y_count = static_cast<size_t>(rows) * static_cast<size_t>(out_features);
  if (!allocate_and_copy(x, &dx, x_count) || !allocate_and_copy(w, &dw, w_count)) {
    cudaFree(dx);
    cudaFree(dw);
    return false;
  }
  if (has_bias && !allocate_and_copy(bias, &db, static_cast<size_t>(out_features))) {
    cudaFree(dx);
    cudaFree(dw);
    return false;
  }
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dy), y_count * sizeof(float)))) {
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(db);
    return false;
  }
  bool ok = microgpt_cuda_linear_forward_buffers(dx, dw, db, dy, rows, in_features, out_features, has_bias) &&
            copy_and_free(y, dy, y_count);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(db);
  return ok;
}

extern "C" bool microgpt_cuda_linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw,
                                               float* db, int rows, int in_features, int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || dy == nullptr || dx == nullptr || dw == nullptr || (has_bias && db == nullptr)) {
    return false;
  }
  float* x_dev = nullptr;
  float* w_dev = nullptr;
  float* dy_dev = nullptr;
  float* dx_dev = nullptr;
  float* dw_dev = nullptr;
  float* db_dev = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(in_features);
  size_t w_count = static_cast<size_t>(in_features) * static_cast<size_t>(out_features);
  size_t dy_count = static_cast<size_t>(rows) * static_cast<size_t>(out_features);
  if (!allocate_and_copy(x, &x_dev, x_count) || !allocate_and_copy(w, &w_dev, w_count) ||
      !allocate_and_copy(dy, &dy_dev, dy_count) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dx_dev), x_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dw_dev), w_count * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(w_dev);
    cudaFree(dy_dev);
    cudaFree(dx_dev);
    cudaFree(dw_dev);
    return false;
  }
  if (has_bias && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&db_dev), static_cast<size_t>(out_features) * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(w_dev);
    cudaFree(dy_dev);
    cudaFree(dx_dev);
    cudaFree(dw_dev);
    return false;
  }
  bool ok = microgpt_cuda_linear_backward_buffers(x_dev, w_dev, dy_dev, dx_dev, dw_dev, db_dev, rows, in_features,
                                                  out_features, has_bias) &&
            cuda_ok(cudaMemcpy(dx, dx_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(dw, dw_dev, w_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            (!has_bias || cuda_ok(cudaMemcpy(db, db_dev, static_cast<size_t>(out_features) * sizeof(float),
                                             cudaMemcpyDeviceToHost)));
  cudaFree(x_dev);
  cudaFree(w_dev);
  cudaFree(dy_dev);
  cudaFree(dx_dev);
  cudaFree(dw_dev);
  cudaFree(db_dev);
  return ok;
}

extern "C" bool microgpt_cuda_gelu_forward(const float* x, float* y, int n) {
  if (x == nullptr || y == nullptr || n < 0) {
    return false;
  }
  float* x_dev = nullptr;
  float* y_dev = nullptr;
  if (!allocate_and_copy(x, &x_dev, static_cast<size_t>(n)) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&y_dev), static_cast<size_t>(n) * sizeof(float)))) {
    cudaFree(x_dev);
    return false;
  }
  gelu_forward_kernel<<<cuda_block_count(n), 256>>>(x_dev, y_dev, n);
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
            copy_and_free(y, y_dev, static_cast<size_t>(n));
  cudaFree(x_dev);
  return ok;
}

extern "C" bool microgpt_cuda_gelu_backward(const float* x, const float* dy, float* dx, int n) {
  if (x == nullptr || dy == nullptr || dx == nullptr || n < 0) {
    return false;
  }
  float* x_dev = nullptr;
  float* dy_dev = nullptr;
  float* dx_dev = nullptr;
  if (!allocate_and_copy(x, &x_dev, static_cast<size_t>(n)) || !allocate_and_copy(dy, &dy_dev, static_cast<size_t>(n)) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dx_dev), static_cast<size_t>(n) * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(dy_dev);
    return false;
  }
  gelu_backward_kernel<<<cuda_block_count(n), 256>>>(x_dev, dy_dev, dx_dev, n);
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
            copy_and_free(dx, dx_dev, static_cast<size_t>(n));
  cudaFree(x_dev);
  cudaFree(dy_dev);
  return ok;
}

extern "C" bool microgpt_cuda_layernorm_forward(const float* x, const float* gamma, const float* beta, float* y,
                                                 float* mean, float* inv_std, float* xhat, int rows, int dim,
                                                 float eps) {
  if (x == nullptr || gamma == nullptr || beta == nullptr || y == nullptr || mean == nullptr || inv_std == nullptr ||
      xhat == nullptr || rows <= 0 || dim <= 0) {
    return false;
  }
  float *x_dev = nullptr, *gamma_dev = nullptr, *beta_dev = nullptr, *y_dev = nullptr, *mean_dev = nullptr;
  float *inv_dev = nullptr, *xhat_dev = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(dim);
  if (!allocate_and_copy(x, &x_dev, x_count) || !allocate_and_copy(gamma, &gamma_dev, static_cast<size_t>(dim)) ||
      !allocate_and_copy(beta, &beta_dev, static_cast<size_t>(dim)) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&y_dev), x_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&mean_dev), static_cast<size_t>(rows) * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&inv_dev), static_cast<size_t>(rows) * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&xhat_dev), x_count * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(gamma_dev);
    cudaFree(beta_dev);
    cudaFree(y_dev);
    cudaFree(mean_dev);
    cudaFree(inv_dev);
    cudaFree(xhat_dev);
    return false;
  }
  layernorm_forward_kernel<<<cuda_block_count(rows), 256>>>(x_dev, gamma_dev, beta_dev, y_dev, mean_dev, inv_dev,
                                                            xhat_dev, rows, dim, eps);
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
            cuda_ok(cudaMemcpy(y, y_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(mean, mean_dev, static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(inv_std, inv_dev, static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(xhat, xhat_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(x_dev);
  cudaFree(gamma_dev);
  cudaFree(beta_dev);
  cudaFree(y_dev);
  cudaFree(mean_dev);
  cudaFree(inv_dev);
  cudaFree(xhat_dev);
  return ok;
}

extern "C" bool microgpt_cuda_layernorm_backward(const float* x, const float* gamma, const float* dy,
                                                  const float* mean, const float* inv_std, const float* xhat,
                                                  float* dx, float* dgamma, float* dbeta, int rows, int dim) {
  if (x == nullptr || gamma == nullptr || dy == nullptr || mean == nullptr || inv_std == nullptr || xhat == nullptr ||
      dx == nullptr || dgamma == nullptr || dbeta == nullptr || rows <= 0 || dim <= 0) {
    return false;
  }
  float *x_dev = nullptr, *gamma_dev = nullptr, *dy_dev = nullptr, *mean_dev = nullptr, *inv_dev = nullptr;
  float *xhat_dev = nullptr, *dx_dev = nullptr, *dgamma_dev = nullptr, *dbeta_dev = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(dim);
  if (!allocate_and_copy(x, &x_dev, x_count) || !allocate_and_copy(gamma, &gamma_dev, static_cast<size_t>(dim)) ||
      !allocate_and_copy(dy, &dy_dev, x_count) || !allocate_and_copy(mean, &mean_dev, static_cast<size_t>(rows)) ||
      !allocate_and_copy(inv_std, &inv_dev, static_cast<size_t>(rows)) || !allocate_and_copy(xhat, &xhat_dev, x_count) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dx_dev), x_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dgamma_dev), static_cast<size_t>(dim) * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dbeta_dev), static_cast<size_t>(dim) * sizeof(float))) ||
      !cuda_ok(cudaMemset(dgamma_dev, 0, static_cast<size_t>(dim) * sizeof(float))) ||
      !cuda_ok(cudaMemset(dbeta_dev, 0, static_cast<size_t>(dim) * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(gamma_dev);
    cudaFree(dy_dev);
    cudaFree(mean_dev);
    cudaFree(inv_dev);
    cudaFree(xhat_dev);
    cudaFree(dx_dev);
    cudaFree(dgamma_dev);
    cudaFree(dbeta_dev);
    return false;
  }
  layernorm_backward_kernel<<<cuda_block_count(rows), 256>>>(x_dev, gamma_dev, dy_dev, mean_dev, inv_dev, xhat_dev,
                                                             dx_dev, dgamma_dev, dbeta_dev, rows, dim);
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
            cuda_ok(cudaMemcpy(dx, dx_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(dgamma, dgamma_dev, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(dbeta, dbeta_dev, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(x_dev);
  cudaFree(gamma_dev);
  cudaFree(dy_dev);
  cudaFree(mean_dev);
  cudaFree(inv_dev);
  cudaFree(xhat_dev);
  cudaFree(dx_dev);
  cudaFree(dgamma_dev);
  cudaFree(dbeta_dev);
  return ok;
}

extern "C" bool microgpt_cuda_feedforward_forward(const float* x, const float* w1, const float* b1, const float* w2,
                                                   const float* b2, float* pre, float* hidden, float* y, int rows,
                                                   int d_model, int d_ff, bool has_b1, bool has_b2) {
  if (x == nullptr || w1 == nullptr || w2 == nullptr || pre == nullptr || hidden == nullptr || y == nullptr ||
      rows <= 0 || d_model <= 0 || d_ff <= 0 || (has_b1 && b1 == nullptr) || (has_b2 && b2 == nullptr)) {
    return false;
  }
  float *x_dev = nullptr, *w1_dev = nullptr, *b1_dev = nullptr, *w2_dev = nullptr, *b2_dev = nullptr;
  float *pre_dev = nullptr, *hidden_dev = nullptr, *y_dev = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(d_model);
  size_t pre_count = static_cast<size_t>(rows) * static_cast<size_t>(d_ff);
  size_t y_count = x_count;
  if (!allocate_and_copy(x, &x_dev, x_count) || !allocate_and_copy(w1, &w1_dev, static_cast<size_t>(d_model) * d_ff) ||
      !allocate_and_copy(w2, &w2_dev, static_cast<size_t>(d_ff) * d_model) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pre_dev), pre_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&hidden_dev), pre_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&y_dev), y_count * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(w1_dev);
    cudaFree(w2_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(y_dev);
    return false;
  }
  if (has_b1 && !allocate_and_copy(b1, &b1_dev, static_cast<size_t>(d_ff))) {
    cudaFree(x_dev);
    cudaFree(w1_dev);
    cudaFree(w2_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(y_dev);
    return false;
  }
  if (has_b2 && !allocate_and_copy(b2, &b2_dev, static_cast<size_t>(d_model))) {
    cudaFree(x_dev);
    cudaFree(w1_dev);
    cudaFree(b1_dev);
    cudaFree(w2_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(y_dev);
    return false;
  }
  linear_forward_kernel<<<cuda_block_count(static_cast<int>(pre_count)), 256>>>(x_dev, w1_dev, b1_dev, pre_dev, rows,
                                                                                d_model, d_ff, has_b1);
  gelu_forward_kernel<<<cuda_block_count(static_cast<int>(pre_count)), 256>>>(pre_dev, hidden_dev,
                                                                              static_cast<int>(pre_count));
  linear_forward_kernel<<<cuda_block_count(static_cast<int>(y_count)), 256>>>(hidden_dev, w2_dev, b2_dev, y_dev, rows,
                                                                               d_ff, d_model, has_b2);
  bool ok = cuda_finish() && cuda_ok(cudaMemcpy(pre, pre_dev, pre_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(hidden, hidden_dev, pre_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(y, y_dev, y_count * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(x_dev);
  cudaFree(w1_dev);
  cudaFree(b1_dev);
  cudaFree(w2_dev);
  cudaFree(b2_dev);
  cudaFree(pre_dev);
  cudaFree(hidden_dev);
  cudaFree(y_dev);
  return ok;
}

extern "C" bool microgpt_cuda_feedforward_backward(const float* x, const float* pre, const float* hidden,
                                                    const float* w1, const float* w2, const float* dy, float* dx,
                                                    float* dw1, float* db1, float* dw2, float* db2, int rows,
                                                    int d_model, int d_ff, bool has_b1, bool has_b2) {
  if (x == nullptr || pre == nullptr || hidden == nullptr || w1 == nullptr || w2 == nullptr || dy == nullptr ||
      dx == nullptr || dw1 == nullptr || dw2 == nullptr || rows <= 0 || d_model <= 0 || d_ff <= 0 ||
      (has_b1 && db1 == nullptr) || (has_b2 && db2 == nullptr)) {
    return false;
  }
  float *x_dev = nullptr, *pre_dev = nullptr, *hidden_dev = nullptr, *w1_dev = nullptr, *w2_dev = nullptr;
  float *dy_dev = nullptr, *dx_dev = nullptr, *dw1_dev = nullptr, *db1_dev = nullptr, *dw2_dev = nullptr;
  float *db2_dev = nullptr, *dhidden_dev = nullptr, *dpre_dev = nullptr;
  size_t x_count = static_cast<size_t>(rows) * static_cast<size_t>(d_model);
  size_t hidden_count = static_cast<size_t>(rows) * static_cast<size_t>(d_ff);
  if (!allocate_and_copy(x, &x_dev, x_count) || !allocate_and_copy(pre, &pre_dev, hidden_count) ||
      !allocate_and_copy(hidden, &hidden_dev, hidden_count) ||
      !allocate_and_copy(w1, &w1_dev, static_cast<size_t>(d_model) * d_ff) ||
      !allocate_and_copy(w2, &w2_dev, static_cast<size_t>(d_ff) * d_model) || !allocate_and_copy(dy, &dy_dev, x_count) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dx_dev), x_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dw1_dev), static_cast<size_t>(d_model) * d_ff * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dw2_dev), static_cast<size_t>(d_ff) * d_model * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dhidden_dev), hidden_count * sizeof(float))) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dpre_dev), hidden_count * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(w1_dev);
    cudaFree(w2_dev);
    cudaFree(dy_dev);
    cudaFree(dx_dev);
    cudaFree(dw1_dev);
    cudaFree(dw2_dev);
    cudaFree(dhidden_dev);
    cudaFree(dpre_dev);
    return false;
  }
  if (has_b1 && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&db1_dev), static_cast<size_t>(d_ff) * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(w1_dev);
    cudaFree(w2_dev);
    cudaFree(dy_dev);
    cudaFree(dx_dev);
    cudaFree(dw1_dev);
    cudaFree(dw2_dev);
    cudaFree(dhidden_dev);
    cudaFree(dpre_dev);
    return false;
  }
  if (has_b2 && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&db2_dev), static_cast<size_t>(d_model) * sizeof(float)))) {
    cudaFree(x_dev);
    cudaFree(pre_dev);
    cudaFree(hidden_dev);
    cudaFree(w1_dev);
    cudaFree(w2_dev);
    cudaFree(dy_dev);
    cudaFree(dx_dev);
    cudaFree(dw1_dev);
    cudaFree(db1_dev);
    cudaFree(dw2_dev);
    cudaFree(dhidden_dev);
    cudaFree(dpre_dev);
    return false;
  }

  linear_backward_dx_kernel<<<cuda_block_count(static_cast<int>(hidden_count)), 256>>>(dy_dev, w2_dev, dhidden_dev,
                                                                                      rows, d_ff, d_model);
  linear_backward_dw_kernel<<<cuda_block_count(d_ff * d_model), 256>>>(hidden_dev, dy_dev, dw2_dev, rows, d_ff,
                                                                       d_model);
  if (has_b2) {
    linear_backward_db_kernel<<<cuda_block_count(d_model), 256>>>(dy_dev, db2_dev, rows, d_model);
  }
  gelu_backward_kernel<<<cuda_block_count(static_cast<int>(hidden_count)), 256>>>(pre_dev, dhidden_dev, dpre_dev,
                                                                                  static_cast<int>(hidden_count));
  linear_backward_dx_kernel<<<cuda_block_count(static_cast<int>(x_count)), 256>>>(dpre_dev, w1_dev, dx_dev, rows,
                                                                                  d_model, d_ff);
  linear_backward_dw_kernel<<<cuda_block_count(d_model * d_ff), 256>>>(x_dev, dpre_dev, dw1_dev, rows, d_model, d_ff);
  if (has_b1) {
    linear_backward_db_kernel<<<cuda_block_count(d_ff), 256>>>(dpre_dev, db1_dev, rows, d_ff);
  }
  bool ok = cuda_finish() && cuda_ok(cudaMemcpy(dx, dx_dev, x_count * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(dw1, dw1_dev, static_cast<size_t>(d_model) * d_ff * sizeof(float),
                               cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(dw2, dw2_dev, static_cast<size_t>(d_ff) * d_model * sizeof(float),
                               cudaMemcpyDeviceToHost)) &&
            (!has_b1 || cuda_ok(cudaMemcpy(db1, db1_dev, static_cast<size_t>(d_ff) * sizeof(float),
                                           cudaMemcpyDeviceToHost))) &&
            (!has_b2 || cuda_ok(cudaMemcpy(db2, db2_dev, static_cast<size_t>(d_model) * sizeof(float),
                                           cudaMemcpyDeviceToHost)));
  cudaFree(x_dev);
  cudaFree(pre_dev);
  cudaFree(hidden_dev);
  cudaFree(w1_dev);
  cudaFree(w2_dev);
  cudaFree(dy_dev);
  cudaFree(dx_dev);
  cudaFree(dw1_dev);
  cudaFree(db1_dev);
  cudaFree(dw2_dev);
  cudaFree(db2_dev);
  cudaFree(dhidden_dev);
  cudaFree(dpre_dev);
  return ok;
}

extern "C" bool microgpt_cuda_adamw_update(float* data, float* grad, float* m, float* v, int n, float lr, float beta1,
                                            float beta2, float eps, float weight_decay, int step, bool decay) {
  if (data == nullptr || grad == nullptr || m == nullptr || v == nullptr || n < 0) {
    return false;
  }
  float *data_dev = nullptr, *grad_dev = nullptr, *m_dev = nullptr, *v_dev = nullptr;
  if (!allocate_and_copy(data, &data_dev, static_cast<size_t>(n)) ||
      !allocate_and_copy(grad, &grad_dev, static_cast<size_t>(n)) || !allocate_and_copy(m, &m_dev, static_cast<size_t>(n)) ||
      !allocate_and_copy(v, &v_dev, static_cast<size_t>(n))) {
    cudaFree(data_dev);
    cudaFree(grad_dev);
    cudaFree(m_dev);
    cudaFree(v_dev);
    return false;
  }
  adamw_update_kernel<<<cuda_block_count(n), 256>>>(data_dev, grad_dev, m_dev, v_dev, n, lr, beta1, beta2, eps,
                                                    weight_decay, step, decay);
  bool ok = cuda_ok(cudaGetLastError()) && cuda_ok(cudaDeviceSynchronize()) &&
            cuda_ok(cudaMemcpy(data, data_dev, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(m, m_dev, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost)) &&
            cuda_ok(cudaMemcpy(v, v_dev, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(data_dev);
  cudaFree(grad_dev);
  cudaFree(m_dev);
  cudaFree(v_dev);
  return ok;
}

extern "C" void microgpt_cuda_command_batch_begin() {}

extern "C" bool microgpt_cuda_command_batch_end() { return true; }

extern "C" size_t microgpt_cuda_command_buffer_submissions() { return cuda_submission_count; }

extern "C" void microgpt_cuda_reset_command_buffer_submissions() { cuda_submission_count = 0; }

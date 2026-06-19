#include <string>

extern "C" void* microgpt_metal_buffer_create(size_t) { return nullptr; }

extern "C" void microgpt_metal_buffer_destroy(void*) {}

extern "C" bool microgpt_metal_buffer_write(void*, const void*, size_t) { return false; }

extern "C" bool microgpt_metal_buffer_read(void*, void*, size_t) { return false; }

extern "C" void* microgpt_metal_buffer_contents(void*) { return nullptr; }

extern "C" bool microgpt_metal_runtime_available() { return false; }

extern "C" bool microgpt_metal_runtime_compiled() { return false; }

extern "C" const char* microgpt_metal_runtime_device_name() { return ""; }

extern "C" const char* microgpt_compiled_acceleration_backend() { return "cpu"; }

extern "C" bool microgpt_metal_linear_forward(const float*, const float*, const float*, float*, int, int, int, bool) {
  return false;
}

extern "C" bool microgpt_metal_linear_backward(const float*, const float*, const float*, float*, float*, float*, int, int,
                                                int, bool) {
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

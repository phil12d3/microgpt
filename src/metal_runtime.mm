#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <cstring>
#include <string>

struct MicrogptLinearShape {
  int rows;
  int in_features;
  int out_features;
  int has_bias;
};

static id<MTLDevice> microgpt_metal_device() {
  static id<MTLDevice> device = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
  });
  return device;
}

static id<MTLCommandQueue> microgpt_metal_command_queue() {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = microgpt_metal_device();
    if (device != nil) {
      queue = [device newCommandQueue];
    }
  });
  return queue;
}

struct MicrogptMetalCommandScope {
  id<MTLCommandBuffer> buffer = nil;
  bool owns = false;
};

static thread_local int microgpt_metal_batch_depth = 0;
static thread_local id<MTLCommandBuffer> microgpt_metal_batch_buffer = nil;
static size_t microgpt_metal_command_buffer_submission_count = 0;

static MicrogptMetalCommandScope microgpt_metal_command_scope() {
  if (microgpt_metal_batch_depth > 0 && microgpt_metal_batch_buffer != nil) {
    return {microgpt_metal_batch_buffer, false};
  }
  id<MTLCommandQueue> queue = microgpt_metal_command_queue();
  if (queue == nil) {
    return {nil, false};
  }
  return {[queue commandBuffer], true};
}

static bool microgpt_metal_command_scope_finish(const MicrogptMetalCommandScope& scope) {
  if (scope.buffer == nil) {
    return false;
  }
  if (scope.owns) {
    [scope.buffer commit];
    [scope.buffer waitUntilCompleted];
    if (scope.buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    ++microgpt_metal_command_buffer_submission_count;
  }
  return true;
}

extern "C" void* microgpt_metal_buffer_create(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return nullptr;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return nullptr;
    }
    return (__bridge void*)buffer;
  }
}

extern "C" void microgpt_metal_buffer_destroy(void* buffer) {
  if (buffer == nullptr) {
    return;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer;
  [metal_buffer release];
}

extern "C" bool microgpt_metal_buffer_write(void* buffer, const void* data, size_t bytes) {
  if (buffer == nullptr || data == nullptr || bytes == 0) {
    return false;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer;
  if ([metal_buffer length] < bytes) {
    return false;
  }
  std::memcpy([metal_buffer contents], data, bytes);
  return true;
}

extern "C" bool microgpt_metal_buffer_read(void* buffer, void* data, size_t bytes) {
  if (buffer == nullptr || data == nullptr || bytes == 0) {
    return false;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer;
  if ([metal_buffer length] < bytes) {
    return false;
  }
  std::memcpy(data, [metal_buffer contents], bytes);
  return true;
}

extern "C" void* microgpt_metal_buffer_contents(void* buffer) {
  if (buffer == nullptr) {
    return nullptr;
  }
  id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)buffer;
  return [metal_buffer contents];
}

extern "C" bool microgpt_metal_runtime_available() {
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    return device != nil;
  }
}

extern "C" bool microgpt_metal_runtime_compiled() { return true; }

extern "C" const char* microgpt_metal_runtime_device_name() {
  static std::string name;
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      name.clear();
    } else {
      name = [[device name] UTF8String];
    }
  }
  return name.c_str();
}

extern "C" const char* microgpt_compiled_acceleration_backend() { return "metal"; }

extern "C" bool microgpt_metal_linear_forward_buffers(void* x_buffer, void* w_buffer, void* bias_buffer, void* y_buffer,
                                                       int rows, int in_features, int out_features, bool has_bias) {
  if (x_buffer == nullptr || w_buffer == nullptr || y_buffer == nullptr || rows <= 0 || in_features <= 0 ||
      out_features <= 0) {
    return false;
  }
  if (has_bias && bias_buffer == nullptr) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }

    static id<MTLLibrary> library = nil;
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LinearShape { int rows; int in_features; int out_features; int has_bias; };\n"
           "kernel void microgpt_linear_forward(device const float* x [[buffer(0)]],\n"
           "                                    device const float* w [[buffer(1)]],\n"
           "                                    device const float* bias [[buffer(2)]],\n"
           "                                    device float* y [[buffer(3)]],\n"
           "                                    constant LinearShape& shape [[buffer(4)]],\n"
           "                                    uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = shape.has_bias ? bias[o] : 0.0f;\n"
           "  for (int i = 0; i < shape.in_features; ++i) {\n"
           "    sum += x[row * shape.in_features + i] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  y[gid] = sum;\n"
           "}\n";
      library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"microgpt_linear_forward"];
      if (function == nil) {
        setup_ok = false;
        return;
      }
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      setup_ok = pipeline != nil;
    });

    if (!setup_ok || pipeline == nil) {
      return false;
    }

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_buffer;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buffer;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)bias_buffer;
    id<MTLBuffer> y_buf = (__bridge id<MTLBuffer>)y_buffer;
    MicrogptLinearShape shape{rows, in_features, out_features, has_bias ? 1 : 0};
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];

    if (x_buf == nil || w_buf == nil || y_buf == nil || shape_buf == nil || (has_bias && b_buf == nil)) {
      return false;
    }

    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    if (queue == nil) {
      return false;
    }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      return false;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:w_buf offset:0 atIndex:1];
    [encoder setBuffer:b_buf offset:0 atIndex:2];
    [encoder setBuffer:y_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];

    NSUInteger total = static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(out_features);
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > total) {
      width = total;
    }
    if (width == 0) {
      return false;
    }
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    return true;
  }
}

extern "C" bool microgpt_metal_linear_forward(const float* x, const float* w, const float* bias, float* y, int rows,
                                               int in_features, int out_features, bool has_bias) {
  if (x == nullptr || w == nullptr || y == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0) {
    return false;
  }
  if (has_bias && bias == nullptr) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    size_t x_bytes = static_cast<size_t>(rows) * static_cast<size_t>(in_features) * sizeof(float);
    size_t w_bytes = static_cast<size_t>(in_features) * static_cast<size_t>(out_features) * sizeof(float);
    size_t y_bytes = static_cast<size_t>(rows) * static_cast<size_t>(out_features) * sizeof(float);
    size_t b_bytes = has_bias ? static_cast<size_t>(out_features) * sizeof(float) : sizeof(float);
    float zero_bias = 0.0f;
    id<MTLBuffer> x_buf =
        [[device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> w_buf =
        [[device newBufferWithBytes:w length:w_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> b_buf =
        [[device newBufferWithBytes:(has_bias ? bias : &zero_bias) length:b_bytes options:MTLResourceStorageModeShared]
            autorelease];
    id<MTLBuffer> y_buf = [[device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || w_buf == nil || b_buf == nil || y_buf == nil) {
      return false;
    }
    bool ok = microgpt_metal_linear_forward_buffers((__bridge void*)x_buf, (__bridge void*)w_buf, (__bridge void*)b_buf,
                                                    (__bridge void*)y_buf, rows, in_features, out_features, has_bias);
    if (!ok) {
      return false;
    }
    std::memcpy(y, [y_buf contents], y_bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_gelu_forward(const float* x, float* y, int n) {
  if (x == nullptr || y == nullptr || n <= 0) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "kernel void microgpt_gelu_forward(device const float* x [[buffer(0)]],\n"
           "                                  device float* y [[buffer(1)]],\n"
           "                                  constant int& n [[buffer(2)]],\n"
           "                                  uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)n) { return; }\n"
           "  float v = x[gid];\n"
           "  float c = sqrt(2.0f / 3.14159265358979323846f);\n"
           "  float t = tanh(c * (v + 0.044715f * v * v * v));\n"
           "  y[gid] = 0.5f * v * (1.0f + t);\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"microgpt_gelu_forward"];
      if (function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      [library release];
      setup_ok = pipeline != nil;
    });
    if (!setup_ok || pipeline == nil) {
      return false;
    }
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> y_buf = [[device newBufferWithLength:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> n_buf = [[device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || y_buf == nil || n_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:y_buf offset:0 atIndex:1];
    [encoder setBuffer:n_buf offset:0 atIndex:2];
    NSUInteger total = static_cast<NSUInteger>(n);
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > total) {
      width = total;
    }
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(y, [y_buf contents], bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_gelu_backward(const float* x, const float* dy, float* dx, int n) {
  if (x == nullptr || dy == nullptr || dx == nullptr || n <= 0) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "kernel void microgpt_gelu_backward(device const float* x [[buffer(0)]],\n"
           "                                   device const float* dy [[buffer(1)]],\n"
           "                                   device float* dx [[buffer(2)]],\n"
           "                                   constant int& n [[buffer(3)]],\n"
           "                                   uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)n) { return; }\n"
           "  float v = x[gid];\n"
           "  float c = sqrt(2.0f / 3.14159265358979323846f);\n"
           "  float v2 = v * v;\n"
           "  float u = c * (v + 0.044715f * v * v2);\n"
           "  float t = tanh(u);\n"
           "  float sech2 = 1.0f - t * t;\n"
           "  float term = c * (1.0f + 3.0f * 0.044715f * v2);\n"
           "  float deriv = 0.5f * (1.0f + t) + 0.5f * v * sech2 * term;\n"
           "  dx[gid] = dy[gid] * deriv;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"microgpt_gelu_backward"];
      if (function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      [library release];
      setup_ok = pipeline != nil;
    });
    if (!setup_ok || pipeline == nil) {
      return false;
    }
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dy_buf = [[device newBufferWithBytes:dy length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dx_buf = [[device newBufferWithLength:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> n_buf = [[device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || dy_buf == nil || dx_buf == nil || n_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:dy_buf offset:0 atIndex:1];
    [encoder setBuffer:dx_buf offset:0 atIndex:2];
    [encoder setBuffer:n_buf offset:0 atIndex:3];
    NSUInteger total = static_cast<NSUInteger>(n);
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > total) {
      width = total;
    }
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(dx, [dx_buf contents], bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_feedforward_forward(const float* x, const float* w1, const float* b1, const float* w2,
                                                    const float* b2, float* pre, float* hidden, float* y, int rows,
                                                    int d_model, int d_ff, bool has_b1, bool has_b2) {
  if (x == nullptr || w1 == nullptr || w2 == nullptr || pre == nullptr || hidden == nullptr || y == nullptr ||
      rows <= 0 || d_model <= 0 || d_ff <= 0) {
    return false;
  }
  if ((has_b1 && b1 == nullptr) || (has_b2 && b2 == nullptr)) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> linear1_pipeline = nil;
    static id<MTLComputePipelineState> gelu_pipeline = nil;
    static id<MTLComputePipelineState> linear2_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct FeedForwardShape { int rows; int d_model; int d_ff; int has_b1; int has_b2; };\n"
           "kernel void microgpt_ff_linear1(device const float* x [[buffer(0)]],\n"
           "                                device const float* w1 [[buffer(1)]],\n"
           "                                device const float* b1 [[buffer(2)]],\n"
           "                                device float* pre [[buffer(3)]],\n"
           "                                constant FeedForwardShape& shape [[buffer(4)]],\n"
           "                                uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_ff);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.d_ff);\n"
           "  int o = (int)(gid % (uint)shape.d_ff);\n"
           "  float sum = shape.has_b1 ? b1[o] : 0.0f;\n"
           "  for (int i = 0; i < shape.d_model; ++i) {\n"
           "    sum += x[row * shape.d_model + i] * w1[i * shape.d_ff + o];\n"
           "  }\n"
           "  pre[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_gelu(device const float* pre [[buffer(0)]],\n"
           "                            device float* hidden [[buffer(1)]],\n"
           "                            constant FeedForwardShape& shape [[buffer(2)]],\n"
           "                            uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_ff);\n"
           "  if (gid >= total) { return; }\n"
           "  float v = pre[gid];\n"
           "  float c = sqrt(2.0f / 3.14159265358979323846f);\n"
           "  float t = tanh(c * (v + 0.044715f * v * v * v));\n"
           "  hidden[gid] = 0.5f * v * (1.0f + t);\n"
           "}\n"
           "kernel void microgpt_ff_linear2(device const float* hidden [[buffer(0)]],\n"
           "                                device const float* w2 [[buffer(1)]],\n"
           "                                device const float* b2 [[buffer(2)]],\n"
           "                                device float* y [[buffer(3)]],\n"
           "                                constant FeedForwardShape& shape [[buffer(4)]],\n"
           "                                uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_model);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.d_model);\n"
           "  int o = (int)(gid % (uint)shape.d_model);\n"
           "  float sum = shape.has_b2 ? b2[o] : 0.0f;\n"
           "  for (int i = 0; i < shape.d_ff; ++i) {\n"
           "    sum += hidden[row * shape.d_ff + i] * w2[i * shape.d_model + o];\n"
           "  }\n"
           "  y[gid] = sum;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> linear1_function = [library newFunctionWithName:@"microgpt_ff_linear1"];
      id<MTLFunction> gelu_function = [library newFunctionWithName:@"microgpt_ff_gelu"];
      id<MTLFunction> linear2_function = [library newFunctionWithName:@"microgpt_ff_linear2"];
      if (linear1_function == nil || gelu_function == nil || linear2_function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      linear1_pipeline = [device newComputePipelineStateWithFunction:linear1_function error:&error];
      gelu_pipeline = [device newComputePipelineStateWithFunction:gelu_function error:&error];
      linear2_pipeline = [device newComputePipelineStateWithFunction:linear2_function error:&error];
      [linear1_function release];
      [gelu_function release];
      [linear2_function release];
      [library release];
      setup_ok = linear1_pipeline != nil && gelu_pipeline != nil && linear2_pipeline != nil;
    });
    if (!setup_ok || linear1_pipeline == nil || gelu_pipeline == nil || linear2_pipeline == nil) {
      return false;
    }
    struct Shape {
      int rows;
      int d_model;
      int d_ff;
      int has_b1;
      int has_b2;
    } shape{rows, d_model, d_ff, has_b1 ? 1 : 0, has_b2 ? 1 : 0};
    size_t x_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d_model) * sizeof(float);
    size_t hidden_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d_ff) * sizeof(float);
    size_t w1_bytes = static_cast<size_t>(d_model) * static_cast<size_t>(d_ff) * sizeof(float);
    size_t w2_bytes = static_cast<size_t>(d_ff) * static_cast<size_t>(d_model) * sizeof(float);
    size_t b1_bytes = has_b1 ? static_cast<size_t>(d_ff) * sizeof(float) : sizeof(float);
    size_t b2_bytes = has_b2 ? static_cast<size_t>(d_model) * sizeof(float) : sizeof(float);
    float zero_bias = 0.0f;
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> w1_buf = [[device newBufferWithBytes:w1 length:w1_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> b1_buf =
        [[device newBufferWithBytes:(has_b1 ? b1 : &zero_bias) length:b1_bytes options:MTLResourceStorageModeShared]
            autorelease];
    id<MTLBuffer> w2_buf = [[device newBufferWithBytes:w2 length:w2_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> b2_buf =
        [[device newBufferWithBytes:(has_b2 ? b2 : &zero_bias) length:b2_bytes options:MTLResourceStorageModeShared]
            autorelease];
    id<MTLBuffer> pre_buf = [[device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> hidden_buf = [[device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> y_buf = [[device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || w1_buf == nil || b1_buf == nil || w2_buf == nil || b2_buf == nil || pre_buf == nil ||
        hidden_buf == nil || y_buf == nil || shape_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };

    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:w1_buf offset:0 atIndex:1];
    [encoder setBuffer:b1_buf offset:0 atIndex:2];
    [encoder setBuffer:pre_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];
    dispatch_1d(linear1_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_ff));

    [encoder setBuffer:pre_buf offset:0 atIndex:0];
    [encoder setBuffer:hidden_buf offset:0 atIndex:1];
    [encoder setBuffer:shape_buf offset:0 atIndex:2];
    dispatch_1d(gelu_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_ff));

    [encoder setBuffer:hidden_buf offset:0 atIndex:0];
    [encoder setBuffer:w2_buf offset:0 atIndex:1];
    [encoder setBuffer:b2_buf offset:0 atIndex:2];
    [encoder setBuffer:y_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];
    dispatch_1d(linear2_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_model));

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(pre, [pre_buf contents], hidden_bytes);
    std::memcpy(hidden, [hidden_buf contents], hidden_bytes);
    std::memcpy(y, [y_buf contents], x_bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_feedforward_backward(const float* x, const float* pre, const float* hidden,
                                                     const float* w1, const float* w2, const float* dy, float* dx,
                                                     float* dw1, float* db1, float* dw2, float* db2, int rows,
                                                     int d_model, int d_ff, bool has_b1, bool has_b2) {
  if (x == nullptr || pre == nullptr || hidden == nullptr || w1 == nullptr || w2 == nullptr || dy == nullptr ||
      dx == nullptr || dw1 == nullptr || dw2 == nullptr || rows <= 0 || d_model <= 0 || d_ff <= 0) {
    return false;
  }
  if ((has_b1 && db1 == nullptr) || (has_b2 && db2 == nullptr)) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> dhidden_pipeline = nil;
    static id<MTLComputePipelineState> dw2_pipeline = nil;
    static id<MTLComputePipelineState> db2_pipeline = nil;
    static id<MTLComputePipelineState> dpre_pipeline = nil;
    static id<MTLComputePipelineState> dx_pipeline = nil;
    static id<MTLComputePipelineState> dw1_pipeline = nil;
    static id<MTLComputePipelineState> db1_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct FeedForwardShape { int rows; int d_model; int d_ff; int has_b1; int has_b2; };\n"
           "static inline float microgpt_gelu_derivative(float v) {\n"
           "  float c = sqrt(2.0f / 3.14159265358979323846f);\n"
           "  float v2 = v * v;\n"
           "  float u = c * (v + 0.044715f * v * v2);\n"
           "  float t = tanh(u);\n"
           "  float sech2 = 1.0f - t * t;\n"
           "  float term = c * (1.0f + 3.0f * 0.044715f * v2);\n"
           "  return 0.5f * (1.0f + t) + 0.5f * v * sech2 * term;\n"
           "}\n"
           "kernel void microgpt_ff_dhidden(device const float* dy [[buffer(0)]],\n"
           "                                device const float* w2 [[buffer(1)]],\n"
           "                                device float* dhidden [[buffer(2)]],\n"
           "                                constant FeedForwardShape& shape [[buffer(3)]],\n"
           "                                uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_ff);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.d_ff);\n"
           "  int i = (int)(gid % (uint)shape.d_ff);\n"
           "  float sum = 0.0f;\n"
           "  for (int o = 0; o < shape.d_model; ++o) {\n"
           "    sum += dy[row * shape.d_model + o] * w2[i * shape.d_model + o];\n"
           "  }\n"
           "  dhidden[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_dw2(device const float* hidden [[buffer(0)]],\n"
           "                            device const float* dy [[buffer(1)]],\n"
           "                            device float* dw2 [[buffer(2)]],\n"
           "                            constant FeedForwardShape& shape [[buffer(3)]],\n"
           "                            uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.d_ff * shape.d_model);\n"
           "  if (gid >= total) { return; }\n"
           "  int i = (int)(gid / (uint)shape.d_model);\n"
           "  int o = (int)(gid % (uint)shape.d_model);\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += hidden[row * shape.d_ff + i] * dy[row * shape.d_model + o];\n"
           "  }\n"
           "  dw2[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_db2(device const float* dy [[buffer(0)]],\n"
           "                            device float* db2 [[buffer(1)]],\n"
           "                            constant FeedForwardShape& shape [[buffer(2)]],\n"
           "                            uint o [[thread_position_in_grid]]) {\n"
           "  if (o >= (uint)shape.d_model) { return; }\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) { sum += dy[row * shape.d_model + (int)o]; }\n"
           "  db2[o] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_dpre(device const float* pre [[buffer(0)]],\n"
           "                             device const float* dhidden [[buffer(1)]],\n"
           "                             device float* dpre [[buffer(2)]],\n"
           "                             constant FeedForwardShape& shape [[buffer(3)]],\n"
           "                             uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_ff);\n"
           "  if (gid >= total) { return; }\n"
           "  dpre[gid] = dhidden[gid] * microgpt_gelu_derivative(pre[gid]);\n"
           "}\n"
           "kernel void microgpt_ff_dx(device const float* dpre [[buffer(0)]],\n"
           "                          device const float* w1 [[buffer(1)]],\n"
           "                          device float* dx [[buffer(2)]],\n"
           "                          constant FeedForwardShape& shape [[buffer(3)]],\n"
           "                          uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.d_model);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.d_model);\n"
           "  int i = (int)(gid % (uint)shape.d_model);\n"
           "  float sum = 0.0f;\n"
           "  for (int o = 0; o < shape.d_ff; ++o) {\n"
           "    sum += dpre[row * shape.d_ff + o] * w1[i * shape.d_ff + o];\n"
           "  }\n"
           "  dx[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_dw1(device const float* x [[buffer(0)]],\n"
           "                            device const float* dpre [[buffer(1)]],\n"
           "                            device float* dw1 [[buffer(2)]],\n"
           "                            constant FeedForwardShape& shape [[buffer(3)]],\n"
           "                            uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.d_model * shape.d_ff);\n"
           "  if (gid >= total) { return; }\n"
           "  int i = (int)(gid / (uint)shape.d_ff);\n"
           "  int o = (int)(gid % (uint)shape.d_ff);\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += x[row * shape.d_model + i] * dpre[row * shape.d_ff + o];\n"
           "  }\n"
           "  dw1[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_ff_db1(device const float* dpre [[buffer(0)]],\n"
           "                            device float* db1 [[buffer(1)]],\n"
           "                            constant FeedForwardShape& shape [[buffer(2)]],\n"
           "                            uint o [[thread_position_in_grid]]) {\n"
           "  if (o >= (uint)shape.d_ff) { return; }\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) { sum += dpre[row * shape.d_ff + (int)o]; }\n"
           "  db1[o] = sum;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> dhidden_function = [library newFunctionWithName:@"microgpt_ff_dhidden"];
      id<MTLFunction> dw2_function = [library newFunctionWithName:@"microgpt_ff_dw2"];
      id<MTLFunction> db2_function = [library newFunctionWithName:@"microgpt_ff_db2"];
      id<MTLFunction> dpre_function = [library newFunctionWithName:@"microgpt_ff_dpre"];
      id<MTLFunction> dx_function = [library newFunctionWithName:@"microgpt_ff_dx"];
      id<MTLFunction> dw1_function = [library newFunctionWithName:@"microgpt_ff_dw1"];
      id<MTLFunction> db1_function = [library newFunctionWithName:@"microgpt_ff_db1"];
      if (dhidden_function == nil || dw2_function == nil || db2_function == nil || dpre_function == nil ||
          dx_function == nil || dw1_function == nil || db1_function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      dhidden_pipeline = [device newComputePipelineStateWithFunction:dhidden_function error:&error];
      dw2_pipeline = [device newComputePipelineStateWithFunction:dw2_function error:&error];
      db2_pipeline = [device newComputePipelineStateWithFunction:db2_function error:&error];
      dpre_pipeline = [device newComputePipelineStateWithFunction:dpre_function error:&error];
      dx_pipeline = [device newComputePipelineStateWithFunction:dx_function error:&error];
      dw1_pipeline = [device newComputePipelineStateWithFunction:dw1_function error:&error];
      db1_pipeline = [device newComputePipelineStateWithFunction:db1_function error:&error];
      [dhidden_function release];
      [dw2_function release];
      [db2_function release];
      [dpre_function release];
      [dx_function release];
      [dw1_function release];
      [db1_function release];
      [library release];
      setup_ok = dhidden_pipeline != nil && dw2_pipeline != nil && db2_pipeline != nil && dpre_pipeline != nil &&
                 dx_pipeline != nil && dw1_pipeline != nil && db1_pipeline != nil;
    });
    if (!setup_ok || dhidden_pipeline == nil || dw2_pipeline == nil || db2_pipeline == nil || dpre_pipeline == nil ||
        dx_pipeline == nil || dw1_pipeline == nil || db1_pipeline == nil) {
      return false;
    }

    struct Shape {
      int rows;
      int d_model;
      int d_ff;
      int has_b1;
      int has_b2;
    } shape{rows, d_model, d_ff, has_b1 ? 1 : 0, has_b2 ? 1 : 0};
    size_t x_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d_model) * sizeof(float);
    size_t hidden_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d_ff) * sizeof(float);
    size_t w1_bytes = static_cast<size_t>(d_model) * static_cast<size_t>(d_ff) * sizeof(float);
    size_t w2_bytes = static_cast<size_t>(d_ff) * static_cast<size_t>(d_model) * sizeof(float);
    size_t db1_bytes = static_cast<size_t>(d_ff) * sizeof(float);
    size_t db2_bytes = static_cast<size_t>(d_model) * sizeof(float);
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> pre_buf = [[device newBufferWithBytes:pre length:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> hidden_buf =
        [[device newBufferWithBytes:hidden length:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> w1_buf = [[device newBufferWithBytes:w1 length:w1_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> w2_buf = [[device newBufferWithBytes:w2 length:w2_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dy_buf = [[device newBufferWithBytes:dy length:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dhidden_buf = [[device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dpre_buf = [[device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dx_buf = [[device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dw1_buf = [[device newBufferWithLength:w1_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> db1_buf = [[device newBufferWithLength:db1_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dw2_buf = [[device newBufferWithLength:w2_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> db2_buf = [[device newBufferWithLength:db2_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || pre_buf == nil || hidden_buf == nil || w1_buf == nil || w2_buf == nil || dy_buf == nil ||
        dhidden_buf == nil || dpre_buf == nil || dx_buf == nil || dw1_buf == nil || db1_buf == nil || dw2_buf == nil ||
        db2_buf == nil || shape_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };

    [encoder setBuffer:dy_buf offset:0 atIndex:0];
    [encoder setBuffer:w2_buf offset:0 atIndex:1];
    [encoder setBuffer:dhidden_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dhidden_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_ff));

    [encoder setBuffer:hidden_buf offset:0 atIndex:0];
    [encoder setBuffer:dy_buf offset:0 atIndex:1];
    [encoder setBuffer:dw2_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dw2_pipeline, static_cast<NSUInteger>(d_ff) * static_cast<NSUInteger>(d_model));

    [encoder setBuffer:dy_buf offset:0 atIndex:0];
    [encoder setBuffer:db2_buf offset:0 atIndex:1];
    [encoder setBuffer:shape_buf offset:0 atIndex:2];
    dispatch_1d(db2_pipeline, static_cast<NSUInteger>(d_model));

    [encoder setBuffer:pre_buf offset:0 atIndex:0];
    [encoder setBuffer:dhidden_buf offset:0 atIndex:1];
    [encoder setBuffer:dpre_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dpre_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_ff));

    [encoder setBuffer:dpre_buf offset:0 atIndex:0];
    [encoder setBuffer:w1_buf offset:0 atIndex:1];
    [encoder setBuffer:dx_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dx_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(d_model));

    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:dpre_buf offset:0 atIndex:1];
    [encoder setBuffer:dw1_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dw1_pipeline, static_cast<NSUInteger>(d_model) * static_cast<NSUInteger>(d_ff));

    [encoder setBuffer:dpre_buf offset:0 atIndex:0];
    [encoder setBuffer:db1_buf offset:0 atIndex:1];
    [encoder setBuffer:shape_buf offset:0 atIndex:2];
    dispatch_1d(db1_pipeline, static_cast<NSUInteger>(d_ff));

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(dx, [dx_buf contents], x_bytes);
    std::memcpy(dw1, [dw1_buf contents], w1_bytes);
    if (has_b1) {
      std::memcpy(db1, [db1_buf contents], db1_bytes);
    }
    std::memcpy(dw2, [dw2_buf contents], w2_bytes);
    if (has_b2) {
      std::memcpy(db2, [db2_buf contents], db2_bytes);
    }
    return true;
  }
}

extern "C" bool microgpt_metal_adamw_update(float* data, float* grad, float* m, float* v, int n, float lr, float beta1,
                                             float beta2, float eps, float weight_decay, int step, bool decay) {
  if (data == nullptr || grad == nullptr || m == nullptr || v == nullptr || n <= 0 || step <= 0) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct AdamWShape { int n; float lr; float beta1; float beta2; float eps; float weight_decay; int step; int decay; };\n"
           "kernel void microgpt_adamw_update(device float* data [[buffer(0)]],\n"
           "                                  device const float* grad [[buffer(1)]],\n"
           "                                  device float* m [[buffer(2)]],\n"
           "                                  device float* v [[buffer(3)]],\n"
           "                                  constant AdamWShape& shape [[buffer(4)]],\n"
           "                                  uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)shape.n) { return; }\n"
           "  float g = grad[gid];\n"
           "  m[gid] = shape.beta1 * m[gid] + (1.0f - shape.beta1) * g;\n"
           "  v[gid] = shape.beta2 * v[gid] + (1.0f - shape.beta2) * g * g;\n"
           "  float b1t = pow(shape.beta1, (float)shape.step);\n"
           "  float b2t = pow(shape.beta2, (float)shape.step);\n"
           "  float mhat = m[gid] / (1.0f - b1t);\n"
           "  float vhat = v[gid] / (1.0f - b2t);\n"
           "  float update = mhat / (sqrt(vhat) + shape.eps);\n"
           "  if (shape.decay != 0) {\n"
           "    update += shape.weight_decay * data[gid];\n"
           "  }\n"
           "  data[gid] -= shape.lr * update;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"microgpt_adamw_update"];
      if (function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      [library release];
      setup_ok = pipeline != nil;
    });
    if (!setup_ok || pipeline == nil) {
      return false;
    }
    struct Shape {
      int n;
      float lr;
      float beta1;
      float beta2;
      float eps;
      float weight_decay;
      int step;
      int decay;
    } shape{n, lr, beta1, beta2, eps, weight_decay, step, decay ? 1 : 0};
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    id<MTLBuffer> data_buf = [[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> grad_buf = [[device newBufferWithBytes:grad length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> m_buf = [[device newBufferWithBytes:m length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> v_buf = [[device newBufferWithBytes:v length:bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (data_buf == nil || grad_buf == nil || m_buf == nil || v_buf == nil || shape_buf == nil) {
      return false;
    }
    MicrogptMetalCommandScope scope = microgpt_metal_command_scope();
    id<MTLCommandBuffer> command_buffer = scope.buffer;
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data_buf offset:0 atIndex:0];
    [encoder setBuffer:grad_buf offset:0 atIndex:1];
    [encoder setBuffer:m_buf offset:0 atIndex:2];
    [encoder setBuffer:v_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];
    NSUInteger total = static_cast<NSUInteger>(n);
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > total) {
      width = total;
    }
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    if (!microgpt_metal_command_scope_finish(scope)) {
      return false;
    }
    std::memcpy(data, [data_buf contents], bytes);
    std::memcpy(m, [m_buf contents], bytes);
    std::memcpy(v, [v_buf contents], bytes);
    return true;
  }
}

extern "C" void microgpt_metal_command_batch_begin() {
  ++microgpt_metal_batch_depth;
  if (microgpt_metal_batch_depth == 1) {
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    microgpt_metal_batch_buffer = queue == nil ? nil : [queue commandBuffer];
  }
}

extern "C" bool microgpt_metal_command_batch_end() {
  if (microgpt_metal_batch_depth <= 0) {
    return false;
  }
  --microgpt_metal_batch_depth;
  if (microgpt_metal_batch_depth == 0) {
    id<MTLCommandBuffer> buffer = microgpt_metal_batch_buffer;
    microgpt_metal_batch_buffer = nil;
    if (buffer == nil) {
      return false;
    }
    [buffer commit];
    [buffer waitUntilCompleted];
    if (buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    ++microgpt_metal_command_buffer_submission_count;
  }
  return true;
}

extern "C" size_t microgpt_metal_command_buffer_submissions() { return microgpt_metal_command_buffer_submission_count; }

extern "C" void microgpt_metal_reset_command_buffer_submissions() {
  microgpt_metal_command_buffer_submission_count = 0;
}

extern "C" bool microgpt_metal_layernorm_forward(const float* x, const float* gamma, const float* beta, float* y,
                                                  float* mean, float* inv_std, float* xhat, int rows, int dim,
                                                  float eps) {
  if (x == nullptr || gamma == nullptr || beta == nullptr || y == nullptr || mean == nullptr || inv_std == nullptr ||
      xhat == nullptr || rows <= 0 || dim <= 0) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LayerNormShape { int rows; int dim; float eps; };\n"
           "kernel void microgpt_layernorm_forward(device const float* x [[buffer(0)]],\n"
           "                                       device const float* gamma [[buffer(1)]],\n"
           "                                       device const float* beta [[buffer(2)]],\n"
           "                                       device float* y [[buffer(3)]],\n"
           "                                       device float* mean [[buffer(4)]],\n"
           "                                       device float* inv_std [[buffer(5)]],\n"
           "                                       device float* xhat [[buffer(6)]],\n"
           "                                       constant LayerNormShape& shape [[buffer(7)]],\n"
           "                                       uint row [[thread_position_in_grid]]) {\n"
           "  if (row >= (uint)shape.rows) { return; }\n"
           "  int base = (int)row * shape.dim;\n"
           "  float mu = 0.0f;\n"
           "  for (int i = 0; i < shape.dim; ++i) { mu += x[base + i]; }\n"
           "  mu /= (float)shape.dim;\n"
           "  float var = 0.0f;\n"
           "  for (int i = 0; i < shape.dim; ++i) { float c = x[base + i] - mu; var += c * c; }\n"
           "  var /= (float)shape.dim;\n"
           "  float inv = rsqrt(var + shape.eps);\n"
           "  mean[row] = mu;\n"
           "  inv_std[row] = inv;\n"
           "  for (int i = 0; i < shape.dim; ++i) {\n"
           "    float xn = (x[base + i] - mu) * inv;\n"
           "    xhat[base + i] = xn;\n"
           "    y[base + i] = xn * gamma[i] + beta[i];\n"
           "  }\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"microgpt_layernorm_forward"];
      if (function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      [function release];
      [library release];
      setup_ok = pipeline != nil;
    });
    if (!setup_ok || pipeline == nil) {
      return false;
    }
    struct Shape {
      int rows;
      int dim;
      float eps;
    } shape{rows, dim, eps};
    size_t data_bytes = static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    size_t row_bytes = static_cast<size_t>(rows) * sizeof(float);
    size_t dim_bytes = static_cast<size_t>(dim) * sizeof(float);
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> gamma_buf = [[device newBufferWithBytes:gamma length:dim_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> beta_buf = [[device newBufferWithBytes:beta length:dim_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> y_buf = [[device newBufferWithLength:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> mean_buf = [[device newBufferWithLength:row_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> inv_buf = [[device newBufferWithLength:row_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> xhat_buf = [[device newBufferWithLength:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || gamma_buf == nil || beta_buf == nil || y_buf == nil || mean_buf == nil || inv_buf == nil ||
        xhat_buf == nil || shape_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:gamma_buf offset:0 atIndex:1];
    [encoder setBuffer:beta_buf offset:0 atIndex:2];
    [encoder setBuffer:y_buf offset:0 atIndex:3];
    [encoder setBuffer:mean_buf offset:0 atIndex:4];
    [encoder setBuffer:inv_buf offset:0 atIndex:5];
    [encoder setBuffer:xhat_buf offset:0 atIndex:6];
    [encoder setBuffer:shape_buf offset:0 atIndex:7];
    NSUInteger total = static_cast<NSUInteger>(rows);
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width > total) {
      width = total;
    }
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(y, [y_buf contents], data_bytes);
    std::memcpy(mean, [mean_buf contents], row_bytes);
    std::memcpy(inv_std, [inv_buf contents], row_bytes);
    std::memcpy(xhat, [xhat_buf contents], data_bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_layernorm_backward(const float* x, const float* gamma, const float* dy,
                                                   const float* mean, const float* inv_std, const float* xhat,
                                                   float* dx, float* dgamma, float* dbeta, int rows, int dim) {
  if (x == nullptr || gamma == nullptr || dy == nullptr || mean == nullptr || inv_std == nullptr || xhat == nullptr ||
      dx == nullptr || dgamma == nullptr || dbeta == nullptr || rows <= 0 || dim <= 0) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    static id<MTLComputePipelineState> dx_pipeline = nil;
    static id<MTLComputePipelineState> param_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LayerNormShape { int rows; int dim; };\n"
           "kernel void microgpt_layernorm_backward_dx(device const float* x [[buffer(0)]],\n"
           "                                          device const float* gamma [[buffer(1)]],\n"
           "                                          device const float* dy [[buffer(2)]],\n"
           "                                          device const float* mean [[buffer(3)]],\n"
           "                                          device const float* inv_std [[buffer(4)]],\n"
           "                                          device float* dx [[buffer(5)]],\n"
           "                                          constant LayerNormShape& shape [[buffer(6)]],\n"
           "                                          uint row [[thread_position_in_grid]]) {\n"
           "  if (row >= (uint)shape.rows) { return; }\n"
           "  int base = (int)row * shape.dim;\n"
           "  float sum_dxhat = 0.0f;\n"
           "  float sum_dxhat_xmu = 0.0f;\n"
           "  for (int i = 0; i < shape.dim; ++i) {\n"
           "    float dxh = dy[base + i] * gamma[i];\n"
           "    float xmu = x[base + i] - mean[row];\n"
           "    sum_dxhat += dxh;\n"
           "    sum_dxhat_xmu += dxh * xmu;\n"
           "  }\n"
           "  float inv = inv_std[row];\n"
           "  float dvar = -0.5f * inv * inv * inv * sum_dxhat_xmu;\n"
           "  float dmu = -inv * sum_dxhat;\n"
           "  float mean_xmu = 0.0f;\n"
           "  for (int i = 0; i < shape.dim; ++i) { mean_xmu += x[base + i] - mean[row]; }\n"
           "  dmu += dvar * (-2.0f * mean_xmu / (float)shape.dim);\n"
           "  for (int i = 0; i < shape.dim; ++i) {\n"
           "    float dxh = dy[base + i] * gamma[i];\n"
           "    float xmu = x[base + i] - mean[row];\n"
           "    dx[base + i] = dxh * inv + dvar * 2.0f * xmu / (float)shape.dim + dmu / (float)shape.dim;\n"
           "  }\n"
           "}\n"
           "kernel void microgpt_layernorm_backward_params(device const float* dy [[buffer(0)]],\n"
           "                                              device const float* xhat [[buffer(1)]],\n"
           "                                              device float* dgamma [[buffer(2)]],\n"
           "                                              device float* dbeta [[buffer(3)]],\n"
           "                                              constant LayerNormShape& shape [[buffer(4)]],\n"
           "                                              uint col [[thread_position_in_grid]]) {\n"
           "  if (col >= (uint)shape.dim) { return; }\n"
           "  float gg = 0.0f;\n"
           "  float gb = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    int idx = row * shape.dim + (int)col;\n"
           "    gg += dy[idx] * xhat[idx];\n"
           "    gb += dy[idx];\n"
           "  }\n"
           "  dgamma[col] = gg;\n"
           "  dbeta[col] = gb;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> dx_function = [library newFunctionWithName:@"microgpt_layernorm_backward_dx"];
      id<MTLFunction> param_function = [library newFunctionWithName:@"microgpt_layernorm_backward_params"];
      if (dx_function == nil || param_function == nil) {
        [library release];
        setup_ok = false;
        return;
      }
      dx_pipeline = [device newComputePipelineStateWithFunction:dx_function error:&error];
      param_pipeline = [device newComputePipelineStateWithFunction:param_function error:&error];
      [dx_function release];
      [param_function release];
      [library release];
      setup_ok = dx_pipeline != nil && param_pipeline != nil;
    });
    if (!setup_ok || dx_pipeline == nil || param_pipeline == nil) {
      return false;
    }
    struct Shape {
      int rows;
      int dim;
    } shape{rows, dim};
    size_t data_bytes = static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    size_t row_bytes = static_cast<size_t>(rows) * sizeof(float);
    size_t dim_bytes = static_cast<size_t>(dim) * sizeof(float);
    id<MTLBuffer> x_buf = [[device newBufferWithBytes:x length:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> gamma_buf = [[device newBufferWithBytes:gamma length:dim_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dy_buf = [[device newBufferWithBytes:dy length:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> mean_buf = [[device newBufferWithBytes:mean length:row_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> inv_buf = [[device newBufferWithBytes:inv_std length:row_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> xhat_buf = [[device newBufferWithBytes:xhat length:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dx_buf = [[device newBufferWithLength:data_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dgamma_buf = [[device newBufferWithLength:dim_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dbeta_buf = [[device newBufferWithLength:dim_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || gamma_buf == nil || dy_buf == nil || mean_buf == nil || inv_buf == nil || xhat_buf == nil ||
        dx_buf == nil || dgamma_buf == nil || dbeta_buf == nil || shape_buf == nil) {
      return false;
    }
    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }
    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };
    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:gamma_buf offset:0 atIndex:1];
    [encoder setBuffer:dy_buf offset:0 atIndex:2];
    [encoder setBuffer:mean_buf offset:0 atIndex:3];
    [encoder setBuffer:inv_buf offset:0 atIndex:4];
    [encoder setBuffer:dx_buf offset:0 atIndex:5];
    [encoder setBuffer:shape_buf offset:0 atIndex:6];
    dispatch_1d(dx_pipeline, static_cast<NSUInteger>(rows));

    [encoder setBuffer:dy_buf offset:0 atIndex:0];
    [encoder setBuffer:xhat_buf offset:0 atIndex:1];
    [encoder setBuffer:dgamma_buf offset:0 atIndex:2];
    [encoder setBuffer:dbeta_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];
    dispatch_1d(param_pipeline, static_cast<NSUInteger>(dim));

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    std::memcpy(dx, [dx_buf contents], data_bytes);
    std::memcpy(dgamma, [dgamma_buf contents], dim_bytes);
    std::memcpy(dbeta, [dbeta_buf contents], dim_bytes);
    return true;
  }
}

extern "C" bool microgpt_metal_linear_backward_buffers(void* x_buffer, void* w_buffer, void* dy_buffer, void* dx_buffer,
                                                        void* dw_buffer, void* db_buffer, int rows, int in_features,
                                                        int out_features, bool has_bias) {
  if (x_buffer == nullptr || w_buffer == nullptr || dy_buffer == nullptr || dx_buffer == nullptr ||
      dw_buffer == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0) {
    return false;
  }
  if (has_bias && db_buffer == nullptr) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }

    static id<MTLComputePipelineState> dx_pipeline = nil;
    static id<MTLComputePipelineState> dw_pipeline = nil;
    static id<MTLComputePipelineState> db_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LinearShape { int rows; int in_features; int out_features; int has_bias; };\n"
           "kernel void microgpt_linear_backward_dx(device const float* dy [[buffer(0)]],\n"
           "                                       device const float* w [[buffer(1)]],\n"
           "                                       device float* dx [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.in_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.in_features);\n"
           "  int i = (int)(gid % (uint)shape.in_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int o = 0; o < shape.out_features; ++o) {\n"
           "    sum += dy[row * shape.out_features + o] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  dx[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_dw(device const float* x [[buffer(0)]],\n"
           "                                       device const float* dy [[buffer(1)]],\n"
           "                                       device float* dw [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.in_features * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int i = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += x[row * shape.in_features + i] * dy[row * shape.out_features + o];\n"
           "  }\n"
           "  dw[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_db(device const float* dy [[buffer(0)]],\n"
           "                                       device float* db [[buffer(1)]],\n"
           "                                       constant LinearShape& shape [[buffer(2)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)shape.out_features) { return; }\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += dy[row * shape.out_features + (int)gid];\n"
           "  }\n"
           "  db[gid] = sum;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> dx_function = [library newFunctionWithName:@"microgpt_linear_backward_dx"];
      id<MTLFunction> dw_function = [library newFunctionWithName:@"microgpt_linear_backward_dw"];
      id<MTLFunction> db_function = [library newFunctionWithName:@"microgpt_linear_backward_db"];
      if (dx_function == nil || dw_function == nil || db_function == nil) {
        setup_ok = false;
        return;
      }
      dx_pipeline = [device newComputePipelineStateWithFunction:dx_function error:&error];
      dw_pipeline = [device newComputePipelineStateWithFunction:dw_function error:&error];
      db_pipeline = [device newComputePipelineStateWithFunction:db_function error:&error];
      setup_ok = dx_pipeline != nil && dw_pipeline != nil && db_pipeline != nil;
    });

    if (!setup_ok || dx_pipeline == nil || dw_pipeline == nil || db_pipeline == nil) {
      return false;
    }

    MicrogptLinearShape shape{rows, in_features, out_features, has_bias ? 1 : 0};

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_buffer;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buffer;
    id<MTLBuffer> dy_buf = (__bridge id<MTLBuffer>)dy_buffer;
    id<MTLBuffer> dx_buf = (__bridge id<MTLBuffer>)dx_buffer;
    id<MTLBuffer> dw_buf = (__bridge id<MTLBuffer>)dw_buffer;
    id<MTLBuffer> db_buf = (__bridge id<MTLBuffer>)db_buffer;
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];

    if (x_buf == nil || w_buf == nil || dy_buf == nil || dx_buf == nil || dw_buf == nil || shape_buf == nil ||
        (has_bias && db_buf == nil)) {
      return false;
    }

    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }

    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      if (width == 0) {
        width = 1;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };

    [encoder setBuffer:dy_buf offset:0 atIndex:0];
    [encoder setBuffer:w_buf offset:0 atIndex:1];
    [encoder setBuffer:dx_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dx_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(in_features));

    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:dy_buf offset:0 atIndex:1];
    [encoder setBuffer:dw_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dw_pipeline, static_cast<NSUInteger>(in_features) * static_cast<NSUInteger>(out_features));

    if (has_bias) {
      [encoder setBuffer:dy_buf offset:0 atIndex:0];
      [encoder setBuffer:db_buf offset:0 atIndex:1];
      [encoder setBuffer:shape_buf offset:0 atIndex:2];
      dispatch_1d(db_pipeline, static_cast<NSUInteger>(out_features));
    }

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }
    return true;
  }
}

extern "C" bool microgpt_metal_linear_forward_backward_buffers(void* x_buffer, void* w_buffer, void* bias_buffer,
                                                                void* y_buffer, void* dy_buffer, void* dx_buffer,
                                                                void* dw_buffer, void* db_buffer, int rows,
                                                                int in_features, int out_features, bool has_bias) {
  if (x_buffer == nullptr || w_buffer == nullptr || y_buffer == nullptr || dy_buffer == nullptr ||
      dx_buffer == nullptr || dw_buffer == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0) {
    return false;
  }
  if (has_bias && (bias_buffer == nullptr || db_buffer == nullptr)) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }

    static id<MTLComputePipelineState> forward_pipeline = nil;
    static id<MTLComputePipelineState> dx_pipeline = nil;
    static id<MTLComputePipelineState> dw_pipeline = nil;
    static id<MTLComputePipelineState> db_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LinearShape { int rows; int in_features; int out_features; int has_bias; };\n"
           "kernel void microgpt_linear_forward(device const float* x [[buffer(0)]],\n"
           "                                    device const float* w [[buffer(1)]],\n"
           "                                    device const float* bias [[buffer(2)]],\n"
           "                                    device float* y [[buffer(3)]],\n"
           "                                    constant LinearShape& shape [[buffer(4)]],\n"
           "                                    uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = shape.has_bias ? bias[o] : 0.0f;\n"
           "  for (int i = 0; i < shape.in_features; ++i) {\n"
           "    sum += x[row * shape.in_features + i] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  y[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_dx(device const float* dy [[buffer(0)]],\n"
           "                                       device const float* w [[buffer(1)]],\n"
           "                                       device float* dx [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.in_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.in_features);\n"
           "  int i = (int)(gid % (uint)shape.in_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int o = 0; o < shape.out_features; ++o) {\n"
           "    sum += dy[row * shape.out_features + o] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  dx[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_dw(device const float* x [[buffer(0)]],\n"
           "                                       device const float* dy [[buffer(1)]],\n"
           "                                       device float* dw [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.in_features * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int i = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += x[row * shape.in_features + i] * dy[row * shape.out_features + o];\n"
           "  }\n"
           "  dw[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_db(device const float* dy [[buffer(0)]],\n"
           "                                       device float* db [[buffer(1)]],\n"
           "                                       constant LinearShape& shape [[buffer(2)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)shape.out_features) { return; }\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += dy[row * shape.out_features + (int)gid];\n"
           "  }\n"
           "  db[gid] = sum;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> forward_function = [library newFunctionWithName:@"microgpt_linear_forward"];
      id<MTLFunction> dx_function = [library newFunctionWithName:@"microgpt_linear_backward_dx"];
      id<MTLFunction> dw_function = [library newFunctionWithName:@"microgpt_linear_backward_dw"];
      id<MTLFunction> db_function = [library newFunctionWithName:@"microgpt_linear_backward_db"];
      if (forward_function == nil || dx_function == nil || dw_function == nil || db_function == nil) {
        setup_ok = false;
        return;
      }
      forward_pipeline = [device newComputePipelineStateWithFunction:forward_function error:&error];
      dx_pipeline = [device newComputePipelineStateWithFunction:dx_function error:&error];
      dw_pipeline = [device newComputePipelineStateWithFunction:dw_function error:&error];
      db_pipeline = [device newComputePipelineStateWithFunction:db_function error:&error];
      setup_ok = forward_pipeline != nil && dx_pipeline != nil && dw_pipeline != nil && db_pipeline != nil;
    });

    if (!setup_ok || forward_pipeline == nil || dx_pipeline == nil || dw_pipeline == nil || db_pipeline == nil) {
      return false;
    }

    MicrogptLinearShape shape{rows, in_features, out_features, has_bias ? 1 : 0};
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_buffer;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buffer;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)bias_buffer;
    id<MTLBuffer> y_buf = (__bridge id<MTLBuffer>)y_buffer;
    id<MTLBuffer> dy_buf = (__bridge id<MTLBuffer>)dy_buffer;
    id<MTLBuffer> dx_buf = (__bridge id<MTLBuffer>)dx_buffer;
    id<MTLBuffer> dw_buf = (__bridge id<MTLBuffer>)dw_buffer;
    id<MTLBuffer> db_buf = (__bridge id<MTLBuffer>)db_buffer;
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || w_buf == nil || y_buf == nil || dy_buf == nil || dx_buf == nil || dw_buf == nil ||
        shape_buf == nil || (has_bias && (b_buf == nil || db_buf == nil))) {
      return false;
    }

    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }

    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      if (width == 0) {
        width = 1;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };

    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:w_buf offset:0 atIndex:1];
    [encoder setBuffer:b_buf offset:0 atIndex:2];
    [encoder setBuffer:y_buf offset:0 atIndex:3];
    [encoder setBuffer:shape_buf offset:0 atIndex:4];
    dispatch_1d(forward_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(out_features));

    [encoder setBuffer:dy_buf offset:0 atIndex:0];
    [encoder setBuffer:w_buf offset:0 atIndex:1];
    [encoder setBuffer:dx_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dx_pipeline, static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(in_features));

    [encoder setBuffer:x_buf offset:0 atIndex:0];
    [encoder setBuffer:dy_buf offset:0 atIndex:1];
    [encoder setBuffer:dw_buf offset:0 atIndex:2];
    [encoder setBuffer:shape_buf offset:0 atIndex:3];
    dispatch_1d(dw_pipeline, static_cast<NSUInteger>(in_features) * static_cast<NSUInteger>(out_features));

    if (has_bias) {
      [encoder setBuffer:dy_buf offset:0 atIndex:0];
      [encoder setBuffer:db_buf offset:0 atIndex:1];
      [encoder setBuffer:shape_buf offset:0 atIndex:2];
      dispatch_1d(db_pipeline, static_cast<NSUInteger>(out_features));
    }

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted;
  }
}

extern "C" bool microgpt_metal_linear_forward_backward_repeat_buffers(void* x_buffer, void* w_buffer, void* bias_buffer,
                                                                       void* y_buffer, void* dy_buffer,
                                                                       void* dx_buffer, void* dw_buffer,
                                                                       void* db_buffer, int rows, int in_features,
                                                                       int out_features, bool has_bias,
                                                                       int iterations) {
  if (iterations <= 0) {
    return false;
  }
  if (x_buffer == nullptr || w_buffer == nullptr || y_buffer == nullptr || dy_buffer == nullptr ||
      dx_buffer == nullptr || dw_buffer == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0) {
    return false;
  }
  if (has_bias && (bias_buffer == nullptr || db_buffer == nullptr)) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }

    static id<MTLComputePipelineState> forward_pipeline = nil;
    static id<MTLComputePipelineState> dx_pipeline = nil;
    static id<MTLComputePipelineState> dw_pipeline = nil;
    static id<MTLComputePipelineState> db_pipeline = nil;
    static dispatch_once_t once;
    static bool setup_ok = false;
    dispatch_once(&once, ^{
      NSError* error = nil;
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct LinearShape { int rows; int in_features; int out_features; int has_bias; };\n"
           "kernel void microgpt_linear_forward(device const float* x [[buffer(0)]],\n"
           "                                    device const float* w [[buffer(1)]],\n"
           "                                    device const float* bias [[buffer(2)]],\n"
           "                                    device float* y [[buffer(3)]],\n"
           "                                    constant LinearShape& shape [[buffer(4)]],\n"
           "                                    uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = shape.has_bias ? bias[o] : 0.0f;\n"
           "  for (int i = 0; i < shape.in_features; ++i) {\n"
           "    sum += x[row * shape.in_features + i] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  y[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_dx(device const float* dy [[buffer(0)]],\n"
           "                                       device const float* w [[buffer(1)]],\n"
           "                                       device float* dx [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.rows * shape.in_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int row = (int)(gid / (uint)shape.in_features);\n"
           "  int i = (int)(gid % (uint)shape.in_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int o = 0; o < shape.out_features; ++o) {\n"
           "    sum += dy[row * shape.out_features + o] * w[i * shape.out_features + o];\n"
           "  }\n"
           "  dx[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_dw(device const float* x [[buffer(0)]],\n"
           "                                       device const float* dy [[buffer(1)]],\n"
           "                                       device float* dw [[buffer(2)]],\n"
           "                                       constant LinearShape& shape [[buffer(3)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = (uint)(shape.in_features * shape.out_features);\n"
           "  if (gid >= total) { return; }\n"
           "  int i = (int)(gid / (uint)shape.out_features);\n"
           "  int o = (int)(gid % (uint)shape.out_features);\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += x[row * shape.in_features + i] * dy[row * shape.out_features + o];\n"
           "  }\n"
           "  dw[gid] = sum;\n"
           "}\n"
           "kernel void microgpt_linear_backward_db(device const float* dy [[buffer(0)]],\n"
           "                                       device float* db [[buffer(1)]],\n"
           "                                       constant LinearShape& shape [[buffer(2)]],\n"
           "                                       uint gid [[thread_position_in_grid]]) {\n"
           "  if (gid >= (uint)shape.out_features) { return; }\n"
           "  float sum = 0.0f;\n"
           "  for (int row = 0; row < shape.rows; ++row) {\n"
           "    sum += dy[row * shape.out_features + (int)gid];\n"
           "  }\n"
           "  db[gid] = sum;\n"
           "}\n";
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        setup_ok = false;
        return;
      }
      id<MTLFunction> forward_function = [library newFunctionWithName:@"microgpt_linear_forward"];
      id<MTLFunction> dx_function = [library newFunctionWithName:@"microgpt_linear_backward_dx"];
      id<MTLFunction> dw_function = [library newFunctionWithName:@"microgpt_linear_backward_dw"];
      id<MTLFunction> db_function = [library newFunctionWithName:@"microgpt_linear_backward_db"];
      if (forward_function == nil || dx_function == nil || dw_function == nil || db_function == nil) {
        setup_ok = false;
        return;
      }
      forward_pipeline = [device newComputePipelineStateWithFunction:forward_function error:&error];
      dx_pipeline = [device newComputePipelineStateWithFunction:dx_function error:&error];
      dw_pipeline = [device newComputePipelineStateWithFunction:dw_function error:&error];
      db_pipeline = [device newComputePipelineStateWithFunction:db_function error:&error];
      [forward_function release];
      [dx_function release];
      [dw_function release];
      [db_function release];
      [library release];
      setup_ok = forward_pipeline != nil && dx_pipeline != nil && dw_pipeline != nil && db_pipeline != nil;
    });

    if (!setup_ok || forward_pipeline == nil || dx_pipeline == nil || dw_pipeline == nil || db_pipeline == nil) {
      return false;
    }

    MicrogptLinearShape shape{rows, in_features, out_features, has_bias ? 1 : 0};
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_buffer;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buffer;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)bias_buffer;
    id<MTLBuffer> y_buf = (__bridge id<MTLBuffer>)y_buffer;
    id<MTLBuffer> dy_buf = (__bridge id<MTLBuffer>)dy_buffer;
    id<MTLBuffer> dx_buf = (__bridge id<MTLBuffer>)dx_buffer;
    id<MTLBuffer> dw_buf = (__bridge id<MTLBuffer>)dw_buffer;
    id<MTLBuffer> db_buf = (__bridge id<MTLBuffer>)db_buffer;
    id<MTLBuffer> shape_buf =
        [[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || w_buf == nil || y_buf == nil || dy_buf == nil || dx_buf == nil || dw_buf == nil ||
        shape_buf == nil || (has_bias && (b_buf == nil || db_buf == nil))) {
      return false;
    }

    id<MTLCommandQueue> queue = microgpt_metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (queue == nil || command_buffer == nil || encoder == nil) {
      return false;
    }

    auto dispatch_1d = ^(id<MTLComputePipelineState> pipeline, NSUInteger total) {
      [encoder setComputePipelineState:pipeline];
      NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
      if (width > total) {
        width = total;
      }
      if (width == 0) {
        width = 1;
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    };

    NSUInteger forward_total = static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(out_features);
    NSUInteger dx_total = static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(in_features);
    NSUInteger dw_total = static_cast<NSUInteger>(in_features) * static_cast<NSUInteger>(out_features);
    NSUInteger db_total = static_cast<NSUInteger>(out_features);
    for (int iter = 0; iter < iterations; ++iter) {
      [encoder setBuffer:x_buf offset:0 atIndex:0];
      [encoder setBuffer:w_buf offset:0 atIndex:1];
      [encoder setBuffer:b_buf offset:0 atIndex:2];
      [encoder setBuffer:y_buf offset:0 atIndex:3];
      [encoder setBuffer:shape_buf offset:0 atIndex:4];
      dispatch_1d(forward_pipeline, forward_total);

      [encoder setBuffer:dy_buf offset:0 atIndex:0];
      [encoder setBuffer:w_buf offset:0 atIndex:1];
      [encoder setBuffer:dx_buf offset:0 atIndex:2];
      [encoder setBuffer:shape_buf offset:0 atIndex:3];
      dispatch_1d(dx_pipeline, dx_total);

      [encoder setBuffer:x_buf offset:0 atIndex:0];
      [encoder setBuffer:dy_buf offset:0 atIndex:1];
      [encoder setBuffer:dw_buf offset:0 atIndex:2];
      [encoder setBuffer:shape_buf offset:0 atIndex:3];
      dispatch_1d(dw_pipeline, dw_total);

      if (has_bias) {
        [encoder setBuffer:dy_buf offset:0 atIndex:0];
        [encoder setBuffer:db_buf offset:0 atIndex:1];
        [encoder setBuffer:shape_buf offset:0 atIndex:2];
        dispatch_1d(db_pipeline, db_total);
      }
    }

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted;
  }
}

extern "C" bool microgpt_metal_linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw,
                                                float* db, int rows, int in_features, int out_features,
                                                bool has_bias) {
  if (x == nullptr || w == nullptr || dy == nullptr || dx == nullptr || dw == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0) {
    return false;
  }
  if (has_bias && db == nullptr) {
    return false;
  }
  @autoreleasepool {
    id<MTLDevice> device = microgpt_metal_device();
    if (device == nil) {
      return false;
    }
    size_t x_bytes = static_cast<size_t>(rows) * static_cast<size_t>(in_features) * sizeof(float);
    size_t w_bytes = static_cast<size_t>(in_features) * static_cast<size_t>(out_features) * sizeof(float);
    size_t dy_bytes = static_cast<size_t>(rows) * static_cast<size_t>(out_features) * sizeof(float);
    size_t db_bytes = static_cast<size_t>(out_features) * sizeof(float);
    id<MTLBuffer> x_buf =
        [[device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> w_buf =
        [[device newBufferWithBytes:w length:w_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dy_buf =
        [[device newBufferWithBytes:dy length:dy_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dx_buf = [[device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> dw_buf = [[device newBufferWithLength:w_bytes options:MTLResourceStorageModeShared] autorelease];
    id<MTLBuffer> db_buf = [[device newBufferWithLength:db_bytes options:MTLResourceStorageModeShared] autorelease];
    if (x_buf == nil || w_buf == nil || dy_buf == nil || dx_buf == nil || dw_buf == nil || db_buf == nil) {
      return false;
    }
    bool ok = microgpt_metal_linear_backward_buffers((__bridge void*)x_buf, (__bridge void*)w_buf,
                                                     (__bridge void*)dy_buf, (__bridge void*)dx_buf,
                                                     (__bridge void*)dw_buf, (__bridge void*)db_buf, rows, in_features,
                                                     out_features, has_bias);
    if (!ok) {
      return false;
    }
    std::memcpy(dx, [dx_buf contents], x_bytes);
    std::memcpy(dw, [dw_buf contents], w_bytes);
    if (has_bias) {
      std::memcpy(db, [db_buf contents], db_bytes);
    }
    return true;
  }
}

extern "C" void* microgpt_cuda_buffer_create(size_t) { return nullptr; }
extern "C" void microgpt_cuda_buffer_destroy(void*) {}
extern "C" bool microgpt_cuda_buffer_write(void*, const void*, size_t) { return false; }
extern "C" bool microgpt_cuda_buffer_read(void*, void*, size_t) { return false; }
extern "C" void* microgpt_cuda_buffer_contents(void*) { return nullptr; }
extern "C" bool microgpt_cuda_runtime_available() { return false; }
extern "C" bool microgpt_cuda_runtime_compiled() { return false; }
extern "C" const char* microgpt_cuda_runtime_device_name() { return ""; }
extern "C" bool microgpt_cuda_linear_forward(const float*, const float*, const float*, float*, int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_cuda_linear_backward(const float*, const float*, const float*, float*, float*, float*, int, int,
                                               int, bool) {
  return false;
}
extern "C" bool microgpt_cuda_linear_forward_buffers(void*, void*, void*, void*, int, int, int, bool) { return false; }
extern "C" bool microgpt_cuda_linear_backward_buffers(void*, void*, void*, void*, void*, void*, int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_cuda_linear_forward_backward_buffers(void*, void*, void*, void*, void*, void*, void*, void*,
                                                               int, int, int, bool) {
  return false;
}
extern "C" bool microgpt_cuda_linear_forward_backward_repeat_buffers(void*, void*, void*, void*, void*, void*, void*,
                                                                      void*, int, int, int, bool, int) {
  return false;
}
extern "C" bool microgpt_cuda_gelu_forward(const float*, float*, int) { return false; }
extern "C" bool microgpt_cuda_gelu_backward(const float*, const float*, float*, int) { return false; }
extern "C" bool microgpt_cuda_layernorm_forward(const float*, const float*, const float*, float*, float*, float*,
                                                 float*, int, int, float) {
  return false;
}
extern "C" bool microgpt_cuda_layernorm_backward(const float*, const float*, const float*, const float*, const float*,
                                                  const float*, float*, float*, float*, int, int) {
  return false;
}
extern "C" bool microgpt_cuda_feedforward_forward(const float*, const float*, const float*, const float*, const float*,
                                                   float*, float*, float*, int, int, int, bool, bool) {
  return false;
}
extern "C" bool microgpt_cuda_feedforward_backward(const float*, const float*, const float*, const float*, const float*,
                                                    const float*, float*, float*, float*, float*, float*, int, int, int,
                                                    bool, bool) {
  return false;
}
extern "C" bool microgpt_cuda_adamw_update(float*, float*, float*, float*, int, float, float, float, float, float, int,
                                            bool) {
  return false;
}
extern "C" void microgpt_cuda_command_batch_begin() {}
extern "C" bool microgpt_cuda_command_batch_end() { return false; }
extern "C" size_t microgpt_cuda_command_buffer_submissions() { return 0; }
extern "C" void microgpt_cuda_reset_command_buffer_submissions() {}

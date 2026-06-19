# Hardware Acceleration Spec

## Goal

Improve training and evaluation throughput while preserving the current CPU
implementation as the correctness reference.

## Position

Adding Metal and CUDA is worthwhile, but only after a backend boundary exists.
The current model code mixes algorithm, tensor storage, allocation, and scalar
loops. Directly adding CUDA or Metal branches inside those loops would make the
code harder to test and maintain.

## Backend Targets

- `cpu`: existing portable C++17 implementation, kept as reference.
- `metal`: Apple hardware backend. This is the first accelerated backend
  because the current development machine is an Apple Mac.
- `cuda`: NVIDIA hardware backend. Keep the architecture ready for it, but do
  not mark it supported until it can be tested on NVIDIA hardware or CI.

The CLI should eventually accept:

```bash
./bin/mgpt train --backend cpu
./bin/mgpt train --backend metal
./bin/mgpt train --backend cuda
```

Build selection should also work at compile time:

```bash
make BACKEND=metal
make deps
./bin/mgpt backends
```

On macOS, `BACKEND=metal` should automatically link:

```text
-framework Metal -framework Foundation -framework QuartzCore
```

## Initial Bottlenecks

- Linear layer forward and backward are scalar nested loops.
- Attention does repeated scalar score, softmax, and value accumulation loops.
- Cross-entropy, gradient clipping, and most attention work are host loops.
- Temporary tensors are allocated repeatedly through `std::vector`.
- Generation recomputes the full context every token.

## Implementation Phases

1. Add benchmark commands or report fields for steps/sec, tokens/sec, eval
   runtime, parameter count, and backend name.
2. Add platform dependency checks for local backend libraries and tools.
3. Add backend selection plumbing while routing all work to CPU.
4. Introduce backend tensor storage and explicit host/device sync points.
5. Move linear forward/backward behind backend operations.
6. Add CPU parity tests for operation outputs and gradients.
7. Move general matmul behind backend operations.
8. Add Metal kernels for the stabilized operation interface.
9. Add CUDA kernels for the stabilized operation interface once hardware is available.
10. Move additional ops behind the backend interface: softmax attention, loss,
   and gradient clipping.
11. Add command-buffer batching and resident activation flow so the accelerated
   backend does not submit one small command per tiny operation.
12. Add generation key/value cache as a separate inference optimization.

## Local Apple Metal Notes

The current Mac has the system Metal, Foundation, and QuartzCore frameworks and
the `metal` compiler. If `metallib` is not available through `xcrun`, the first
Metal pass should compile Metal shader source at runtime through the Metal API
instead of requiring an offline `.metallib` artifact.

The current Metal build can link the frameworks, detect the default device, and
accept `--backend metal`. Linear forward and backward have first Metal kernels.
The wider model still uses CPU storage as the source of truth, so unsupported
operations and cross-operation tensor flow can bring data back to the CPU.

Current operation boundary:

- `BackendBuffer` can own a persistent Metal buffer and upload/download data.
- Linear forward and backward are routed through `include/microgpt/backend_ops.hpp`.
- CPU remains the reference implementation.
- General matmul has a backend operation boundary with CPU coverage.
- Metal linear forward and backward have first runtime-compiled kernels and
  fall back to CPU when Metal is unavailable.
- Cached linear layers reuse Metal buffers and avoid re-uploading forward input
  and weights for the matching backward pass.
- AdamW updates now batch all parameter work into one Metal command submission
  before downloading updated weights and optimizer state.
- `mgpt bench --resident-linear` isolates repeated linear forward/backward work
  with a single upload phase and a single final download phase.
- The lab suite includes linear operation output/gradient coverage, matmul
  coverage, conditional Metal linear forward/backward parity coverage, and a
  command-buffer submission regression test for Metal optimizer batching.
- The lab suite also includes a staged parity trace that compares embeddings,
  block outputs, logits, gradients, and AdamW updates on a fixed trace and
  reports the first divergent stage.

## Correctness Rules

- CPU remains the reference backend.
- Checkpoints must stay backend-independent.
- Tokenization, data import, validation, and eval reports remain shared.
- Backend results should match CPU within documented float tolerances.
- Training curves should be comparable for the same seed, config, and dataset.

## Non-Goals For The First Pass

- Mixed precision.
- Distributed training.
- Replacing the project with a third-party tensor framework.
- Changing model behavior or checkpoint format just to support a backend.

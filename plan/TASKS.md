# microgpt Task List

## Done

- [x] Reorganize planning material into `plan/`.
- [x] Reorganize static HTML documentation into `docs/`.
- [x] Add `docs/index.html` as the documentation overview.
- [x] Add docs for data format, experiment workflow, and roadmap.
- [x] Fix generation prompt alignment so inference uses the same absolute positions as training.
- [x] Add a standalone CLI options reference page.
- [x] Add a stronger generation regression test that catches prompt-position mistakes.
- [x] Add reproducible arithmetic dataset generation.
- [x] Add an evaluation command for checkpoints.
- [x] Add quiet generation output for cleaner scripts and evaluations.
- [x] Rebuild and run the full test suite after each code change.
- [x] Add explicit train/test dataset split support.
- [x] Add a greedy decoding shorthand flag.
- [x] Create a repeatable experiment matrix for data size, training steps, model width, layers, and context length.
- [x] Add assistant dataset validation and counting.
- [x] Add dataset split tooling for instruction examples.
- [x] Add JSONL import for `user` and `assistant` pairs.
- [x] Add richer evaluation modes beyond exact match.
- [x] Expand checkpoint metadata with dataset and command details.
- [x] Add dataset round-trip and split contract tests.
- [x] Split executables into `mgpt`, `lab`, and `chat` entrypoints.
- [x] Add checkpoint metadata history with model size and per-run timing.

## Documentation Rule

- [x] Every user-facing code or data-format change should update the relevant HTML page under `docs/`.
- [x] Document the canonical single-turn data contract and its validation rule.

## Next Tiny Assistant Phase

- [ ] Revisit multi-turn session context after single-turn behavior is reliable.

## Harness Refactor

- [x] Add a harness framework for composing built-in and custom tools from other codebases.
- [x] Move the CLI entrypoint to a thin registry-backed dispatcher.
- [x] Split the remaining command implementations into smaller tool-specific headers.
- [x] Add a small example of embedding the harness in another program.

## Artifact Organization

- [x] Define a structured layout for generated datasets and checkpoints.
- [x] Move loose generated root artifacts into `artifacts/`.
- [x] Document that `.bin` is required and `.json` is metadata only.
- [x] Add a cleanup or retention workflow for generated artifacts.
- [x] Add an ignored `artifacts/evals/` folder for generated evaluation reports.

## Core Library Split

- [x] Split `include/microgpt.hpp` into smaller concern-based headers for model, training, generation, checkpointing, and tests.
- [x] Keep `microgpt.hpp` as a compatibility aggregation header until downstream uses migrate.
- [x] Make sure reusable headers are usable from other programs without the CLI.
- [x] Define a stable public API boundary separate from internal implementation headers.
- [x] Add a C-compatible or otherwise stable embedding API for external applications.
- [x] Add library build targets once the public API boundary is defined.
- [x] Add a style policy and formatter configuration.
- [x] Audit header-only inline/static state for ODR and embedding risks.

## Build And Test Hardening

- [x] Add strict and sanitizer Makefile test targets.
- [x] Fail explicitly when `BACKEND=metal` is requested on non-macOS platforms.
- [x] Add bounded file reads and byte-tokenizer vocabulary contract checks.
- [x] Add lab coverage for IO/tokenizer safety contracts.
- [x] Add CI workflow covering CPU build, tests, strict build, and sanitizer build.
- [x] Add CLI-level tests for train/generate/eval/data commands.
- [x] Add negative-path tests for malformed data, bad checkpoints, and invalid CLI arguments.
- [x] Improve CLI error reporting with command context and distinct exit codes.
- [x] Add a documented policy for maximum input sizes and checkpoint compatibility.

## Hardware Acceleration

- [x] Add benchmark reporting for train steps/sec, generation tokens/sec, backend, and model config.
- [x] Add a local acceleration dependency check for platform libraries and tools.
- [x] Add Makefile backend framework linking for `BACKEND=metal` on macOS.
- [x] Add a backend selection surface such as `--backend cpu|metal|cuda`.
- [x] Add runtime backend capability reporting with `mgpt backends`.
- [x] Introduce backend-owned tensor buffer scaffolding while keeping CPU tensors as the reference path.
- [x] Add persistent Metal buffer create/read/write/destroy primitives.
- [x] Add cached Metal buffer usage for linear forward/backward.
- [x] Avoid redundant cached linear input/weight uploads between Metal forward and backward.
- [x] Add a resident linear benchmark that uploads once, runs repeated Metal work, and downloads once.
- [x] Reuse a shared Metal command queue and batch resident linear benchmark iterations into one command buffer.
- [x] Add backend upload/download counters to benchmarks so CPU synchronization overhead is visible.
- [x] Move linear forward/backward behind a backend operation interface with CPU fallback.
- [x] Add parity tests comparing linear operation outputs and gradients against CPU tolerances.
- [x] Add CPU/Metal train parity and Metal-checkpoint-on-CPU generation interop lab tests.
- [x] Move general matmul operations behind the backend operation interface.
- [x] Add a first Metal linear-forward kernel with CPU fallback.
- [x] Add Metal linear-backward kernels.
- [x] Move GELU behind backend operations and add Metal forward/backward kernels.
- [x] Move LayerNorm behind backend operations and add Metal forward/backward kernels.
- [x] Add feed-forward gradient coverage to catch activation derivative regressions.
- [x] Add a fused Metal feed-forward forward path that batches `linear -> GELU -> linear` in one command buffer.
- [x] Add a fused Metal feed-forward backward path that batches linear/GELU gradient work in one command buffer.
- [x] Add a staged CPU-vs-Metal parity harness that compares embeddings, block outputs, logits, gradients, and AdamW updates on fixed traces.
- [ ] Collect and review a CPU-vs-Metal staged comparison report on real Metal hardware and use it to pinpoint the first divergent stage.
- [ ] Add a strict no-fallback Metal mode so selected Metal runs fail when any phase silently drops back to CPU.
- [x] Add compact trace reporting for parity debugging so the first divergent tensor or update is easy to isolate.
- [ ] Add command-buffer batching to the full model execution path so small Metal operations do not pay one command submission per kernel.
- [x] Keep more activation tensors resident across adjacent backend operations.
- [x] Move optimizer updates behind the backend interface so Metal training does not download gradients before AdamW.
- [x] Test Metal locally on Apple hardware.
- [ ] Add broader Metal backend support for Apple hardware after the backend interface is stable.
- [ ] Add CUDA backend support for NVIDIA hardware after the backend interface is stable.
- [ ] Test CUDA only on NVIDIA hardware or CI with CUDA available.
- [ ] Keep checkpoints, data parsing, tokenizer behavior, and eval reports portable across backends.

## Tiny Assistant Roadmap

- [x] Add a dedicated `chat` executable for interactive assistant-style use.
- [x] Move chat prompt formatting and interactive loop into a reusable header.
- [x] Define the minimum feature set for the tiny assistant beyond single-turn chat.
- [x] Add a tracked single-turn assistant seed dataset.
- [x] Generate and validate assistant train/validation artifacts from the seed dataset.
- [x] Add assistant training and chat launcher scripts.
- [x] Train an assistant baseline checkpoint from the seed train split.
- [x] Evaluate the assistant baseline against train and validation splits.
- [ ] Improve the assistant baseline until single-turn train accuracy shows reliable memorization.
- [ ] Improve the assistant baseline until validation examples show useful held-out behavior.
- [ ] Add session-aware prompt formatting only after single-turn reliability stays stable.
- [x] Decide how assistant-style evaluation should report quality beyond exact match.
- [x] Document how other applications can embed the library or register custom tools.

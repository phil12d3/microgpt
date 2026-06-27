# microgpt

microgpt is a small C++17 language-model playground and assistant runtime.
It includes:

- a CLI for training, resuming, generating, evaluating, and data conversion
- a chat front end
- a tiny lab/test harness
- a C API and example embeddings
- CPU, Metal, and CUDA backends

## Project Layout

- `src/` contains the CLI entry points and runtime code.
- `include/microgpt/` contains the public headers and command implementations.
- `examples/` contains C and C++ embedding examples.
- `scripts/` contains repeatable training, chat, and smoke-test workflows.
- `sample_data/` contains tracked starter datasets.
- `artifacts/` stores generated datasets, checkpoints, and evaluation reports.
- `docs/` contains the longer-form project documentation.

## Build

Build everything with:

```bash
make BACKEND=cpu all
```

Useful variants:

```bash
make BACKEND=metal all
make BACKEND=cuda all
make test
make cli-test
make clean
```

Notes:

- `BACKEND=metal` requires macOS.
- `BACKEND=cuda` requires CUDA and the CUDA runtime libraries.
- `make test` runs the lab harness through `bin/lab`.
- `make cli-test` runs the CLI smoke test script.

## Binaries

The default build produces:

- `bin/microgpt`
- `bin/mgpt`
- `bin/lab`
- `bin/chat`
- `bin/embed_api_example`
- `bin/embed_c_example`
- `bin/libmicrogpt.a`

The `mgpt` binary exposes the main command set:

- `train`
- `resume`
- `generate`
- `eval`
- `validate-data`
- `split-data`
- `import-jsonl`
- `make-arithmetic-data`
- `list-artifacts`
- `clean-artifacts`
- `backends`
- `parity`
- `bench`
- `test`

Run `./bin/mgpt` with no arguments to print the built-in usage summary.

## Quick Start

Train on the bundled sample data:

```bash
./bin/mgpt train \
  --input sample_data/data.txt \
  --checkpoint artifacts/checkpoints/model.bin
```

Generate text from a checkpoint:

```bash
./bin/mgpt generate \
  --checkpoint artifacts/checkpoints/model.bin \
  --prompt "What is 1+1?"
```

Launch chat:

```bash
./bin/chat --checkpoint artifacts/checkpoints/model.bin
```

## Assistant Workflows

The repository includes scripted pipelines for the tracked assistant seed data.

Train from the JSONL seed:

```bash
./scripts/train_assistant.sh
```

Resume from an existing checkpoint:

```bash
./scripts/resume_assistant.sh
```

Open an interactive chat session against the assistant checkpoint:

```bash
./scripts/chat_assistant.sh
```

These scripts default to:

- `sample_data/assistant/seed.jsonl`
- `artifacts/datasets/assistant/`
- `artifacts/checkpoints/assistant-seed-bpe.bin`
- `artifacts/evals/`

You can override those paths and training parameters with environment
variables. See the script headers for the available knobs.

## Data Format

microgpt accepts both plain instruction datasets and session datasets.

Instruction example:

```text
<BOS><USER>
question or command
<ASSISTANT>
response
<EOS>
```

Session example:

```json
{"turns":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."}]}
```

Common data commands:

- `import-jsonl` converts JSONL source data into canonical training text
- `validate-data` checks a dataset before training
- `split-data` creates train and validation splits
- `make-arithmetic-data` generates a small synthetic corpus for smoke tests

## Public API

The library target can be linked from C++ or C.

- `include/microgpt/api.hpp` exposes the C++ API.
- `include/microgpt/api_c.h` exposes the C API.
- `examples/embed_api.cpp` shows C++ embedding.
- `examples/embed_c.c` shows C embedding.

## Documentation

The best starting point for the longer docs is:

- `docs/index.html`

That page links to the CLI reference, data format guide, workflow docs,
public API notes, and roadmap.

## Generated Artifacts

Generated files live under `artifacts/` by default:

- datasets in `artifacts/datasets/`
- checkpoints in `artifacts/checkpoints/`
- evaluation reports in `artifacts/evals/`

If you need to inspect or clean them, use the built-in artifact commands
instead of deleting files manually.


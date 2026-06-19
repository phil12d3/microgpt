# Tiny Assistant Spec

## Current Scope

The initial assistant surface is `bin/chat`. It loads a checkpoint and runs a
single-turn instruction prompt for each user input:

```text
<BOS><USER>
user prompt
<ASSISTANT>
```

The executable supports:

- `--checkpoint PATH`
- `--max-new-tokens N`
- `--temperature T`
- `--top-k K`
- `--greedy`
- `/exit` and `/quit`

## Minimum Useful Assistant

- Keep single-turn prompting as the default until the model reliably handles
  curated single-turn examples.
- Use `sample_data/assistant/seed.jsonl` as the tracked starter assistant
  dataset.
- Generate canonical artifacts under `artifacts/datasets/assistant/`.
- Current seed split: 94 total examples, 85 train examples, 9 validation
  examples with `--ratio 0.9 --seed 42`.
- Add `/reset` when session history exists.
- Add `/settings` or equivalent only after settings become mutable at runtime.
- Keep assistant training data in the canonical instruction format or JSONL
  import format.
- Evaluate assistant behavior with a mix of exact, prefix, and contains checks.

## Deferred Session Context

Multi-turn context should wait until there is:

- a documented prompt format for history,
- enough multi-turn training examples,
- a context-window policy for trimming old turns,
- evaluation data that checks whether history actually helps.

## Embedding Direction

Other applications should use `include/microgpt/chat.hpp` for chat behavior,
`include/microgpt/generation.hpp` for lower-level generation, and
`include/microgpt/tools.hpp` for registry-based CLI-style tools.

# Assistant Seed Data

This folder contains tracked single-turn assistant seed examples. The JSONL file
is the editable source of truth; generated training and validation files should
live under `artifacts/datasets/assistant/`.

Each line in `seed.jsonl` has this shape:

```json
{"user":"Prompt text","assistant":"Expected answer text"}
```

Build the canonical instruction dataset with:

```bash
./scripts/train_assistant.sh
```

The script imports, validates, splits, trains, and writes eval reports. To run
only the data preparation commands manually:

```bash
./bin/mgpt import-jsonl \
  --input sample_data/assistant/seed.jsonl \
  --output artifacts/datasets/assistant/all.txt

./bin/mgpt validate-data \
  --input artifacts/datasets/assistant/all.txt

./bin/mgpt split-data \
  --input artifacts/datasets/assistant/all.txt \
  --train artifacts/datasets/assistant/train.txt \
  --val artifacts/datasets/assistant/val.txt \
  --ratio 0.9 \
  --seed 42
```

Keep this dataset single-turn until the current assistant can reliably answer
held-out examples. Multi-turn data should use a separate documented format.

# Sample Data

This directory contains a tiny starter corpus for quick smoke tests.

Format:

```text
<BOS><USER>
question or command
<ASSISTANT>
response
<EOS>
```

Recommended first command for the tiny smoke corpus:

```bash
./bin/mgpt train --input sample_data/data.txt --checkpoint artifacts/checkpoints/smoke.bin --steps 1000
```

You can also point `--input` at `sample_data/data.txt` if you want to keep
generated checkpoints and training data separated.

The model will treat `<EOS>` as the stop sequence during generation.

The `assistant/` subdirectory contains the first tracked assistant seed dataset
in JSONL form. Convert it into canonical instruction text before training:

```bash
./bin/mgpt import-jsonl \
  --input sample_data/assistant/seed.jsonl \
  --output artifacts/datasets/assistant/all.txt
```

Note: checkpoints created before the special-token change are no longer
compatible. Retrain from this corpus to create a fresh `model.bin`.

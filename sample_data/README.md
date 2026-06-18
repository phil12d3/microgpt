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

Recommended first command:

```bash
./bin/microgpt train --input data.txt --checkpoint model.bin --steps 1000
```

You can also point `--input` at `sample_data/data.txt` if you want to keep
generated checkpoints and training data separated.

The model will treat `<EOS>` as the stop sequence during generation.

Note: checkpoints created before the special-token change are no longer
compatible. Retrain from this corpus to create a fresh `model.bin`.

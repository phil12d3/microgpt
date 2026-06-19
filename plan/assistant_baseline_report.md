# Assistant Baseline Report

## 2026-06-18 Seed Baseline

Dataset:

- Source: `sample_data/assistant/seed.jsonl`
- Imported: `artifacts/datasets/assistant/all.txt`
- Total examples: 94
- Train split: 85 examples
- Validation split: 9 examples
- Split command: `--ratio 0.9 --seed 42`

Training command:

```bash
./bin/mgpt train \
  --input artifacts/datasets/assistant/train.txt \
  --val-input artifacts/datasets/assistant/val.txt \
  --checkpoint artifacts/checkpoints/assistant-seed.bin \
  --steps 3000 \
  --context 128 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --ff 128 \
  --batch-size 8 \
  --lr 0.0008 \
  --eval-interval 500 \
  --save-interval 1000 \
  --progress-interval 500
```

Training result:

- Runtime: about 4 minutes 34 seconds.
- Final train loss: 1.1015.
- Final average loss: 1.1144.
- Best observed validation loss: 2.1099 at step 1500.
- Final validation loss: 2.2114.

Evaluation:

```bash
./bin/mgpt eval \
  --checkpoint artifacts/checkpoints/assistant-seed.bin \
  --input artifacts/datasets/assistant/train.txt \
  --max-new-tokens 80 \
  --match contains \
  --greedy \
  --hide-failures \
  --output artifacts/evals/assistant-seed-train.json

./bin/mgpt eval \
  --checkpoint artifacts/checkpoints/assistant-seed.bin \
  --input artifacts/datasets/assistant/val.txt \
  --max-new-tokens 80 \
  --match contains \
  --greedy \
  --output artifacts/evals/assistant-seed-val.json
```

Results:

- Train contains accuracy: 11/85, 12.94%.
- Validation contains accuracy: 0/9, 0.00%.

Conclusion:

The assistant data pipeline is now real and measurable, but this baseline is
not ready for session context. The next step is to make the single-turn baseline
memorize or reliably answer this seed set by trying a smaller curriculum,
longer training, or a larger model.

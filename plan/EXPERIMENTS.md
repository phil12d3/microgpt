# microgpt Experiment Matrix

Use this file to run scaling experiments in a repeatable way. Each experiment should record:

- Dataset command
- Train command
- Final train and validation loss
- Exact-match train accuracy
- Exact-match validation accuracy
- Notes about generation failures

Keep generated datasets and checkpoints out of git unless there is a deliberate reason to preserve them.

## Dataset Setup

Small training split:

```sh
./bin/mgpt make-arithmetic-data \
  --output /private/tmp/microgpt-train-small.txt \
  --max-a 9 \
  --max-b 9
```

Small validation split:

```sh
./bin/mgpt make-arithmetic-data \
  --output /private/tmp/microgpt-val-small.txt \
  --max-a 12 \
  --max-b 9
```

The validation file intentionally includes examples beyond the training range. Use exact-match eval to distinguish memorization from extrapolation.

## Baseline Tiny

```sh
./bin/mgpt train \
  --input /private/tmp/microgpt-train-small.txt \
  --val-input /private/tmp/microgpt-val-small.txt \
  --checkpoint /private/tmp/microgpt-baseline-tiny.bin \
  --steps 3000 \
  --context 64 \
  --d-model 32 \
  --layers 1 \
  --heads 4 \
  --ff 64 \
  --batch-size 4 \
  --lr 0.001 \
  --eval-interval 500 \
  --save-interval 1000 \
  --progress-interval 500
```

```sh
./bin/mgpt eval \
  --checkpoint /private/tmp/microgpt-baseline-tiny.bin \
  --input /private/tmp/microgpt-train-small.txt \
  --max-new-tokens 20 \
  --greedy \
  --hide-failures
```

```sh
./bin/mgpt eval \
  --checkpoint /private/tmp/microgpt-baseline-tiny.bin \
  --input /private/tmp/microgpt-val-small.txt \
  --max-new-tokens 20 \
  --greedy \
  --hide-failures
```

## Baseline Medium

```sh
./bin/mgpt train \
  --input /private/tmp/microgpt-train-small.txt \
  --val-input /private/tmp/microgpt-val-small.txt \
  --checkpoint /private/tmp/microgpt-baseline-medium.bin \
  --steps 6000 \
  --context 64 \
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

Evaluate train and validation accuracy with the same commands as above, swapping in `microgpt-baseline-medium.bin`.

## Scaling Axes

Change one axis at a time:

| Axis | Values |
| --- | --- |
| Data range | `9x9`, `19x9`, `49x9`, `99x9` |
| Steps | `1000`, `3000`, `6000`, `10000` |
| Width | `32`, `64`, `128` |
| Layers | `1`, `2`, `4` |
| Context | `32`, `64`, `128` |
| Batch size | `4`, `8`, `16` |

## Interpretation

- High train accuracy and low validation accuracy means memorization.
- High train and validation accuracy within the same numeric range means interpolation.
- Accuracy on held-out larger numbers or changed wording is extrapolation and is harder.
- If train accuracy does not rise, reduce the dataset size or increase model capacity before running longer.

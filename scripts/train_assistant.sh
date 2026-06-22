#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${ASSISTANT_SEED:=sample_data/assistant/seed.jsonl}"
: "${ASSISTANT_DATA_DIR:=artifacts/datasets/assistant}"
: "${ASSISTANT_CHECKPOINT:=artifacts/checkpoints/assistant-seed-bpe.bin}"
: "${ASSISTANT_EVAL_DIR:=artifacts/evals}"
: "${ASSISTANT_BACKEND:=cuda}"
: "${ASSISTANT_FORMAT:=single}"
: "${ASSISTANT_TOKENIZER:=bpe}"
: "${ASSISTANT_SPLIT_RATIO:=0.9}"
: "${ASSISTANT_SPLIT_SEED:=42}"
: "${STEPS:=12000}"
: "${CONTEXT:=128}"
: "${D_MODEL:=64}"
: "${LAYERS:=2}"
: "${HEADS:=4}"
: "${FF:=128}"
: "${BATCH_SIZE:=8}"
: "${LR:=0.0008}"
: "${EVAL_INTERVAL:=500}"
: "${SAVE_INTERVAL:=1000}"
: "${PROGRESS_INTERVAL:=500}"
: "${MAX_NEW_TOKENS:=80}"
: "${MATCH:=contains}"

ALL_DATA="$ASSISTANT_DATA_DIR/all.txt"
TRAIN_DATA="$ASSISTANT_DATA_DIR/train.txt"
VAL_DATA="$ASSISTANT_DATA_DIR/val.txt"
CHECKPOINT_NAME="$(basename "$ASSISTANT_CHECKPOINT")"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.*}"
TRAIN_REPORT="$ASSISTANT_EVAL_DIR/$CHECKPOINT_NAME-train.json"
VAL_REPORT="$ASSISTANT_EVAL_DIR/$CHECKPOINT_NAME-val.json"

run_eval() {
  set +e
  "$@"
  local status=$?
  set -e
  if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then
    exit "$status"
  fi
}

echo "building tools"
make BACKEND="$ASSISTANT_BACKEND" all

mkdir -p "$ASSISTANT_DATA_DIR" "$(dirname "$ASSISTANT_CHECKPOINT")" "$ASSISTANT_EVAL_DIR"

echo "importing assistant JSONL"
./bin/mgpt import-jsonl \
  --input "$ASSISTANT_SEED" \
  --output "$ALL_DATA" \
  --format "$ASSISTANT_FORMAT"

echo "validating assistant dataset"
./bin/mgpt validate-data \
  --input "$ALL_DATA" \
  --format "$ASSISTANT_FORMAT"

echo "splitting assistant dataset"
./bin/mgpt split-data \
  --input "$ALL_DATA" \
  --train "$TRAIN_DATA" \
  --val "$VAL_DATA" \
  --ratio "$ASSISTANT_SPLIT_RATIO" \
  --seed "$ASSISTANT_SPLIT_SEED" \
  --format "$ASSISTANT_FORMAT"

echo "training assistant checkpoint"
./bin/mgpt train \
  --input "$TRAIN_DATA" \
  --val-input "$VAL_DATA" \
  --checkpoint "$ASSISTANT_CHECKPOINT" \
  --steps "$STEPS" \
  --context "$CONTEXT" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --ff "$FF" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --tokenizer "$ASSISTANT_TOKENIZER" \
  --eval-interval "$EVAL_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --progress-interval "$PROGRESS_INTERVAL" \
  --backend "$ASSISTANT_BACKEND"

echo "evaluating train split"
run_eval ./bin/mgpt eval \
  --checkpoint "$ASSISTANT_CHECKPOINT" \
  --input "$TRAIN_DATA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --match "$MATCH" \
  --greedy \
  --hide-failures \
  --output "$TRAIN_REPORT" \
  --format "$ASSISTANT_FORMAT" \
  --backend "$ASSISTANT_BACKEND"

echo "evaluating validation split"
run_eval ./bin/mgpt eval \
  --checkpoint "$ASSISTANT_CHECKPOINT" \
  --input "$VAL_DATA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --match "$MATCH" \
  --greedy \
  --output "$VAL_REPORT" \
  --format "$ASSISTANT_FORMAT" \
  --backend "$ASSISTANT_BACKEND"

echo "assistant training pipeline complete"
echo "checkpoint: $ASSISTANT_CHECKPOINT"
echo "train report: $TRAIN_REPORT"
echo "validation report: $VAL_REPORT"
echo "note: eval exits with status 2 when accuracy is below 100%; this script keeps the reports."

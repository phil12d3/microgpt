#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN="$ROOT_DIR/bin/mgpt"

if [ ! -x "$BIN" ]; then
  echo "missing executable: $BIN" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/microgpt-cli.XXXXXX")
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT HUP INT TERM

DATA="$TMP_DIR/arithmetic.txt"
TRAIN="$TMP_DIR/train.txt"
VAL="$TMP_DIR/val.txt"
JSONL="$TMP_DIR/pairs.jsonl"
IMPORTED="$TMP_DIR/imported.txt"
MODEL="$TMP_DIR/model.bin"
FULL_MODEL="$TMP_DIR/model-full.bin"
REPORT="$TMP_DIR/eval.json"
FULL_REPORT="$TMP_DIR/eval-full.json"
OUT="$TMP_DIR/out.txt"
ERR="$TMP_DIR/err.txt"

"$BIN" make-arithmetic-data --output "$DATA" --max-a 2 --max-b 2 >"$OUT"
"$BIN" validate-data --input "$DATA" >"$OUT"
"$BIN" split-data --input "$DATA" --train "$TRAIN" --val "$VAL" --ratio 0.7 --seed 7 >"$OUT"

printf '%s\n' \
  '{"user":"Say one","assistant":"one"}' \
  '{"user":"Say two","assistant":"two"}' >"$JSONL"
"$BIN" import-jsonl --input "$JSONL" --output "$IMPORTED" >"$OUT"
"$BIN" validate-data --input "$IMPORTED" >"$OUT"

"$BIN" train \
  --input "$TRAIN" \
  --val-input "$VAL" \
  --checkpoint "$MODEL" \
  --steps 2 \
  --context 8 \
  --d-model 8 \
  --layers 0 \
  --heads 1 \
  --ff 16 \
  --batch-size 2 \
  --lr 0.001 \
  --eval-interval 100 \
  --save-interval 100 \
  --progress-interval 2 >"$OUT"

test -s "$MODEL"
test -s "${MODEL%.bin}.json"

"$BIN" train \
  --input "$DATA" \
  --no-val-split \
  --checkpoint "$FULL_MODEL" \
  --steps 1 \
  --context 8 \
  --d-model 8 \
  --layers 0 \
  --heads 1 \
  --ff 16 \
  --batch-size 2 \
  --lr 0.001 \
  --eval-interval 100 \
  --save-interval 100 \
  --progress-interval 1 >"$OUT"

test -s "$FULL_MODEL"
grep -q '"validation_source": "full_data"' "${FULL_MODEL%.bin}.json"

"$BIN" generate --checkpoint "$MODEL" --prompt "What is 1+1?" --max-new-tokens 2 --greedy --quiet >"$OUT"
"$BIN" bench --train-steps 1 --gen-tokens 1 >"$OUT"
"$BIN" list-artifacts --root "$TMP_DIR" >"$OUT"

set +e
"$BIN" eval --checkpoint "$MODEL" --input "$VAL" --max-examples 1 --max-new-tokens 1 --greedy --hide-failures --output "$REPORT" >"$OUT"
eval_status=$?
set -e
if [ "$eval_status" -ne 0 ] && [ "$eval_status" -ne 2 ]; then
  echo "eval smoke command failed with unexpected status $eval_status" >&2
  exit 1
fi
test -s "$REPORT"

set +e
"$BIN" eval --checkpoint "$FULL_MODEL" --input "$DATA" --max-examples 1 --max-new-tokens 1 --greedy --hide-failures --output "$FULL_REPORT" >"$OUT"
full_eval_status=$?
set -e
if [ "$full_eval_status" -ne 0 ] && [ "$full_eval_status" -ne 2 ]; then
  echo "full-data eval smoke command failed with unexpected status $full_eval_status" >&2
  exit 1
fi
test -s "$FULL_REPORT"

printf 'not a valid instruction dataset\n' >"$TMP_DIR/bad.txt"
set +e
"$BIN" validate-data --input "$TMP_DIR/bad.txt" >"$OUT" 2>"$ERR"
bad_validate_status=$?
"$BIN" generate --checkpoint "$TMP_DIR/missing.bin" --prompt x --max-new-tokens 1 >"$OUT" 2>"$ERR"
missing_checkpoint_status=$?
"$BIN" split-data --input "$DATA" --train "$TRAIN" --val "$VAL" --ratio 2.0 >"$OUT" 2>"$ERR"
bad_arg_status=$?
"$BIN" >"$OUT" 2>"$ERR"
usage_status=$?
set -e

if [ "$bad_validate_status" -eq 0 ]; then
  echo "validate-data unexpectedly accepted malformed input" >&2
  exit 1
fi
if [ "$missing_checkpoint_status" -eq 0 ]; then
  echo "generate unexpectedly accepted missing checkpoint" >&2
  exit 1
fi
if [ "$bad_arg_status" -eq 0 ]; then
  echo "split-data unexpectedly accepted invalid ratio" >&2
  exit 1
fi
if [ "$usage_status" -ne 64 ]; then
  echo "missing command returned $usage_status, expected 64" >&2
  exit 1
fi

echo "cli_smoke_test pass"

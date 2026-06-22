#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${ASSISTANT_CHECKPOINT:=artifacts/checkpoints/assistant-seed-bpe.bin}"
: "${MAX_NEW_TOKENS:=120}"
: "${TEMPERATURE:=0.2}"
: "${TOP_K:=1}"
: "${GREEDY:=1}"
: "${ASSISTANT_BACKEND:=cuda}"

if [ ! -f "$ASSISTANT_CHECKPOINT" ]; then
  echo "missing checkpoint: $ASSISTANT_CHECKPOINT" >&2
  echo "run ./scripts/train_assistant.sh first, or set ASSISTANT_CHECKPOINT=/path/to/model.bin" >&2
  exit 1
fi

make BACKEND="$ASSISTANT_BACKEND" all

if [ "$GREEDY" = "1" ]; then
  exec ./bin/chat \
    --checkpoint "$ASSISTANT_CHECKPOINT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --backend "$ASSISTANT_BACKEND" \
    --greedy
fi

exec ./bin/chat \
  --checkpoint "$ASSISTANT_CHECKPOINT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --backend "$ASSISTANT_BACKEND"

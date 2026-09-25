#!/usr/bin/env bash
# Full research pipeline, in order:
#   data + target construction
#   -> simple baselines (record naive-split result, set aside)   [Phase 2]
#   -> complex model                                              [Phase 3]
#   -> walk-forward validation applied to both                    [Phase 4]
#   -> naive vs. walk-forward comparison and verdict              [Phase 5]
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "== Phase 1: data =="
"$PYTHON" -m src.data "$@"

echo "Later stages are not implemented yet (see TASKS.md)."

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WINDOWS=(2 3 4 5)
FOLDS=(0 1 2 3 4)

for window in "${WINDOWS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "Training RidgeCV for window ${window} fold ${fold}"
    python "${ROOT}/scripts/train_linear.py" --window "${window}" --fold "${fold}"
  done
done


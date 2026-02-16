#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Resolve Python interpreter for environments where `python` is unavailable
# (e.g., Git Bash on Windows often exposes `python3` or `python.exe` only).
PYTHON_CMD=()
if command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v py.exe >/dev/null 2>&1; then
  PYTHON_CMD=(py.exe -3)
else
  echo "[ERROR] Could not find a Python interpreter (python/python3/python.exe/py)." >&2
  exit 1
fi
echo "Using Python interpreter: ${PYTHON_CMD[*]}"

# Figure 3 corresponds to experiment 1:
# - 5 tasks, 5 epochs per task
# - MLP with 2 hidden layers of 100 units
# - Compare Plastic (Naive) vs Stable regimes on Rotated/Permuted MNIST
# - Compute top-20 Hessian eigenvalues for the middle row in Fig.3
TASKS=5
EPOCHS_PER_TASK=5
HIDDENS=100
NUM_EIGENTHINGS=20
SEEDS=(1234)

run_regime () {
  local dataset="$1"
  local regime_name="$2"
  local lr="$3"
  local gamma="$4"
  local batch_size="$5"
  local dropout="$6"

  echo " >>>>>>>> ${regime_name} (${dataset})"
  for seed in "${SEEDS[@]}"; do
    echo "seed=${seed}, lr=${lr}, gamma=${gamma}, bs=${batch_size}, dropout=${dropout}"
    "${PYTHON_CMD[@]}" -m stable_sgd.main \
      --dataset "${dataset}" \
      --tasks "${TASKS}" \
      --epochs-per-task "${EPOCHS_PER_TASK}" \
      --lr "${lr}" \
      --gamma "${gamma}" \
      --hiddens "${HIDDENS}" \
      --batch-size "${batch_size}" \
      --dropout "${dropout}" \
      --seed "${seed}" \
      --compute-eigenspectrum \
      --num-eigenthings "${NUM_EIGENTHINGS}"
  done
  echo ""
}

echo "************************ replicating Figure 3 (rotated MNIST) ***********************"
run_regime "rot-mnist" "Plastic (Naive) SGD" "0.01" "1.0" "64" "0.0"
run_regime "rot-mnist" "Stable SGD" "0.1" "0.4" "16" "0.25"

echo "************************ replicating Figure 3 (permuted MNIST) ***********************"
run_regime "perm-mnist" "Plastic (Naive) SGD" "0.01" "1.0" "64" "0.0"
run_regime "perm-mnist" "Stable SGD" "0.1" "0.4" "16" "0.25"

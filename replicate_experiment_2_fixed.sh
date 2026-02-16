#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Resolve Python interpreter for environments where `python` is unavailable.
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

# Experiment 2: 20 tasks, 1 epoch/task
TASKS=20
EPOCHS_PER_TASK=1
HIDDENS=256

# Paper reports 5-run averages.
SEEDS=(1234)

run_sgd () {
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
      --seed "${seed}"
  done
  echo ""
}

echo "************************ replicating experiment 2 (rotated MNIST) ***********************"
run_sgd "rot-mnist" "Plastic (Naive) SGD" "0.01" "1.0" "10" "0.0"
# Stable params follow the authors' issue #5 update (2020-10-12):
# lr=0.1, gamma=0.55, batch_size=64, dropout=0.25
run_sgd "rot-mnist" "Stable SGD" "0.1" "0.55" "64" "0.25"

echo "************************ replicating experiment 2 (permuted MNIST) ***********************"
run_sgd "perm-mnist" "Plastic (Naive) SGD" "0.01" "1.0" "10" "0.0"
# No dedicated perm-mnist stable setting was posted in issue #5.
# Use a stability-oriented setting compatible with 20-task MNIST runs.
run_sgd "perm-mnist" "Stable SGD" "0.1" "0.8" "64" "0.25"

echo "************************ replicating experiment 2 (Split CIFAR-100) ***********************"
run_sgd "cifar100" "Plastic (Naive) SGD" "0.01" "1.0" "10" "0.0"
# Stable params follow the authors' issue #5 update (2020-10-12):
# lr=0.15, gamma=0.85, batch_size=10, dropout=0.1
run_sgd "cifar100" "Stable SGD" "0.15" "0.85" "10" "0.1"

if [[ "${RUN_BASELINES:-0}" == "1" ]]; then
  echo ""
  echo ">>>>>>>>> Other Methods (ER, A-GEM, EWC)"
  echo "NOTE: external baseline scripts are TF-based and may require additional fixes."
  cd ./external_libs/continual_learning_algorithms
  bash replicate_mnist.sh rot-mnist
  bash replicate_mnist.sh perm-mnist
  bash replicate_cifar.sh
  cd ../..
fi

# bash replicate_experiment_2_fixed.sh

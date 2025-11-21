#!/usr/bin/env bash


set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------- base setup -----------------------------------

DATASET="rot-mnist"      # you can change to perm-mnist if desired
TASKS=2                  # as requested: 2-task setup
EPOCHS_PER_TASK=3        # 3 epochs per task, like the short cd experiments
HIDDENS=100              # same scale as Experiment 1 MLP

# Hyperparameter sweeps:
# - LR / GAMMA roughly move between naive-SGD and stable-SGD regimes
# - BATCH_SIZE / DROPOUT also interpolate plastic vs. stable behavior
LR_LIST=(0.01 0.1)
GAMMA_LIST=(1.0 0.4)
BATCH_SIZE_LIST=(64 16)
DROPOUT_LIST=(0.0 0.5)
SEEDS=(7891 1145 9723)

# Where to record mapping from hyperparameters to outputs directory
RESULTS_CSV="fig2_cd_runs.csv"

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "dataset,tasks,epochs_per_task,lr,gamma,batch_size,dropout,hiddens,seed,experiment_dir" > "${RESULTS_CSV}"
fi

echo "************************ running Fig.2 (c,d) cd-style grid ************************"
echo "Results mapping will be appended to ${RESULTS_CSV}"
echo ""

# ---------------------------- main sweep loop --------------------------------

for SEED in "${SEEDS[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for GAMMA in "${GAMMA_LIST[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
        for DROPOUT in "${DROPOUT_LIST[@]}"; do

          echo "----------------------------------------------------------------"
          echo "Running run with:"
          echo "  dataset=${DATASET}"
          echo "  tasks=${TASKS}, epochs_per_task=${EPOCHS_PER_TASK}"
          echo "  lr=${LR}, gamma=${GAMMA}"
          echo "  batch_size=${BATCH_SIZE}, dropout=${DROPOUT}"
          echo "  hiddens=${HIDDENS}, seed=${SEED}"
          echo "----------------------------------------------------------------"

          # Run one Stable-SGD experiment.
          # NOTE: --compute-eigenspectrum is needed so that analyze_fig2_point.py
          # can read hessian_eigs.csv and compute λ_max for task 1.
          python -m stable_sgd.main \
            --dataset "${DATASET}" \
            --tasks "${TASKS}" \
            --epochs-per-task "${EPOCHS_PER_TASK}" \
            --lr "${LR}" \
            --gamma "${GAMMA}" \
            --hiddens "${HIDDENS}" \
            --batch-size "${BATCH_SIZE}" \
            --dropout "${DROPOUT}" \
            --seed "${SEED}" \
            --compute-eigenspectrum

          # After the run finishes, the newest directory under ./outputs
          # corresponds to this experiment (TRIAL_ID in stable_sgd.utils).
          # We reuse analyze_fig2_point.find_latest_experiment_dir to resolve it.
          EXP_DIR=$(python - << 'EOF'
from analyze_fig2_point import find_latest_experiment_dir
print(find_latest_experiment_dir("outputs"))
EOF
)

          echo "Latest experiment directory: ${EXP_DIR}"

          # Append a single CSV row with hyperparameters + experiment directory.
          echo "${DATASET},${TASKS},${EPOCHS_PER_TASK},${LR},${GAMMA},${BATCH_SIZE},${DROPOUT},${HIDDENS},${SEED},${EXP_DIR}" >> "${RESULTS_CSV}"

        done
      done
    done
  done
done

echo ""
echo "All runs finished. Hyperparameter-to-output mapping saved in ${RESULTS_CSV}."



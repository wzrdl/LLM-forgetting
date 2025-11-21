import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analyze_fig2_point import (
    compute_F1,
    compute_F1_loss,
    compute_delta_w_sq,
    load_lambda_max_task1,
)


def main(
    runs_csv: str = "fig2_cd_runs.csv",
    out_png: str = "fig2c_from_runs.png",
    y_metric: str = "acc",  # "acc" (original) or "loss" (loss-based forgetting)
):
    """
    Aggregate all runs recorded in `fig2_cd_runs.csv` and draw a Fig.2(c)-style scatter plot:
        x = λ_max^(1) * ||Δw||^2
        y = F1 (accuracy-based)  or  F1_loss (loss-based), depending on `y_metric`.

    Each point corresponds to one experiment (one row in runs_csv).
    """
    if not os.path.isfile(runs_csv):
        raise RuntimeError(f"{runs_csv} not found. Make sure you ran replicate_fig2_cd.sh first.")

    runs_df = pd.read_csv(runs_csv)

    records = []
    for _, row in runs_df.iterrows():
        exp_dir = row["experiment_dir"]

        try:
            # Filter: only keep runs whose final accuracy (averaged over tasks)
            # is at least 90%. This mirrors the paper's practice of discarding
            # clearly under-trained runs.
            acc_path = os.path.join(exp_dir, "accs.csv")
            if not os.path.isfile(acc_path):
                print(f"[WARN] Skipping {exp_dir} (no accs.csv)")
                continue
            acc_df = pd.read_csv(acc_path, index_col=0)
            # final accuracy per task = last time index for each task column
            final_accs = []
            for col in acc_df.columns:
                try:
                    final_accs.append(float(acc_df[col].iloc[-1]))
                except Exception:
                    continue
            if not final_accs:
                print(f"[WARN] Skipping {exp_dir} (no valid accuracy entries)")
                continue
            avg_final_acc = sum(final_accs) / len(final_accs)
            if avg_final_acc < 90.0:
                print(f"[INFO] Skipping {exp_dir} due to avg final acc {avg_final_acc:.2f} < 90")
                continue

            F1 = compute_F1(exp_dir)
            F1_loss = compute_F1_loss(exp_dir)
            delta_w_sq = compute_delta_w_sq(exp_dir)
            lambda_max_1 = load_lambda_max_task1(exp_dir)
        except Exception as e:
            print(f"[WARN] Skipping {exp_dir} due to error: {e}")
            continue

        x = lambda_max_1 * delta_w_sq * 0.01
        # Choose which forgetting metric to plot on the y-axis.
        if y_metric == "loss":
            # Loss-based forgetting: F1_loss = L_1(w*_2) - L_1(w*_1)
            y = F1_loss
        else:
            # Default: accuracy-based forgetting as in the original script.
            y = F1

        records.append(
            {
                "x": x,
                "y": y,
                "F1_acc": F1,
                "F1_loss": F1_loss,
                "lr": row["lr"],
                "gamma": row["gamma"],
                "batch_size": row["batch_size"],
                "dropout": row["dropout"],
                "seed": row["seed"],
            }
        )

    if not records:
        raise RuntimeError("No valid experiment entries found to plot.")

    df = pd.DataFrame(records)

    # Draw a scatter plot mimicking Fig.2(c):
    # - x-axis: λ_max^(1) * ||Δw||^2 (log scale)
    # - y-axis: forgetting on task 1 (metric chosen by `y_metric`)
    # - color: gamma; marker style: batch size; size: dropout
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))

    ax = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="gamma",
        style="batch_size",
        size="dropout",
        palette="viridis",
        sizes=(40, 120),
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\max}^{(1)} \cdot \|\Delta w\|^2 (\times 10^{-2})$")
    if y_metric == "loss":
        ax.set_ylabel(r"$F_1^{\mathrm{loss}}$ (loss-based forgetting on task 1)")
        ax.set_title("Fig.2(c)-style plot (Rot-MNIST, 2 tasks, loss-based)")
    else:
        ax.set_ylabel(r"$F_1$ (forgetting on task 1)")
        ax.set_title("Fig.2(c)-style plot (Rot-MNIST, 2 tasks, accuracy-based)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    # Keep default behavior (no args) as the original accuracy-based plot.
    # You can call, e.g., main(y_metric="loss", out_png="fig2c_from_runs_loss.png")
    # from Python, or modify this block to parse CLI arguments if desired.
    main()



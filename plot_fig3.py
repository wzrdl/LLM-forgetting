import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_sgd.data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks
from stable_sgd.models import MLP


@dataclass(frozen=True)
class Condition:
    key: str
    dataset: str
    regime: str


CONDITIONS: Sequence[Condition] = (
    Condition("rot_plastic", "rot-mnist", "Plastic"),
    Condition("rot_stable", "rot-mnist", "Stable"),
    Condition("perm_plastic", "perm-mnist", "Plastic"),
    Condition("perm_stable", "perm-mnist", "Stable"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 3 from recent replicate outputs.")
    parser.add_argument("--outputs-root", default="outputs", help="Outputs directory root.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help=(
            "Explicit run directories in order: rot_plastic, rot_stable, perm_plastic, perm_stable. "
            "For multiple seeds, provide groups in that order per seed count."
        ),
    )
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--num-eigenthings", type=int, default=20)
    parser.add_argument("--hiddens", type=int, default=100)
    parser.add_argument("--dropout-plastic", type=float, default=0.0)
    parser.add_argument("--dropout-stable", type=float, default=0.25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--out", default="fig3_reproduced.png")
    return parser.parse_args()


def find_recent_fig3_runs(
    outputs_root: str, tasks: int, epochs_per_task: int, num_conditions: int, num_seeds: int
) -> List[str]:
    expected_cols = tasks
    expected_last_epoch = tasks * epochs_per_task - 1
    candidates: List[Tuple[float, str]] = []

    for run_dir in glob.glob(os.path.join(outputs_root, "*")):
        h_path = os.path.join(run_dir, "hessian_eigs.csv")
        if not os.path.isfile(h_path):
            continue
        try:
            h_df = pd.read_csv(h_path, index_col=0)
        except Exception:
            continue
        if h_df.shape[1] != expected_cols:
            continue
        cols = list(h_df.columns)
        if f"task-{tasks}-epoch-{expected_last_epoch}" not in cols:
            continue
        candidates.append((os.path.getmtime(h_path), run_dir))

    needed = num_conditions * num_seeds
    if len(candidates) < needed:
        raise RuntimeError(
            f"Not enough Fig.3-like runs under {outputs_root}. Needed {needed}, found {len(candidates)}."
        )

    candidates.sort(key=lambda x: x[0])
    return [d for _, d in candidates[-needed:]]


def chunk_runs(run_dirs: Sequence[str], num_seeds: int) -> Dict[str, List[str]]:
    if len(run_dirs) != len(CONDITIONS) * num_seeds:
        raise RuntimeError(
            f"Expected {len(CONDITIONS) * num_seeds} run dirs, got {len(run_dirs)}."
        )
    grouped: Dict[str, List[str]] = {}
    i = 0
    for cond in CONDITIONS:
        grouped[cond.key] = list(run_dirs[i : i + num_seeds])
        i += num_seeds
    return grouped


def load_acc_curve(run_dir: str, tasks: int, epochs_per_task: int) -> np.ndarray:
    acc_path = os.path.join(run_dir, "accs.csv")
    acc_df = pd.read_csv(acc_path, index_col=0)
    curve = []
    for t in range(tasks * epochs_per_task):
        current_task = min(tasks, t // epochs_per_task + 1)
        cols = [str(i) for i in range(1, current_task + 1)]
        vals = [float(acc_df[c].iloc[t]) for c in cols]
        curve.append(float(np.mean(vals)))
    return np.asarray(curve, dtype=np.float32)


def load_spectrum_per_task(run_dir: str, tasks: int, epochs_per_task: int, num_eigs: int) -> np.ndarray:
    h_path = os.path.join(run_dir, "hessian_eigs.csv")
    h_df = pd.read_csv(h_path, index_col=0)
    rows = []
    for task_id in range(1, tasks + 1):
        epoch = task_id * epochs_per_task - 1
        key = f"task-{task_id}-epoch-{epoch}"
        if key not in h_df.columns:
            raise RuntimeError(f"Missing column {key} in {h_path}")
        vals = np.asarray(h_df[key].dropna().values, dtype=np.float32)
        rows.append(vals[:num_eigs])
    return np.stack(rows, axis=0)


def flatten_model_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def build_task1_loader(dataset: str, seed: int, batch_size: int = 256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dataset == "rot-mnist":
        tasks = get_rotated_mnist_tasks(num_tasks=5, batch_size=batch_size)
    elif dataset == "perm-mnist":
        tasks = get_permuted_mnist_tasks(num_tasks=5, batch_size=batch_size)
    else:
        raise RuntimeError(f"Unsupported dataset: {dataset}")
    return tasks[1]["test"]


def evaluate_task1_loss(model: nn.Module, loader, task_id: int = 1) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss().to("cpu")
    total = 0.0
    count = 0
    with torch.no_grad():
        for data, target in loader:
            out = model(data, task_id)
            loss = criterion(out, target)
            bs = int(data.shape[0])
            total += float(loss.item()) * bs
            count += bs
    return total / max(count, 1)


def load_checkpoint_vec(run_dir: str, time_step: int, hiddens: int, dropout: float) -> torch.Tensor:
    trial_id = os.path.basename(run_dir)
    ckpt_path = os.path.join(run_dir, f"model-{trial_id}-{time_step}.pth")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Missing checkpoint: {ckpt_path}")
    model = MLP(hiddens, {"dropout": dropout}).to("cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return flatten_model_params(model)


def compute_exact_and_quadratic(
    run_dir: str,
    dataset: str,
    seed: int,
    tasks: int,
    epochs_per_task: int,
    hiddens: int,
    dropout: float,
) -> Tuple[np.ndarray, np.ndarray]:
    task1_loader = build_task1_loader(dataset, seed, batch_size=256)

    t_points = [i * epochs_per_task for i in range(1, tasks + 1)]
    model = MLP(hiddens, {"dropout": dropout}).to("cpu")
    criterion = nn.CrossEntropyLoss().to("cpu")

    losses = []
    params = []
    trial_id = os.path.basename(run_dir)
    for t in t_points:
        ckpt_path = os.path.join(run_dir, f"model-{trial_id}-{t}.pth")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Missing checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for data, target in task1_loader:
                out = model(data, 1)
                loss = criterion(out, target)
                bs = int(data.shape[0])
                total += float(loss.item()) * bs
                count += bs
        losses.append(total / max(count, 1))
        params.append(flatten_model_params(model).numpy())

    base_loss = losses[0]
    exact_delta = np.asarray([l - base_loss for l in losses], dtype=np.float32)

    h_df = pd.read_csv(os.path.join(run_dir, "hessian_eigs.csv"), index_col=0)
    eigvals = np.asarray(h_df["task-1-epoch-4"].dropna().values, dtype=np.float32)
    eigvecs = np.load(os.path.join(run_dir, "task-1-epoch-4-vec.npy")).astype(np.float32)

    w1 = params[0]
    quad_delta = []
    for wt in params:
        dw = wt - w1
        coeffs = eigvecs @ dw
        quad = 0.5 * np.sum(eigvals * np.square(coeffs))
        quad_delta.append(float(quad))
    return exact_delta, np.asarray(quad_delta, dtype=np.float32)


def aggregate_mean_std(arrs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(arrs, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def main() -> None:
    args = parse_args()

    if args.run_dirs:
        run_dirs = args.run_dirs
    else:
        run_dirs = find_recent_fig3_runs(
            outputs_root=args.outputs_root,
            tasks=args.tasks,
            epochs_per_task=args.epochs_per_task,
            num_conditions=len(CONDITIONS),
            num_seeds=len(args.seeds),
        )

    grouped = chunk_runs(run_dirs, num_seeds=len(args.seeds))
    print("Using run directories:")
    for cond in CONDITIONS:
        print(f"  {cond.key}:")
        for d in grouped[cond.key]:
            print(f"    - {d}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    task_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x_acc = np.arange(1, args.tasks * args.epochs_per_task + 1)
    x_rank = np.arange(1, args.num_eigenthings + 1)
    x_task = np.arange(1, args.tasks + 1)

    for col, cond in enumerate(CONDITIONS):
        cond_runs = grouped[cond.key]
        is_stable = cond.regime.lower() == "stable"
        dropout = args.dropout_stable if is_stable else args.dropout_plastic

        acc_curves = [load_acc_curve(d, args.tasks, args.epochs_per_task) for d in cond_runs]
        acc_mean, acc_std = aggregate_mean_std(acc_curves)
        ax = axes[0, col]
        ax.plot(x_acc, acc_mean, color="#111111", linewidth=2)
        ax.fill_between(x_acc, acc_mean - acc_std, acc_mean + acc_std, color="#999999", alpha=0.25)
        for b in range(args.epochs_per_task, args.tasks * args.epochs_per_task, args.epochs_per_task):
            ax.axvline(b, linestyle="--", linewidth=0.8, color="#cccccc")
        ax.set_ylim(60, 100)
        ax.set_title(f"{cond.dataset} | {cond.regime}")
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Avg test acc (%)")

        spec_arrs = [
            load_spectrum_per_task(d, args.tasks, args.epochs_per_task, args.num_eigenthings) for d in cond_runs
        ]
        spec_mean, _ = aggregate_mean_std(spec_arrs)
        ax = axes[1, col]
        for task_idx in range(args.tasks):
            ax.plot(x_rank, spec_mean[task_idx], color=task_colors[task_idx], linewidth=1.8, label=f"T{task_idx+1}")
        ax.set_xlabel("Eigenvalue rank")
        if col == 0:
            ax.set_ylabel("Hessian eigenvalue")
        if col == 3:
            ax.legend(frameon=False, fontsize=8)

        exact_all = []
        quad_all = []
        for seed, run_dir in zip(args.seeds, cond_runs):
            exact, quad = compute_exact_and_quadratic(
                run_dir=run_dir,
                dataset=cond.dataset,
                seed=seed,
                tasks=args.tasks,
                epochs_per_task=args.epochs_per_task,
                hiddens=args.hiddens,
                dropout=dropout,
            )
            exact_all.append(exact)
            quad_all.append(quad)
        exact_mean, exact_std = aggregate_mean_std(exact_all)
        quad_mean, quad_std = aggregate_mean_std(quad_all)
        ax = axes[2, col]
        ax.plot(x_task, exact_mean, color="#2ca02c", linewidth=2, label="Exact")
        ax.fill_between(x_task, exact_mean - exact_std, exact_mean + exact_std, color="#2ca02c", alpha=0.2)
        ax.plot(x_task, quad_mean, color="#d62728", linewidth=2, linestyle="--", label="Quadratic")
        ax.fill_between(x_task, quad_mean - quad_std, quad_mean + quad_std, color="#d62728", alpha=0.15)
        ax.set_xlabel("Task index")
        if col == 0:
            ax.set_ylabel(r"$\Delta L_1$")
        if col == 3:
            ax.legend(frameon=False, fontsize=8)

    axes[0, 0].set_title(f"{CONDITIONS[0].dataset} | {CONDITIONS[0].regime}")
    fig.suptitle("Figure 3 (reproduced from local runs)", fontsize=14)
    fig.savefig(args.out, dpi=250)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()

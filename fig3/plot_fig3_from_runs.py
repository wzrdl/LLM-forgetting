import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


TASK_COLORS = {
    1: "#0B57D0",  # blue
    2: "#D93025",  # red
    3: "#E0A800",  # yellow/orange
    4: "#12B886",  # green
    5: "#A020F0",  # purple
}


def nanmean_no_warn(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.sum(~np.isnan(arr), axis=axis)
    total = np.nansum(arr, axis=axis)
    out = total / np.maximum(valid, 1)
    out = np.where(valid == 0, np.nan, out)
    return out


def nanstd_no_warn(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = np.expand_dims(nanmean_no_warn(arr, axis=axis), axis=axis)
    diff = arr - mean
    diff[np.isnan(arr)] = np.nan
    valid = np.sum(~np.isnan(arr), axis=axis)
    var = np.nansum(diff ** 2, axis=axis) / np.maximum(valid, 1)
    out = np.sqrt(var)
    out = np.where(valid == 0, np.nan, out)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a Figure-3-style reproduction from Stable-CL experiment outputs."
    )
    parser.add_argument("--outputs-root", default="outputs", type=str)
    parser.add_argument("--save-path", default="fig3/fig3_reproduced.png", type=str)
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help=(
            "Optional explicit run directories in replicate_fig3 order: "
            "rot-plastic, rot-stable, perm-plastic, perm-stable (repeat by seed)."
        ),
    )
    parser.add_argument("--tasks", default=5, type=int)
    parser.add_argument("--eigenthings", default=20, type=int)
    return parser.parse_args()


def parse_seed_count_from_replicate_script(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"SEEDS=\(([^)]*)\)", text)
    if not m:
        return 0
    payload = m.group(1).strip()
    if not payload:
        return 0
    values = [x for x in payload.split() if x.strip()]
    return len(values)


def is_fig3_candidate(exp_dir: Path, tasks: int, eigenthings: int) -> bool:
    acc_path = exp_dir / "accs.csv"
    hess_path = exp_dir / "hessian_eigs.csv"
    if not (acc_path.exists() and hess_path.exists()):
        return False
    try:
        hdf = pd.read_csv(hess_path, index_col=0)
    except Exception:
        return False

    task_cols = [c for c in hdf.columns if c.startswith("task-")]
    if len(task_cols) != tasks:
        return False
    if hdf.shape[0] < eigenthings:
        return False

    model_files = list(exp_dir.glob("model-*.pth"))
    if len(model_files) < tasks:
        return False
    return True


def discover_runs(outputs_root: Path, tasks: int, eigenthings: int) -> list[Path]:
    candidates = []
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        if is_fig3_candidate(d, tasks, eigenthings):
            candidates.append(d)
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates


def select_recent_block(candidates: list[Path], seed_count_hint: int) -> list[Path]:
    if len(candidates) < 4:
        raise RuntimeError(
            f"Need at least 4 Figure-3-style runs, but found only {len(candidates)}."
        )

    if seed_count_hint > 0:
        expected = 4 * seed_count_hint
        if len(candidates) >= expected:
            return candidates[-expected:]

    # Fallback: use latest 4 runs.
    return candidates[-4:]


def split_by_regime(run_dirs: list[Path]) -> dict[tuple[str, str], list[Path]]:
    if len(run_dirs) % 4 != 0:
        raise RuntimeError(
            "Number of selected runs must be divisible by 4 "
            "(rot-plastic, rot-stable, perm-plastic, perm-stable)."
        )
    n = len(run_dirs) // 4
    groups = {
        ("rotated", "plastic"): run_dirs[0:n],
        ("rotated", "stable"): run_dirs[n : 2 * n],
        ("permuted", "plastic"): run_dirs[2 * n : 3 * n],
        ("permuted", "stable"): run_dirs[3 * n : 4 * n],
    }
    return groups


def parse_task_end_indices_from_hessian(hess_columns: list[str], tasks: int) -> list[int]:
    indices = []
    for t in range(1, tasks + 1):
        cols = [c for c in hess_columns if c.startswith(f"task-{t}-epoch-")]
        if not cols:
            raise RuntimeError(f"Missing Hessian column for task {t}.")
        m = re.search(rf"task-{t}-epoch-(\d+)", cols[0])
        if not m:
            raise RuntimeError(f"Could not parse epoch index from {cols[0]}.")
        indices.append(int(m.group(1)))
    return indices


def load_top_accuracy_matrix(exp_dir: Path, tasks: int) -> np.ndarray:
    acc = pd.read_csv(exp_dir / "accs.csv", index_col=0)
    hess = pd.read_csv(exp_dir / "hessian_eigs.csv", index_col=0)
    end_indices = parse_task_end_indices_from_hessian(list(hess.columns), tasks)

    mat = np.full((tasks, tasks), np.nan, dtype=float)
    for learned_task in range(1, tasks + 1):
        row_idx = end_indices[learned_task - 1]
        if row_idx >= len(acc):
            raise RuntimeError(
                f"Row index {row_idx} out of bounds in {exp_dir / 'accs.csv'}."
            )
        for eval_task in range(1, learned_task + 1):
            mat[eval_task - 1, learned_task - 1] = float(acc[str(eval_task)].iloc[row_idx])
    return mat


def load_hessian_spectra(exp_dir: Path, tasks: int, eigenthings: int) -> np.ndarray:
    hess = pd.read_csv(exp_dir / "hessian_eigs.csv", index_col=0)
    arr = np.full((tasks, eigenthings), np.nan, dtype=float)
    for t in range(1, tasks + 1):
        cols = [c for c in hess.columns if c.startswith(f"task-{t}-epoch-")]
        if not cols:
            raise RuntimeError(f"Missing Hessian column for task {t} in {exp_dir}.")
        vec = hess[cols[0]].to_numpy(dtype=float)
        arr[t - 1, :] = vec[:eigenthings]
    return arr


def parse_time_from_model_filename(filename: str) -> int | None:
    m = re.search(r"model-[A-Z0-9]+-(\d+)\.pth", filename)
    if not m:
        return None
    return int(m.group(1))


def l2_norm_between_state_dicts(sd_a: dict, sd_b: dict) -> float:
    sq = 0.0
    for key in sd_a:
        diff = (sd_b[key] - sd_a[key]).reshape(-1)
        sq += float(torch.dot(diff, diff))
    # Euclidean (L2) distance: ||w_b - w_a||_2
    # return math.sqrt(sq)
    return sq


def load_delta_w_matrix(exp_dir: Path, tasks: int) -> np.ndarray:
    files = sorted(exp_dir.glob("model-*.pth"))
    time_to_path = {}
    for f in files:
        t = parse_time_from_model_filename(f.name)
        if t is not None:
            time_to_path[t] = f
    times = sorted(time_to_path.keys())
    if len(times) < tasks:
        raise RuntimeError(
            f"Expected at least {tasks} checkpoints in {exp_dir}, found {len(times)}."
        )
    times = times[:tasks]
    states = [torch.load(time_to_path[t], map_location="cpu") for t in times]

    # Rows: reference task 1..(tasks-1), Cols: learned task 1..tasks
    mat = np.full((tasks - 1, tasks), np.nan, dtype=float)
    for ref in range(tasks - 1):
        for cur in range(ref + 1, tasks):
            mat[ref, cur] = l2_norm_between_state_dicts(states[ref], states[cur])
    return mat


def aggregate_group(run_dirs: list[Path], tasks: int, eigenthings: int) -> dict[str, np.ndarray]:
    top_list = []
    eig_list = []
    delta_list = []
    for d in run_dirs:
        top_list.append(load_top_accuracy_matrix(d, tasks))
        eig_list.append(load_hessian_spectra(d, tasks, eigenthings))
        delta_list.append(load_delta_w_matrix(d, tasks))
    top = np.stack(top_list, axis=0)
    eig = np.stack(eig_list, axis=0)
    delta = np.stack(delta_list, axis=0)
    return {
        "top_mean": nanmean_no_warn(top, axis=0),
        "top_std": nanstd_no_warn(top, axis=0),
        "eig_mean": nanmean_no_warn(eig, axis=0),
        "eig_std": nanstd_no_warn(eig, axis=0),
        "delta_mean": nanmean_no_warn(delta, axis=0),
        "delta_std": nanstd_no_warn(delta, axis=0),
        "n_runs": np.array([top.shape[0]], dtype=int),
    }


def draw_figure(panel_data: dict[tuple[str, str], dict[str, np.ndarray]], save_path: Path, tasks: int, eigenthings: int):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.2))

    panel_order = [
        ("permuted", "stable"),
        ("permuted", "plastic"),
        ("rotated", "stable"),
        ("rotated", "plastic"),
    ]
    panel_titles = [
        "(a) Permuted - Stable",
        "(b) Permuted - Plastic",
        "(c) Rotated - Stable",
        "(d) Rotated - Plastic",
    ]

    x_tasks = np.arange(1, tasks + 1)
    x_eigs = np.arange(1, eigenthings + 1)

    # Keep a unified y-scale across all four columns in each row.
    eig_row_max = max(float(np.nanmax(panel_data[key]["eig_mean"])) for key in panel_order)
    eig_row_ylim_max = max(2.0, eig_row_max * 1.15)

    delta_row_candidates = []
    for key in panel_order:
        mean = panel_data[key]["delta_mean"]
        std = panel_data[key]["delta_std"]
        if np.all(np.isnan(std)):
            delta_row_candidates.append(float(np.nanmax(mean)))
        else:
            delta_row_candidates.append(float(np.nanmax(mean + std)))
    delta_row_ylim_max = max(10.0, max(delta_row_candidates) * 1.2)

    for col_idx, key in enumerate(panel_order):
        data = panel_data[key]
        n_runs = int(data["n_runs"][0])

        # Top row: accuracy evolution
        ax = axes[0, col_idx]
        for t in range(1, tasks + 1):
            y = data["top_mean"][t - 1]
            ax.plot(
                x_tasks,
                y,
                marker="o",
                linewidth=2.0,
                color=TASK_COLORS[t],
                label=f"Task {t}",
            )
        ax.set_xlim(0.8, tasks + 0.2)
        ax.set_ylim(50, 100)
        ax.set_xticks(x_tasks)
        ax.set_xlabel("Tasks", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

        # Middle row: Hessian eigenspectrum
        ax = axes[1, col_idx]
        for t in range(1, tasks + 1):
            y = data["eig_mean"][t - 1]
            ax.plot(
                x_eigs,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=TASK_COLORS[t],
                label=f"Task {t}",
            )
        ax.set_xlim(0.8, eigenthings + 0.2)
        ax.set_ylim(0, eig_row_ylim_max)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Eigenvalue Index", fontsize=10)
        ax.set_ylabel("Eigenvalue", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

        # Bottom row: parameter change ||Delta w||
        ax = axes[2, col_idx]
        for t in range(1, tasks):
            y = data["delta_mean"][t - 1]
            ax.plot(
                x_tasks,
                y,
                marker="o",
                linewidth=2.0,
                color=TASK_COLORS[t],
                label=f"Task {t}",
            )
            if n_runs > 1:
                std = data["delta_std"][t - 1]
                ax.fill_between(
                    x_tasks,
                    y - std,
                    y + std,
                    color=TASK_COLORS[t],
                    alpha=0.18,
                    linewidth=0,
                )
        ax.set_xlim(0.8, tasks + 0.2)
        ax.set_ylim(0, delta_row_ylim_max)
        ax.set_xticks(x_tasks)
        ax.set_xlabel("Tasks", fontsize=10)
        ax.set_ylabel(r"$||\Delta w||$", fontsize=10)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Layout first, then place subtitles/caption in free bottom area.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.30, wspace=0.28, hspace=0.42)

    # Column subtitles under bottom row
    for col_idx, title in enumerate(panel_titles):
        pos = axes[2, col_idx].get_position()
        x = (pos.x0 + pos.x1) / 2
        y = pos.y0 - 0.055
        fig.text(x, y, title, ha="center", va="top", fontsize=12, family="serif")

    caption = (
        "Figure 3: Comparison of training regimes for MNIST datasets:\n"
        "(Top) Evolution of validation accuracy for each task; "
        "(Middle) Spectrum of the Hessian for each task;\n"
        "(Bottom) Degree of parameter change."
    )
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=11, family="serif")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise RuntimeError(f"Outputs directory not found: {outputs_root}")

    if args.run_dirs:
        selected = [Path(d) if Path(d).is_absolute() else Path(args.outputs_root) / d for d in args.run_dirs]
    else:
        candidates = discover_runs(outputs_root, args.tasks, args.eigenthings)
        seed_count_hint = parse_seed_count_from_replicate_script(Path("replicate_fig3.sh"))
        selected = select_recent_block(candidates, seed_count_hint)

    selected = [p.resolve() for p in selected]
    groups = split_by_regime(selected)

    print("Selected runs (in replicate_fig3 order):")
    for p in selected:
        print(f"  {p}")

    panel_data = {}
    for key, dirs in groups.items():
        panel_data[key] = aggregate_group(dirs, tasks=args.tasks, eigenthings=args.eigenthings)

    save_path = Path(args.save_path)
    draw_figure(panel_data, save_path, tasks=args.tasks, eigenthings=args.eigenthings)
    print(f"Saved Figure-3-style plot to: {save_path}")


if __name__ == "__main__":
    main()

# python fig3/plot_fig3_from_runs.py --outputs-root outputs --run-dirs 527382 A913AE 91E5D3 FC82F0 --save-path fig3/fig3_reproduced_fixed_mapping.png

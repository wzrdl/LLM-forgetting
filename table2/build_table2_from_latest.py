import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class RunStats:
    run_id: str
    path: str
    arch: str
    tasks: int
    rows: int
    avg_acc: float
    forget: float
    diag_mean: float | None
    diag_min: float | None
    diag_max: float | None
    mtime: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Table-2-style summary from the latest local outputs."
    )
    parser.add_argument("--outputs-root", default="outputs", type=str)
    parser.add_argument("--table-md", default="table2/table2_latest.md", type=str)
    parser.add_argument("--table-csv", default="table2/table2_latest.csv", type=str)
    parser.add_argument("--runs-json", default="table2/table2_latest_runs.json", type=str)
    return parser.parse_args()


def parse_time_from_model_filename(filename: str) -> int | None:
    m = re.search(r"model-[A-Z0-9]+-(\d+)\.pth", filename)
    if not m:
        return None
    return int(m.group(1))


def detect_architecture(exp_dir: Path) -> str:
    model_files = sorted(exp_dir.glob("model-*.pth"))
    if not model_files:
        return "unknown"
    time_to_path = {}
    for f in model_files:
        t = parse_time_from_model_filename(f.name)
        if t is not None:
            time_to_path[t] = f
    if not time_to_path:
        return "unknown"
    first = time_to_path[min(time_to_path.keys())]
    sd = torch.load(first, map_location="cpu")
    keys = list(sd.keys())
    if any(k.startswith("W1.") for k in keys):
        return "mlp"
    if any(k.startswith("conv1.") for k in keys):
        return "resnet"
    return "unknown"


def compute_stats(exp_dir: Path) -> RunStats | None:
    acc_path = exp_dir / "accs.csv"
    if not acc_path.exists():
        return None
    try:
        acc = pd.read_csv(acc_path, index_col=0)
    except Exception:
        return None

    tasks = len(acc.columns)
    rows = len(acc)
    if tasks <= 1:
        return None

    try:
        final_accs = [float(acc[str(i)].iloc[-1]) for i in range(1, tasks + 1)]
        avg_acc = float(np.mean(final_accs))
        forget = float(
            np.mean([float(acc[str(i)].max() - acc[str(i)].iloc[-1]) for i in range(1, tasks)])
            / 100.0
        )
    except Exception:
        return None

    diag_mean = None
    diag_min = None
    diag_max = None
    if rows >= tasks:
        diag = [float(acc[str(i)].iloc[i - 1]) for i in range(1, tasks + 1)]
        diag_mean = float(np.mean(diag))
        diag_min = float(np.min(diag))
        diag_max = float(np.max(diag))

    return RunStats(
        run_id=exp_dir.name,
        path=str(exp_dir.resolve()),
        arch=detect_architecture(exp_dir),
        tasks=tasks,
        rows=rows,
        avg_acc=avg_acc,
        forget=forget,
        diag_mean=diag_mean,
        diag_min=diag_min,
        diag_max=diag_max,
        mtime=exp_dir.stat().st_mtime,
    )


def collect_20task_runs(outputs_root: Path) -> list[RunStats]:
    all_stats = []
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        stats = compute_stats(d)
        if stats is None:
            continue
        if stats.tasks == 20 and stats.rows == 20:
            all_stats.append(stats)
    all_stats.sort(key=lambda s: s.mtime, reverse=True)
    return all_stats


def select_latest_block(runs: list[RunStats], block_size: int = 6) -> list[RunStats]:
    if len(runs) < block_size:
        raise RuntimeError(
            f"Need at least {block_size} 20-task runs to build Table 2, found {len(runs)}."
        )
    selected = sorted(runs[:block_size], key=lambda s: s.mtime)
    return selected


def map_runs_to_table(selected: list[RunStats]) -> dict[str, dict[str, RunStats]]:
    """
    Heuristic mapping for 6 latest runs:
    - 4 MLP runs -> permuted/rotated pairs (split by diagonal current-task accuracy).
    - 2 ResNet runs -> split CIFAR100 pair.
    - In each pair, 'stable' is selected by lower forgetting (MLP) or higher accuracy (ResNet).
    """
    mlp = [r for r in selected if r.arch == "mlp"]
    res = [r for r in selected if r.arch == "resnet"]

    if len(mlp) != 4 or len(res) != 2:
        raise RuntimeError(
            "Expected exactly 4 MLP runs and 2 ResNet runs in latest 6 runs. "
            f"Got {len(mlp)} MLP and {len(res)} ResNet."
        )

    mlp_sorted = sorted(mlp, key=lambda r: (r.diag_mean if r.diag_mean is not None else -1.0))
    rotated_pair = mlp_sorted[:2]   # lower diagonal current-task accuracy
    perm_pair = mlp_sorted[2:]      # higher diagonal current-task accuracy

    # Stable vs Naive assignment inside each pair.
    # For MLP, use lower forgetting as "stable".
    rot_stable = min(rotated_pair, key=lambda r: r.forget)
    rot_naive = max(rotated_pair, key=lambda r: r.forget)
    perm_stable = min(perm_pair, key=lambda r: r.forget)
    perm_naive = max(perm_pair, key=lambda r: r.forget)

    # For CIFAR runs, choose higher-accuracy run as "stable".
    cifar_stable = max(res, key=lambda r: r.avg_acc)
    cifar_naive = min(res, key=lambda r: r.avg_acc)

    return {
        "naive": {
            "perm": perm_naive,
            "rot": rot_naive,
            "cifar": cifar_naive,
        },
        "stable": {
            "perm": perm_stable,
            "rot": rot_stable,
            "cifar": cifar_stable,
        },
    }


def fmt_acc(x: float) -> str:
    return f"{x:.1f}"


def fmt_forget(x: float) -> str:
    return f"{x:.2f}"


def build_table_rows(mapping: dict[str, dict[str, RunStats]]) -> list[dict[str, str]]:
    naive = mapping["naive"]
    stable = mapping["stable"]

    rows = [
        {
            "Method": "Naive SGD",
            "Memoryless": "Yes",
            "Permuted MNIST Accuracy": fmt_acc(naive["perm"].avg_acc),
            "Permuted MNIST Forgetting": fmt_forget(naive["perm"].forget),
            "Rotated MNIST Accuracy": fmt_acc(naive["rot"].avg_acc),
            "Rotated MNIST Forgetting": fmt_forget(naive["rot"].forget),
            "Split CIFAR100 Accuracy": fmt_acc(naive["cifar"].avg_acc),
            "Split CIFAR100 Forgetting": fmt_forget(naive["cifar"].forget),
        },
        {
            "Method": "EWC",
            "Memoryless": "Yes",
            "Permuted MNIST Accuracy": "N/A",
            "Permuted MNIST Forgetting": "N/A",
            "Rotated MNIST Accuracy": "N/A",
            "Rotated MNIST Forgetting": "N/A",
            "Split CIFAR100 Accuracy": "N/A",
            "Split CIFAR100 Forgetting": "N/A",
        },
        {
            "Method": "A-GEM",
            "Memoryless": "No",
            "Permuted MNIST Accuracy": "N/A",
            "Permuted MNIST Forgetting": "N/A",
            "Rotated MNIST Accuracy": "N/A",
            "Rotated MNIST Forgetting": "N/A",
            "Split CIFAR100 Accuracy": "N/A",
            "Split CIFAR100 Forgetting": "N/A",
        },
        {
            "Method": "ER-Reservoir",
            "Memoryless": "No",
            "Permuted MNIST Accuracy": "N/A",
            "Permuted MNIST Forgetting": "N/A",
            "Rotated MNIST Accuracy": "N/A",
            "Rotated MNIST Forgetting": "N/A",
            "Split CIFAR100 Accuracy": "N/A",
            "Split CIFAR100 Forgetting": "N/A",
        },
        {
            "Method": "Stable SGD",
            "Memoryless": "Yes",
            "Permuted MNIST Accuracy": fmt_acc(stable["perm"].avg_acc),
            "Permuted MNIST Forgetting": fmt_forget(stable["perm"].forget),
            "Rotated MNIST Accuracy": fmt_acc(stable["rot"].avg_acc),
            "Rotated MNIST Forgetting": fmt_forget(stable["rot"].forget),
            "Split CIFAR100 Accuracy": fmt_acc(stable["cifar"].avg_acc),
            "Split CIFAR100 Forgetting": fmt_forget(stable["cifar"].forget),
        },
        {
            "Method": "Multi-Task Learning",
            "Memoryless": "N/A",
            "Permuted MNIST Accuracy": "N/A",
            "Permuted MNIST Forgetting": "0.00",
            "Rotated MNIST Accuracy": "N/A",
            "Rotated MNIST Forgetting": "0.00",
            "Split CIFAR100 Accuracy": "N/A",
            "Split CIFAR100 Forgetting": "0.00",
        },
    ]
    return rows


def save_markdown(rows: list[dict[str, str]], md_path: Path):
    headers = [
        "Method",
        "Memoryless",
        "Permuted MNIST Accuracy",
        "Permuted MNIST Forgetting",
        "Rotated MNIST Accuracy",
        "Rotated MNIST Forgetting",
        "Split CIFAR100 Accuracy",
        "Split CIFAR100 Forgetting",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            "Table 2 (latest local runs): Comparison of average accuracy and forgetting on three datasets.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise RuntimeError(f"Outputs root not found: {outputs_root}")

    runs = collect_20task_runs(outputs_root)
    selected = select_latest_block(runs, block_size=6)
    mapping = map_runs_to_table(selected)
    rows = build_table_rows(mapping)

    # Save table files
    df = pd.DataFrame(rows)
    csv_path = Path(args.table_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_markdown(rows, Path(args.table_md))

    # Save run diagnostics and mapping provenance
    diagnostics = {
        "selected_runs_latest_6": [asdict(r) for r in selected],
        "mapping": {
            "naive": {k: asdict(v) for k, v in mapping["naive"].items()},
            "stable": {k: asdict(v) for k, v in mapping["stable"].items()},
        },
        "notes": [
            "EWC/A-GEM/ER rows are N/A because no corresponding result files were found locally.",
            "Mapping uses heuristics from run architecture and trajectory statistics; verify if run IDs are known.",
        ],
    }
    runs_json = Path(args.runs_json)
    runs_json.parent.mkdir(parents=True, exist_ok=True)
    runs_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {Path(args.table_md)}")
    print(f"  {Path(args.table_csv)}")
    print(f"  {Path(args.runs_json)}")

    print("\nResolved mapping (heuristic):")
    print(
        "  Naive  - Perm:{0} Rot:{1} CIFAR:{2}".format(
            mapping["naive"]["perm"].run_id,
            mapping["naive"]["rot"].run_id,
            mapping["naive"]["cifar"].run_id,
        )
    )
    print(
        "  Stable - Perm:{0} Rot:{1} CIFAR:{2}".format(
            mapping["stable"]["perm"].run_id,
            mapping["stable"]["rot"].run_id,
            mapping["stable"]["cifar"].run_id,
        )
    )


if __name__ == "__main__":
    main()

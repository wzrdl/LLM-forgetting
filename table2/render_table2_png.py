import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Render a Table-2-style PNG from CSV.")
    parser.add_argument("--csv", default="table2/table2_latest.csv", type=str)
    parser.add_argument("--out", default="table2/table2_latest.png", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=False)
    df = df.fillna("N/A")

    col_labels = [
        "Method",
        "Memoryless",
        "Permuted MNIST\nAccuracy",
        "Permuted MNIST\nForgetting",
        "Rotated MNIST\nAccuracy",
        "Rotated MNIST\nForgetting",
        "Split CIFAR100\nAccuracy",
        "Split CIFAR100\nForgetting",
    ]

    cell_text = df.values.tolist()

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    ax.set_title(
        "Table 2: Comparison of the average accuracy and forgetting of several methods on three datasets.",
        fontsize=14,
        pad=14,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    # Header styling
    for c in range(len(col_labels)):
        cell = table[(0, c)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
        cell.set_linewidth(0.8)

    # Body borders
    n_rows = len(cell_text)
    n_cols = len(col_labels)
    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            table[(r, c)].set_linewidth(0.5)

    # Slight emphasis on Stable row if present
    for r_idx, row in enumerate(cell_text, start=1):
        if str(row[0]).strip() == "Stable SGD":
            for c in range(n_cols):
                table[(r_idx, c)].set_text_props(weight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table image: {out_path}")


if __name__ == "__main__":
    main()

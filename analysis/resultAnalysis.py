"""Result analysis utilities for experiment outputs.

Usage:
    python -m analysis.resultAnalysis --results results/linear_regression_scz/
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str | Path) -> list[dict]:
    """Load all JSON result files from a directory."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON result files found in {results_dir}")
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def boxplot_r2_scores(results_dir: str | Path, save: bool = True) -> None:
    """
    Create a boxplot of per-seed R² scores from all result files in a directory.

    Each result file contributes one set of per-seed R² values.
    If there is only one file, the boxplot shows the spread across seeds.
    If there are multiple files (e.g., repeated runs), they are shown side by side.
    """
    results_dir = Path(results_dir)
    results = load_results(results_dir)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    labels = []
    data = []

    for res in results:
        experiment = res["experiment"]
        timestamp = res.get("timestamp", "")
        label = f"{experiment}\n({timestamp})" if len(results) > 1 else experiment

        r2_scores = [seed["r2"] for seed in res["per_seed"] if "r2" in seed]
        if not r2_scores:
            print(f"  Skipping {label}: no R² scores found")
            continue

        labels.append(label)
        data.append(r2_scores)

    if not data:
        print("No R² data to plot.")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(data)), 5))
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("#fafafa")

    colors = sns.color_palette("Set2", len(data))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2, color="#555555"),
                    capprops=dict(linewidth=1.2, color="#555555"),
                    medianprops=dict(linewidth=2, color="#333333"),
                    flierprops=dict(marker="o", markersize=5, alpha=0.5,
                                    markerfacecolor="#999999", markeredgecolor="none"))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")

    # Overlay individual seed points
    #for i, scores in enumerate(data):
    #    x = np.random.normal(i + 1, 0.04, size=len(scores))
    #    ax.scatter(x, scores, alpha=0.7, s=30, color="#333333", zorder=5)

    ax.set_title("R² Scores per Seed", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()

    if save:
        out_path = results_dir / "r2_boxplot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Boxplot saved to {out_path}")

    plt.show()


def combined_boxplot(results_root: str | Path, save: bool = True) -> None:
    """
    Scan all sub-directories of *results_root*, pick the newest JSON file per
    method (based on the timestamp in the filename), and draw one boxplot per
    method side by side.
    """
    results_root = Path(results_root)
    method_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not method_dirs:
        raise FileNotFoundError(f"No sub-directories found in {results_root}")

    labels: list[str] = []
    data: list[list[float]] = []

    for method_dir in method_dirs:
        json_files = sorted(method_dir.glob("*.json"))
        if not json_files:
            print(f"  Skipping {method_dir.name}: no JSON files found")
            continue

        # pick the file with the largest timestamp (last when sorted lexicographically)
        newest_file = max(json_files, key=lambda f: f.stem.split("_", maxsplit=len(f.stem))[-2] + "_" + f.stem.split("_")[-1])

        with open(newest_file) as fh:
            res = json.load(fh)

        r2_scores = [seed["r2"] for seed in res.get("per_seed", []) if "r2" in seed]
        if not r2_scores:
            print(f"  Skipping {method_dir.name}: no R² scores found in {newest_file.name}")
            continue

        # build a readable label from the experiment name
        label = res.get("experiment", method_dir.name)
        label = label.replace("_scz", "").replace("_regression", "\nRegression").replace("_dnn", "\nDNN").replace("_", " ").title()
        print(f"  {method_dir.name}: {newest_file.name}  (n={len(r2_scores)} seeds, median R²={float(np.median(r2_scores)):.3f})")

        labels.append(label)
        data.append(r2_scores)

    if not data:
        print("No R² data to plot.")
        return

    # sort by median R² descending
    order = sorted(range(len(data)), key=lambda i: float(np.median(data[i])), reverse=True)
    labels = [labels[i] for i in order]
    data   = [data[i]   for i in order]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("#fafafa")

    colors = sns.color_palette("Set2", len(data))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, color="#555555"),
        capprops=dict(linewidth=1.2, color="#555555"),
        medianprops=dict(linewidth=2, color="#333333"),
        flierprops=dict(marker="o", markersize=5, alpha=0.5,
                        markerfacecolor="#999999", markeredgecolor="none"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")

    ax.axhline(0, color="#cc0000", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("R² Scores per Method (newest run, all seeds)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()

    if save:
        out_path = results_root / "combined_r2_boxplot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nCombined boxplot saved to {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse experiment results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", type=str,
                       help="Path to a single results directory (e.g. results/linear_regression_scz/)")
    group.add_argument("--combined", type=str,
                       help="Path to the root results directory to combine all methods (e.g. results/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the plot to disk")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.combined:
        combined_boxplot(args.combined, save=not args.no_save)
    else:
        boxplot_r2_scores(args.results, save=not args.no_save)


if __name__ == "__main__":
    main()

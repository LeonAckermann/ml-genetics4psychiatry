"""Result analysis utilities for experiment outputs.

Usage:
    python -m analysis.resultAnalysis --results results/linear_regression_scz/
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _summary_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of floats."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute summary stats for empty list")

    q1, median, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(arr)),
    }


def _write_results_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_results(results_dir: str | Path) -> tuple[list[dict], list[Path]]:
    """Load all JSON result files from a directory and return (results, files)."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON result files found in {results_dir}")
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results, files


def boxplot_r2_scores(results_dir: str | Path, save: bool = True) -> None:
    """
    Create a boxplot of per-seed R² scores from all result files in a directory.

    Each result file contributes one set of per-seed R² values.
    If there is only one file, the boxplot shows the spread across seeds.
    If there are multiple files (e.g., repeated runs), they are shown side by side.
    """
    results_dir = Path(results_dir)
    results, files = load_results(results_dir)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    labels: list[str] = []
    data: list[list[float]] = []
    files_used: list[str] = []

    for res, src_file in zip(results, files):
        experiment = res["experiment"]
        timestamp = res.get("timestamp", "")
        label = f"{experiment}\n({timestamp})" if len(results) > 1 else experiment

        r2_scores = [seed["r2"] for seed in res["per_seed"] if "r2" in seed]
        if not r2_scores:
            print(f"  Skipping {label}: no R² scores found")
            continue

        labels.append(label)
        data.append(r2_scores)
        files_used.append(Path(src_file).name)

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

        # Also write a machine-readable summary (one entry per JSON file).
        summary_rows = []
        for label, r2_scores, filename in zip(labels, data, files_used):
            summary_rows.append({
                "label": label,
                "file": filename,
                "r2": _summary_stats(r2_scores),
            })
        _write_results_json(results_dir / "results.json", {
            "mode": "single",
            "results_dir": str(results_dir),
            "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "runs": summary_rows,
        })
        print(f"Summary written to {results_dir / 'results.json'}")

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
    summary: list[dict] = []

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

        summary.append({
            "method_dir": method_dir.name,
            "experiment": res.get("experiment", method_dir.name),
            "file": newest_file.name,
            "timestamp": res.get("timestamp"),
            "r2": _summary_stats(r2_scores),
        })

    if not data:
        print("No R² data to plot.")
        return

    # sort by median R² descending
    order = sorted(range(len(data)), key=lambda i: float(np.median(data[i])), reverse=True)
    labels = [labels[i] for i in order]
    data   = [data[i]   for i in order]
    summary = [summary[i] for i in order]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    # increase resolution
    fig, ax = plt.subplots(figsize=(10, 5))
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
        fig.savefig(out_path, dpi=350, bbox_inches="tight")
        print(f"\nCombined boxplot saved to {out_path}")

        # Also write a machine-readable summary.
        _write_results_json(results_root / "results.json", {
            "mode": "combined",
            "results_root": str(results_root),
            "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "methods": summary,
        })
        print(f"Summary written to {results_root / 'results.json'}")

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

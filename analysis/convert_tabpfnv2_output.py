"""Convert TabPFNv2 print-output logs into the project's standard JSON results format.

This repo's experiment runner (main.py) writes JSON files with:
  - experiment
  - config
  - timestamp
  - per_seed: list[{seed, r2, mse, ...}]
  - aggregated: {r2, mse, ...}

The TabPFNv2 batch script currently prints per-seed metrics like:
  Results for random seed 42:
  Mean Squared Error (MSE): ...
  R² Score: ...

This script parses those blocks and writes an equivalent JSON file so that
analysis/resultAnalysis.py can include TabPFNv2 in combined plots.

Usage:
  python -m analysis.convert_tabpfnv2_output \
    --input results/tabpfnv2_scz/results.txt \
    --experiment tabpfnv2_scz \
    --illness SCZ

By default, output is written to:
  results/<experiment>/<experiment>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


_SEED_RE = re.compile(r"^\s*Results\s+for\s+random\s+seed\s+(\d+)\s*:\s*$")
_MSE_RE = re.compile(r"Mean\s+Squared\s+Error\s*\(MSE\)\s*:\s*([-+0-9.eE]+)")
_R2_RE = re.compile(r"R\s*.*?Score\s*:\s*([-+0-9.eE]+)")


@dataclass(frozen=True)
class ParsedSeedResult:
    seed: int
    mse: float
    r2: float


def _parse_results_text(text: str) -> list[ParsedSeedResult]:
    lines = text.splitlines()

    current_seed: int | None = None
    current_mse: float | None = None
    current_r2: float | None = None

    parsed: list[ParsedSeedResult] = []

    def maybe_commit() -> None:
        nonlocal current_seed, current_mse, current_r2
        if current_seed is None:
            return
        if current_mse is None or current_r2 is None:
            # Incomplete block; keep scanning.
            return
        parsed.append(ParsedSeedResult(seed=current_seed, mse=float(current_mse), r2=float(current_r2)))
        current_seed = None
        current_mse = None
        current_r2 = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = _SEED_RE.match(line)
        if m:
            # New block starts; commit previous (if complete).
            maybe_commit()
            current_seed = int(m.group(1))
            current_mse = None
            current_r2 = None
            continue

        if current_seed is None:
            continue

        mse_m = _MSE_RE.search(line)
        if mse_m:
            current_mse = float(mse_m.group(1))
            continue

        r2_m = _R2_RE.search(line)
        if r2_m:
            current_r2 = float(r2_m.group(1))
            continue

    maybe_commit()

    if not parsed:
        raise ValueError(
            "No seed results found. Expected blocks like 'Results for random seed <N>:' "
            "followed by 'Mean Squared Error (MSE):' and 'R² Score:'."
        )

    # Sort deterministically by seed.
    parsed.sort(key=lambda r: r.seed)
    return parsed


def _default_timestamp(input_path: Path) -> str:
    # Prefer file mtime so repeated conversions are stable.
    ts = datetime.fromtimestamp(input_path.stat().st_mtime)
    return ts.strftime("%Y%m%d_%H%M%S")


def convert_file(
    input_path: str | Path,
    experiment: str,
    illness: str,
    target_column: str | None = None,
    timestamp: str | None = None,
    output_root: str | Path = "results",
) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8", errors="replace")
    seed_results = _parse_results_text(text)

    target = target_column or f"Z_scores_{illness}"

    per_seed = [
        {"r2": float(r.r2), "mse": float(r.mse), "seed": int(r.seed)}
        for r in seed_results
    ]

    r2s = np.array([r.r2 for r in seed_results], dtype=float)
    mses = np.array([r.mse for r in seed_results], dtype=float)

    aggregated = {
        # Note: without raw predictions, we cannot reproduce the exact
        # concatenated-evaluation used in main.py. We instead aggregate across
        # seeds (splits) by arithmetic mean.
        "r2": float(np.mean(r2s)),
        "mse": float(np.mean(mses)),
    }

    cfg = {
        "data": {
            "illness": illness,
            "target": target,
            "test_size": None,
            "n_splits": len(seed_results),
            "save_splits": None,
        },
        "model": {"name": "tabpfnv2"},
        "evaluation": {"metrics": ["r2", "mse"], "binary_threshold": None},
    }

    ts = timestamp or _default_timestamp(input_path)

    output = {
        "experiment": experiment,
        "config": cfg,
        "timestamp": ts,
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    out_dir = Path(output_root) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert TabPFNv2 stdout logs to JSON results")
    p.add_argument("--input", required=True, help="Path to results.txt (TabPFNv2 stdout)")
    p.add_argument("--experiment", default="tabpfnv2_scz", help="Experiment name / results subfolder")
    p.add_argument("--illness", default="SCZ", help="Illness code, e.g. SCZ")
    p.add_argument("--target", default=None, help="Target column name (default: Z_scores_<ILLNESS>)")
    p.add_argument(
        "--timestamp",
        default=None,
        help="Override timestamp (format: YYYYMMDD_HHMMSS). Default: input file mtime.",
    )
    p.add_argument("--output-root", default="results", help="Root results directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = convert_file(
        input_path=args.input,
        experiment=args.experiment,
        illness=args.illness,
        target_column=args.target,
        timestamp=args.timestamp,
        output_root=args.output_root,
    )
    print(f"Wrote JSON results: {out_path}")


if __name__ == "__main__":
    main()

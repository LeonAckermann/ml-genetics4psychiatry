"""Entry point for running experiments from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataloader import DataConfig, load_csv
from model import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML genetics experiments")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--sep", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--index-col", type=int, default=None, help="Index column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(data_path=args.data, sep=args.sep, index_col=args.index_col)
    df = load_csv(config)

    model = BaselineModel(target_column=args.target).fit(df)
    preds = model.predict(df)

    print(f"Loaded {len(df)} rows from {config.data_path}")
    print(f"Baseline mean prediction for '{args.target}': {model.mean_}")
    print(f"Predictions shape: {preds.shape}")


if __name__ == "__main__":
    main()

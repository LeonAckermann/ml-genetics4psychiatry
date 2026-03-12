"""Analysis utilities for datasets."""

from __future__ import annotations

import pandas as pd


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic summary statistics for numeric columns."""

    return df.describe().T


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values per column."""

    missing = df.isna().sum()
    pct = (missing / len(df)) * 100 if len(df) else 0
    return pd.DataFrame({"missing": missing, "missing_pct": pct}).sort_values(
        "missing", ascending=False
    )

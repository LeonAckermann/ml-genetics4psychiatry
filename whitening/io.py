"""Loaders for whitening diagnostics: covariance/intercept matrices and Z-score matrices.

Everything downstream assumes two objects:

* ``Sigma``  – a ``(K, K)`` float array **with a label vector**.  The labels are
  not decoration: check 1 (column alignment) asserts on them, never on shapes.
* ``Z``      – an ``(n_snps, K)`` float array of z-scores whose columns carry the
  same kind of labels, plus an optional allele frame used by the sign checks.

The loaders are format-tolerant on purpose so that pointing the config at the
toy matrix or at the full 192-trait intercept matrix is a one-line change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Strings that appear in this project's matrices where a value is absent.
# The joined GWAS matrices carry the literal string "Null" alongside real NaN;
# reading without this list silently yields an object-dtype column.
NULL_STRINGS = ["", "NA", "NaN", "nan", "Null", "NULL", "null", "None", "."]


# ---------------------------------------------------------------------------
# Sigma
# ---------------------------------------------------------------------------

def load_matrix(
    path: str | Path,
    labels_path: str | Path | None = None,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load a labelled square matrix.

    Supported layouts:

    * ``.csv`` / ``.tsv`` / ``.txt`` – first column holds the row labels, the
      header row holds the column labels (the layout of
      ``official_intercept_matrix.csv``).
    * ``.parquet`` – same, first column is labels.
    * ``.npy`` – unlabelled; ``labels_path`` must point at a text file with one
      label per line.

    Parameters
    ----------
    columns
        Optional subset (and ordering) of traits to keep.  Missing names raise —
        a silently dropped trait is exactly the failure mode check 1 exists for.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        A = np.load(path)
        if labels_path is None:
            raise ValueError(
                f"{path} is unlabelled .npy — supply sigma.labels_path so the "
                "alignment check has label vectors to assert on."
            )
        labels = [ln.strip() for ln in Path(labels_path).read_text().splitlines() if ln.strip()]
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df.set_index(df.columns[0])
        A, labels = df.to_numpy(dtype=float), [str(c) for c in df.columns]
        row_labels = [str(i) for i in df.index]
        _check_square_labels(row_labels, labels, path)
    else:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0, na_values=NULL_STRINGS)
        A = df.to_numpy(dtype=float)
        labels = [str(c) for c in df.columns]
        row_labels = [str(i) for i in df.index]
        _check_square_labels(row_labels, labels, path)

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: expected a square matrix, got shape {A.shape}")
    if len(labels) != A.shape[0]:
        raise ValueError(f"{path}: {len(labels)} labels for a {A.shape} matrix")

    if columns is not None:
        idx = _label_index(labels, columns, str(path))
        A = A[np.ix_(idx, idx)]
        labels = list(columns)

    return A, labels


def _check_square_labels(row_labels: list[str], col_labels: list[str], path: Path) -> None:
    if row_labels != col_labels:
        n_bad = sum(r != c for r, c in zip(row_labels, col_labels))
        raise ValueError(
            f"{path}: row labels do not match column labels ({n_bad} positions "
            "differ). The matrix is not indexed consistently; fix before whitening."
        )


def _label_index(available: list[str], wanted: list[str], where: str) -> list[int]:
    pos = {lab: i for i, lab in enumerate(available)}
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise KeyError(
            f"{where}: {len(missing)} requested trait(s) absent: "
            f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}"
        )
    return [pos[w] for w in wanted]


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def load_z(
    path: str | Path,
    id_cols: list[str] | None = None,
    columns: list[str] | None = None,
    max_snps: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict]:
    """Load an SNP x trait z-score matrix.

    Returns ``(Z, labels, meta, stats)`` where ``meta`` holds the id/allele
    columns for the retained SNPs and ``stats`` records the thinning and the
    complete-case drop.

    Thinning is a deterministic stride (every ``m``-th row), applied *before*
    materialising the value columns, so the 9.6M-row matrix never lands in
    memory whole.  Stride rather than random sampling keeps the subset stable
    across runs and keeps SNPs spread over the genome.
    """
    path = Path(path)
    id_cols = list(id_cols or [])
    stats: dict = {"path": str(path)}

    try:
        import polars as pl
    except ImportError:  # pragma: no cover - polars is in requirements
        pl = None

    if pl is not None and path.suffix.lower() in {".parquet", ".txt", ".tsv", ".csv"}:
        Z, labels, meta, stats = _load_z_polars(pl, path, id_cols, columns, max_snps, stats)
    else:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, na_values=NULL_STRINGS)
        stats["n_rows_file"] = int(len(df))
        if max_snps is not None and len(df) > max_snps:
            stride = int(np.ceil(len(df) / max_snps))
            df = df.iloc[::stride]
            stats["stride"] = stride
        value_cols = columns or [c for c in df.columns if c not in id_cols]
        meta = df[[c for c in id_cols if c in df.columns]].reset_index(drop=True)
        Z = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        labels = list(value_cols)

    # Complete-case: pairwise-complete estimation is a different estimand and
    # would quietly change what the calibration check is testing.
    keep = ~np.isnan(Z).any(axis=1)
    stats["n_rows_loaded"] = int(Z.shape[0])
    stats["n_rows_complete"] = int(keep.sum())
    stats["n_rows_dropped_incomplete"] = int((~keep).sum())
    Z = Z[keep]
    meta = meta.loc[keep].reset_index(drop=True) if len(meta) else meta

    rng = np.random.default_rng(seed)
    stats["seed"] = seed
    del rng  # kept for signature stability; thinning is deterministic

    return Z, labels, meta, stats


def _load_z_polars(pl, path, id_cols, columns, max_snps, stats):
    if path.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        lf = pl.scan_csv(path, separator=sep, null_values=NULL_STRINGS,
                         infer_schema_length=10000)

    schema_names = lf.collect_schema().names()
    value_cols = columns or [c for c in schema_names if c not in id_cols]
    missing = [c for c in value_cols if c not in schema_names]
    if missing:
        raise KeyError(f"{path}: columns absent from z-score file: {', '.join(missing[:10])}")

    n_rows = lf.select(pl.len()).collect().item()
    stats["n_rows_file"] = int(n_rows)

    if max_snps is not None and n_rows > max_snps:
        stride = int(np.ceil(n_rows / max_snps))
        lf = lf.with_row_index("__row").filter(pl.col("__row") % stride == 0).drop("__row")
        stats["stride"] = stride

    keep_ids = [c for c in id_cols if c in schema_names]
    df = lf.select(keep_ids + value_cols).collect()
    meta = df.select(keep_ids).to_pandas() if keep_ids else pd.DataFrame(index=range(len(df)))
    Z = df.select([pl.col(c).cast(pl.Float64, strict=False) for c in value_cols]).to_numpy()
    return Z, list(value_cols), meta, stats


# ---------------------------------------------------------------------------
# Trait groups (optional, used by the sign-coherence check)
# ---------------------------------------------------------------------------

def load_groups(
    path: str | Path,
    key_col: str | list[str],
    group_col: str,
    key_sep: str = "_",
) -> dict[str, str]:
    """Load a trait -> group mapping from a CSV (e.g. reference.csv Category).

    ``key_col`` may be a list (e.g. ``[Consortium, Outcome]``) when the trait
    label used elsewhere is a composite of several reference columns — as it
    is for ``official_intercept_matrix.csv``, whose columns are
    ``{Consortium}_{Outcome}`` (e.g. ``BCT_BASO``) but ``reference.csv`` only
    has them split out.
    """
    df = pd.read_csv(path)
    key_cols = [key_col] if isinstance(key_col, str) else list(key_col)
    missing = [c for c in key_cols + [group_col] if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}, have {list(df.columns)[:15]}")

    keys = df[key_cols[0]].astype(str)
    for c in key_cols[1:]:
        keys = keys + key_sep + df[c].astype(str)

    valid = df[key_cols].notna().all(axis=1)
    return {k: str(v) for k, v, ok in zip(keys, df[group_col], valid) if ok}

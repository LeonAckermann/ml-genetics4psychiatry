"""Data loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import tqdm
import polars as pl

@dataclass
class DataConfig:
    """Configuration for loading datasets."""

    data_path: Path
    sep: str = ","
    index_col: Optional[int] = None
    header: Optional[int] = 0


def load_csv(config: DataConfig) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""

    path = Path(config.data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(
        path,
        sep=config.sep,
        index_col=config.index_col,
        header=config.header,
    )

def load_txt_polars(
    data_path: Path | str,
    *,
    sep: Optional[str] = None,
    chunk_size: Optional[int] = 100000, 
    max_chunks: Optional[int] = None,
    total_chunks: Optional[int] = None,
    null_values: Optional[list[str]] = None,
    ignore_errors: bool = False,
    force_string_columns: Optional[list[str]] = None,
    verbose: bool = False,
    return_polars: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Load a TXT/CSV file using Polars.

    Notes:
    - Many genetics text formats use '.' to denote missing values. Polars will
      error if it infers a numeric dtype but encounters '.' in that column.
      Use ``null_values`` (defaults include '.') to treat those as nulls.
    - If parsing still fails, this function retries with more permissive
      settings (e.g. ``infer_schema_length=0``) and can optionally fall back
      to reading all columns as strings.
    - If you plan to merge/join in pandas, it helps to force key columns (e.g.
      'ID') to string to avoid `int64` vs `object` key mismatches.
    """

    path = Path(data_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if verbose:
        print(f"Starting to load {path} with Polars...")
    
    # Backwards-compat
    if max_chunks is None and total_chunks is not None:
        max_chunks = total_chunks

    # Polars caveat: it does not support regex separators (like r"\s+") natively in read_csv.
    # If your data is standard CSV or TSV, set the exact character.
    separator = sep if sep is not None else "\t"

    # Common missing tokens in GWAS/genotype exports
    effective_null_values = (
        null_values
        if null_values is not None
        else [".", "NA", "N/A", "NaN", "nan", "NULL", "null", ""]
    )

    # FAST PATH: No limits requested. Let Polars use full nativeif 
    # if max_chunks is None and chunk_size is None:
    if verbose:
        print(f"Loading {path.name} (Native Polars Speed)...")

    # BATCHED PATH: Use batched reading for progress bars and max limits
    n_rows = max_chunks * chunk_size if max_chunks and chunk_size else None

    read_kwargs: dict = {
        "separator": separator,
        "batch_size": chunk_size if chunk_size else 100000,
        "n_rows": n_rows,
        "null_values": effective_null_values,
        "ignore_errors": ignore_errors,
    }

    if force_string_columns:
        read_kwargs["schema_overrides"] = {c: pl.Utf8 for c in force_string_columns}

    try:
        df_polars = pl.read_csv(path, **read_kwargs)
    except pl.exceptions.ComputeError as exc:
        if verbose:
            print(
                f"Polars failed to parse {path.name}: {exc}. Retrying with more permissive settings..."
            )

        # Try harder: scan more/entire file to infer schema and allow ragged lines
        try:
            df_polars = pl.read_csv(
                path,
                **read_kwargs,
                infer_schema_length=0,
                truncate_ragged_lines=True,
            )
        except pl.exceptions.ComputeError:
            if verbose:
                print(
                    f"Still failing to parse {path.name}. Falling back to reading all columns as strings."
                )
            df_polars = pl.read_csv(
                path,
                **read_kwargs,
                infer_schema_length=0,
                truncate_ragged_lines=True,
                schema_overrides={},
                dtypes=pl.Utf8,
            )

    if verbose:
        print(f"Finished loading {path.name} with Polars. Total rows: {df_polars.shape[0]}")
    
    if return_polars:
        return df_polars
    # Convert to pandas DataFrame for downstream compatibility
    return df_polars.to_pandas()


def load_txt(
    data_path: Path,
    *,
    sep: Optional[str] = None,
    index_col: Optional[int] = None,
    header: Optional[int] = 0,
    chunk_size: Optional[int] = None,
    max_chunks: Optional[int] = None,
    total_chunks: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a TXT file with a header row and data in rows.

    If ``sep`` is None, the file is treated as whitespace-delimited.

    For quick testing on large files, you can combine ``chunk_size`` with
    ``max_chunks`` (or the backwards-compatible alias ``total_chunks``) to
    stop after reading only the first N chunks.
    """

    if verbose:
        print(f"Starting to load {data_path} with pandas...")

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Backwards-compat: older code used `total_chunks` intending "limit number of chunks".
    if max_chunks is None and total_chunks is not None:
        max_chunks = total_chunks

    if chunk_size is not None:

        chunks = []      

        reader = pd.read_csv(
            path,
            sep=sep if sep is not None else r"\s+",
            index_col=index_col,
            header=header,
            engine="python" if sep is None else "c",
            chunksize=chunk_size,
        )

        if max_chunks is not None:
            with tqdm.tqdm(total=max_chunks, unit="chunk", desc=f"Loading {path.name}") as pbar:
                for i, chunk in enumerate(reader):
                    chunks.append(chunk)
                    pbar.update(1)
                    if i + 1 >= max_chunks:
                        break
        else:
            total_size = path.stat().st_size
            with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Loading {path.name}") as pbar:
                for chunk in reader:
                    chunks.append(chunk)
                    pbar.update(int(chunk.memory_usage(deep=True).sum()))
                #if i == 0:
                #    pbar.set_postfix({"rows": chunk.shape[0], "cols": chunk.shape[1]})
                #else:
                #    pbar.set_postfix({"rows": sum(c.shape[0] for c in chunks), "cols": chunk.shape[1]}) 

        #for chunk in pd.read_csv(
        #    path,
        #    sep=sep if sep is not None else r"\s+",
        #    index_col=index_col,
        #    header=header,
        #    engine="python" if sep is None else "c",
        #    chunksize=chunk_size,
        #):
        #    chunks.append(chunk)
            
        return pd.concat(chunks, ignore_index=True)
    
    else:
        return pd.read_csv(
            path,
            sep=sep if sep is not None else r"\s+",
            index_col=index_col,
            header=header,
            engine="python" if sep is None else "c",
        )



def preprocess(df, target, testsize, seed=42, binary=False, p_value_binary=0.05):
    df = df.drop(columns=["ID"])  # Replace 'ID' with your actual ID column name
    X = df.drop(columns=[target])  # Replace 'target' with your actual target column name
    y = df[target]
    if binary:
        # convert z score to p value on the fly
        from scipy.stats import norm
        import numpy as np
        y = norm.sf(abs(y)) * 2  # two-tailed p-value
        y = (y <= p_value_binary).astype(int)  # binary labels: 1 if significant, 0 otherwise
        # print class distribution 
        print(f"Class distribution (binary with p <= {p_value_binary} as positive):")
        print(sum(y)/len(y)*100, "%")  # print as percentage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def load_phenotype_clumped_data(
    phenotype: str,
    gwas_pheno_path,
    clumps_path=None,
    chunk_size: int = 100000,
    total_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """Build a phenotype-prediction dataset directly from clumps + the raw
    gwas_pheno Z-score matrix, without a precomputed 'final' file.

    Target: ``phenotype``'s own Z-score, renamed to ``Z``.
    Features: every other phenotype's Z-score column in the matrix.
    Rows: restricted to ``phenotype``'s own clumped (LD-independent,
    significant) SNPs, so the row set is driven entirely by that phenotype's
    own GWAS signal and stays valid regardless of which other phenotypes
    have since been added to the matrix.
    """
    gwas_pheno_path = Path(gwas_pheno_path).expanduser().resolve()
    if clumps_path is None:
        clumps_path = Path(f"./data/pipeline/output/clumped_{phenotype}.clumps")
    clumps_path = Path(clumps_path).expanduser().resolve()

    df_clumped = load_txt_polars(clumps_path, chunk_size=chunk_size, total_chunks=total_chunks)
    df_clumped = df_clumped[["ID", "P"]]

    clumped_ids = df_clumped["ID"].tolist()
    # "Null" is the missing-cell token construct_gwas_phenotype writes; declare
    # it (and friends) so polars parses the Z-score columns as f64.
    null_vals = ["Null", ".", "NA", "N/A", "NaN", "nan", "NULL", "null", ""]
    df_pheno = (
        pl.scan_csv(str(gwas_pheno_path), separator="\t", null_values=null_vals)
        .filter(pl.col("ID").is_in(clumped_ids))
        .collect()
    )
    drop_cols = [c for c in ["chrom", "pos", "A0", "A1"] if c in df_pheno.columns]
    df_pheno = df_pheno.drop(drop_cols)

    if phenotype not in df_pheno.columns:
        raise ValueError(f"Phenotype {phenotype!r} not found as a column in {gwas_pheno_path}")
    df_pheno = df_pheno.rename({phenotype: "Z"}).to_pandas()

    return df_clumped.merge(df_pheno, on="ID", how="inner")


_COL_FILTER_PROTECT = ("ID", "Z", "P", "Z_bin")


def filter_columns_by_missing(df, threshold, protect=_COL_FILTER_PROTECT):
    """Drop feature columns whose fraction of missing values exceeds ``threshold``.

    Operates on the assembled sample matrix (rows = SNPs, columns = phenotype
    features). ``protect`` columns (ID, target Z, P, Z_bin) are never dropped.
    Returns ``(filtered_df, stats)``.
    """
    R = len(df)
    feat_cols = [c for c in df.columns if c not in protect]
    dropped = [c for c in feat_cols if (df[c].isnull().mean() if R else 0.0) > threshold]
    keep_cols = [c for c in df.columns if c not in dropped]
    stats = {
        "strategy": "max_col_missing",
        "threshold": threshold,
        "n_features_before": len(feat_cols),
        "n_features_kept": len(feat_cols) - len(dropped),
        "n_features_dropped": len(dropped),
        "dropped_columns": dropped,
    }
    return df[keep_cols], stats


def pick_densest_features(df, min_complete_frac, protect=_COL_FILTER_PROTECT):
    """Pick the largest set of densest feature columns such that the fraction of
    rows complete (non-null across every kept feature) stays >= ``min_complete_frac``.

    Adds features densest-first; the complete-row count only falls as features
    are added, so this walks that curve and stops at the last feature count that
    still clears the target fraction. Returns ``(filtered_df, stats)``.
    """
    R = len(df)
    feat_cols = [c for c in df.columns if c not in protect]
    if R == 0 or not feat_cols:
        return df, {"strategy": "min_complete_frac", "threshold": min_complete_frac,
                    "n_features_before": len(feat_cols), "n_features_kept": len(feat_cols),
                    "n_features_dropped": 0, "achieved_complete_frac": None,
                    "dropped_columns": []}

    present = df[feat_cols].notnull().to_numpy()          # (R, F) bool
    order = np.argsort(present.mean(axis=0), kind="stable")[::-1]   # densest first
    keep_mask = np.ones(R, dtype=bool)
    best_k, best_frac = 0, 1.0
    for k, idx in enumerate(order, start=1):
        keep_mask = keep_mask & present[:, idx]
        frac = keep_mask.sum() / R
        if frac >= min_complete_frac:
            best_k, best_frac = k, frac
        else:
            break                                          # monotonic: never recovers
    if best_k == 0:                                        # even densest single column misses target
        best_k, best_frac = 1, float(present[:, order[0]].mean())
        print(f"  WARNING: min_complete_frac={min_complete_frac} unreachable; "
              f"densest single feature gives only {best_frac:.1%} complete rows")

    kept_feats = {feat_cols[i] for i in order[:best_k]}
    keep_cols = [c for c in df.columns if c in protect or c in kept_feats]
    stats = {
        "strategy": "min_complete_frac",
        "threshold": min_complete_frac,
        "n_features_before": len(feat_cols),
        "n_features_kept": best_k,
        "n_features_dropped": len(feat_cols) - best_k,
        "achieved_complete_frac": best_frac,
        "dropped_columns": [c for c in feat_cols if c not in kept_feats],
    }
    return df[keep_cols], stats


def sample(p_value, sample_size=None, distribution="uniform", seed=42, illness="SCZ", data_path=None, max_bins=100, polars=True, chunk_size=100000, total_chunks=None, sample_p=False, gwas_pheno_path=None, clumps_path=None, max_col_missing=None, min_complete_frac=None):

    if gwas_pheno_path is not None:
        df = load_phenotype_clumped_data(
            illness, gwas_pheno_path, clumps_path=clumps_path,
            chunk_size=chunk_size, total_chunks=total_chunks,
        )
    else:
        if data_path is None:
            data_path = "./data"
        else:
            data_path = "../../data"

        if sample_p:
            data_path = Path(f"{data_path}/pipeline/final_p/aligned_clumped_{illness}.txt").expanduser().resolve()
        else:
            data_path = Path(f"{data_path}/pipeline/final/aligned_clumped_{illness}.txt").expanduser().resolve()
        if polars:
            df = load_txt_polars(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)
        else:
            df = load_txt(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)

    #df = load_txt(Path(f"{data_path}/pipeline/final/aligned_clumped_{illness}.txt"), chunk_size=10000)
    #df = df.copy()
    #significant = df.where(df["P"] <= p_value).dropna()  # Replace 'P' with your actual p-value column name
#
    #non_significant = df.where(df["P"] > p_value).dropna()  # Replace 'P' with your actual p-value column name

    significant     = df.loc[df["P"] <= p_value].copy()
    non_significant = df.loc[df["P"] > p_value].copy()

    # Each branch builds the final sampled `df` and its `output_path`; the shared
    # tail below drops P, prunes sparse feature columns, and writes the file.
    n_bins = min_samples = None
    base_dir = "data/sampled_p" if sample_p else "data/sampled"

    if distribution == "uninformed":
        if sample_size is None:
            sample_size = len(significant)
        sampled_non_significant = non_significant.sample(n=sample_size, random_state=42)
        df = pd.concat([significant, sampled_non_significant], ignore_index=True)
        output_path = Path(f"data/sampled/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
    elif distribution == "uniform":
        n_significant = len(significant)
        n_bins = min(max_bins, n_significant)  # ensure we don't have more bins than samples
        non_significant["Z_bin"] = pd.cut(
            non_significant["Z"],
            bins=n_bins,
            labels=False,          # gives bin indices 0..9
            include_lowest=True
        )
        min_samples = n_significant // n_bins
        sampled_non_significant = non_significant.groupby("Z_bin").apply(
            lambda x: x.sample(n=min_samples, replace=True) if len(x) >= min_samples else x
        ).reset_index(drop=True)
        df = pd.concat([significant, sampled_non_significant], ignore_index=True)
        output_path = Path(f"data/sampled/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
    elif distribution == "low_high":
        non_significant = non_significant.sort_values(by="P", ascending=False)
        non_significant_sampled = non_significant.head(len(significant))
        df = pd.concat([significant, non_significant_sampled], ignore_index=True)
        output_path = Path(f"{base_dir}/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
    elif distribution == "low":
        df = significant.copy()
        output_path = Path(f"{base_dir}/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # Drop the p-value column (no longer needed for modelling).
    if "P" in df.columns:
        df = df.drop(columns=["P"])

    # ── Optional feature-column pruning by missingness ────────────────────────
    # min_complete_frac (auto-pick densest features to hit a complete-row target)
    # takes precedence over max_col_missing (drop columns over a null-rate floor).
    col_filter_stats = None
    if min_complete_frac is not None:
        df, col_filter_stats = pick_densest_features(df, float(min_complete_frac))
        print(f"  column filter (min_complete_frac={min_complete_frac}): kept "
              f"{col_filter_stats['n_features_kept']}/{col_filter_stats['n_features_before']} "
              f"features -> {col_filter_stats['achieved_complete_frac']:.1%} of rows complete")
    elif max_col_missing is not None:
        df, col_filter_stats = filter_columns_by_missing(df, float(max_col_missing))
        print(f"  column filter (max_col_missing={max_col_missing}): kept "
              f"{col_filter_stats['n_features_kept']}/{col_filter_stats['n_features_before']} "
              f"features ({col_filter_stats['n_features_dropped']} dropped)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved sampled data with {distribution} distribution at {output_path} \n ")

    # construct output dictionary with metadata
    output = {
        "illness": illness,
        "p_value": p_value,
        "distribution": distribution,
        "sample_size": len(df),
        "num_significant": len(significant),
        "num_non_significant": len(non_significant),
        "num_bins": n_bins,
        "min_samples_per_bin": min_samples,
        "column_filter": col_filter_stats,
        "output_path": str(output_path),
    }

    return output





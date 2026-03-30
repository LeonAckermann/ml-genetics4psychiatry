"""Data loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    verbose: bool = True,
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



def preprocess(df, target, testsize, seed=42):
    df = df.drop(columns=["ID"])  # Replace 'ID' with your actual ID column name
    X = df.drop(columns=[target])  # Replace 'target' with your actual target column name
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def sample(p_value, sample_size=None, distribution="uniform", seed=42, illness="SCZ", data_path=None, max_bins=100):

    if data_path is None:
        data_path = "./data"
    else:
        data_path = "../../data"

    df = load_txt(Path(f"{data_path}/pipeline/final/aligned_clumped_{illness}.txt"), chunk_size=10000)
    df = df.copy()
    significant = df.where(df["P"] <= p_value).dropna()  # Replace 'P' with your actual p-value column name
    non_significant = df.where(df["P"] > p_value).dropna()  # Replace 'P' with your actual p-value column name

    if distribution == "uninformed":
        if sample_size is None:
            sample_size = len(significant)
        sampled_non_significant = non_significant.sample(n=sample_size, random_state=42)
        df = pd.concat([significant, sampled_non_significant], ignore_index=True)
        df.drop(columns=["P"], inplace=True)  # Drop the p-value column as it's no longer needed
        # save as txt file
        output_path = Path(f"{data_path}/sampled/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved sampled data with uninformed distribution at {output_path} \n ")
    elif distribution == "uniform":
        n_significant = len(significant)
        max_bins = max_bins        
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
        # remove Z_bin column
        sampled_non_significant.drop(columns=["Z_bin"], inplace=True)
        df = pd.concat([significant, sampled_non_significant], ignore_index=True)
        df.drop(columns=["P"], inplace=True)  # Drop the p-value column as it's no longer needed
        # save as txt file
        output_path = Path(f"{data_path}/sampled/{distribution}/sampled_{illness}_p{p_value}.txt").expanduser().resolve()
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved sampled data with uniform distribution at {output_path} \n ")
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # construct output dictionary with metadata
    output = {
        "illness": illness,
        "p_value": p_value,
        "distribution": distribution,
        "sample_size": len(df),
        "num_significant": len(significant),
        "num_non_significant": len(non_significant),
        "num_bins": n_bins if distribution == "uniform" else None,
        "min_samples_per_bin": min_samples if distribution == "uniform" else None,
        "output_path": str(output_path),
    }

    return output





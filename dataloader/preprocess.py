"""Data loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


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


def load_txt(
    data_path: Path,
    *,
    sep: Optional[str] = None,
    index_col: Optional[int] = None,
    header: Optional[int] = 0,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load a TXT file with a header row and data in rows.

    If ``sep`` is None, the file is treated as whitespace-delimited.
    """

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if chunk_size is not None:
        chunks = []
        for chunk in pd.read_csv(
            path,
            sep=sep if sep is not None else r"\s+",
            index_col=index_col,
            header=header,
            engine="python" if sep is None else "c",
            chunksize=chunk_size,
        ):
            chunks.append(chunk)
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





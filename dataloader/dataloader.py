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
    max_chunks: Optional[int] = None,
    columns: bool = False
) -> pd.DataFrame:
    """Load a TXT file with a header row and data in rows.

    If ``sep`` is None, the file is treated as whitespace-delimited.
    """

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    #if chunk_size is None:
    #     return pd.read_csv(
    #        path,
    #        sep=sep if sep is not None else r"\s+",
    #        index_col=index_col,
    #        header=header,
    #        engine="python" if sep is None else "c",
    #    )
    if columns == True:
        return pd.read_csv(path,
            sep=sep if sep is not None else r"\s+",
            index_col=index_col,
            header=header,
            engine="python" if sep is None else "c"
            , nrows=0).columns

def preprocess(df, target, testsize, random_state=42):
    df = df.drop(columns=["ID"])  # Replace 'ID' with your actual ID column name
    X = df.drop(columns=[target])  # Replace 'target' with your actual target column name
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


class GWASDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

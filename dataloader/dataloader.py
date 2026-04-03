
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import polars as pl

from dataloader.preprocess import DataConfig, load_csv, load_txt, preprocess, load_txt_polars
from dataloader.GWASDataset import GWASDataset

from dataloader.preprocess import preprocess

def load_illness_data(illness, in_notebook=True, polars=False, distribution="low", chunk_size=100000, total_chunks=None, p_value="0.001"):
    illnesses = {"MDD": "0.001", "ADHD": "0.001", "ASD": "0.001", "OCD": "0.001", "SCZ": "0.0001", "BIP": "0.001", "AZ": "0.001"}

    if illness not in illnesses:
        raise ValueError(f"Unknown illness: {illness}. Valid options are: {', '.join(illnesses.keys())}")
    pval_threshold = illnesses[illness]
    data_path = f"./data/sampled/{distribution}/sampled_{illness}_p{p_value}.txt"
    if in_notebook:
        data_path = Path("../..") / data_path
    else:
        data_path = Path(data_path).expanduser().resolve()
    print(f"Loading data for illness {illness} at {data_path}"  )
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found for illness {illness} at {data_path}")
    
    if polars:
        df_illness = load_txt_polars(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_illness = load_txt(Path(data_path), chunk_size=chunk_size, total_chunks=total_chunks)
    #df_illness = load_txt(data_path)
    return df_illness

def prepare_data_splits(df, testsize, illness, nsplits, save=True):
    """
    Input: DataFrame, target column name, test size, illness name, number of splits, whether to save splits
    Output: Saves train/test splits as .pt files in data/splits/{illness}_{nsplits}/seed_{seed}/
    """
    seeds = [42 + i for i in range(nsplits)]
    target = f"Z"
    
    for seed in seeds:
        X_train, y_train, X_test, y_test = preprocess(df, target, testsize, seed)
        output_dir = Path(f"./data/splits/{illness}_{nsplits}/seed_{seed}").expanduser().resolve()
        if seed == 42:
            print(f"saved splits for seed {seed} at {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if save:
            import numpy as np
            np.savez(output_dir / f"train_split_{seed}.npz", X=X_train, y=y_train)
            np.savez(output_dir / f"test_split_{seed}.npz", X=X_test, y=y_test)

def load_data_split(illness, nsplits, seed):
    output_dir = Path(f"./data/splits/{illness}_{nsplits}/seed_{seed}").expanduser().resolve()
    train_path = output_dir / f"train_split_{seed}.npz"
    test_path = output_dir / f"test_split_{seed}.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Data splits not found for seed {seed} at {output_dir}")
    import numpy as np
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    return train_data["X"], train_data["y"], test_data["X"], test_data["y"]
